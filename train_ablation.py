"""
Orange Defect Segmentation - Warmup -> Joint KD training
DataLoader outputs 1024; Student runs at 512; Teacher runs at 1024 then downsample to 512 for KD.
"""

import os
import inspect
import time
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import gc
from models import model_dict
from models.load_ckpt import load_checkpoint
from dataset.OrangeDefectDataloader1 import OrangeDefectLoader
from helper.util import AverageMeter, pred, Distill_one_epoch, train_one_epoch
from helper.loss import IOU
from helper.util import build_target_for_loss,build_logit_for_loss
import random
import shutil


def _get_env(name, default, cast_fn):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return cast_fn(value)
    except ValueError as exc:
        raise ValueError(f"Invalid value for env `{name}`: {value}") from exc


def _extract_logits(output):
    if isinstance(output, (list, tuple)):
        return output[0]
    return output


def build_phase1_optim_sched(model):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=phase1_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay,
        amsgrad=False
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=phase1_epochs, eta_min=phase1_eta_min, last_epoch=-1
    )
    return optimizer, scheduler


def build_phase2_optim_sched(model):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=phase2_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay,
        amsgrad=False
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=phase2_tmax, eta_min=phase2_eta_min, last_epoch=-1
    )
    return optimizer, scheduler


def set_seed(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = False
    if deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.deterministic = False


def worker_init_fn(worker_id):
    """
    DataLoader worker init to make worker RNGs deterministic and different across workers.
    """
    base_seed = torch.initial_seed()
    seed = (base_seed + worker_id) % (2**32 - 1)
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torchvision  # noqa: F401
    except Exception:
        pass


def cleanup():
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


# =========================
# Env-configurable configs
# =========================
inp_size = _get_env("INP_SIZE", 1024, int)
model_name = os.getenv("MODEL_NAME", "mobile_sam_adapter")
ckpt_path = os.getenv("CKPT_PATH", "./checkpoints/mobile_sam.pt")

batch_size = _get_env("BATCH_SIZE", 4, int)
num_workers = _get_env("NUM_WORKERS", 8, int)
epochs_total = _get_env("EPOCHS_TOTAL", 500, int)
phase1_epochs = _get_env("PHASE1_EPOCHS", 200, int)
phase2_epochs = epochs_total - phase1_epochs
test_freq = _get_env("TEST_FREQ", 30, int)

phase1_lr = _get_env("PHASE1_LR", 1e-3, float)
phase2_lr = _get_env("PHASE2_LR", 1e-4, float)
weight_decay = _get_env("WEIGHT_DECAY", 1e-2, float)
phase1_eta_min = _get_env("PHASE1_ETA_MIN", 1e-5, float)
phase2_eta_min = _get_env("PHASE2_ETA_MIN", 1e-6, float)
phase2_tmax = _get_env("PHASE2_TMAX", 50, int)

img_size = inp_size
is_onehot = False

# 按你现在的数据结构默认到 data/orange
dataset_path = os.getenv("DATASET_PATH", "./data/orange")
gt_folder = os.getenv("GT_FOLDER", "./data/orange/masks/")
test_list_file = os.getenv("TEST_LIST_FILE", "./data/orange/imageset/test.txt")
exp_name = os.getenv("EXP_NAME", "phaseA_ablation")


def main():
    set_seed(seed=42, deterministic=True)
    cleanup()

    # 优先给支持 inp_size 的模型传参，避免尺寸不匹配
    # try:
    #     model = model_dict[model_name](inp_size=inp_size)
    # except TypeError:
    #     model = model_dict[model_name]()
    # 1) 一定要是 int，不要字符串
    inp_size = int(os.getenv("INP_SIZE", 1024))

    model_cls = model_dict[model_name]
    sig = inspect.signature(model_cls.__init__)

    if "inp_size" in sig.parameters:
        model = model_cls(inp_size=inp_size)
        print(f"[Init] {model_name} with inp_size={inp_size}")
    else:
        raise RuntimeError(
            f"{model_name}.__init__ does not accept inp_size, "
            f"but ablation is changing INP_SIZE={inp_size}. "
            f"Please patch model constructor first."
        )

    # 2) 强校验：防止默默还是1024
    if hasattr(model, "inp_size"):
        assert int(model.inp_size) == int(inp_size), \
            f"model.inp_size={model.inp_size} != INP_SIZE={inp_size}"

    print("model.inp_size:", model.inp_size)
    print("encoder.img_size:", model.image_encoder.img_size)
    print("encoder.feature_size:", model.image_encoder.feature_size)

    criterion_bce = torch.nn.BCEWithLogitsLoss()
    criterion_iou = IOU(is_onehot)

    optimizer, scheduler = build_phase1_optim_sched(model)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        model.cuda()
        criterion_bce.cuda()
        criterion_iou.cuda()
        cudnn.benchmark = True
        print('cuda available')
    else:
        device = torch.device('cpu')

    if ckpt_path is not None:
        load_checkpoint(model, ckpt_path, device, model_name)

    save_path = './save/{}/{}/'.format(model_name, exp_name)
    os.makedirs(save_path, exist_ok=True)
    pred_path = './save/{}/{}/pred/'.format(model_name, exp_name)
    os.makedirs(pred_path, exist_ok=True)

    copy_files = [
        './models/mobile_sam_adapter.py',
        './models/MobileSAMv2/mobilesamv2/build_sam.py',
        './models/MobileSAMv2/tinyvit/tiny_vit_change.py',
        './models/MobileSAMv2/mobilesamv2/modeling/mask_decoder.py',
        './helper/util.py',
        './train.py'
    ]
    for file in copy_files:
        if os.path.exists(file):
            name = file.split('/')[-1]
            save_file = os.path.join(save_path, name)
            shutil.copy2(file, save_file)

    trainset = OrangeDefectLoader(dataset_path, train=True, test=False, size=img_size, num_classes=2)
    traindataloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, worker_init_fn=worker_init_fn, pin_memory=True
    )

    testset = OrangeDefectLoader(dataset_path, train=False, test=True, size=img_size, num_classes=2)
    testdataloader = DataLoader(
        testset, batch_size=1, shuffle=False,
        num_workers=num_workers, worker_init_fn=worker_init_fn, pin_memory=True
    )

    with open(test_list_file, "r") as f:
        val_img_list = [line.strip() for line in f.readlines()]

    best_loss = 999
    log_file = open(os.path.join(save_path, "train_log.txt"), "a")

    # 初始验证
    log_line = "===================  validation of model  ===================\n"
    print(log_line.strip())
    log_file.write(log_line)
    iou = pred(testdataloader, model, val_img_list, pred_path, gt_folder, inp_size, is_onehot)
    log_line = '✅ Background IoU: {:.4f}\n✅ Foreground IoU: {:.4f}\n✅ Mean IoU (mIoU): {:.4f}\n'.format(
        iou[0], iou[1], iou[2]
    )
    print(log_line.strip())
    log_file.write(log_line)

    print("==> training...")
    current_phase = 1

    for epoch in range(1, epochs_total + 1):
        if epoch == phase1_epochs + 1 and current_phase == 1:
            optimizer, scheduler = build_phase2_optim_sched(model)
            current_phase = 2
            print("==== Switched to Phase-2 optimizer/scheduler (epoch {}) ====".format(epoch))
            log_file.write("==== Switched to Phase-2 optimizer/scheduler (epoch {}) ====\n".format(epoch))

        current_lr = optimizer.param_groups[0]['lr']
        time1 = time.time()
        losses = train_one_epoch(model, traindataloader, is_onehot, criterion_bce, criterion_iou, optimizer)
        scheduler.step()

        time2 = time.time()
        log_line = (
            f"epoch {epoch} (phase {current_phase}), train, lr={current_lr:.6f}, "
            f"mean loss {losses.avg:.3f}, total time {time2 - time1:.2f}\n"
        )
        print(log_line.strip())
        log_file.write(log_line)

        # --------------------- 验证 ---------------------
        time1 = time.time()
        model.eval()
        val_losses = AverageMeter()
        with torch.no_grad():
            for idx, data in enumerate(testdataloader):
                input, target, onehot = data
                if torch.cuda.is_available():
                    input = input.cuda()
                    target = target.unsqueeze(1).cuda()
                    onehot = onehot.cuda()

                # logit = _extract_logits(model(input))
                # if is_onehot:
                #     loss = criterion_bce(logit, onehot.float()) + criterion_iou(logit, onehot.float())
                # else:
                #     #loss = criterion_bce(logit, target.float()) + criterion_iou(logit, target.float())
                #     target_for_loss = build_target_for_loss(logit, target, onehot,is_onehot)
                #     loss = criterion_bce(logit, target_for_loss) + criterion_iou(logit, target_for_loss)
                model_inp_size = getattr(model, "inp_size", None)
                if (
                        isinstance(model_inp_size, int)
                        and (input.size(2) != model_inp_size or input.size(3) != model_inp_size)
                ):
                    input = F.interpolate(input, size=model_inp_size, mode='bilinear', align_corners=False)
                    target = F.interpolate(target.float(), size=model_inp_size, mode='nearest')
                    onehot = F.interpolate(onehot.float(), size=model_inp_size, mode='nearest')
                logit=build_logit_for_loss(_extract_logits(model(input)),is_onehot)
                target_for_loss = build_target_for_loss(logit,target, onehot,is_onehot)
                loss = criterion_bce(logit, target_for_loss)+criterion_iou(logit, target_for_loss)
                val_losses.update(loss.item(), input.size(0))

        time2 = time.time()
        log_line = f"epoch {epoch}, val, mean loss {val_losses.avg:.3f}, total time {time2 - time1:.2f}\n"
        print(log_line.strip())
        log_file.write(log_line)

        if epoch % test_freq == 0:
            log_line = f"========  get student model iou (epoch {epoch})  ========:\n"
            print(log_line.strip())
            log_file.write(log_line)
            iou_list = pred(testdataloader, model, val_img_list, pred_path, gt_folder, inp_size, is_onehot)
            log_line = (
                f"✅ Background IoU: {iou_list[0]:.4f}\n"
                f"✅ Foreground IoU: {iou_list[1]:.4f}\n"
                f"✅ Mean IoU (mIoU): {iou_list[2]:.4f}\n"
            )
            print(log_line.strip())
            log_file.write(log_line)

        # 以验证集loss为准保存最佳
        if best_loss > val_losses.avg:
            best_loss = val_losses.avg
            torch.save(model.state_dict(), os.path.join(save_path, f"{model_name}_best.pth"))

    # 训练结束后对最佳模型做一次最终评测与重命名
    if os.path.exists(os.path.join(save_path, f"{model_name}_best.pth")):
        log_line = f"\n========  test of the best model  ========:\n"
        print(log_line.strip())
        log_file.write(log_line)
        load_checkpoint(model, os.path.join(save_path, f"{model_name}_best.pth"), device, model_name)
        iou_list = pred(testdataloader, model, val_img_list, pred_path, gt_folder, inp_size, is_onehot)
        log_line = (
            f"✅ Background IoU: {iou_list[0]:.4f}\n"
            f"✅ Foreground IoU: {iou_list[1]:.4f}\n"
            f"✅ Mean IoU (mIoU): {iou_list[2]:.4f}\n"
        )
        print(log_line.strip())
        log_file.write(log_line)

        os.rename(
            os.path.join(save_path, f"{model_name}_best.pth"),
            os.path.join(save_path, f'{model_name}_best_loss{best_loss:.4f}.pth')
        )

    log_file.close()


if __name__ == '__main__':
    main()