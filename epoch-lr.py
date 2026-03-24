"""
Orange Defect Segmentation - Warmup -> Joint KD training
DataLoader outputs 1024; Student runs at 512; Teacher runs at 1024 then downsample to 512 for KD.
"""

import os
import time
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import gc
import matplotlib.pyplot as plt

from models import model_dict
from models.load_ckpt import load_checkpoint
from dataset.OrangeDefectDataloader1 import OrangeDefectLoader
from helper.util import AverageMeter, pred_train, Distill_one_epoch, train_one_epoch
from helper.loss import IOU
import random
import shutil

def build_phase1_optim_sched(model):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=phase1_epochs, eta_min=1e-5, last_epoch=-1
    )
    return optimizer, scheduler


def build_phase2_optim_sched(model):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False
    )
    # 按你的要求使用 CosineAnnealingLR(T_max=50)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=1e-6, last_epoch=-1
    )
    return optimizer, scheduler

def set_seed(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn settings
    cudnn.benchmark = False
    if deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.deterministic = False

# def worker_init_fn():
#     seed = 42
#     np.random.seed(seed)
#     random.seed(seed)

def worker_init_fn(worker_id):
    """
    DataLoader worker init to make worker RNGs deterministic and different across workers.

    Called in each worker process with that worker's id (0..num_workers-1).
    """
    # torch.initial_seed() returns a different base seed for each worker/process.
    base_seed = torch.initial_seed()  # 64-bit value
    # make it fit into 32-bit numpy/python seeds and mix with worker_id to avoid collisions
    seed = (base_seed + worker_id) % (2**32 - 1)
    np.random.seed(seed)
    random.seed(seed)
    # If you use libraries that require their own seeds, set them here (e.g., for pillow/augmentations)
    try:
        import torchvision
        # torchvision transforms that rely on random will follow python/np seeds
    except Exception:
        pass


def cleanup():
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


inp_size = 1024
model_name = 'mobile_sam_adapter'
'''
[
'PVMNet': PVMNet,
'PVMNetPlus': PVMNet_plus,
'sam2_adapter_tiny': './checkpoints/sam2.1_hiera_tiny.pt',
'SegFormerB0': make_SegFormerB0,
'SegFormerB1': make_SegFormerB1,
'sam2_adapter_light': SAM2_Adapter_Light,
'unet': UNet,
'mobile_sam_adapter': './checkpoints/mobile_sam.pt',
]
'''
ckpt_path = './checkpoints/mobile_sam.pt'

# train set
batch_size = 4
num_workers = 8
epochs_total = 500
phase1_epochs = 200                # 你的两阶段优化器/调度器切换点
phase2_epochs = epochs_total - phase1_epochs
test_freq = 30

# data set
img_size = inp_size
is_onehot = False
dataset_path = "./data/orange"
gt_folder = "./data/orange/masks/"
test_list_file = "./data/orange/imageset/test.txt"

def main():
    set_seed(seed=42, deterministic=True)
    cleanup()
    model = model_dict[model_name]()
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


    print("==> training...")
    current_phase = 1
    lr = []
    for epoch in range(1, epochs_total + 1):

        if epoch == phase1_epochs + 1 and current_phase == 1:
            optimizer, scheduler = build_phase2_optim_sched(model)
            current_phase = 2

            print("==== Switched to Phase-2 optimizer/scheduler (epoch {}) ====".format(epoch))

        current_lr = optimizer.param_groups[0]['lr']
        print(epoch, current_lr)
        lr.append(current_lr)
        scheduler.step()
    return lr


if __name__ == '__main__':
    lr = main()
    epochs = list(range(1, len(lr) + 1))


    plt.figure()
    plt.plot(epochs, lr)
    plt.axvline(
        x=phase1_epochs,
        linestyle='--',
        linewidth=1
    )
    plt.text(
        phase1_epochs + 2,
        max(lr) * 0.8,
        "Phase-2 Start",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule (Phase-1 → Phase-2)")
    plt.grid(True)
    plt.savefig("lr_schedule.png", dpi=300, bbox_inches="tight")
    plt.show()

