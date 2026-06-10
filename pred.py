import os
import time
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from helper.util import pred
from models import model_dict
from dataset.OrangeDefectDataloader1 import OrangeDefectLoader
from helper.loss import IOU
def main():
    # --------------------- 固定参数 ---------------------

    num_workers = 8
    model_name = 'mobile_sam_adapter'
    '''
    choices=['PVMNet', 'mobile_sam_adapter', 'sam2_adapter_tiny', 'SegFormerB0', 'SegFormerB1']
    '''
    ckpt_path = './save/mobile_sam_adapter/adapter_with_AGFF4/mobile_sam_adapter_best_loss0.3563.pth'
    model = model_dict[model_name]()

    criterion_bce = torch.nn.BCEWithLogitsLoss()
    criterion_iou = IOU()
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model.cuda()
        criterion_bce.cuda()
        criterion_iou.cuda()
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    weights = torch.load(ckpt_path)
    model.load_state_dict(weights)
    dataset_path = "./data/orange"
    global gt_folder
    gt_folder = "./data/orange/masks/"
    test_list = "./data/orange/imageset/test.txt"
    with open(test_list, "r") as f:
        val_img_list = [line.strip() for line in f.readlines()]
    global pred_save_folder
    pred_save_folder = './save/{}/pred/'.format(model_name)
    pred_gray_folder = './save/{}/gray/'.format(model_name)
    os.makedirs(pred_save_folder, exist_ok=True)

    # --------------------- 数据加载 ---------------------
    testset = OrangeDefectLoader(dataset_path, train=False, test=True, size=1024, num_classes=2)
    testdataloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=num_workers)

    iou_list = pred(testdataloader, model, val_img_list, pred_save_folder, pred_gray_folder, gt_folder, 1024, False)
    log_line = f"✅ Background IoU: {iou_list[0]:.4f}\n✅ Foreground IoU: {iou_list[1]:.4f}\n✅ Mean IoU (mIoU): {iou_list[2]:.4f}\n"
    print(log_line.strip())

if __name__ == '__main__':
    main()
