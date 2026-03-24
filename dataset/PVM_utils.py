import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import numpy as np
import os
import math
import random
import cv2
from PIL import Image
import logging
import logging.handlers
from matplotlib import pyplot as plt

from scipy.ndimage import zoom
import SimpleITK as sitk
from medpy import metric


def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_logger(name, log_dir):
    '''
    Args:
        name(str): name of logger
        log_dir(str): path of log
    '''

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = logging.handlers.TimedRotatingFileHandler(info_name,
                                                             when='D',
                                                             encoding='utf-8')
    info_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    info_handler.setFormatter(formatter)

    logger.addHandler(info_handler)

    return logger


def log_config_info(config, logger):
    config_dict = config.__dict__
    log_info = f'#----------Config info----------#'
    logger.info(log_info)
    for k, v in config_dict.items():
        if k[0] == '_':
            continue
        else:
            log_info = f'{k}: {v},'
            logger.info(log_info)


def get_optimizer(config, model):
    assert config.opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop',
                          'SGD'], 'Unsupported optimizer!'

    if config.opt == 'Adadelta':
        return torch.optim.Adadelta(
            model.parameters(),
            lr=config.lr,
            rho=config.rho,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'Adagrad':
        return torch.optim.Adagrad(
            model.parameters(),
            lr=config.lr,
            lr_decay=config.lr_decay,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad
        )
    elif config.opt == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad
        )
    elif config.opt == 'Adamax':
        return torch.optim.Adamax(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'ASGD':
        return torch.optim.ASGD(
            model.parameters(),
            lr=config.lr,
            lambd=config.lambd,
            alpha=config.alpha,
            t0=config.t0,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'RMSprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            alpha=config.alpha,
            eps=config.eps,
            centered=config.centered,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'Rprop':
        return torch.optim.Rprop(
            model.parameters(),
            lr=config.lr,
            etas=config.etas,
            step_sizes=config.step_sizes,
        )
    elif config.opt == 'SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            dampening=config.dampening,
            nesterov=config.nesterov
        )
    else:  # default opt is SGD
        return torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=0.05,
        )


def get_scheduler(config, optimizer):
    assert config.sch in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
                          'CosineAnnealingWarmRestarts', 'WP_MultiStepLR', 'WP_CosineLR'], 'Unsupported scheduler!'
    if config.sch == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.milestones,
            gamma=config.gamma,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.gamma,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.T_max,
            eta_min=config.eta_min,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.mode,
            factor=config.factor,
            patience=config.patience,
            threshold=config.threshold,
            threshold_mode=config.threshold_mode,
            cooldown=config.cooldown,
            min_lr=config.min_lr,
            eps=config.eps
        )
    elif config.sch == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.T_0,
            T_mult=config.T_mult,
            eta_min=config.eta_min,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'WP_MultiStepLR':
        lr_func = lambda \
            epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else config.gamma ** len(
            [m for m in config.milestones if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    elif config.sch == 'WP_CosineLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else 0.5 * (
                math.cos((epoch - config.warm_up_epochs) / (config.epochs - config.warm_up_epochs) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    return scheduler


def save_imgs(img, msk, msk_pred, i, save_path, datasets, threshold=0.5, test_data_name=None):
    img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = img / 255. if img.max() > 1.1 else img
    if datasets == 'retinal':
        msk = np.squeeze(msk, axis=0)
        msk_pred = np.squeeze(msk_pred, axis=0)
    else:
        msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)
        msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)

    plt.figure(figsize=(7, 15))

    plt.subplot(3, 1, 1)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(3, 1, 2)
    plt.imshow(msk, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 1, 3)
    plt.imshow(msk_pred, cmap='gray')
    plt.axis('off')

    if test_data_name is not None:
        save_path = save_path + test_data_name + '_'
    plt.savefig(save_path + str(i) + '.png')
    plt.close()


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth) / (pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum() / size

        return dice_loss


class nDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(nDiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class CeDiceLoss(nn.Module):
    def __init__(self, num_classes, loss_weight=[0.4, 0.6]):
        super(CeDiceLoss, self).__init__()
        self.celoss = nn.CrossEntropyLoss()
        self.diceloss = nDiceLoss(num_classes)
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        loss_ce = self.celoss(pred, target[:].long())
        loss_dice = self.diceloss(pred, target, softmax=True)
        loss = self.loss_weight[0] * loss_ce + self.loss_weight[1] * loss_dice
        return loss


class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss


class GT_BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(GT_BceDiceLoss, self).__init__()
        self.bcedice = BceDiceLoss(wb, wd)

    def forward(self, gt_pre, out, target):
        bcediceloss = self.bcedice(out, target)
        gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1 = gt_pre
        gt_loss = self.bcedice(gt_pre5, target) * 0.1 + self.bcedice(gt_pre4, target) * 0.2 + self.bcedice(gt_pre3,
                                                                                                           target) * 0.3 + self.bcedice(
            gt_pre2, target) * 0.4 + self.bcedice(gt_pre1, target) * 0.5
        return bcediceloss + gt_loss


class myToTensor:
    def __init__(self):
        pass

    def __call__(self, data):
        image, mask = data
        return torch.FloatTensor(image).permute(2, 0, 1), torch.IntTensor(mask)  # .permute(2,0,1)


class WeightedIoULoss(nn.Module):
    def __init__(self, class_weights, wb, wd):
        super(WeightedIoULoss, self).__init__()
        self.class_weights = class_weights  # 输入的类别权重
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, logits, labels):
        bceloss = self.bce(logits, labels)
        diceloss = self.dice(logits, labels)

        loss = self.wd * diceloss + self.wb * bceloss
        smooth = 1e-6  # 防止分母为0
        # 对logits应用sigmoid，将logits转化为概率
        probs = torch.sigmoid(logits)

        # 将labels转换为float类型
        labels = labels.float()

        # 逐类别计算交集
        intersection = (probs * labels).sum(dim=(2, 3))  # 对 height 和 width 维度求和
        # 逐类别计算并集
        union = (probs + labels).sum(dim=(2, 3)) - intersection

        # 逐类别IoU
        iou_per_class = (intersection + smooth) / (union + smooth)

        # 计算加权IoU损失
        weighted_iou_loss = (1 - iou_per_class) * self.class_weights

        # 对所有类别取平均损失
        return 0.2 * weighted_iou_loss.mean() + 0.8 * loss


class myRandomResizedCrop:
    def __init__(self, size_h, size_w, p=0.5, scale=(0.08, 1.0)):
        """
        Args:
            size (tuple or int): 输出图像的目标大小，通常是一个元组 (height, width)。
            scale (tuple of float): 随机裁剪的比例范围。
            ratio (tuple of float): 随机裁剪区域的宽高比范围。
        """
        self.p = p
        self.size_h = size_h
        self.size_w = size_w
        self.scale = scale

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            width, height = image.size()[1], image.size()[2]
            h = math.floor(random.uniform(*self.scale) * height)
            w = h
            top = random.randint(0, height - h)
            left = random.randint(0, width - w)

            image = TF.crop(image, top, left, h, w)
            mask = TF.crop(mask, top, left, h, w)

        image = TF.resize(image, [self.size_h, self.size_w], interpolation=TF._interpolation_modes_from_int(0))
        mask = TF.resize(mask, [self.size_h, self.size_w], interpolation=TF._interpolation_modes_from_int(0))
        return image, mask


# def create_circular_mask(height, width, center=None, radius=170):
#     if center is None:
#         center = (int(width / 2), int(height / 2))
#     Y, X = np.ogrid[:height, :width]
#     dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
#     mask = dist_from_center <= radius
#     return mask.astype(np.uint8)


class myColoraug:
    def __init__(self, brightness, contrast, saturation, p=0.5):
        self.color_jitter_transform = transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=0
        )
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p and mask.max() == 3:
            return self.color_jitter_transform(image), mask
        else:
            return image, mask


# class myfade:
#     def __init__(self, p=0.3, radius=170, blend_ratio=[0.2, 0.5]):
#         self.p = p
#         self.radius = radius
#         self.blend_ratio = blend_ratio
#
#     def __call__(self, data):
#         image, mask = data  # 待增强的数据
#         if random.random() < self.p and (mask == 2).any():
#             ratio = random.uniform(self.blend_ratio[0], self.blend_ratio[1])
#             background = (mask <= 0).astype(np.uint8)  # 背景部分，0 或 1
#             defect = (mask == 2).astype(np.uint8)  # 缺陷部分，0 或 1
#             image_background = image * np.expand_dims(1 - defect, axis=2)
#             image_defect = image * np.expand_dims(defect, axis=2)
#             circular_mask = create_circular_mask(mask.shape[0], mask.shape[1], radius=int(0.33*min(mask.shape[0], mask.shape[1])))
#             # circular_mask = create_circular_mask(512, 512, radius=self.radius)
#             # circular_mask长宽为512，在以radius为半径的圆内数值为1，圆外为0
#             background = background * circular_mask
#             image_aug = image * np.expand_dims(background, axis=2)
#             # 1. 计算image_aug三个通道的均值
#             image_aug_mean = np.sum(image_aug, axis=(0, 1)) / np.sum(background)
#             # 2. 将均值按比例（self.blend_ratio）与image_defect叠加
#             blended_defect = ratio * image_aug_mean * np.expand_dims(defect, axis=2) + (1 - ratio) * image_defect
#             # 3. 将得到的与image_defect叠加与image_background相加得到新的image
#             image = image_background + blended_defect
#         return image, mask



def create_circular_mask(h, w, center=None, radius=None):
    # 你原来有的 create_circular_mask 函数；如果没有，请用这个实现
    if center is None:  # use center of image
        center = (int(w/2), int(h/2))
    if radius is None:
        radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = (X - center[0])**2 + (Y - center[1])**2
    mask = dist_from_center <= radius*radius
    return mask.astype(np.uint8)  # 0/1 mask

class myfade:
    def __init__(self, p=0.3, radius=170, blend_ratio=(0.2, 0.5), debug=False):
        self.p = p
        self.radius = radius
        self.blend_ratio = blend_ratio
        self.debug = debug

    def __call__(self, data):
        image, mask = data  # image: HxWx3 (uint8 or float), mask: HxW (values 0,1,2,3)
        # 要在 float 上做运算
        img = image.astype(np.float32)
        msk = mask  # keep int

        if random.random() < self.p and (msk == 2).any():
            ratio = random.uniform(self.blend_ratio[0], self.blend_ratio[1])

            # defect 和 background 二值 mask (uint8 0/1)
            defect = (msk == 2).astype(np.uint8)            # 1 where defect
            background = (msk <= 0).astype(np.uint8)        # 1 where background (按你的定义)

            # circular mask (0/1)
            h, w = msk.shape
            circular_mask = create_circular_mask(h, w, radius=int(0.33 * min(h, w)))
            # apply circular mask only to background
            background_mask = background * circular_mask   # 0/1

            # 如果 background_mask 中没有 1，就跳过增强以避免除0
            bg_count = int(background_mask.sum())
            if bg_count <= 0:
                if self.debug:
                    print("myfade: no background pixels under circular mask; skip augmentation")
                return image, mask

            # image_background: pixels outside defect remain; image_defect: defect area
            image_background = img * (1 - defect[:, :, None])  # [H,W,3], float
            image_defect = img * (defect[:, :, None])          # [H,W,3], float

            # image_aug = image * background_mask  -> but need channel expand
            bg_mask_3c = np.expand_dims(background_mask, axis=2)  # [H,W,1]

            # sum across pixels where background_mask==1, per channel
            # safer: use boolean indexing to avoid dividing by zero
            # shape: (n_pixels, 3)
            pixels = img[background_mask.astype(bool)]  # shape (bg_count, 3)
            if pixels.size == 0:
                if self.debug:
                    print("myfade: no pixels found after indexing, skip")
                return image, mask

            image_aug_mean = pixels.mean(axis=0)  # [3,] float

            # blended_defect: ratio * image_aug_mean * defect + (1-ratio)*image_defect
            # image_aug_mean broadcast to HxWx3 via defect mask
            blended_defect = ratio * image_aug_mean.reshape(1, 1, 3) * (defect[:, :, None].astype(np.float32)) \
                             + (1.0 - ratio) * image_defect

            # combine background and blended defect
            out_img = image_background + blended_defect

            # clamp to valid range (assume input 0..255)
            out_img = np.clip(out_img, 0.0, 255.0)

            # cast back to original dtype (if original was uint8)
            if image.dtype == np.uint8:
                out_img = out_img.astype(np.uint8)
            else:
                # keep float32
                out_img = out_img.astype(image.dtype)

            if self.debug:
                print(f"myfade applied: ratio={ratio:.3f}, bg_count={bg_count}, mean={image_aug_mean}")

            return out_img, mask

        # not applied -> return original
        return image, mask


class mydata_aug:
    def __init__(self, folder, image_list, p=0.1, radius=170, blend_ratio=0.6):
        self.folder = folder
        self.image_list = image_list
        self.p = p
        self.radius = radius
        self.blend_ratio = blend_ratio

    def __call__(self, data):
        image, mask = data  # 待增强的数据
        if random.random() < self.p:
            image_name = random.choice(self.image_list)
            self.image_path = os.path.join(self.folder, image_name)
            image_defect = np.array(Image.open(self.image_path).convert('RGB'))
            mask_defect = np.array(Image.open(self.image_path.replace('images', 'masks').replace('jpg', 'png')))
            # 加载风伤类型的图片和掩码

            mask_aug = (mask_defect > 0).astype(np.uint8)  # 0 和 1
            circular_mask = create_circular_mask(512, 512, radius=self.radius)
            # circular_mask长宽为512，在以radius为半径的圆内数值为1，圆外为0

            mask_aug = self.blend_ratio * mask_aug * circular_mask
            # 限制增强范围和粘贴比例，mask_aug数值范围为（0，1）

            mask_aug[(mask_aug < mask)] = 0
            # 设置待增强数据中缺陷处增强效果为0，
            # mask中背景为0、三类缺陷数值分别为1，2，3，所以mask中缺陷处的数值一定大于mask_aug

            image = image * (1 - np.expand_dims(mask_aug, axis=2)) + image_defect * np.expand_dims(mask_aug, axis=2)
            mask_aug = np.where(mask_aug > 0.5, 1, 0)
            mask = mask * (1 - mask_aug) + mask_defect * mask_aug
        return (image, mask)


class myResize:
    def __init__(self, size_h=256, size_w=256):
        self.size_h = size_h
        self.size_w = size_w

    def __call__(self, data):
        image, mask = data
        if mask.size().__len__() == 2:
            mask = mask.unsqueeze(0)
        return TF.resize(image, [self.size_h, self.size_w]), TF.resize(mask, [self.size_h, self.size_w],
                                                                       interpolation=TF._interpolation_modes_from_int(
                                                                           0))


class mytestResize:
    def __init__(self, size_h=256, size_w=256):
        self.size_h = size_h
        self.size_w = size_w

    def __call__(self, data):
        image, mask = data
        mask = mask.unsqueeze(0)
        return TF.resize(image, [self.size_h, self.size_w]), TF.resize(mask, [self.size_h, self.size_w],
                                                                       interpolation=TF._interpolation_modes_from_int(
                                                                           0))


class myRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            # return np.flip(image, axis=1), np.flip(mask, axis=1)
            return TF.hflip(image), TF.hflip(mask)
        else:
            return image, mask


class myRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return TF.vflip(image), TF.vflip(mask)
            # return np.flip(image, axis=0), np.flip(mask, axis=0)
        else:
            return image, mask


class myRandomRotation:
    def __init__(self, p=0.5, degree=[0, 360]):
        self.degree = degree
        self.p = p

    def __call__(self, data):
        image, mask = data
        mask = mask.unsqueeze(0)
        if random.random() < self.p:
            angle = random.uniform(self.degree[0], self.degree[1])
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)
        return image, mask


import cv2
import numpy as np
import random


class myNormalize:
    def __init__(self, data_name, model):
        self.data_name = data_name
        if data_name == 'VOC2007' or data_name == 'grape':
            if model == 'train':
                self.mean = 131.44703211343557
                self.std = 58.763782109802165
            elif model == 'val':
                self.mean = 131.15040504192484
                self.std = 58.645976338912014
            else:
                self.mean = 131.03664428009384
                self.std = 58.36421969848735
        else:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]

    def __call__(self, data):
        img, msk = data
        if self.data_name == 'coco':
            img_normalized = TF.normalize(img, mean=self.mean, std=self.std)
        else:
            img_normalized = (img - self.mean) / self.std
        return img_normalized, msk


import numpy as np


class RGB_HSV:
    def __call__(self, data):
        """
        将 RGB 图像 NumPy 数组转换为 HSV 图像 NumPy 数组。

        Args:
            data (tuple): 包含图像和掩码的元组 (image, mask)
                          image: NumPy array, shape (H, W, C), RGB, range [0, 1]
                          mask: NumPy array, shape (H, W)

        Returns:
            tuple: 转换后的 HSV 图像和掩码
        """
        image, mask = data

        # 确保图像在 [0, 1] 范围内，并转换为浮点型
        image = image.astype(np.float32)
        image = np.clip(image, 0, 1)

        # 分离 R, G, B 通道
        R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]

        # 计算 Cmax, Cmin, Delta
        Cmax = np.maximum.reduce([R, G, B])
        Cmin = np.minimum.reduce([R, G, B])
        Delta = Cmax - Cmin

        # 初始化 H, S, V
        H = np.zeros_like(Cmax)
        S = np.zeros_like(Cmax)
        V = Cmax.copy()

        # 防止除以零
        Delta_nonzero = Delta > 1e-6  # 使用一个小阈值防止浮点误差

        # Hue calculation
        # Mask for different conditions
        mask_r = (Cmax == R) & Delta_nonzero
        mask_g = (Cmax == G) & Delta_nonzero
        mask_b = (Cmax == B) & Delta_nonzero

        # Compute Hue
        H[mask_r] = (60 * ((G[mask_r] - B[mask_r]) / Delta[mask_r])) % 360
        H[mask_g] = (60 * ((B[mask_g] - R[mask_g]) / Delta[mask_g]) + 120) % 360
        H[mask_b] = (60 * ((R[mask_b] - G[mask_b]) / Delta[mask_b]) + 240) % 360

        # Normalize Hue to [0, 1]
        H = H / 360.0

        # Saturation calculation
        S[Cmax > 0] = Delta[Cmax > 0] / Cmax[Cmax > 0]

        # Stack H, S, V channels
        hsv_image = np.stack([H, S, V], axis=2)  # Shape: (H, W, C)

        return hsv_image, mask


# class myNormalize:
#     def __init__(self, data_name, train=True):
#         self.data_name = data_name
#         if data_name == 'isic18':
#             if train:
#                 self.mean = 157.561
#                 self.std = 26.706
#             else:
#                 self.mean = 149.034
#                 self.std = 32.022
#         elif data_name == 'isic17':
#             if train:
#                 self.mean = 159.922
#                 self.std = 28.871
#             else:
#                 self.mean = 148.429
#                 self.std = 25.748
#         elif data_name == 'isic18_82':
#             if train:
#                 self.mean = 156.2899
#                 self.std = 26.5457
#             else:
#                 self.mean = 149.8485
#                 self.std = 35.3346
#
#     def __call__(self, data):
#         img, msk = data
#         if self.data_name == 'orange':
#             # my_transforms_Normalize = transforms.Normalize((0.4740, 0.4283, 0.2585), (0.2735, 0.2197, 0.2326))
#             my_transforms_Normalize = transforms.Normalize((0.4740, 0.4477, 0.3148), (0.2761, 0.2208, 0.2309))
#             my_totensor = transforms.ToTensor()
#             img = my_totensor(img)
#             img_normalized = my_transforms_Normalize(img)
#             img_normalized = img_normalized.numpy()
#             img_normalized = np.transpose(img_normalized, (1, 2, 0))
#
#         elif self.data_name == 'VOC2007':
#             # my_transforms_Normalize = transforms.Normalize((0.6908, 0.5431, 0.3110), (0.1025, 0.1204, 0.2454))
#             my_totensor = transforms.ToTensor()
#             img = my_totensor(img)
#             # img_normalized = my_transforms_Normalize(img)
#             img_normalized = img.numpy()
#             img_normalized = np.transpose(img_normalized, (1, 2, 0))
#
#         else:
#             img_normalized = (img-self.mean)/self.std
#             img_normalized = ((img_normalized - np.min(img_normalized))
#                                 / (np.max(img_normalized)-np.min(img_normalized))) * 255.
#         return img_normalized, msk


from thop import profile  ## 导入thop模块


def cal_params_flops(model, size, logger):
    input = torch.randn(1, 3, size, size).cuda()
    flops, params = profile(model, inputs=(input,))
    print('flops', flops / 1e9)  ## 打印计算量
    print('params', params / 1e6)  ## 打印参数量

    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total / 1e6))
    logger.info(f'flops: {flops / 1e9}, params: {params / 1e6}, Total params: : {total / 1e6:.4f}')


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256],
                       test_save_path=None, case=None, z_spacing=1, val_or_test=False):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None and val_or_test is True:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/' + case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/' + case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/' + case + "_gt.nii.gz")
        # cv2.imwrite(test_save_path + '/'+case + '.png', prediction*255)
    return metric_list