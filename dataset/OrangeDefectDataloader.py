import numpy as np
import random
from scipy import ndimage
import torch
from torch.utils.data import Dataset, DataLoader

class Resize:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, img, mask):
        img = ndimage.zoom(img, (self.height / img.shape[0], self.width / img.shape[1], 1), order=1)
        mask = ndimage.zoom(mask, (self.height / mask.shape[0], self.width / mask.shape[1], 1), order=0)
        return img, mask

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img, mask):
        if random.random() < self.p:
            img = np.flip(img, axis=1)
            mask = np.flip(mask, axis=1)
        return img, mask

class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img, mask):
        if random.random() < self.p:
            img = np.flip(img, axis=0)
            mask = np.flip(mask, axis=0)
        return img, mask

class RandomRotation:
    def __init__(self, p=0.5, degree=(0, 360)):
        self.p = p
        self.degree = degree
    def __call__(self, img, mask):
        if random.random() < self.p:
            angle = random.uniform(*self.degree)
            img = ndimage.rotate(img, angle, reshape=False, order=1, mode='reflect')
            mask = ndimage.rotate(mask, angle, reshape=False, order=0, mode='nearest')
        return img, mask

class ColorAug:
    def __init__(self, brightness=0.3, contrast=0.2, saturation=0.2, p=0.2):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.p = p
    def __call__(self, img, mask):
        if random.random() < self.p:
            # 亮度
            img = img * (1 + random.uniform(-self.brightness, self.brightness))
            # 对比度
            mean = img.mean(axis=(0, 1), keepdims=True)
            img = (img - mean) * (1 + random.uniform(-self.contrast, self.contrast)) + mean
            # 饱和度 (仅适用于彩色图像)
            gray = img.mean(axis=2, keepdims=True)
            img = (img - gray) * (1 + random.uniform(-self.saturation, self.saturation)) + gray
            img = np.clip(img, 0, 1)
        return img, mask
import os
import cv2
from tqdm import tqdm
from torchvision import transforms

class OrangeDefectLoader(Dataset):
    def __init__(self, data_dir, train=True, test=False, size=1024, num_classes=2):
        self.data_dir = data_dir
        self.train = train
        self.test = test
        self.size = size
        self.num_classes = num_classes
        train_txt_path = './data/orange/imageset/train.txt'
        test_txt_path = './data/orange/imageset/test.txt'
        val_txt_path = './data/orange/imageset/test.txt'
        self.img_path = './data/orange/images/'
        self.mask_path = './data/orange/masks/'
        compute_mean_std = False
        if compute_mean_std:
            self.imgs, self.masks = self.__load_txt_list__(train_txt_path)
            self.mean, self.std = self._compute_mean_std()
        else:
            self.mean = [0.46301203, 0.44775238, 0.31114299]
            self.std = [0.28111694, 0.22697894, 0.23791684]

        if train:
            self.imgs, self.masks = self.__load_txt_list__(train_txt_path)
        elif test:
            self.imgs, self.masks = self.__load_txt_list__(test_txt_path)
        else:
            self.imgs, self.masks = self.__load_txt_list__(val_txt_path)

        # self.color_jitter = transforms.ColorJitter(
        #     brightness=0.3,
        #     contrast=0.2,
        #     saturation=0.2,
        #     hue=0
        # )
        self.color_jitter = ColorAug(brightness=0.3, contrast=0.2, saturation=0.2, p=0.2)
        # 归一化（ImageNet 或你自己数据集统计）
        self.normalize = transforms.Normalize(self.mean, self.std)

    def __getitem__(self, idx):
        # 读取 image
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if self.train:
            if random.random() < 0.2:
                image, mask = self.color_jitter(image, mask)
        if self.train:
            image, mask = self._augment(image, mask)
        else:
            image = cv2.resize(image, (self.size, self.size))
            mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        # numpy -> tensor
        image = transforms.ToTensor()(image)   # [3, H, W]
        # if self.train:
        #     if random.random() < 0.2:
        #         image = self.color_jitter(image)
        image = self.normalize(image)

        mask = torch.from_numpy(mask).long()   # [H, W]

        # 二分类（缺陷 / 背景）
        if self.num_classes == 2:
            mask = (mask > 0).long()
        # 计算 one-hot 编码 [num_classes, H, W]
        one_hot = torch.nn.functional.one_hot(mask, num_classes=self.num_classes).permute(2, 0, 1).float()

        return image.float(), mask, one_hot

    def __load_txt_list__(self, txt_path):
        with open(txt_path, "r") as f:
            names = [line.strip() for line in f.readlines()]
        imgs = [os.path.join(self.img_path, name+".png") for name in names]
        masks = [os.path.join(self.mask_path, name+".png") for name in names]
        return imgs, masks

    def __len__(self):
        return len(self.imgs)

    def _compute_mean_std(self):
        print("Computing dataset mean & std...")
        channel_sum = np.zeros(3)
        channel_squared_sum = np.zeros(3)
        num_pixels = 0

        for img_path in tqdm(self.imgs):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.size, self.size))
            img = img.astype(np.float32) / 255.0  # [0,1]

            channel_sum += img.sum(axis=(0, 1))
            channel_squared_sum += (img ** 2).sum(axis=(0, 1))
            num_pixels += img.shape[0] * img.shape[1]

        mean = channel_sum / num_pixels
        std = np.sqrt(channel_squared_sum / num_pixels - mean ** 2)

        print("Mean:", mean)
        print("Std :", std)

        return mean.tolist(), std.tolist()

    def _augment(self, image, mask):
        # Resize
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        # 随机翻转
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        if random.random() < 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)

        # 随机旋转
        if random.random() < 0.5:
            angle = random.uniform(-15, 15)
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)

        return image, mask

if __name__ == '__main__':
    trainset = OrangeDefectLoader("../data/orange", train=False, test=False, size=224, num_classes=2)
    trainloader = DataLoader(trainset, batch_size=8, shuffle=True)

    for img, mask, onehot in trainloader:
        print(img.shape, mask.shape, onehot.shape)
        print(mask.max(), mask.min())
        print(onehot.max(), onehot.min())
        break