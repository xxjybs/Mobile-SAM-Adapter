from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import random
# import h5py
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from scipy import ndimage
from PIL import Image
import cv2


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def preprocess_input(image):
    image /= 255.0
    return image

from dataset.PVM_utils import *
class NPY_datasets(Dataset):
    def __init__(self, path_Data, num_classes=3, train=True, test=False):
        super(NPY_datasets, self)
        self.train = train
        self.input_shape = [256, 256]
        self.num_classes = num_classes
        if train:
            images_list = sorted(os.listdir(path_Data + 'train/images/'))
            masks_list = sorted(os.listdir(path_Data + 'train/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data + 'train/images/' + images_list[i]
                mask_path = path_Data + 'train/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
                self.transformer = transforms.Compose([
                    # mydata_aug(folder, image_list),
                    myfade(p=0.2, radius=150, blend_ratio=[0.2, 0.5]),
                    myNormalize('VOC2007', model='train'),
                    myToTensor(),
                    myRandomHorizontalFlip(p=0.5),
                    myRandomVerticalFlip(p=0.5),
                    myRandomRotation(p=0.5, degree=[0, 360]),
                    myResize(1024, 1024),
                    myColoraug(0.3, 0.2, 0.2, p=0.2),
                ])
        elif test:
            images_list = sorted(os.listdir(path_Data + 'test/images/'))
            masks_list = sorted(os.listdir(path_Data + 'test/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data + 'test/images/' + images_list[i]
                mask_path = path_Data + 'test/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = transforms.Compose([
                myNormalize('VOC2007', model='test'),
                myToTensor(),
                mytestResize(1024, 1024),
            ])

        else:
            images_list = sorted(os.listdir(path_Data + 'val/images/'))
            masks_list = sorted(os.listdir(path_Data + 'val/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data + 'val/images/' + images_list[i]
                mask_path = path_Data + 'val/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = transforms.Compose([
                myNormalize('VOC2007', model='val'),
                myToTensor(),
                mytestResize(1024, 1024),
            ])

    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.array(Image.open(msk_path))
        # if self.config.datasets == 'coco':
        #     msk = msk.astype(np.int32) - 91
        #     np.where(msk > 0, msk, 0)
        mask = np.array(Image.open(msk_path).convert('RGB'))
        img, msk = self.transformer((img, msk))
        seg_labels = np.eye(self.num_classes)[msk.reshape([-1])]
        seg_labels = seg_labels.reshape((int(msk.shape[1]), int(msk.shape[2]), self.num_classes))
        seg_labels = np.transpose(seg_labels, (2, 0, 1))
        return img, seg_labels  # , msk

    def __len__(self):
        return len(self.data)


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


# class Synapse_dataset(Dataset):
#     def __init__(self, base_dir, list_dir, split, transform=None):
#         self.transform = transform  # using transform in torch!
#         self.split = split
#         self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
#         self.data_dir = base_dir
#
#     def __len__(self):
#         return len(self.sample_list)
#
#     def __getitem__(self, idx):
#         if self.split == "train":
#             slice_name = self.sample_list[idx].strip('\n')
#             data_path = os.path.join(self.data_dir, slice_name + '.npz')
#             data = np.load(data_path)
#             image, label = data['image'], data['label']
#         else:
#             vol_name = self.sample_list[idx].strip('\n')
#             filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
#             data = h5py.File(filepath)
#             image, label = data['image'][:], data['label'][:]
#
#         sample = {'image': image, 'label': label}
#         if self.transform:
#             sample = self.transform(sample)
#         sample['case_name'] = self.sample_list[idx].strip('\n')
#         return sample

