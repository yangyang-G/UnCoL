import os
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.ndimage.interpolation import zoom
from scipy import ndimage
import random
class ACDC(Dataset):
    """ 加载数据"""
    def __init__(self, base_dir=None, split="train", num=None, transform=None, fold_num=0):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        if self.split == "train":
            with open(os.path.join(self._base_dir, "slicelist/fold_{}/train.txt".format(fold_num)), "r") as f:
                self.sample_list = f.readlines()
        elif self.split == "val":
            with open(os.path.join(self._base_dir, "slicelist/fold_{}/val.txt".format(fold_num)), "r") as f:
                self.sample_list = f.readlines()
        else:
            with open(os.path.join(self._base_dir, "slicelist/fold_{}/test.txt".format(fold_num)), "r") as f:
                self.sample_list = f.readlines()
        
        self.sample_list = [item.strip() for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))
    def __len__(self):
        return len(self.sample_list)
    def __getitem__(self, idx):
        case = self.sample_list[idx]
        h5f = h5py.File(os.path.join(self._base_dir, "{}.h5".format(case)), "r")
        image = h5f["image"][:].squeeze()
        label = h5f["label"][:].squeeze()
        sample = {"image": image, "label": label.astype(np.float32)}
        if self.transform:
            sample = self.transform(sample)
        sample["idx"] = idx
        sample["name"] = case
        return sample
def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image
def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label
class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size
    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.long))
        sample = {"image": image, "label": label}
        return sample