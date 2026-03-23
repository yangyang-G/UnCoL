import os
import random

import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset


class KiPA22(Dataset):
    def __init__(self, base_dir=None, split="train", num=None, transform=None, fold_num=0):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform

        split_file = self._resolve_split_file(fold_num)
        with open(split_file, "r") as f:
            self.sample_list = [item.strip() for item in f.readlines()]

        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def _resolve_split_file(self, fold_num):
        split_files = [
            os.path.join(self._base_dir, "slicelist", "fold_{}".format(fold_num), "{}.txt".format(self.split)),
            os.path.join(self._base_dir, "datalist", "fold_{}".format(fold_num), "{}.txt".format(self.split)),
            os.path.join(self._base_dir, "{}.list".format(self.split)),
            os.path.join(self._base_dir, "{}.txt".format(self.split)),
        ]
        for path in split_files:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            "Cannot find split file for '{}' under '{}'".format(self.split, self._base_dir)
        )

    def _resolve_case_file(self, case):
        case_files = [
            os.path.join(self._base_dir, "{}.h5".format(case)),
            os.path.join(self._base_dir, "data", "{}.h5".format(case)),
        ]
        for path in case_files:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            "Cannot find sample '{}' under '{}'".format(case, self._base_dir)
        )

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        h5f = h5py.File(self._resolve_case_file(case), "r")
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
        label = torch.from_numpy(label.astype(np.int64))
        return {"image": image, "label": label}
