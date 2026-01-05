import os
import torch
from torch.utils.data import Dataset
import h5py
from torch.utils.data.sampler import Sampler
from torchvision.transforms import Compose
import torchio as tio
import numpy as np

class AortaDissection(Dataset):
    """ Aorta Dissection Dataset """

    def __init__(self, data_dir, split, fold_num=0, num=None, scale_range = (0.9,1.1), rotate_degree = 10):
        self.data_dir = data_dir
        self.split = split
        self.scale_range = scale_range
        self.rotate_degree = rotate_degree
        
        if split == 'train':
            self.data_path = os.path.join(data_dir,'ImageTBADlist/AD_{}/train.txt'.format(fold_num))
            self.transform = True
        elif split == 'test':
            self.data_path = os.path.join(data_dir,'ImageTBADlist/AD_{}/test.txt'.format(fold_num))
            self.transform = False
        else:
            self.data_path = os.path.join(data_dir,'ImageTBADlist/AD_{}/val.txt'.format(fold_num))
            self.transform = False

        with open(self.data_path, 'r') as f:
            self.image_list = f.readlines()

        self.image_list = [item.replace('\n', '') for item in self.image_list]
        self.image_list = [os.path.join(self.data_dir, item + '.h5') for item in self.image_list]

        if num is not None:
            self.image_list = self.image_list[:num]

        print("{} set: total {} samples".format(split, len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx] 
        h5f = h5py.File(image_path, 'r')
        image, label = h5f['image'][:], h5f['label'][:].astype(np.float32)
        subject = tio.Subject(image=tio.ScalarImage(tensor=image[np.newaxis,]), \
            label=tio.LabelMap(tensor=label[np.newaxis,]))
        subject = tio.Clamp(-1000,1000)(subject)

        if self.transform:
            RandomAffine = tio.RandomAffine(scales=self.scale_range,degrees=self.rotate_degree)
            subject = RandomAffine(subject)
        image =  subject['image']['data']
        label =  torch.unsqueeze(subject['label']['data'], 0)

        return {'image':image.float(), 'label':label.squeeze().long(), 'name': os.path.basename(image_path)[:-3]}

if __name__ == '__main__':

    data_dir = "ImageTBAD_PATH"

    trainset = AortaDissection(data_dir, fold_num=0, split='train')
    testset = AortaDissection(data_dir, fold_num=0, split='test')
 
    train_sample = trainset[0] 
    test_sample = testset[0]
 
    print(len(trainset), train_sample['image'].shape, train_sample['label'].shape) # 100 torch.Size([1, 128, 128, 128]) torch.Size([128, 128, 128])
    print(len(testset), test_sample['image'].shape, test_sample['label'].shape) # 24 torch.Size([1, 128, 128, 128]) torch.Size([128, 128, 128])

