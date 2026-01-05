from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchio as tio
import torch
import numpy as np
import os
import torch
import SimpleITK as sitk
from prefetch_generator import BackgroundGenerator
import h5py
from torchvision.transforms import Compose


class Dataset_Union_ALL(Dataset): 
    def __init__(self, data_dir, mode='train', image_size=128, 
                 transform=None, num=None, 
                 split_num=1, split_idx=0):
        # self.paths = paths
        self.data_dir = data_dir
        self.split_num=split_num
        self.split_idx=split_idx

        self.image_size = image_size
        self.transform = transform
        self.mode = mode
        self.num = num

        self._get_data_list(self.data_dir, self.mode)
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        h5f = h5py.File(self.data_list[index], 'r')
        image, label = h5f['image'][:], h5f['label'][:].astype(np.float32)
        
        # subject = tio.Subject(
        #     image=tio.ScalarImage(tensor=image[np.newaxis,]), 
        #     label=tio.LabelMap(tensor=label[np.newaxis,]))
        
        # if 'Pancreas' or 'TBAD' in self.data_list[index]:
        #     subject = tio.Clamp(-1000,1000)(subject)

        samples = image, label
        if self.transform:
            tr_samples = self.transform(samples)
        image_, label_ = tr_samples
        if self.mode == "train":
            
            return image_.float().clone().detach(), label_.unsqueeze(0).long().clone().detach()
        else:
            return image_.float().clone().detach(), label_.unsqueeze(0).long().clone().detach(), self.data_list[index]   

        # if self.transform:
        #     try:
        #         subject = self.transform(subject)
        #     except:
        #         print(self.data_list[index])
        
        # if self.mode == "train":
        #     return subject.image.data.clone().detach(), subject.label.data.clone().detach()
        # else:
        #     return subject.image.data.clone().detach(), subject.label.data.clone().detach(), self.data_list[index]   

    def _get_data_list(self, data_dir, mode):
        list_path = os.path.join(data_dir,'{}.txt'.format(mode))
        with open(list_path, 'r') as f:
            self.data_list = f.readlines() 

        self.data_list = [item.replace('\n', '') for item in self.data_list]
        if 'LA' in data_dir:
            self.data_list = [os.path.join(self.data_dir, item+'/mri_norm2.h5') for item in self.data_list]
        else:
            self.data_list = [os.path.join(self.data_dir, item+'.h5') for item in self.data_list]

        if self.num is not None:
            self.data_list = self.data_list[:self.num]

class Dataset_Union_ALL_Val(Dataset_Union_ALL):
    def _get_data_list(self, data_dir, mode):
        self.data_list = []

        list_path = os.path.join(data_dir,'{}.txt'.format(mode))
        with open(list_path, 'r') as f:
            self.data_list = f.readlines() 
        self.data_list = [item.replace('\n', '') for item in self.data_list]
        if 'LA' in data_dir:
            self.data_list = [os.path.join(self.data_dir, item+'/mri_norm2.h5') for item in self.data_list]
        else:
            self.data_list = [os.path.join(self.data_dir, item+'.h5') for item in self.data_list]

        self.data_list = self.data_list[self.split_idx::self.split_num]
        if self.num is not None:
            self.data_list = self.data_list[:self.num]


class Union_Dataloader(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class Test_Single(Dataset): 
    def __init__(self, paths, image_size=128, transform=None, threshold=500):
        self.paths = paths

        self._set_file_paths(self.paths)
        self.image_size = image_size
        self.transform = transform
        self.threshold = threshold
    
    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, index):

        sitk_image = sitk.ReadImage(self.image_paths[index])
        sitk_label = sitk.ReadImage(self.label_paths[index])

        if sitk_image.GetOrigin() != sitk_label.GetOrigin():
            sitk_image.SetOrigin(sitk_label.GetOrigin())
        if sitk_image.GetDirection() != sitk_label.GetDirection():
            sitk_image.SetDirection(sitk_label.GetDirection())

        subject = tio.Subject(
            image = tio.ScalarImage.from_sitk(sitk_image),
            label = tio.LabelMap.from_sitk(sitk_label),
        )

        if '/ct_' in self.image_paths[index]:
            subject = tio.Clamp(-1000,1000)(subject)

        if self.transform:
            try:
                subject = self.transform(subject)
            except:
                print(self.image_paths[index])


        if subject.label.data.sum() <= self.threshold:
            return self.__getitem__(np.random.randint(self.__len__()))
        

        return subject.image.data.clone().detach(), subject.label.data.clone().detach(), self.image_paths[index]
    
    def _set_file_paths(self, paths):
        self.image_paths = []
        self.label_paths = []

        self.image_paths.append(paths)
        self.label_paths.append(paths.replace('images', 'labels'))



class CreateNewDataset(Dataset):
    def __init__(self, imgs, plabs, masks, labs, crop_size = (112, 112, 80)):
        self.img = [img.cpu().squeeze().numpy() for img in imgs]
        self.plab = [np.squeeze(plab.cpu().numpy()) for plab in plabs]
        self.mask = [np.squeeze(mask.cpu().numpy()) for mask in masks]
        self.lab = [np.squeeze(lab.cpu().numpy()) for lab in labs]
        self.num = len(self.img)
        self.tr_transform = Compose([
            CenterCrop(crop_size),
            ToTensor()
        ])

    def __getitem__(self, idx):
        samples = self.img[idx], self.plab[idx], self.mask[idx], self.lab[idx]
        samples = self.tr_transform(samples)
        imgs, plab, mask, labs = samples
        return {'image': imgs, 'pseudo':plab.long(), 'mask':plab.long(), 'label': labs.long()}

    def __len__(self):
        return self.num

class MaxCenterCrop(object):
    def __init__(self, scale=16):
        self.output_scale = scale

    def _get_transform(self, label):
        max_v = max(label.shape)
        n = (max_v // self.output_scale)
        output_size = n * self.output_scale

        if label.shape[0] <= output_size[0] or label.shape[1] <= output_size[1] or label.shape[2] <= output_size[2]:
            pw = max((output_size[0] - label.shape[0]) // 2 + 1, 0)
            ph = max((output_size[1] - label.shape[1]) // 2 + 1, 0)
            pd = max((output_size[2] - label.shape[2]) // 2 + 1, 0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        else:
            pw, ph, pd = 0, 0, 0

        (w, h, d) = label.shape
        w1 = int(round((w - output_size[0]) / 2.))
        h1 = int(round((h - output_size[1]) / 2.))
        d1 = int(round((d - output_size[2]) / 2.))

        def do_transform(x):
            if x.shape[0] <= output_size[0] or x.shape[1] <= output_size[1] or x.shape[2] <= output_size[2]:
                x = np.pad(x, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            x = x[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]]
            return x
        return do_transform

    def __call__(self, samples):
        transform = self._get_transform(samples[0])
        return [transform(s) for s in samples]


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def _get_transform(self, label):
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 1, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 1, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 1, 0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        else:
            pw, ph, pd = 0, 0, 0

        (w, h, d) = label.shape
        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        def do_transform(x):
            if x.shape[0] <= self.output_size[0] or x.shape[1] <= self.output_size[1] or x.shape[2] <= self.output_size[2]:
                x = np.pad(x, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            x = x[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return x
        return do_transform

    def __call__(self, samples):
        transform = self._get_transform(samples[0])
        return [transform(s) for s in samples]


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def _get_transform(self, x):
        if x.shape[0] <= self.output_size[0] or x.shape[1] <= self.output_size[1] or x.shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - x.shape[0]) // 2 + 1, 0)
            ph = max((self.output_size[1] - x.shape[1]) // 2 + 1, 0)
            pd = max((self.output_size[2] - x.shape[2]) // 2 + 1, 0)
            x = np.pad(x, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        else:
            pw, ph, pd = 0, 0, 0

        (w, h, d) = x.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        def do_transform(image):
            if image.shape[0] <= self.output_size[0] or image.shape[1] <= self.output_size[1] or image.shape[2] <= self.output_size[2]:
                try:
                    image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
                except Exception as e:
                    print(e)
            image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return image
        return do_transform

    def __call__(self, samples):
        transform = self._get_transform(samples[0])
        return [transform(s) for s in samples]


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def _get_transform(self, x):
        k = np.random.randint(0, 4)
        axis = np.random.randint(0, 2)
        def do_transform(image):
            image = np.rot90(image, k)
            image = np.flip(image, axis=axis).copy()
            return image
        return do_transform

    def __call__(self, samples):
        transform = self._get_transform(samples[0])
        return [transform(s) for s in samples]


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def _get_transform(self, x):
        noise = np.clip(self.sigma * np.random.randn(x.shape[0], x.shape[1], x.shape[2]), -2 * self.sigma, 2 * self.sigma)
        noise = noise + self.mu
        def do_transform(image):
            image = image + noise
            return image
        return do_transform

    def __call__(self, samples):
        transform = self._get_transform(samples[0])
        return [transform(s) if i == 0 else s for i, s in enumerate(samples)]


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample[0]
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        sample = [image] + [*sample[1:]]
        return [torch.from_numpy(s.astype(np.float32)) for s in sample]

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    train_dataset = Dataset_Union_ALL(
        data_dir='/amax/data/luwenjing/P1_UPCoL/Datasets/LA_dataset', 
        mode='train', 
        transform=Compose([
            # RandomRotFlip(),
            RandomCrop([128,128,128]),
            # RandomNoise(),
            ToTensor()
        ]), num=4)

    train_dataloader = Union_Dataloader(
        dataset=train_dataset,
        sampler=None,
        batch_size=1, 
        shuffle=True
    )
    f = plt.figure(figsize=(10,15), dpi = 100)
    ax = 1

    for i,j in train_dataloader:
        print(i.shape)
        print(j.shape)
        f.add_subplot(4, 2, ax); ax+=1
        plt.imshow(i[0,0,:,64,:])
        f.add_subplot(4, 2, ax); ax+=1
        plt.imshow(j[0,0,:,64,:])
        # print(n)
        # continue
    f.savefig('/amax/data/luwenjing/P3_SemiMedSAM/codes/results/SAM-Med3D/test/test.png')
