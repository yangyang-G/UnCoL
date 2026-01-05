#%% set environment
import numpy as np
import matplotlib.pyplot as plt
from os import makedirs
from os.path import join
from tqdm import tqdm
from skimage import transform
import torch
import sys
sys.path.append('.')
from segment_anything import sam_model_registry, SamPredictor
import glob
import os
import argparse
import h5py
import re
import nibabel as nib
import cv2
import matplotlib 
from medpy import metric


color_list = ["gray", "red", "yellow","blue","green"]
cmap = matplotlib.colors.ListedColormap(color_list)
alpha = 0.4

def show_mask(mask, ax, label_id=None, random_color=False):
    # if random_color:
    #     color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    # else:
    #     color = np.array([251/255, 252/255, 30/255, 0.5])
    # h, w = mask.shape[-2:]
    # mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    # ax.imshow(mask_image)
    if len(np.unique(mask)) > 2:
        ax.imshow(mask, cmap=cmap, alpha=alpha)
    else:
        label_id = int(label_id)
        cmap_id = matplotlib.colors.ListedColormap(["gray", color_list[label_id]])
        ax.imshow(mask, cmap=cmap_id, alpha=alpha)


def show_box(box, ax, label_id):
    label_id = int(label_id)

    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color_list[label_id], facecolor=(0,0,0,0), lw=2))     

def cal_metric(gt, pred):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        jc = metric.binary.jc(pred,gt)
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred,gt)
        return np.array([dice, jc, hd95, asd])
    else:
        return np.zeros(4)

def load_nifti_file(file_path):
    """
    Load a NIfTI file and return the image data.
    
    Args:
    file_path (str): Path to the NIfTI file.
    
    Returns:
    numpy.ndarray: The image data.
    """
    nifti_img = nib.load(file_path)
    return nifti_img.get_fdata()

# %%
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--img_path', type=str, default="/amax/data/luwenjing/P1_UPCoL/Datasets/OASIS2d1024")
parser.add_argument('-o', '--seg_path', type=str, default="/amax/data/luwenjing/P4_UKD-SAM/results/SAM")
parser.add_argument('-m', '--model_path', type=str, default="/amax/data/luwenjing/P2_UPCoL_v2/CompExp/MedSAM/work_dir/SAM/sam_vit_b_01ec64.pth")
args = parser.parse_args()
img_path = args.img_path
seg_path = os.path.join(args.seg_path, 'OASIS2D')
makedirs(seg_path, exist_ok=True)

SAM_MODEL_TYPE = "vit_b"
SAM_CKPT_PATH = args.model_path
device = torch.device("cuda:0")


directory = '/amax/data/luwenjing/P1_UPCoL/Datasets/neurite-oasis/oasis' 
image_file_list = []
label_file_list = []
for folder in os.listdir(directory):
    if 'OASIS' in folder:
        for filename in os.listdir(os.path.join(directory,folder)):
            if filename.endswith('.nii') or filename.endswith('.nii.gz'):
                if 'slice_norm' in filename:
                    norm_path = os.path.join(directory, folder, filename)
                    mask4_path = norm_path.replace('norm', 'seg4')
                    if os.path.exists(mask4_path):
                        label_file_list.append(mask4_path)

print('find {} files'.format(len(label_file_list)))
image_size = 1024
bbox_shift = 20


# %% set up model
sam_model = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CKPT_PATH)
sam_model.to(device=device)
predictor = SamPredictor(sam_model)

total_scores = []

#%% predict npz files and save results
for gt_path_file in tqdm(label_file_list):
    gt_file_name = os.path.basename(gt_path_file)
    task_folder = os.path.basename(os.path.dirname(gt_path_file))
    os.makedirs(join(seg_path, task_folder), exist_ok=True)

    gt = load_nifti_file(gt_path_file).squeeze()
    img_3c = load_nifti_file(re.sub('seg4', 'norm', gt_path_file)).squeeze()

    segs = np.zeros_like(gt, dtype=np.uint8) 
    if len(img_3c.shape) == 2:
        img_3c = np.repeat(img_3c[:,:, None], 3, axis=-1)

    resize_img_1024 = cv2.resize(img_3c, (1024, 1024), interpolation=cv2.INTER_CUBIC)
    predictor.set_image(resize_img_1024.astype(np.uint8)) # conpute the image embedding only once

    gt_1024 = cv2.resize(gt, (1024, 1024), interpolation=cv2.INTER_NEAREST) # (1024, 1024)
    segs_1024 = np.zeros_like(gt_1024)
    label_ids = np.unique(gt)[1:]

    f = plt.figure(figsize=(12,6), dpi=300)
    num_class = len(label_ids)+1
    f.add_subplot(3,num_class,1)
    plt.xticks([]) 
    plt.yticks([]) 
    plt.imshow(resize_img_1024,cmap='gray')

    scores = []

    with open(join(seg_path,'SAM_results.txt'), "a") as file:
        file.write("{}\n".format(join(seg_path, task_folder)))

    for label_id in label_ids:
        gt_1024_label_id = np.uint8(gt_1024 == label_id) # only one label, (256, 256)
        y_indices, x_indices = np.where(gt_1024_label_id > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt_1024_label_id.shape
        x_min = max(0, x_min - bbox_shift)
        x_max = min(W, x_max + bbox_shift)
        y_min = max(0, y_min - bbox_shift)
        y_max = min(H, y_max + bbox_shift)
        bboxes1024 = np.array([x_min, y_min, x_max, y_max])
        
        sam_mask, _, _ = predictor.predict(point_coords=None, point_labels=None, box=bboxes1024[None, :], multimask_output=False) #1024x1024, bool
        segs_1024[sam_mask[0]>0] = label_id
        gt_mask = np.uint8(gt_1024 == label_id)
        segs_mask = np.uint8(segs_1024 == label_id)

        sam_mask = transform.resize(sam_mask[0].astype(np.uint8), (gt.shape[-2], gt.shape[-1]), order=0, preserve_range=True, mode='constant', anti_aliasing=False) # (256, 256)
        segs[sam_mask > 0] = label_id

        score = cal_metric(gt_mask, segs_mask)
        scores.append(score)
        with open(join(seg_path,'SAM_results.txt'), "a") as file:
            file.write("class {}, {:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n".format(label_id, score[0]*100, score[1]*100, score[2], score[3]))

        ax = int(1+label_id)
        f.add_subplot(3,num_class,ax)
        plt.xticks([]) 
        plt.yticks([]) 
        plt.imshow(resize_img_1024,cmap='gray')
        show_box(bboxes1024,plt.gca(),label_id) 

        ax = int(1+label_id+num_class)
        f.add_subplot(3,num_class,ax)
        plt.xticks([]) 
        plt.yticks([]) 
        plt.imshow(resize_img_1024,cmap='gray')
        show_mask(segs_mask,plt.gca(), label_id, random_color=True) 

        ax = int(1+label_id+2*num_class)
        f.add_subplot(3,num_class,ax)
        plt.xticks([]) 
        plt.yticks([]) 
        plt.imshow(resize_img_1024,cmap='gray')
        plt.xlabel('class {}: {:.2f}%'.format(label_id, score[0]*100))
        show_mask(gt_mask,plt.gca(), label_id, random_color=True) 

    scores = np.array(scores)

    ax = int(1+num_class)
    f.add_subplot(3,num_class,ax)
    plt.xticks([]) 
    plt.yticks([]) 
    plt.imshow(resize_img_1024,cmap='gray')
    show_mask(segs_1024,plt.gca(), random_color=True) 

    ax = int(1+2*num_class)
    f.add_subplot(3,num_class,ax)
    plt.xticks([]) 
    plt.yticks([]) 
    plt.imshow(resize_img_1024,cmap='gray')
    plt.xlabel('avg: {:.2f}%'.format(scores[:, 0].mean()*100))
    show_mask(gt_1024,plt.gca(), random_color=True) 
    
    total_scores.append(scores)
    f.suptitle('{} - SAM'.format(task_folder))
    plt.savefig(join(seg_path, task_folder, 'SAM_plt.png'))
    print('Save at: {}'.format(join(seg_path, task_folder, 'SAM_plt.png')))
    with open(join(seg_path,'SAM_results.txt'), "a") as file:
        file.write("avg, {:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n\n".format(
            scores[:, 0].mean()*100, scores[:, 1].mean()*100, 
            scores[:, 2].mean(), scores[:, 3].mean()))

total_scores = np.array(total_scores)

for i in rang(len(label_ids)):
    with open(join(seg_path,'SAM_results.txt'), "a") as file:
        file.write("class {}, {:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n\n".format(i+1,
            total_scores[:, i, 0].mean()*100, total_scores[:, i, 1].mean()*100, 
            total_scores[:, i, 2].mean(), total_scores[:, i, 2].mean()))

with open(join(seg_path,'SAM_results.txt'), "a") as file:
    file.write("overall, {:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n".format(
        total_scores[:, :, 0].mean()*100, total_scores[:, :, 1].mean()*100, 
        total_scores[:, :, 2].mean(), total_scores[:, :, 3].mean()))
# #         np.savez_compressed(join(seg_path, task_folder, npz_name), segs=segs, gts=gt)
 