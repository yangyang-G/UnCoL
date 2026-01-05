# -*- coding: utf-8 -*-

"""
usage example:
python MedSAM_Inference.py -i assets/img_demo.png -o ./ --box "[95,255,190,350]"

"""

# %% load environment
import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
import torch
from MedSAM.segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
import argparse

import glob
import os
import argparse
import h5py
import re
import nibabel as nib
import cv2
import matplotlib 
from medpy import metric
from tqdm import tqdm
from skimage import transform
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

# visualization functions
# source: https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
# change color to avoid red and green


image_size = 1024
bbox_shift = 20

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _, upsampling_embeddings = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
        middle_output=True,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)
    return low_res_pred, upsampling_embeddings


def preprocess_image(img_tensor, gt_tensor):
    img_tensor = img_tensor.float()
    if img_tensor.size(1) == 1:  # 假设数据格式为(B, C, H, W)
        img_tensor = img_tensor.repeat(1, 3, 1, 1)
    resize = transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.BICUBIC)
    img_1024 = resize(img_tensor)

    min_vals = img_1024.view(img_1024.size(0), -1).min(1, keepdim=True)[0]
    max_vals = img_1024.view(img_1024.size(0), -1).max(1, keepdim=True)[0]
    img_1024 = (img_1024 - min_vals.view(img_1024.size(0), 1, 1, 1)) / torch.clamp(max_vals.view(img_1024.size(0), 1, 1, 1) - min_vals.view(img_1024.size(0), 1, 1, 1), min=1e-8)
    
    gt_1024 = TF.resize(gt_tensor, (1024, 1024), interpolation=transforms.InterpolationMode.NEAREST)

    return img_1024, gt_1024



def get_medsam_features(image, label, medsam_model, num_classes):
    batch_size, _, HH, WW = image.shape
    img_1024, gt_1024 = preprocess_image(image, label)

    bg_predictions = torch.zeros_like(label.unsqueeze(1)).to(label.device)+0.5
    fg_predictions = []
    posterior_features = []

    with torch.no_grad():
        # image_embedding = medsam_model.image_encoder(img_1024)  # (B, 256, 64, 64) (B, 768, 64, 64)
        image_embedding, encoder_features = medsam_model.image_encoder(img_1024, hidden_out=True)  # (B, 256, 64, 64) (B, 768, 64, 64)
    label_ids = torch.unique(gt_1024)[1:]

    for label_id in range(1, num_classes):
        bboxes1024 = []
        for i_batch in range(batch_size):
            gt_1024_label_id = (gt_1024[i_batch] == label_id).float() # only one label, (1024, 1024)
            y_indices, x_indices = torch.where(gt_1024_label_id > 0)
            H, W = gt_1024_label_id.shape

            if (len(y_indices) > 0) and (len(x_indices) > 0):
                x_min, x_max = torch.min(x_indices).item(), torch.max(x_indices).item()
                y_min, y_max = torch.min(y_indices).item(), torch.max(y_indices).item()
                # add perturbation to bounding box coordinates
                x_min = max(0, x_min - bbox_shift)
                x_max = min(W, x_max + bbox_shift)
                y_min = max(0, y_min - bbox_shift)
                y_max = min(H, y_max + bbox_shift)
            else:
                x_min, x_max, y_min, y_max = 0, W, 0, H
            bboxes1024.append([[x_min, y_min, x_max, y_max]])
        bboxes1024 = np.array(bboxes1024)
        medsam_seg, upsampling_embeddings = medsam_inference(medsam_model, image_embedding, bboxes1024, HH, WW) # B, 1, H, W
        fg_predictions.append(medsam_seg)
        posterior_features.append(upsampling_embeddings)
    fg_predictions = torch.cat(fg_predictions, dim=1)
    sum_foreground = torch.sum(fg_predictions, dim=1, keepdim=True)
    bg_predictions = torch.clamp(1.0 - sum_foreground, min=0.0)
    p_all = torch.cat([bg_predictions, fg_predictions], dim=1)
    p_normalized = torch.softmax(p_all, dim=1)
    masks = torch.argmax(p_normalized, dim=1)
    
    if label_id < 3:
        posterior_features = torch.stack(posterior_features, dim=1).squeeze(1)
    else:
        posterior_features = torch.stack(posterior_features, dim=1).flatten(1,2)
    return encoder_features, posterior_features, masks, p_normalized
        


def get_medsam_hidden(image, medsam_model):
    batch_size, _, HH, WW = image.shape

    img_tensor = image.float()
    if img_tensor.size(1) == 1:  # 假设数据格式为(B, C, H, W)
        img_tensor = img_tensor.repeat(1, 3, 1, 1)
    resize = transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.BICUBIC)
    img_1024 = resize(img_tensor)

    min_vals = img_1024.view(img_1024.size(0), -1).min(1, keepdim=True)[0]
    max_vals = img_1024.view(img_1024.size(0), -1).max(1, keepdim=True)[0]
    img_1024 = (img_1024 - min_vals.view(img_1024.size(0), 1, 1, 1)) / torch.clamp(max_vals.view(img_1024.size(0), 1, 1, 1) - min_vals.view(img_1024.size(0), 1, 1, 1), min=1e-8)

    with torch.no_grad():
        # image_embedding = medsam_model.image_encoder(img_1024)  # (B, 256, 64, 64) (B, 768, 64, 64)
        _, encoder_features = medsam_model.image_encoder(img_1024, hidden_out=True)  # (B, 256, 64, 64) (B, 768, 64, 64)

    return encoder_features
