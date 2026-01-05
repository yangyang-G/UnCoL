import math
from glob import glob

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from medpy import metric
from tqdm import tqdm
import logging
import os

from monai.losses import DiceCELoss
import time
from utils.train_utils import get_sam_img_embed
import pandas as pd

sam_dice_ce_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1, AMC=False, BCP=False):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                               (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0],
                                   ys:ys+patch_size[1], zs:zs+patch_size[2]]

                test_patch = np.expand_dims(np.expand_dims(
                    test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y1 = net(test_patch)
                    # y1, _ = net(test_patch)
                    # ensemble
                    if AMC:
                        y2 = 0.25 * (y1[0]+ y1[1]+ y1[2]+ y1[3])
                        y = torch.softmax(y2, dim=1)
                    elif BCP:
                        y = torch.softmax(y1[0], dim=1)
                    else:
                        y = torch.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,
                              hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map = score_map[:, wl_pad:wl_pad +
                              w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
    return label_map


def cal_metric(gt, pred):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        jc = metric.binary.jc(pred,gt)
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred,gt)
        return np.array([dice, jc, hd95, asd])
    else:
        return np.zeros(4)

def test_single_case_3d(net, sam_model, image, stride_xy, stride_z, patch_size, num_classes=1, AMC=False):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                               (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0],
                                   ys:ys+patch_size[1], zs:zs+patch_size[2]]

                test_patch = np.expand_dims(np.expand_dims(
                    test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    test_img_embed = get_sam_img_embed(test_patch, sam_model)
                    y1 = net(test_patch, test_img_embed)
                    # ensemble
                    if AMC:
                        y2 = 0.25 * (y1[0]+ y1[1]+ y1[2]+ y1[3])
                        y = torch.softmax(y2, dim=1)
                    else:
                        y = torch.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,
                              hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map = score_map[:, wl_pad:wl_pad +
                              w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
    return label_map

def test_all_case_3d(net, sam_model, data_dir, test_list="full_test.list", num_classes=3, 
    patch_size=(48, 160, 160), stride_xy=32, stride_z=24, AMC=False, test_save_path=None):

    if 'TBAD' in data_dir:
        with open(test_list, 'r') as f:
            image_list = f.readlines()
    elif 'OASIS' in data_dir:
        with open(test_list, 'r') as f:
            image_list = f.readlines()
    elif 'BTCV' in data_dir:
        with open(os.path.join(data_dir, 'datalist', test_list), 'r') as f:
            image_list = f.readlines()
    else:
        with open(data_dir + '/{}'.format(test_list), 'r') as f:
            image_list = f.readlines()

    if 'LA' in data_dir:
        image_list = [data_dir + "/{}/mri_norm2.h5".format(
            item.replace('\n', '').split(",")[0]) for item in image_list]
    elif 'BraTS2021' in data_dir:
        image_list = [data_dir + "/data/{}.h5".format(
            item.replace('\n', '').split(",")[0]) for item in image_list]
    else:
        image_list = [data_dir + "/{}.h5".format(
            item.replace('\n', '').split(",")[0]) for item in image_list]
    total_metric = np.zeros((num_classes-1, 4))

    print("Validation begin")
    for image_path in tqdm(image_list):
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        prediction = test_single_case_3d(
            net, sam_model, image, stride_xy, stride_z, patch_size, num_classes=num_classes, AMC=AMC)
        for i in range(1, num_classes):
            total_metric[i-1, :] += cal_metric(label == i, prediction == i)

    print("Validation end")
    return total_metric / len(image_list)

def test_all_case(net, data_dir, test_list="full_test.list", num_classes=3, 
    patch_size=(48, 160, 160), stride_xy=32, stride_z=24, AMC=False, test_save_path=None):

    if 'TBAD' in data_dir:
        with open(test_list, 'r') as f:
            image_list = f.readlines()
    elif 'OASIS' in data_dir:
        with open(test_list, 'r') as f:
            image_list = f.readlines()
    elif 'BTCV' in data_dir:
        with open(os.path.join(data_dir, 'datalist', test_list), 'r') as f:
            image_list = f.readlines()
    else:
        with open(data_dir + '/{}'.format(test_list), 'r') as f:
            image_list = f.readlines()

    if 'LA' in data_dir:
        image_list = [data_dir + "/{}/mri_norm2.h5".format(
            item.replace('\n', '').split(",")[0]) for item in image_list]
    elif 'BraTS2021' in data_dir:
        image_list = [data_dir + "/data/{}.h5".format(
            item.replace('\n', '').split(",")[0]) for item in image_list]
    else:
        image_list = [data_dir + "/{}.h5".format(
            item.replace('\n', '').split(",")[0]) for item in image_list]
    total_metric = np.zeros((num_classes-1, 4))

    print("Validation begin")
    for image_path in tqdm(image_list):
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        prediction = test_single_case(
            net, image, stride_xy, stride_z, patch_size, num_classes=num_classes, AMC=AMC)
        for i in range(1, num_classes):
            total_metric[i-1, :] += cal_metric(label == i, prediction == i)

    print("Validation end")
    return total_metric / len(image_list)

@torch.no_grad()
def test_AD(model, data_dir, test_list, num_classes=3, AMC=False):
    with open(test_list, 'r') as f:
        image_list = f.readlines()

    image_list = [data_dir + "/{}.h5".format(
        item.replace('\n', '').split(",")[0]) for item in image_list]
    total_metric = np.zeros((num_classes-1, 4))

    print("Validation begin")
    for image_path in tqdm(image_list):
        h5f = h5py.File(image_path, 'r')
        image = torch.from_numpy(h5f['image'][:]).type(torch.FloatTensor).cuda()
        label = h5f['label'][:]
        out = model(image.unsqueeze(0).unsqueeze(0))
        if AMC:
            outputs = (out[0] + out[1] + out[2] + out[3]) / 4
        else:
            outputs = out
        outputs = torch.softmax(outputs, dim = 1)
        outputs = torch.argmax(outputs, dim = 1).cpu().numpy()

        for c in range(1, num_classes):
            pred_test_data_tr = outputs.squeeze().copy()
            pred_test_data_tr[pred_test_data_tr != c] = 0

            pred_gt_data_tr = label.copy()
            pred_gt_data_tr[pred_gt_data_tr != c] = 0

            score = cal_metric(pred_gt_data_tr, pred_test_data_tr)
            total_metric[c-1, :] += score

    print("Validation end")
    return total_metric / len(image_list)

def test_all_case_print_single(net, data_dir, test_list="full_test.list", num_classes=3, 
    patch_size=(48, 160, 160), stride_xy=32, stride_z=24, AMC=False, test_save_path=None, save_result=False, BCP=False):

    if 'TBAD' in data_dir:
        with open(test_list, 'r') as f:
            image_list = f.readlines()
    elif 'OASIS' in data_dir:
        with open(test_list, 'r') as f:
            image_list = f.readlines()
    elif 'BTCV' in data_dir:
        with open(os.path.join(data_dir, 'datalist', test_list), 'r') as f:
            image_list = f.readlines()
    else:
        with open(data_dir + '/{}'.format(test_list), 'r') as f:
            image_list = f.readlines()
    if 'LA' in data_dir:
        image_list = [data_dir + "/{}/mri_norm2.h5".format(
            item.replace('\n', '').split(",")[0]) for item in image_list]
    else:
        image_list = [data_dir + "/{}.h5".format(
            item.replace('\n', '').split(",")[0]) for item in image_list]
    total_metric = np.zeros((num_classes-1, 4))

    idx = 0
    total_metric_all = np.zeros((num_classes-1, len(image_list), 4))
    total_num = len(image_list)

    print("Validation begin")
    time_start = time.time()

    for image_path in tqdm(image_list):
        h5f = h5py.File(image_path, 'r')
        if 'LA' in image_path:
            ids = image_path.split("/")[-2]
        else:
            ids = image_path.split("/")[-1].replace(".h5", "")
        image = h5f['image'][:]
        label = h5f['label'][:]
        prediction = test_single_case(
            net, image, stride_xy, stride_z, patch_size, num_classes=num_classes, AMC=AMC, BCP=BCP)
        logging.info('\n[{}/{}] {}%:\t{}'.format(idx+1,total_num,round((idx+1)/total_num*100),ids))

        for i in range(1, num_classes):
            score = cal_metric(label == i, prediction == i)
            total_metric[i-1, :] += score
            total_metric_all[i-1, idx, :] = score
            logging.info('Class: {}: dice {:.2f}% | jaccard {:.2f}% | hd95 {:.2f} | asd {:.2f}'
                            .format(i, score[0]*100,score[1]*100,score[2],score[3]))
        if save_result:
            np.save(os.path.join(test_save_path, ids+'_UPSAM_results.npy'), {'img':image, 'lab': label, 'pred':prediction})
            logging.info('Save at: {}'.format(os.path.join(test_save_path, ids +'_UPSAM_results.npy')))
        idx += 1
    time_end = time.time()

    for i in range(num_classes-1):
        logging.info('\nTotal Performance: \nClass: {}: dice {:.2f}% | jaccard {:.2f}% | hd95 {:.2f} | asd {:.2f}'
                        .format(i+1, total_metric_all[i,:,0].mean()*100,
                                total_metric_all[i,:,1].mean()*100, total_metric_all[i,:,2].mean(),
                                total_metric_all[i,:,3].mean(),))
        logging.info('{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'
                        .format(total_metric_all[i,:,0].mean()*100,
                                total_metric_all[i,:,1].mean()*100, total_metric_all[i,:,2].mean(),
                                total_metric_all[i,:,3].mean(),))
        logging.info('{:.2f}({:.2f})\t{:.2f}({:.2f})\t{:.2f}({:.2f})\t{:.2f}({:.2f})'
                        .format(total_metric_all[i,:,0].mean()*100, total_metric_all[i,:,0].std()*100,
                                total_metric_all[i,:,1].mean()*100, total_metric_all[i,:,1].std()*100,
                                total_metric_all[i,:,2].mean(), total_metric_all[i,:,2].std(),
                                total_metric_all[i,:,3].mean(), total_metric_all[i,:,3].std()))
        logging.info('Inference time: {:.2f} s'.format((time_end-time_start)/len(image_list)))
    logging.info('{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'
                    .format(total_metric_all[:,:,0].mean()*100,
                            total_metric_all[:,:,1].mean()*100, total_metric_all[:,:,2].mean(),
                            total_metric_all[:,:,3].mean(),))
    logging.info('{:.2f}({:.2f})\t{:.2f}({:.2f})\t{:.2f}({:.2f})\t{:.2f}({:.2f})'
                    .format(total_metric_all[:,:,0].mean()*100, total_metric_all[:,:,0].std()*100,
                            total_metric_all[:,:,1].mean()*100, total_metric_all[:,:,1].std()*100,
                            total_metric_all[:,:,2].mean(), total_metric_all[:,:,2].std(),
                            total_metric_all[:,:,3].mean(), total_metric_all[:,:,3].std()))

    row_data = []

    for i in range(num_classes - 1):
        dice = total_metric_all[i, :, 0].mean() * 100
        jaccard = total_metric_all[i, :, 1].mean() * 100
        hd95 = total_metric_all[i, :, 2].mean()
        asd = total_metric_all[i, :, 3].mean()
        row_data.extend([dice, jaccard, hd95, asd])

    mean_dice = total_metric_all[:, :, 0].mean() * 100
    mean_jaccard = total_metric_all[:, :, 1].mean() * 100
    mean_hd95 = total_metric_all[:, :, 2].mean()
    mean_asd = total_metric_all[:, :, 3].mean()
    row_data.extend([mean_dice, mean_jaccard, mean_hd95, mean_asd])

    columns = []
    mean_row = []
    std_row = []

    latex_output = "\n"
    for i in range(1, num_classes):  
        columns.extend([
            f'Class{i}_Dice(%)',
            f'Class{i}_Jaccard(%)',
            f'Class{i}_95HD(voxel)',
            f'Class{i}_ASD(voxel)',
        ])
        mean_row.extend([
            f'{total_metric_all[i-1, :, 0].mean()*100:.2f}',
            f'{total_metric_all[i-1, :, 1].mean()*100:.2f}',
            f'{total_metric_all[i-1, :, 2].mean():.2f}',
            f'{total_metric_all[i-1, :, 3].mean():.2f}',
        ])
        std_row.extend([
            f'{total_metric_all[i-1, :, 0].std()*100:.2f}',
            f'{total_metric_all[i-1, :, 1].std()*100:.2f}',
            f'{total_metric_all[i-1, :, 2].std():.2f}',
            f'{total_metric_all[i-1, :, 3].std():.2f}',
        ])
        latex_output += f"{total_metric_all[i-1, :, 0].mean()*100:.2f}\\tiny{{$\\pm${total_metric_all[i-1, :, 0].std()*100:.2f}}} & "
        latex_output += f"{total_metric_all[i-1, :, 1].mean()*100:.2f}\\tiny{{$\\pm${total_metric_all[i-1, :, 1].std()*100:.2f}}} & "
        latex_output += f"{total_metric_all[i-1, :, 2].mean():.2f}\\tiny{{$\\pm${total_metric_all[i-1, :, 2].std():.2f}}} & "
        latex_output += f"{total_metric_all[i-1, :, 3].mean():.2f}\\tiny{{$\\pm${total_metric_all[i-1, :, 3].std():.2f}}} & "
    
    latex_output += "\n\n"

    columns.extend([
        'Avg_Dice(%)',
        'Avg_Jaccard(%)',
        'Avg_95HD(voxel)',
        'Avg_ASD(voxel)'
    ])
    mean_row.extend([
        f'{total_metric_all[:, :, 0].mean()*100:.2f}',
        f'{total_metric_all[:, :, 1].mean()*100:.2f}',
        f'{total_metric_all[:, :, 2].mean():.2f}',
        f'{total_metric_all[:, :, 3].mean():.2f}',
    ])
    std_row.extend([
        f'{total_metric_all[:, :, 0].std()*100:.2f}',
        f'{total_metric_all[:, :, 1].std()*100:.2f}',
        f'{total_metric_all[:, :, 2].std():.2f}',
        f'{total_metric_all[:, :, 3].std():.2f}',
    ])

    latex_output += f"{total_metric_all[:, :, 0].mean()*100:.2f}\\tiny{{$\\pm${total_metric_all[:, :, 0].std()*100:.2f}}} & "
    latex_output += f"{total_metric_all[:, :, 1].mean()*100:.2f}\\tiny{{$\\pm${total_metric_all[:, :, 1].std()*100:.2f}}} & "
    latex_output += f"{total_metric_all[:, :, 2].mean():.2f}\\tiny{{$\\pm${total_metric_all[:, :, 2].std():.2f}}} & "
    latex_output += f"{total_metric_all[:, :, 3].mean():.2f}\\tiny{{$\\pm${total_metric_all[:, :, 3].std():.2f}}} & "

    logging.info("\t".join(columns))
    logging.info("\t".join(mean_row))
    logging.info("\t".join(std_row))

    logging.info("".join(latex_output))

    if test_save_path is not None:
        df = pd.DataFrame([row_data], columns=columns)
        df.to_excel(test_save_path + '/metric_summary.xlsx', index=False)

    print("Validation end")
    return total_metric / len(image_list)

