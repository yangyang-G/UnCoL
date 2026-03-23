import argparse
import os
import shutil
import csv
import h5py
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm

from networks.net_factory import net_factory
import logging
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../../../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='MT2D_ACDC', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4, help='output channel of network')
parser.add_argument('--labelnum', type=int, default=7, help='labeled data')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--label_ratio', type=str, default='5%', help='label ratio')
parser.add_argument('--dataset_name', type=str, default='ACDC' ,help='dataset_name')

args = parser.parse_args()


label_num_mapping = {
    "ACDC": {'5%': 3, '10%': 7, '20%': 14, '100%': 70},
    "Prostate": {'5%': 2, '10%': 4, '20%': 7, '100%': 35},
    "Hippocampus": {'5%': 8, '10%': 16, '20%': 31, '100%': 156},"Vertebral": {'5%': 5, '10%': 10, '20%': 21, '100%': 106},     "HepaticVessel": {'5%': 9, '10%': 18, '20%': 36, '100%':181},  "ATLAS": {'5%': 2, '10%': 4, '20%': 7, '100%': 36},         "WORD": {'5%': 4, '10%': 7, '20%': 14, '100%': 72},
    "BTCV": {'5%': 1, '10%': 2, '20%': 4, '100%': 18},
}
def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"1": 34, "3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Prostate"in dataset:
        ref_dict = {"2": 47, "4": 111, "7": 191,
                    "11": 306, "14": 391, "18": 478, "35": 940}
    elif "Hippocampus" in dataset:
        ref_dict = {"8": 288, "16": 581,"31": 1115,"156": 5559}
    elif "Vertebral" in dataset:
        ref_dict = {"5": 63, "10": 123,"21": 261,"103": 1302}
    elif "HepaticVessel" in dataset:
        ref_dict = {"9": 926, "18": 1708,"36": 3251,"181": 12881}
    elif "ATLAS" in dataset:
        ref_dict = {"2": 120, "4": 272,"7": 528,"36": 2864}
    elif "WORD" in dataset:
        ref_dict = {"4": 874, "7": 1359,"14": 2767,"72": 14412}
    elif "BTCV" in dataset:
        ref_dict = {"1": 143, "2": 236,"4": 476,"18": 2320}
    else:
        print("patients_to_slices——Error")

    return ref_dict[str(patiens_num)]
if "ACDC" in args.root_path:
    args.labelnum = label_num_mapping["ACDC"][args.label_ratio]
    args.num_classes = 4
    args.dataset_name="ACDC"
elif "Prostate" in args.root_path:
    args.labelnum = label_num_mapping["Prostate"][args.label_ratio]
    args.num_classes = 2
    args.dataset_name="Prostate"
elif "Hippocampus" in args.root_path:
    args.labelnum = label_num_mapping["Hippocampus"][args.label_ratio]
    args.num_classes = 3
    args.dataset_name="Hippocampus"
elif "Vertebral" in args.root_path:
    args.labelnum = label_num_mapping["Vertebral"][args.label_ratio]
    args.num_classes = 20
    args.dataset_name="Vertebral"
elif "ATLAS" in args.root_path:
    args.labelnum = label_num_mapping["ATLAS"][args.label_ratio]
    args.num_classes = 3
    args.dataset_name="ATLAS"
elif "WORD" in args.root_path:
    args.labelnum = label_num_mapping["WORD"][args.label_ratio]
    args.num_classes = 17
    args.dataset_name="WORD"
elif "BTCV" in args.root_path:
    args.labelnum = label_num_mapping["BTCV"][args.label_ratio]
    args.num_classes = 14
    args.dataset_name="BTCV"
elif "HepaticVessel" in args.root_path:
    args.labelnum = label_num_mapping["HepaticVessel"][args.label_ratio]
    args.num_classes = 3
    args.dataset_name="HepaticVessel"
else:
    print("Error")
    sys.exit(1)

def calculate_metric_percase(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    if gt.sum() == 0:  # 如果GT为空
        return None, None, None, None

    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        jc = metric.binary.jc(pred, gt)
        asd = metric.binary.asd(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, jc, hd95, asd
    else:
        if pred.sum() == 0 and gt.sum() == 0:
            return 1.0, 1.0, 0.0, 0.0
        else:
            return 0.0, 0.0, 0.0, 0.0


def test_single_volume(case, net, test_save_path, args):
    h5f = h5py.File(args.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out_main = net(input)
            if len(out_main)>1:
                out_main=out_main[0]
            out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred
    per_class_metrics = []
    for cls in range(1, args.num_classes):  # skip background (0)
        if np.sum(label == cls) == 0:
            # 如果 GT 中没有该类别，则跳过指标计算
            per_class_metrics.append((None, None, None, None))
        else:
            metrics = calculate_metric_percase(prediction == cls, label == cls)
            per_class_metrics.append(metrics)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return per_class_metrics


class _OnlyMainOutput(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (list, tuple)):
            return out[0]
        return out


def _compute_model_complexity(model: nn.Module, input_shape=(1, 1, 256, 256)):
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    flops_val = None
    flops_str = 'N/A'
    try:
        from thop import profile, clever_format
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')
        m = _OnlyMainOutput(model).to(device)
        m.eval()
        with torch.no_grad():
            dummy = torch.randn(*input_shape, device=device)
            macs_val, _ = profile(m, inputs=(dummy,), verbose=False)
        flops_val = 2 * macs_val
        flops_str, _ = clever_format([flops_val, total_params], '%.3f')
    except Exception:
        flops_val = None
        flops_str = 'N/A'
    return total_params, flops_val, flops_str


def Inference(args):
    with open(args.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0] for item in image_list])
    snapshot_path = "../model/Pilot_experiment/{}_{}_{}_labeled/".format(args.exp,args.dataset_name, args.labelnum)
    test_save_path = "../model/Pilot_experiment/{}_{}_{}_labeled/{}_predictions_model/".format(args.exp,args.dataset_name,args.labelnum, args.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)

    net = net_factory(net_type='unet', in_chns=1, class_num=args.num_classes)
    save_model_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
    net.load_state_dict(torch.load(save_model_path), strict=False)
    print("init weight from {}".format(save_model_path))
    net.eval()

    total_metric_per_class = np.zeros((args.num_classes - 1, 4), dtype=np.float64)  # 初始化为 float64 类型
    all_metrics = []

    for case in tqdm(image_list):
        case_metrics = test_single_volume(case, net, test_save_path, args)
        case_metrics = np.asarray(case_metrics, dtype=np.float64)  # 确保是 float64 类型
        total_metric_per_class += case_metrics  # 应该可以进行加法操作

        print(f"\n{case} results:")
        row_metrics = [case]  # 样本名称
        for cls in range(1, args.num_classes):
            dice, jc, hd95, asd = case_metrics[cls - 1]
            print(f"Class {cls:2d}: Dice: {dice:.4f}, JC: {jc:.4f}, HD95: {hd95:.2f}, ASD: {asd:.2f}")
            row_metrics.extend([dice, jc, hd95, asd])  # 将每个类别的指标保存到行中
        all_metrics.append(row_metrics)

    avg_metric_per_class = total_metric_per_class / len(image_list)

    # 保存到CSV文件
    header = ['Sample']  # 样本名称
    for cls in range(1, args.num_classes):
        header.extend([f'Class {cls} Dice', f'Class {cls} JC', f'Class {cls} HD95', f'Class {cls} ASD'])
    
    csv_file = os.path.join(test_save_path, 'metrics_per_sample.csv')
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)  # 写入表头
        writer.writerows(all_metrics)  # 写入所有样本的指标

    return avg_metric_per_class, test_save_path


if __name__ == '__main__':
    
    metric, test_save_path = Inference(args)

    for cls in range(1, args.num_classes):
        print(f"Class {cls:2d} => Dice: {metric[cls - 1, 0]:.4f}, JC: {metric[cls - 1, 1]:.4f}, HD95: {metric[cls - 1, 2]:.2f}, ASD: {metric[cls - 1, 3]:.2f}")

    mean_avg = np.mean(metric, axis=0)
    print(f"\nOverall Average => Dice: {mean_avg[0]:.4f}, JC: {mean_avg[1]:.4f}, HD95: {mean_avg[2]:.2f}, ASD: {mean_avg[3]:.2f}")

    with open(test_save_path + '../performance.txt', 'w') as f:
        f.writelines("Per-class metrics:\n")
        for cls in range(1, args.num_classes):
            f.writelines(f"Class {cls:2d} => Dice: {metric[cls - 1, 0]:.4f}, JC: {metric[cls - 1, 1]:.4f}, HD95: {metric[cls - 1, 2]:.2f}, ASD: {metric[cls - 1, 3]:.2f}\n")
        f.writelines(f"\nOverall Average => Dice: {mean_avg[0]:.4f}, JC: {mean_avg[1]:.4f}, HD95: {mean_avg[2]:.2f}, ASD: {mean_avg[3]:.2f}\n")
