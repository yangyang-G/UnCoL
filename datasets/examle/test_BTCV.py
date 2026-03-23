import argparse
import os
import shutil
import h5py
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from networks.config import get_config
from networks.net_factory import net_factory
from networks.vision_transformer import SwinUnet as ViT_seg
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../../data/BTCV', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='RemainingConsistency', help='experiment_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--model_1', type=str,
                    default='unet', help='model_name')
parser.add_argument('--model_2', type=str,
                    default='swin_unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=14,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=2,
                    help='labeled data')
parser.add_argument('--image_size', type=list, default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--cfg', type=str,
                    default="./configs/swin_tiny_patch4_window7_224_lite.yaml",
                    help='path to config file', )
parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ",
                    default=None, nargs='+', )
parser.add_argument('--batch_size', type=int, default=12,
                    help='batch_size per gpu')
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, ''full: cache all data, ''part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')
args = parser.parse_args() 
config = get_config(args)

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
            return 0.0, 0.0, None, None

def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (224 / x, 224 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            if FLAGS.model_1 == "unet_urds":
                out_main, _, _, _ = net(input)
            else:
                out_main = net(input)
            out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 224, y / 224), order=0)
            prediction[ind] = pred
            
    per_class_metrics = []
    for cls in range(1, FLAGS.num_classes):  # skip background (0)
        if np.sum(label == cls) == 0:
            # 如果 GT 中没有该类别，则跳过指标计算
            per_class_metrics.append((None, None, None, None))
        else:
            metrics = calculate_metric_percase(prediction == cls, label == cls)
            per_class_metrics.append(metrics)
            
    # 保存图像
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

def Inference_model1(FLAGS):
    print("——Starting the Model1 Prediction——")
    with open(FLAGS.root_path + '/test.list', 'r') as f:image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]for item in image_list])
    # 使用图片中显示的实际路径
    snapshot_path = "/public/home/wyf_cdu/Project/Semi-Supervised/ABD/model/Cross_Teaching/BTCV_train_BTCV_Cross_Teaching_RemainingConsistency_2_labeled"
    test_save_path = "/public/home/wyf_cdu/Project/Semi-Supervised/ABD/model/Cross_Teaching/BTCV_train_BTCV_Cross_Teaching_RemainingConsistency_2_labeled/unet_predictions/"
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model_1, in_chns=1,class_num=FLAGS.num_classes)
    
    # 根据图片，模型文件可能是model1_iter_10000_dice_xxx.pth格式
    # 我们需要找到最佳模型文件，先尝试加载unet_best_model.pth
    best_model_path = os.path.join(snapshot_path, 'unet_best_model.pth')
    if os.path.exists(best_model_path):
        print(f"Loading best model from: {best_model_path}")
        net.load_state_dict(torch.load(best_model_path))
    else:
        # 如果没有找到best模型，查找最新的model1模型
        import glob
        model_files = glob.glob(os.path.join(snapshot_path, 'model1_iter_*.pth'))
        if model_files:
            # 按文件名排序，选择最新的
            model_files.sort(key=os.path.getmtime, reverse=True)
            best_model_path = model_files[0]
            print(f"Best model not found. Loading latest model: {best_model_path}")
            net.load_state_dict(torch.load(best_model_path))
        else:
            raise FileNotFoundError(f"No model files found in {snapshot_path}")
    
    print("init weight from {}".format(best_model_path))
    net.eval()

    total_metric_per_class = np.zeros((FLAGS.num_classes - 1, 4), dtype=np.float64)
    all_metrics = []

    for case in tqdm(image_list):
        case_metrics = test_single_volume(case, net, test_save_path, FLAGS)
        case_metrics = np.asarray(case_metrics, dtype=np.float64)
        total_metric_per_class += case_metrics

        print(f"\n{case} results:")
        row_metrics = [case]
        for cls in range(1, FLAGS.num_classes):
            dice, jc, hd95, asd = case_metrics[cls - 1]
            print(f"Class {cls:2d}: Dice: {dice:.4f}, JC: {jc:.4f}, HD95: {hd95:.2f}, ASD: {asd:.2f}")
            row_metrics.extend([dice, jc, hd95, asd])
        all_metrics.append(row_metrics)

    avg_metric_per_class = total_metric_per_class / len(image_list)

    # 保存到CSV文件
    header = ['Sample']
    for cls in range(1, FLAGS.num_classes):
        header.extend([f'Class {cls} Dice', f'Class {cls} JC', f'Class {cls} HD95', f'Class {cls} ASD'])
    
    csv_file = os.path.join(test_save_path, 'metrics_per_sample.csv')
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_metrics)

    return avg_metric_per_class, test_save_path


def Inference_model2(FLAGS):
    print("——Starting the Model2 Prediction——")
    with open(FLAGS.root_path + '/test.list', 'r') as f:image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]for item in image_list])
    # 使用图片中显示的实际路径
    snapshot_path = "/public/home/wyf_cdu/Project/Semi-Supervised/ABD/model/Cross_Teaching/BTCV_train_BTCV_Cross_Teaching_RemainingConsistency_2_labeled"
    test_save_path = "/public/home/wyf_cdu/Project/Semi-Supervised/ABD/model/Cross_Teaching/BTCV_train_BTCV_Cross_Teaching_RemainingConsistency_2_labeled/swin_unet_predictions/"
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)
    net = ViT_seg(config, img_size=FLAGS.image_size, num_classes=FLAGS.num_classes).cuda()
    net.load_from(config)
    
    # 查找model2的最佳模型文件
    best_model_path = os.path.join(snapshot_path, 'swin_unet_best_model.pth')
    if os.path.exists(best_model_path):
        print(f"Loading best model from: {best_model_path}")
        net.load_state_dict(torch.load(best_model_path))
    else:
        # 如果没有找到best模型，查找最新的model2模型
        import glob
        model_files = glob.glob(os.path.join(snapshot_path, 'model2_iter_*.pth'))
        if model_files:
            # 按文件名排序，选择最新的
            model_files.sort(key=os.path.getmtime, reverse=True)
            best_model_path = model_files[0]
            print(f"Best model not found. Loading latest model: {best_model_path}")
            net.load_state_dict(torch.load(best_model_path))
        else:
            raise FileNotFoundError(f"No model files found in {snapshot_path}")
    
    print("init weight from {}".format(best_model_path))
    net.eval()

    # 这里保持原有的test_single_volume调用方式
    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric = test_single_volume(case, net, test_save_path, FLAGS)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
    avg_metric = [first_total / len(image_list), second_total / len(image_list), third_total / len(image_list)]
    average = (avg_metric[0] + avg_metric[1] + avg_metric[2]) / 3
    print(avg_metric)
    print(average)
    with open(os.path.join(test_save_path, 'performance.txt'), 'w') as file:
        file.write(str(avg_metric) + '\n')
        file.write(str(average) + '\n')
    return avg_metric

  
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    FLAGS = parser.parse_args()
    metric, test_save_path = Inference_model1(FLAGS)
    # metric_model2 = Inference_model2(FLAGS)
    
    for cls in range(1, FLAGS.num_classes):
        print(f"Class {cls:2d} => Dice: {metric[cls - 1, 0]:.4f}, JC: {metric[cls - 1, 1]:.4f}, HD95: {metric[cls - 1, 2]:.2f}, ASD: {metric[cls - 1, 3]:.2f}")

    mean_avg = np.mean(metric, axis=0)
    print(f"\nOverall Average => Dice: {mean_avg[0]:.4f}, JC: {mean_avg[1]:.4f}, HD95: {mean_avg[2]:.2f}, ASD: {mean_avg[3]:.2f}")

    with open(test_save_path + '../performance.txt', 'w') as f:
        f.writelines("Per-class metrics:\n")
        for cls in range(1, FLAGS.num_classes):
            f.writelines(f"Class {cls:2d} => Dice: {metric[cls - 1, 0]:.4f}, JC: {metric[cls - 1, 1]:.4f}, HD95: {metric[cls - 1, 2]:.2f}, ASD: {metric[cls - 1, 3]:.2f}\n")
        f.writelines(f"\nOverall Average => Dice: {mean_avg[0]:.4f}, JC: {mean_avg[1]:.4f}, HD95: {mean_avg[2]:.2f}, ASD: {mean_avg[3]:.2f}\n")