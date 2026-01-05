import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

sys.path.append('./')
from networks.CA_UNet import Prob_UNet
from utils import losses, ramps
from utils.val_2D import test_single_volume_pred, test_single_volume_ds, test_single_volume
from utils.train_utils import load_dataset, get_sam_features, cal_dice, load_from, update_ema_variables
from datasets.oasis2d import BaseDataSets, RandomGenerator, TwoStreamBatchSampler

from MedSAM.segment_anything import sam_model_registry

from medpy import metric
from monai.losses import DiceCELoss
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib
import pandas as pd
import random

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='data dir')
parser.add_argument('--fold_num', type=int, help='fold number of the datalist')
parser.add_argument('--gpu', type=str, default='3',
                    help='gpu id')
parser.add_argument('--seed', type=int,  default=3315, help='GPU to use')
parser.add_argument('--num_classes', type=int,  default=5,
                    help='output channel of network')
parser.add_argument('--checkpoint_dir', type=str, help='model_name')

FLAGS = parser.parse_args()
seed = FLAGS.seed
# Use CUDA

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda")

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda")
test_save_path = os.path.join(FLAGS.checkpoint_dir, f'test_prediction-{seed}')
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

logging.basicConfig(filename=test_save_path + "/record_log.txt", level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info('Split path: {}\nModel path: {}'.format(FLAGS.data_dir, FLAGS.checkpoint_dir))
logging.info('Save at: {}'.format(test_save_path))


def create_model(ema=False):
    # model = UNet(in_chns=1, class_num=FLAGS.num_classes).cuda()
    model = Prob_UNet(in_chns=1, class_num=FLAGS.num_classes).cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def save_dice_results_to_csv_transposed(num_classes, total_metric_all, model_name, output_file):
    """
    Save dice results of each class for a single model in a transposed CSV file.
    
    Args:
        num_classes (int): Number of classes.
        total_metric_all (numpy array): Array containing metrics for each class.
        model_name (str): Name of the model.
        output_file (str): Path to save the CSV file.
    """
    # Collect results for each class
    results = []
    class_names = []
    dice_means = []
    dice_stds = []
    
    for i in range(num_classes - 1):  # Assuming 0-indexed classes, skipping background
        class_name = f"Class_{i+1}"
        dice_mean = total_metric_all[i, :, 0].mean() * 100  # Dice mean
        dice_std = total_metric_all[i, :, 0].std() * 100  # Dice std deviation
        
        class_names.append(class_name)
        dice_means.append(dice_mean)
        dice_stds.append(dice_std)
    
    # Transpose results into a dictionary
    transposed_results = {
        "Metric": ["Dice Mean (%)", "Dice Std (%)"],
        **{class_name: [dice_means[i], dice_stds[i]] for i, class_name in enumerate(class_names)},
    }
    
    # Convert transposed results to DataFrame
    df = pd.DataFrame(transposed_results)
    df["Model"] = [model_name, ""]  # Add model name column with blank in the second row

    # Save to CSV
    df.to_csv(output_file, index=False)
    return df

if __name__ == "__main__":
    model = create_model()
    # model.load_state_dict(torch.load(os.path.join(FLAGS.checkpoint_dir,'unet_best_model.pth')))
    model.load_state_dict(torch.load(os.path.join(FLAGS.checkpoint_dir,'best_model.pth')))
    model.eval()

    db_test = BaseDataSets(base_dir=FLAGS.data_dir, split="test", fold_num=FLAGS.fold_num)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False,
                           num_workers=1)
    # total_metric = np.zeros((len(db_test), FLAGS.num_classes-1, 4))
    total_metric_all = np.zeros((FLAGS.num_classes-1, len(db_test), 4))
    total_num = len(db_test)

    for i_batch, sampled_batch in enumerate(testloader):
        total_metric = test_single_volume(
            sampled_batch["image"], sampled_batch["label"], model, classes=FLAGS.num_classes, AMC=True)

        patient_folder = sampled_batch["name"][0]
        logging.info('\n[{}/{}] {}%:\t{}'.format(i_batch+1,total_num,round((i_batch+1)/total_num*100),patient_folder))

        for i in range(FLAGS.num_classes-1):
            total_metric_all[i, i_batch, :] = total_metric[i]
            logging.info('Class: {}: dice {:.2f}% | jaccard {:.2f}% | hd95 {:.2f} | asd {:.2f}'
                            .format(i, total_metric[i, 0]*100, total_metric[i, 1]*100,
                            total_metric[i, 2], total_metric[i, 3]))

    for i in range(FLAGS.num_classes-1):
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
                                
    logging.info('{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'
                    .format(total_metric_all[:,:,0].mean()*100,
                            total_metric_all[:,:,1].mean()*100, total_metric_all[:,:,2].mean(),
                            total_metric_all[:,:,3].mean(),))
    logging.info('{:.2f}({:.2f})\t{:.2f}({:.2f})\t{:.2f}({:.2f})\t{:.2f}({:.2f})'
                    .format(total_metric_all[:,:,0].mean()*100, total_metric_all[:,:,0].std()*100,
                            total_metric_all[:,:,1].mean()*100, total_metric_all[:,:,1].std()*100,
                            total_metric_all[:,:,2].mean(), total_metric_all[:,:,2].std(),
                            total_metric_all[:,:,3].mean(), total_metric_all[:,:,3].std()))
    save_dice_results_to_csv_transposed(FLAGS.num_classes, total_metric_all, 'UnCoL', test_save_path + '/results.csv')

    row_data = []

    for i in range(FLAGS.num_classes - 1):
        # 每一类的四个平均指标
        dice = total_metric_all[i, :, 0].mean() * 100
        jaccard = total_metric_all[i, :, 1].mean() * 100
        hd95 = total_metric_all[i, :, 2].mean()
        asd = total_metric_all[i, :, 3].mean()
        row_data.extend([dice, jaccard, hd95, asd])

    # 添加最后四列平均结果
    mean_dice = total_metric_all[:, :, 0].mean() * 100
    mean_jaccard = total_metric_all[:, :, 1].mean() * 100
    mean_hd95 = total_metric_all[:, :, 2].mean()
    mean_asd = total_metric_all[:, :, 3].mean()
    row_data.extend([mean_dice, mean_jaccard, mean_hd95, mean_asd])

    # 构造列名
    columns = []
    # 构造均值和标准差行
    mean_row = []
    std_row = []

    latex_output = "\n"
    for i in range(1, FLAGS.num_classes):  # 类别从1开始到 num_classes-1
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

    # 保存为 DataFrame 并写入 Excel 文件
    if test_save_path is not None:
        df = pd.DataFrame([row_data], columns=columns)
        df.to_excel(test_save_path + '/metric_summary.xlsx', index=False)