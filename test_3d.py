import argparse
import os
import shutil
from glob import glob
import numpy

import torch
import torch.nn as nn

import sys
import logging

sys.path.append('.')
from networks.CA_VNet import Prob_VNet
from thop import clever_format

from utils.val_3D import test_all_case_print_single
import numpy as np
from pathlib import Path
import random

def Inference(FLAGS, model_path):
    # snapshot_path = os.path.dirname(os.path.dirname(model_path))
    snapshot_path = os.path.dirname(model_path)
    fold_num = snapshot_path.split('/')[-5][-1]

    if 'TBAD' in FLAGS.data_dir:
        test_list = os.path.join(FLAGS.data_dir, 'ImageTBADlist', 'AD_{}'.format(FLAGS.fold_num), 'test.txt')
        test_save_path = os.path.join(snapshot_path, 'test_prediction-{}'.format(seed))
    elif 'OASIS' in FLAGS.data_dir:
        test_list = os.path.join(FLAGS.data_dir, 'datalist', 'fold_{}'.format(FLAGS.fold_num), 'test.txt')
        test_save_path = os.path.join(snapshot_path, 'test_prediction-{}'.format(seed))
    else:
        test_list = 'test.txt'
        test_save_path = os.path.join(snapshot_path, 'test_prediction')
    os.makedirs(test_save_path, exist_ok=True)

    logging.basicConfig(filename=test_save_path + "/record_log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info('Split path: {}\nModel path: {}'.format(FLAGS.data_dir, model_path))
    logging.info('Save at: {}'.format(test_save_path))

    logging.basicConfig(filename=test_save_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    net = Prob_VNet(n_channels=1, n_classes=FLAGS.num_classes, n_branches=4,).cuda()
    
    for n, q in net.named_parameters():
        print(n, clever_format([q.numel()], "%.3f"))
    net.load_state_dict(torch.load(model_path))

    net.eval()

    logging.info("init weight from {}".format(model_path))

    net.eval()
    avg_metric =  test_all_case_print_single(
                        net, FLAGS.data_dir, test_list=test_list, num_classes=num_classes, 
                        patch_size=FLAGS.patch_size,
                        stride_xy=16, stride_z=4, AMC=True, test_save_path=test_save_path,
                        save_result=False, BCP=False)

    return avg_metric

def load_net_opt(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    logging.info('Loaded from {}'.format(path))

if __name__ == '__main__':

    fold_num = 2
    parser = argparse.ArgumentParser()
    # Pancreas-processed
    parser.add_argument("--data_dir", type=str,
                        help="Path to the dataset.")
    parser.add_argument("--fold_num", type=int, default=fold_num)
    parser.add_argument('--model_path', type=str, help='model_path')
    parser.add_argument('--gpu', type=str,  default='3', help='GPU to use')
    parser.add_argument('--seed', type=int,  default=3315, help='GPU to use')
    parser.add_argument('--num_classes', type=int,  default=2, help='GPU to use')
    parser.add_argument('--patch_size', type=list,  default=[128,128,128], help='patch size of network input')
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

    model_path = Path(FLAGS.model_path) / 'best_model.pth'
    
    num_classes = FLAGS.num_classes
    total_metric = Inference(FLAGS, model_path)
