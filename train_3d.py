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
from datasets.brats19 import (CenterCrop, RandomCrop,
                                   RandomRotFlip, ToTensor,
                                   TwoStreamBatchSampler, TwoStreamBatchSampler_new)

from networks.CA_VNet import Prob_VNet

from utils import losses, ramps
from utils.val_3D import test_all_case, test_AD
from utils.train_utils import (load_dataset, get_sam_features, cal_dice, load_from,
                                update_ema_variables, compute_softmax_entropy, 
                                compute_batch_rank_maps, generate_mask, compute_consistency_loss,
                                get_sam_hidden)

from SAM_Med3D.segment_anything.build_sam3D import sam_model_registry3D
from SAM_Med3D.fine_tune_3D_func import finetune_model_predict3D

from utils.BCP_utils import context_mask, mix_loss, parameter_sharing, masked_mix_loss
from skimage.measure import label

from medpy import metric
import math

parser = argparse.ArgumentParser()
# default information
parser.add_argument('--method', type=str,
                    default='UnCoL', help='method name')
parser.add_argument('--mode', type=str,
                    default='PRETRAIN', help='stage')
parser.add_argument('--num_classes', type=int,
                    default=2, help='number of classes')
parser.add_argument('--dataset', type=str,
                    default='Pancrease-CT', help='dataset name')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')

# dataset
parser.add_argument('--data_dir', type=str, help='Data root')
parser.add_argument('--save_path', type=str,help='Save path')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--patch_size', type=list,  default=[128,128,128],
                    help='patch size of network input')
parser.add_argument('--fold_num', type=int, default=0,
                    help='fold num of the dataset')

# implementation details
parser.add_argument('--max_iterations', type=int,
                    default=15000, help='maximum epoch number to train')
parser.add_argument('--warmup_iters', type=int,
                    default=2000, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--pretrain_base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=4,
                    help='labeled data')

# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--weight_decay', type=float,  default=0.1, help='ema_decay')

# loss function
parser.add_argument('--consistency', type=float,
                    default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=250.0, help='consistency_rampup')
# sam 
parser.add_argument('-pm', '--point_method', type=str, default='uncertain')
parser.add_argument('--crop_size', type=int, default=128)
parser.add_argument('-nc', '--num_clicks', type=int, default=5)
parser.add_argument('--gpu', type=str, default='0')

# checkpoints
parser.add_argument('--checkpoint_sam', type=str)
parser.add_argument('--checkpoint_dir', type=str)

parser.add_argument('--mask_ratio', type=float, default=2/3, help='ratio of mask/image')
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')

args = parser.parse_args()

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda")


def get_cut_mask(out, thres=0.5, nms=0):
    # probs = F.softmax(out, 1)
    # masks = (probs >= thres).type(torch.int64)
    # masks = masks[:, 1, :, :].contiguous()
    # _, probs = torch.max(probs, dim=1)
    probs = F.softmax(out, dim=1)
    masks = torch.argmax(probs, dim=1)
    if nms == 1:
        masks = LargestCC_3Ddatasets(masks)
        # masks = get_ACDC_2DLargestCC(probs)
    return masks

def LargestCC_pancreas(segmentation):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob)
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)
    
    return torch.Tensor(batch_list).cuda()

def LargestCC_3Ddatasets(segmentation):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        seg_np = segmentation[n].detach().cpu().numpy()
        new_seg = np.zeros_like(seg_np)
        for c in range(1, args.num_classes): 
            class_mask = (seg_np == c).astype(np.uint8)
            labels = label(class_mask)
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1 
                new_seg[largestCC] = c
        batch_list.append(new_seg)
    return torch.Tensor(batch_list).cuda()

def get_current_consistency_weight(epoch, consistency):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def create_model(ema=False):
    # Network definition
    model = Prob_VNet(n_channels=1, n_classes=args.num_classes, n_branches=4).cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

def pretrain(args, snapshot_path):
    base_lr = args.pretrain_base_lr
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    db_train, test_list  = load_dataset(args, num=args.labeled_num)
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn, drop_last=True)

    model = create_model()
    ema_model = create_model(ema=True)

    sam_model = sam_model_registry3D['vit_b_ori'](checkpoint=None)
    sam_model = load_from(sam_model, args.checkpoint_sam).cuda()

    model.train()
    ema_model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    # loss func
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(nclass=args.num_classes)
    focal_loss = losses.FocalLoss()
    iou_loss = losses.SoftIoULoss(nclass=args.num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    sub_bs = int(args.labeled_bs/2)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:]
            lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:]
            with torch.no_grad():
                img_mask, loss_mask = context_mask(img_a, args.mask_ratio)

            """Mix Input"""
            volume_batch = img_a * img_mask + img_b * (1 - img_mask)
            label_batch = lab_a * img_mask + lab_b * (1 - img_mask)

            with torch.no_grad():
                hidden_states_out_sam, posterior_features, _ = get_sam_features(
                    volume_batch, label_batch, sam_model, args, device=device, 
                    click_method='random', num_clicks=args.num_clicks, 
                    prev_masks=None) # torch.Size([B, 1, 128, 128, 128]) (B, 128, 128, 128)
            outputs, loss_kd, loss_infomax = model(volume_batch, hidden_states_out_sam=hidden_states_out_sam,
                                                sam_post_embed=posterior_features)
            loss_ce = ce_loss(outputs[0], label_batch)
            loss_dice = dice_loss(torch.softmax(outputs[1], dim=1), label_batch)
            loss_focal = focal_loss(outputs[2], label_batch)
            loss_iou = iou_loss(outputs[3], label_batch)
            loss_sup = 0.25 * (loss_ce+loss_dice+loss_focal+loss_iou)

            consistency_weight = get_current_consistency_weight(iter_num//30, args.consistency)
            loss_distillation = loss_kd + loss_infomax
            loss = loss_sup + consistency_weight * loss_distillation

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('pretrain/lr', lr_, iter_num)
            writer.add_scalar('pretrain/total_loss', loss, iter_num)
            

            logging.info(
                'iteration {} : loss : {:.4f}, '
                'loss_sup: {:.4f}, loss_kd: {:.4f}, loss_infomax: {:.4f}'.format(
                iter_num, loss.item(), 
                loss_sup.item(), loss_kd.item(), loss_infomax.item()))

            if iter_num > 500 and iter_num % 200 == 0:
                model.eval()
                avg_metric = test_all_case(
                    model, args.data_dir, test_list=test_list, num_classes=args.num_classes, patch_size=args.patch_size,
                    stride_xy=16, stride_z=4, AMC=True)

                if avg_metric[:, 0].mean() > best_performance:
                    best_performance = avg_metric[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    torch.save(model.state_dict(), save_mode_path)
                    save_best = os.path.join(snapshot_path, 'best_model.pth')
                    save_best_ema = os.path.join(snapshot_path, 'best_ema_model.pth')
                    torch.save(model.state_dict(), save_best)
                    torch.save(ema_model.state_dict(), save_best_ema)

                for i in range(args.num_classes-1):
                    writer.add_scalar('val/class{}_val_dice_score'.format(i+1),
                                    avg_metric[i, 0], iter_num)
                    writer.add_scalar('val/class{}_val_hd95'.format(i+1),
                                    avg_metric[i, 2], iter_num)
                    logging.info(
                        'Class {}, iteration {} : dice_score : {:.2f}% jaccard : {:.2f}%  hd95 : {:.2f} assd : {:.2f}'.format(
                            i+1, iter_num, avg_metric[i, 0].mean()*100, avg_metric[i, 1].mean()*100,
                            avg_metric[i, 2].mean(), avg_metric[i, 3].mean()))

                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"

def fuse_pseudo_label_with_mask_fast(p_foundation, p_teacher, u_foundation, u_teacher, uncertainty_threshold=1.0):
    trust_foundation = (u_foundation <= uncertainty_threshold).float()  
    trust_teacher = (u_teacher <= uncertainty_threshold).float()
    trust_mask = ((trust_foundation + trust_teacher) > 0).float()

    both_trust = trust_foundation * trust_teacher
    only_teacher = (1.0 - trust_foundation) * trust_teacher
    only_foundation = trust_foundation * (1.0 - trust_teacher)

    w_foundation = torch.exp(-u_foundation)
    w_teacher = torch.exp(-u_teacher)
    w_sum = w_foundation + w_teacher + 1e-8
    w_foundation /= w_sum
    w_teacher = 1.0 - w_foundation
    fused = w_foundation * p_foundation + w_teacher * p_teacher

    pseudo_fused = (
        both_trust * fused +
        only_teacher * p_teacher +
        only_foundation * p_foundation
    )
    new_mask = torch.argmax(pseudo_fused, dim=1)
    return new_mask, trust_mask

def ssl_train(args, snapshot_path):
    base_lr = args.pretrain_base_lr
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    num_classes = args.num_classes
    
    db_train, test_list  = load_dataset(args)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    labeled_idxs = list(range(0, args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num, len(db_train)))
    batch_sampler = TwoStreamBatchSampler_new(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs, drop_last=True, shuffle=True)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model = create_model()
    ema_model = create_model(ema=True)
    if args.checkpoint_dir is not None:
        model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir,'best_model.pth')))
        ema_model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir,'best_ema_model.pth')))

    sam_model = sam_model_registry3D['vit_b_ori'](checkpoint=None)
    sam_model = load_from(sam_model, args.checkpoint_sam).cuda()

    model.train()
    ema_model.train()

    model.train()
    # optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    def entropy(p, eps=1e-8, softmax=False):
        if softmax:
            p = torch.softmax(p, dim=1)
        entropy_max = math.log(args.num_classes)
        return -torch.sum(p * torch.log(p + eps), dim=1) / (entropy_max + eps)  

    def linear_warmup_cosine_annealing(iteration, warmup_iters, max_iter, base_lr, min_lr):
        if iteration < warmup_iters:
            return base_lr * (iteration / warmup_iters)
        else:
            return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * (iteration - warmup_iters) / (max_iter - warmup_iters)))

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)

    # loss func
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(nclass=args.num_classes)
    maskdice_loss = losses.mask_DiceLoss(nclass=args.num_classes)
    focal_loss = losses.FocalLoss()
    iou_loss = losses.SoftIoULoss(nclass=args.num_classes)
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    sub_bs = int(args.labeled_bs/2)
    min_lr = base_lr * 0.01
    
    def weighted_mask(u_foundation, u_teacher):
        w_foundation = torch.exp(-u_foundation)
        w_teacher = torch.exp(-u_teacher)
        w_sum = w_foundation + w_teacher
        w_foundation /= w_sum
        w_teacher = 1.0 - w_foundation
        return w_foundation, w_teacher

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:args.labeled_bs]
            unimg_a, unimg_b = volume_batch[args.labeled_bs:args.labeled_bs+sub_bs], volume_batch[args.labeled_bs+sub_bs:]

            with torch.no_grad():
                hiddenstates_u_a = get_sam_hidden(unimg_a, sam_model)
                hiddenstates_u_b = get_sam_hidden(unimg_b, sam_model)
                unoutput_a, _ = ema_model(unimg_a, hidden_states_out_sam=hiddenstates_u_a, unlab=True)
                unoutput_b, _ = ema_model(unimg_b, hidden_states_out_sam=hiddenstates_u_b, unlab=True)
                plab_a = get_cut_mask(unoutput_a, nms=1)
                plab_b = get_cut_mask(unoutput_b, nms=1)
                u_teacher_a = entropy(unoutput_a, softmax=True)
                u_teacher_b = entropy(unoutput_b, softmax=True)
                img_mask, loss_mask = context_mask(img_a, args.mask_ratio)
                _, _, mask_foundation_a, p_foundation_a = get_sam_features(
                    unimg_a, plab_a, sam_model, args, device=device, 
                    click_method='uncertain', num_clicks=args.num_clicks, 
                    prev_masks=None, GTM_rank_map=True, entropy_maps=u_teacher_a) 
                _, _, mask_foundation_b, p_foundation_b = get_sam_features(
                    unimg_b, plab_b, sam_model, args, device=device, 
                    click_method='uncertain', num_clicks=args.num_clicks, 
                    prev_masks=None, GTM_rank_map=True, entropy_maps=u_teacher_b) 
                u_foundation_a = entropy(p_foundation_a, softmax=False)
                u_foundation_b = entropy(p_foundation_b, softmax=False)

            consistency_weight = get_current_consistency_weight(iter_num // 30, args.consistency)

            mixl_img = img_a * img_mask + unimg_a * (1 - img_mask)
            mixu_img = unimg_b * img_mask + img_b * (1 - img_mask)

            with torch.no_grad():
                hiddenstates_mixl = get_sam_hidden(mixl_img, sam_model)
                hiddenstates_mixu = get_sam_hidden(mixu_img, sam_model)

            outputs_l, loss_infomax_mixl = model(mixl_img, hidden_states_out_sam=hiddenstates_mixl, unlab=True)
            outputs_u, loss_infomax_mixu = model(mixu_img, hidden_states_out_sam=hiddenstates_mixu, unlab=True)

            p_teacher_a = torch.softmax(unoutput_a, dim=1)
            p_teacher_b = torch.softmax(unoutput_b, dim=1)
            threshold = 0.75+0.25*ramps.sigmoid_rampup(iter_num,
                                                        max_iterations)
            newplab_a, plab_mask_a = fuse_pseudo_label_with_mask_fast(p_foundation_a, p_teacher_a, u_foundation_a, u_teacher_a, uncertainty_threshold=threshold)
            newplab_b, plab_mask_b = fuse_pseudo_label_with_mask_fast(p_foundation_b, p_teacher_b, u_foundation_b, u_teacher_b, uncertainty_threshold=threshold)

            loss_l = masked_mix_loss(outputs_l, lab_a, newplab_a, loss_mask, plab_mask_a, u_weight=args.u_weight, num_classes=args.num_classes)
            loss_u = masked_mix_loss(outputs_u, newplab_b, lab_b, loss_mask, plab_mask_b, u_weight=args.u_weight, unlab=True, num_classes=args.num_classes)

            loss_infomax = loss_infomax_mixl + loss_infomax_mixu
            loss = loss_l + loss_u + 0.1*loss_infomax

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_


            iter_num = iter_num + 1
            writer.add_scalar('ssl/lr', lr_, iter_num)
            writer.add_scalar('ssl/total_loss', loss, iter_num)

            logging.info(
                'iteration {} : loss : {:.4f}, lr : {:.4f}, loss_infomax : {:.4f}, '
                'loss_l : {:.4f},  loss_u : {:.4f}, '.format(
                iter_num, loss.item(), lr_, loss_infomax.item(),
                loss_l.item(), loss_u.item(), ))

            if iter_num > 0 and iter_num % 100 == 0:
                model.eval()
                avg_metric = test_all_case(
                    model, args.data_dir, test_list=test_list, num_classes=args.num_classes, patch_size=args.patch_size,
                    stride_xy=16, stride_z=4, AMC=True)

                if avg_metric[:, 0].mean() > best_performance:
                    best_performance = avg_metric[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path, 'best_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                for i in range(args.num_classes-1):
                    writer.add_scalar('val/class{}_val_dice_score'.format(i+1),
                                    avg_metric[i, 0], iter_num)
                    writer.add_scalar('val/class{}_val_hd95'.format(i+1),
                                    avg_metric[i, 2], iter_num)
                    logging.info(
                        'Class {}, iteration {} : dice_score : {:.2f}% jaccard : {:.2f}%  hd95 : {:.2f} assd : {:.2f}'.format(
                            i+1, iter_num, avg_metric[i, 0].mean()*100, avg_metric[i, 1].mean()*100,
                            avg_metric[i, 2].mean(), avg_metric[i, 3].mean()))
                    logging.info(
                        'Class {}:\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'.format(
                            i+1, avg_metric[i, 0].mean()*100, avg_metric[i, 1].mean()*100,
                            avg_metric[i, 2].mean(), avg_metric[i, 3].mean()))

                logging.info(
                    '{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'.format(
                        avg_metric[:, 0].mean()*100, avg_metric[:, 1].mean()*100,
                        avg_metric[:, 2].mean(), avg_metric[:, 3].mean()))

                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = os.path.join(args.save_path, "{}/lab-{}/{}/{}".format(
        args.dataset, args.labeled_num, args.method, args.mode))
    if 'TBAD' in args.data_dir:
        snapshot_path = os.path.join(args.save_path, "{}/fold-{}/lab-{}/{}/{}".format(
            args.dataset, args.fold_num, args.labeled_num, args.method, args.mode))

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/codes'):
        shutil.rmtree(snapshot_path + '/codes')
    shutil.copytree('.', snapshot_path + '/codes',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s %(message)s')

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(snapshot_path+'/'+'snapshot.log', encoding='utf8')
    fh.setFormatter(formatter) 
    logger.addHandler(fh)
    logging.info('Code launch on device: {}'.format(device))
    logging.info(str(args))
    logging.info('Save at : {}'.format(snapshot_path))
    if args.mode == 'PRETRAIN':
        pretrain(args, snapshot_path)
    else:
        ssl_train(args, snapshot_path)
