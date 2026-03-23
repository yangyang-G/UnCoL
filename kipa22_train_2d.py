import argparse
import logging
import os
import random
import shutil
import sys
import time
from skimage.measure import label
from tqdm import tqdm
import math
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
sys.path.append('.')
from datasets.oasis2d import TwoStreamBatchSampler
from datasets.kipa22 import KiPA22, RandomGenerator as KiPA22RandomGenerator
from networks.CA_UNet import Prob_UNet
from MedSAM.segment_anything import sam_model_registry
from MedSAM.MedSAM_2D_inference import get_medsam_features, get_medsam_hidden
from utils import losses, ramps
from utils.val_2D import test_single_volume, test_single_volume_ds
from utils.train_utils import cal_dice, update_ema_variables
from utils.BCP_utils import generate_mask, masked_mix_loss, context_mask, mix_loss, parameter_sharing
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str,
                    default='UnCoL', help='method_name')
parser.add_argument('--dataset', type=str,
                    default='KiPA22', help='dataset name')
parser.add_argument('--mode', type=str,
                    default='PRETRAIN', help='mode type')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')
parser.add_argument('--fold_num', type=int, default=0,
                    help='fold num of dataset')
parser.add_argument('--data_dir', type=str, default='../../data/KiPA22', help='Data source')
parser.add_argument('--save_path', type=str, default='./model', help='Save path')
parser.add_argument('--max_iterations', type=int,
                    default=15000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=6, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')

parser.add_argument('--labeled_bs', type=int, default=3,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=10,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=250.0, help='consistency_rampup')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--mask_ratio', type=float, default=2/3, help='ratio of mask/image')
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
parser.add_argument(
    "-chk",
    "--checkpoint_sam",
    type=str,
    default='/public/home/wyf_cdu/Project/Semi-Supervised/UnCol/code/datasets/weight/medsam_vit_b.pth',
    help="path to the trained MedSAM model",
)
parser.add_argument(
    "-chk_u",
    "--checkpoint_unet",
    type=str,
    help="path to the pretrained model in stage 1",
)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda")
def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)
def get_2DLargestCC(segmentation, num_classes):
    batch_list = []
    N = segmentation.shape[0]
    for i in range(0, N):
        class_list = []
        for c in range(1, num_classes):
            temp_seg = segmentation[i]
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
                class_list.append(largestCC * c)
            else:
                class_list.append(temp_prob)
        
        n_batch = sum(class_list)
        batch_list.append(n_batch)
    return torch.Tensor(batch_list).cuda()
def get_masks(output, nms=0, num_classes=4):
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)
    if nms == 1:
        probs = get_2DLargestCC(probs, num_classes)      
    return probs
def pretrain(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    def create_model(ema=False):
        # Network definition
        model = Prob_UNet(in_chns=1, class_num=args.num_classes, n_branches=4).cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model
    model = create_model()
    ema_model = create_model(ema=True)
    medsam_model = sam_model_registry["vit_b"](checkpoint=args.checkpoint_sam)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    db_train = KiPA22(base_dir=args.data_dir, split="train", num=args.labeled_num, transform=transforms.Compose([
        KiPA22RandomGenerator(args.patch_size),
    ]))
    db_val = KiPA22(base_dir=args.data_dir, split="val")
    total_slices = len(db_train)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, args.labeled_num))
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn, drop_last=True)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
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
                img_mask, loss_mask = generate_mask(img_a, args.mask_ratio)
            """Mix Input"""
            volume_batch = img_a * img_mask + img_b * (1 - img_mask)
            label_batch = lab_a * img_mask + lab_b * (1 - img_mask)
            with torch.no_grad():
                hidden_states_out_sam, posterior_features, _, _ = get_medsam_features(
                    volume_batch, label_batch, medsam_model, num_classes=args.num_classes) # torch.Size([B, 1, 128, 128, 128]) (B, 128, 128, 128)
            outputs, loss_kd, loss_infomax = model(volume_batch, hidden_states_out_sam=hidden_states_out_sam,
                                                sam_post_embed=posterior_features)
            loss_ce = ce_loss(outputs[0], label_batch)
            loss_dice = dice_loss(torch.softmax(outputs[1], dim=1), label_batch)
            loss_focal = focal_loss(outputs[2], label_batch)
            loss_iou = iou_loss(outputs[3], label_batch)
            loss_sup = 0.25 * (loss_ce+loss_dice+loss_focal+loss_iou)
            
            consistency_weight = get_current_consistency_weight(iter_num//30)
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
            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                total_metric = np.zeros((num_classes-1, 4))
                for i_batch, sampled_batch in tqdm(enumerate(valloader)):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes, AMC=True)
                    total_metric += np.array(metric_i)
                total_metric = total_metric / len(db_val)
                if total_metric[:, 0].mean() > best_performance:
                    best_performance = total_metric[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    torch.save(model.state_dict(), save_mode_path)
                    save_best = os.path.join(snapshot_path, 'best_model.pth')
                    save_best_ema = os.path.join(snapshot_path, 'best_ema_model.pth')
                    torch.save(model.state_dict(), save_best)
                    torch.save(ema_model.state_dict(), save_best_ema)
                for i in range(num_classes-1):
                    writer.add_scalar('val/class{}_val_dice_score'.format(i+1),
                                    total_metric[i, 0], iter_num)
                    writer.add_scalar('val/class{}_val_hd95'.format(i+1),
                                    total_metric[i, 2], iter_num)
                    logging.info(
                        'Class {}, iteration {} : dice_score : {:.2f}% jaccard : {:.2f}%  hd95 : {:.2f} assd : {:.2f}'.format(
                            i+1, iter_num, total_metric[i, 0]*100, total_metric[i, 1]*100,
                            total_metric[i, 2], total_metric[i, 3]))
                    logging.info(
                        'Class {}:\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'.format(
                            i+1, total_metric[i, 0]*100, total_metric[i, 1]*100,
                            total_metric[i, 2], total_metric[i, 3]))
                logging.info(
                    '{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'.format(
                        total_metric[:, 0].mean()*100, total_metric[:, 1].mean()*100,
                        total_metric[:, 2].mean(), total_metric[:, 3].mean()))
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
    new_mask = torch.argmax(fused, dim=1)
    return new_mask, trust_mask
def ssl_train(args, snapshot_path):
    base_lr = args.base_lr
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    db_train = KiPA22(base_dir=args.data_dir, split="train", transform=transforms.Compose([
        KiPA22RandomGenerator(args.patch_size),
    ]))
    db_val = KiPA22(base_dir=args.data_dir, split="val")
    labeled_idxs = list(range(0, args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num, len(db_train)))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)
    unlabeled_bs = batch_size-args.labeled_bs
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)
    total_slices = len(db_train)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, args.labeled_num))
    def create_model(ema=False):
        # Network definition
        model = Prob_UNet(in_chns=1, class_num=args.num_classes, n_branches=4).cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model
    def entropy(p, eps=1e-8, softmax=False):
        if softmax:
            p = torch.softmax(p, dim=1)
        entropy_max = math.log(args.num_classes)
        return -torch.sum(p * torch.log(p + eps), dim=1, keepdim=True) / (entropy_max + eps)
    model = create_model()
    ema_model = create_model(ema=True)
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_unet,'best_model.pth')))
    ema_model.load_state_dict(torch.load(os.path.join(args.checkpoint_unet,'best_ema_model.pth')))
    medsam_model = sam_model_registry["vit_b"](checkpoint=args.checkpoint_sam)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()
    model.train()
    ema_model.train()
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
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
    
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:args.labeled_bs]
            unimg_a, unimg_b = volume_batch[args.labeled_bs:args.labeled_bs+sub_bs], volume_batch[args.labeled_bs+sub_bs:]
            # === 1. Get predictions & uncertainty from two teachers ===
            with torch.no_grad():
                hiddenstates_u_a = get_medsam_hidden(unimg_a, medsam_model)
                hiddenstates_u_b = get_medsam_hidden(unimg_b, medsam_model)
                unoutput_a, _ = ema_model(unimg_a, hidden_states_out_sam=hiddenstates_u_a, unlab=True)
                unoutput_b, _ = ema_model(unimg_b, hidden_states_out_sam=hiddenstates_u_b, unlab=True)
                plab_a = get_masks(unoutput_a, nms=1, num_classes=args.num_classes)
                plab_b = get_masks(unoutput_b, nms=1, num_classes=args.num_classes)
                img_mask, loss_mask = generate_mask(img_a, args.mask_ratio)
                u_teacher_a = entropy(unoutput_a, softmax=True)
                u_teacher_b = entropy(unoutput_b, softmax=True)
                # Foundation model predictions (prompt-guided)
                _, _, mask_foundation_a, p_foundation_a = get_medsam_features(
                    unimg_a, plab_a, medsam_model, num_classes=args.num_classes) 
                _, _, mask_foundation_b, p_foundation_b = get_medsam_features(
                    unimg_b, plab_b, medsam_model, num_classes=args.num_classes) 
                u_foundation_a = entropy(p_foundation_a, softmax=False)
                u_foundation_b = entropy(p_foundation_b, softmax=False)
            
            consistency_weight = get_current_consistency_weight(iter_num // 30)
            mixl_img = img_a * img_mask + unimg_a * (1 - img_mask)
            mixu_img = unimg_b * img_mask + img_b * (1 - img_mask)
            with torch.no_grad():
                hiddenstates_mixl = get_medsam_hidden(mixl_img, medsam_model)
                hiddenstates_mixu = get_medsam_hidden(mixu_img, medsam_model)
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
                loss_l.item(), loss_u.item()))
            if iter_num > 0 and iter_num % 100 == 0:
                model.eval()
                total_metric = np.zeros((args.num_classes-1, 4))
                for i_batch, sampled_batch in tqdm(enumerate(valloader)):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model, classes=args.num_classes, AMC=True)
                    total_metric += np.array(metric_i)
                total_metric = total_metric / len(db_val)
                if total_metric[:, 0].mean() > best_performance:
                    best_performance = total_metric[:, 0].mean()
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
                                    total_metric[i, 0], iter_num)
                    writer.add_scalar('val/class{}_val_hd95'.format(i+1),
                                    total_metric[i, 2], iter_num)
                    logging.info(
                        'Class {}, iteration {} : dice_score : {:.2f}% jaccard : {:.2f}%  hd95 : {:.2f} assd : {:.2f}'.format(
                            i+1, iter_num, total_metric[i, 0]*100, total_metric[i, 1]*100,
                            total_metric[i, 2], total_metric[i, 3]))
                    logging.info(
                        'Class {}:\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'.format(
                            i+1, total_metric[i, 0]*100, total_metric[i, 1]*100,
                            total_metric[i, 2], total_metric[i, 3]))
                logging.info(
                    '{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'.format(
                        total_metric[:, 0].mean()*100, total_metric[:, 1].mean()*100,
                        total_metric[:, 2].mean(), total_metric[:, 3].mean()))
                model.train()
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