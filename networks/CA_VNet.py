from typing import Tuple, Union

import torch
import torch.nn as nn
from torch.distributions import LowRankMultivariateNormal, kl

from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.nets import ViT

import sys 
sys.path.append('.')
from SAM_Med3D.segment_anything.build_sam3D import sam_model_registry3D
from SAM_Med3D.segment_anything.modeling.image_encoder3D import ImageEncoderViT3D
from networks.simple3dvit import SimpleViT
from functools import partial

from networks.vnet import ConvBlock, ResidualConvBlock, DownsamplingConvBlock, UpsamplingDeconvBlock, Upsampling
from utils.kf_losses import InfoMax_loss
import torch.nn.functional as F
from utils import losses
import itertools


class LowRankGaussianMixtureModel(nn.Module):
    def __init__(self, n_components, n_channels, hidden_dim, epsilon=1e-5, rank=10, mode='posterior'):
        super(LowRankGaussianMixtureModel, self).__init__()
        self.n_components = n_components
        self.n_channels = n_channels
        self.rank = rank
        self.mode = mode
        self.epsilon = epsilon

        if mode == 'posterior':
            self.mean = nn.Conv3d(n_components * hidden_dim, n_components * n_channels, kernel_size=1, bias=True)
            self.log_cov_diag = nn.Conv3d(n_components * hidden_dim, n_components * n_channels, kernel_size=1, bias=True)
            self.cov_factor = nn.Conv3d(n_components * hidden_dim, n_components * n_channels * rank, kernel_size=1, bias=True)
        else:
            self.mean = nn.Conv3d(hidden_dim, n_components * n_channels, kernel_size=1, bias=True)
            self.log_cov_diag = nn.Conv3d(hidden_dim, n_components * n_channels, kernel_size=1, bias=True)
            self.cov_factor = nn.Conv3d(hidden_dim, n_components * n_channels * rank, kernel_size=1, bias=True)

    def forward(self, x, samples = None, sample_num=None):
        batch_size, hidden_dim, height, width, depth = x.size()
        mean = self.mean(x).reshape((batch_size, self.n_components, -1))
        
        log_cov_diag = self.log_cov_diag(x).reshape((batch_size, self.n_components, -1))
        cov_diag = torch.exp(log_cov_diag) + self.epsilon
        
        cov_factor = self.cov_factor(x).reshape((batch_size, self.n_components, self.n_channels, self.rank, -1))
        cov_factor = cov_factor.transpose(3,4)
        cov_factor = cov_factor.flatten(2,3) + self.epsilon
        # print(mean.shape, cov_diag.shape, cov_factor.shape)
        distributions = []
        samples = []
        if sample_num is not None:
            for i in range(self.n_components):
                dist = LowRankMultivariateNormal(mean[:, i, ...], cov_factor=cov_factor[:, i, ...], cov_diag=cov_diag[:, i, ...])
                sample = dist.rsample((sample_num,))
                samples.append(sample)
                distributions.append(dist)
        else:
            for i in range(self.n_components):
                dist = LowRankMultivariateNormal(mean[:, i, ...], cov_factor=cov_factor[:, i, ...], cov_diag=cov_diag[:, i, ...])
                sample = dist.rsample()
                samples.append(sample)
                distributions.append(dist)
        samples = torch.stack(samples, dim=1)
        return samples, distributions
    

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim_q=256, dim_k=32, heads=4):
        super(CrossAttentionFusion, self).__init__()
        self.q_proj = nn.Linear(dim_q, dim_q)
        self.k_proj = nn.Linear(dim_k, dim_q)
        self.v_proj = nn.Linear(dim_k, dim_q)
        self.out_proj = nn.Linear(dim_q, dim_q)
        self.scale = (dim_q // heads) ** -0.5

    def forward(self, F_VNet, F_Knowledge):
        B, C, H, W, D = F_VNet.shape
        F_VNet_flatten = F_VNet.view(B, C, -1).transpose(-2, -1)  # (B, H*W*D, C)
        C_Knowledge = F_Knowledge.shape[1]
        F_Knowledge_flatten = F_Knowledge.view(B, C_Knowledge, -1).transpose(-2, -1)
        Q = self.q_proj(F_VNet_flatten)  # (B, H*W*D, C)
        K = self.k_proj(F_Knowledge_flatten)  # (B, N_tokens, C)
        V = self.v_proj(F_Knowledge_flatten)  # (B, N_tokens, C)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H*W*D, N_tokens)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_out = torch.matmul(attn_weights, V)  # (B, H*W*D, C)
        attn_out = self.out_proj(attn_out).transpose(-2, -1).view(B, C, H, W, D)  # 恢复形状

        return F_VNet + attn_out  # 残差连接

class Prob_VNet(nn.Module):

    '''
    VNet_AMC modified from https://github.com/grant-jpg/FUSSNet
    '''
    
    def __init__(self, n_channels=1, n_classes=2, n_filters=16, 
                    normalization='instancenorm', has_dropout=False, n_branches = 3,):
        super(Prob_VNet, self).__init__()
        self.has_dropout = has_dropout

        # VNet Encoder
        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        # VNet decoder
        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        self.branchs = nn.ModuleList()

        for i in range(n_branches):
            if has_dropout:
                seq = nn.Sequential(
                    ConvBlock(1, n_filters, n_filters, normalization=normalization),
                    nn.Dropout3d(p=0.5),
                    nn.Conv3d(n_filters, n_classes, 1, padding=0)
                )
            else:
                seq = nn.Sequential(
                    ConvBlock(1, n_filters, n_filters, normalization=normalization),
                    nn.Conv3d(n_filters, n_classes, 1, padding=0)
                )
            self.branchs.append(seq)

        self.s_prior_encoder = SimpleViT(
                    image_size = 128,          # image size
                    frames = 128,               # number of frames
                    image_patch_size = n_filters,     # image patch size
                    frame_patch_size = n_filters,      # frame patch size
                    dim = 256,
                    depth = 4,
                    heads = 4,
                    channels = 1,
                    mlp_dim = 1024              # mlp_ratio * embed_dim = 4 * 256
                    )
        self.num_classes = n_classes

        self.cross_attn = CrossAttentionFusion(dim_q=n_filters * 16, dim_k=256, heads=4)
        self.proj = nn.Conv3d(768, 256, kernel_size=1, bias=False)
        self.proj_align = nn.Conv3d(384 * (n_classes-1), 256, kernel_size=1, bias=False)

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res


    def decoder(self, features, input_features=None):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]
        
        x55 = self.cross_attn(x5, input_features) 
        x5_up = self.block_five_up(x55)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        out = []
        for branch in self.branchs:
            o = branch(x8_up)
            out.append(o)
        return out
    
    def compute_maxinfo(self, h_sam, h_enc):
        loss_infomax1 = nn.MSELoss()(self.proj(h_sam[3]), h_enc[0])
        loss_infomax2 = nn.MSELoss()(self.proj(h_sam[7]), h_enc[1])
        loss_infomax3 = nn.MSELoss()(self.proj(h_sam[11]), h_enc[2])
        loss_infomax = loss_infomax1 + loss_infomax2 + loss_infomax3
        return loss_infomax/3

    def kl_divergence_loss(self, F_VNet, F_Knowledge, T=2.0):
        P_VNet = F.log_softmax(F_VNet / T, dim=-1)
        P_Knowledge = F.softmax(F_Knowledge / T, dim=-1)
        return F.kl_div(P_VNet, P_Knowledge, reduction='batchmean') * (T * T)

    def extract_prior_predictions(self, input):
        features = self.encoder(input)
        _, s_prior_features = self.s_prior_encoder(input, return_mid=True)

        out = self.decoder(features, s_prior_features)
        outputs = 0.25*(out[0]+out[1]+out[2]+out[3])
        # out_masks = torch.argmax(torch.softmax(outputs, dim = 1), dim=1)
        return outputs
        
    def forward(self, input, sam_post_embed=None, hidden_states_out_sam=None, turnoff_drop=False, unlab=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False

        features = self.encoder(input)
        hidden_states_out, s_prior_features = self.s_prior_encoder(input, return_mid=True)

        if self.training:
            loss_infomax = self.compute_maxinfo(hidden_states_out_sam, hidden_states_out)

            if unlab:
                out = self.decoder(features, s_prior_features)
                output = 0.25 * (out[0]+out[1]+out[2]+out[3])
                return output, loss_infomax

            else:
                F_posterior = self.proj_align(sam_post_embed)
                out = self.decoder(features, F_posterior)
            
                if turnoff_drop:
                    self.has_dropout = has_dropout
                    
                loss_kd = nn.MSELoss()(s_prior_features, F_posterior)
                return out, loss_kd, loss_infomax
        else:
            out = self.decoder(features, s_prior_features)
            return out

if __name__ == '__main__':

    # compute FLOPS & PARAMETERS
    from thop import profile
    from thop import clever_format


    model = Prob_VNet(n_channels=1, n_classes=3, n_branches=4).to("cuda:1")

    input = torch.randn(2, 1, 128, 128, 128).to("cuda:1")
    sam_post_embed = torch.randn(2, 384*2, 8, 8, 8).to("cuda:1")
    hidden_states_out_sam = [torch.randn(2, 768, 8, 8, 8).to("cuda:1") for i in range(12)]
    flops, params = profile(model, inputs=(input,sam_post_embed,hidden_states_out_sam, ))
    print(flops, params)
    macs, params = clever_format([flops, params], "%.3f")
    print(macs, params)
    print("Prob_VNet have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_params = clever_format([trainable_params], "%.3f")
    print(f"Trainable parameters: {trainable_params}")

    trainable_params = sum(p.numel() for (n, p) in model.named_parameters() if 'cross_attn' in n)
    trainable_params = clever_format([trainable_params], "%.3f")
    print(f"Encoder Posterior parameters: {trainable_params}")
