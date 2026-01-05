# -*- coding: utf-8 -*-
"""
The implementation is borrowed from: https://github.com/HiLab-git/PyMIC
"""
from __future__ import division, print_function

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform
import sys
sys.path.append('.')
from networks.simple2dvit import SimpleViT
import torch.nn.functional as F

def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def sparse_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.sparse_(m.weight, sparsity=0.1)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

    
class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)

        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output


class Decoder_AMC(nn.Module):
    def __init__(self, params):
        super(Decoder_AMC, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.n_branches = self.params['n_branches']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.branchs = nn.ModuleList()
        for i in range(self.n_branches):
            seq = nn.Conv2d(self.ft_chns[0], self.n_class,
                                    kernel_size=3, padding=1)
            self.branchs.append(seq)
        self.cross_attn = CrossAttentionFusion(dim_q=self.ft_chns[4], dim_k=256, heads=4)

    def forward(self, feature, z):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x44 = self.cross_attn(x4, z) 
        x = self.up1(x44, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        # output = self.out_conv(x)
        output = []
        for branch in self.branchs:
            o = branch(x)
            output.append(o)
        return output


def Dropout(x, p=0.3):
    x = torch.nn.functional.dropout(x, p)
    return x


def FeatureDropout(x):
    attention = torch.mean(x, dim=1, keepdim=True)
    max_val, _ = torch.max(attention.view(
        x.size(0), -1), dim=1, keepdim=True)
    threshold = max_val * np.random.uniform(0.7, 0.9)
    threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
    drop_mask = (attention < threshold).float()
    x = x.mul(drop_mask)
    return x


class FeatureNoise(nn.Module):
    def __init__(self, uniform_range=0.3):
        super(FeatureNoise, self).__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(
            x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        return x

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim_q=256, dim_k=32, heads=4):
        super(CrossAttentionFusion, self).__init__()
        self.q_proj = nn.Linear(dim_q, dim_q)
        self.k_proj = nn.Linear(dim_k, dim_q)
        self.v_proj = nn.Linear(dim_k, dim_q)
        self.out_proj = nn.Linear(dim_q, dim_q)
        self.scale = (dim_q // heads) ** -0.5

    def forward(self, F_VNet, F_Knowledge):
        B, C, H, W  = F_VNet.shape
        F_VNet_flatten = F_VNet.view(B, C, -1).transpose(-2, -1)  # (B, H*W, C)
        C_Knowledge = F_Knowledge.shape[1]
        F_Knowledge_flatten = F_Knowledge.view(B, C_Knowledge, -1).transpose(-2, -1)
        Q = self.q_proj(F_VNet_flatten)  # (B, H*W, C)
        K = self.k_proj(F_Knowledge_flatten)  # (B, N_tokens, C)
        V = self.v_proj(F_Knowledge_flatten)  # (B, N_tokens, C)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H*W, N_tokens)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_out = torch.matmul(attn_weights, V)  # (B, H*W, C)
        attn_out = self.out_proj(attn_out).transpose(-2, -1).view(B, C, H, W)  # 恢复形状

        return F_VNet + attn_out  # 残差连接

class Prob_UNet(nn.Module):
    def __init__(self, in_chns, class_num, n_branches=4):
        super(Prob_UNet, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu',
                  'n_branches': n_branches}

        self.encoder = Encoder(params)
        self.decoder = Decoder_AMC(params)

        self.s_prior_encoder = SimpleViT(
                                image_size = 256,
                                patch_size = 16,
                                dim = 128,
                                depth = 4,
                                heads = 4,
                                channels = 1,
                                mlp_dim = 256,
                                out_chans = 256,
                                )


        # self.cross_attn = CrossAttentionFusion(dim_q=n_filters * 16, dim_k=128, heads=4)
        self.proj = nn.Sequential(
                    nn.Conv2d(768, 128, kernel_size=1),     
                    nn.AvgPool2d(kernel_size=4)               
                )
        self.proj_align = nn.Sequential(
                    nn.Conv2d(256 * (class_num-1), 256, kernel_size=1),     
                    nn.AvgPool2d(kernel_size=4)               
                )

    def compute_maxinfo(self, h_sam, h_enc):
        loss_infomax1 = nn.MSELoss()(self.proj(h_sam[5]), h_enc[1])
        loss_infomax2 = nn.MSELoss()(self.proj(h_sam[11]), h_enc[2])
        loss_infomax = loss_infomax1 + loss_infomax2
        return loss_infomax/2
        
    def extract_prior_predictions(self, input):
        features = self.encoder(input)
        _, s_prior_features = self.s_prior_encoder(input, return_mid=True)

        out = self.decoder(features, s_prior_features)
        outputs = 0.25*(out[0]+out[1]+out[2]+out[3])
        # out_masks = torch.argmax(torch.softmax(outputs, dim = 1), dim=1)
        return outputs

    def forward(self, input, sam_post_embed=None, hidden_states_out_sam=None, turnoff_drop=False, unlab=False):
        features = self.encoder(input)
        hidden_states_out, s_prior_features = self.s_prior_encoder(input, return_mid=True)

        if self.training:
            loss_infomax = self.compute_maxinfo(hidden_states_out_sam, hidden_states_out)
            if unlab:
                out = self.decoder(features, s_prior_features)
                outputs = 0.25*(out[0]+out[1]+out[2]+out[3])
                return outputs, loss_infomax
            else:
                F_posterior = self.proj_align(sam_post_embed)
                out = self.decoder(features, F_posterior)
                    
                loss_kd = nn.MSELoss()(s_prior_features, F_posterior)
                return out, loss_kd, loss_infomax
        else:
            out = self.decoder(features, s_prior_features)
            return out



if __name__ == '__main__':
    # compute FLOPS & PARAMETERS
    from thop import profile
    from thop import clever_format
    model = Prob_UNet(in_chns=1, class_num=5).to("cuda:3")
    input = torch.randn(2, 1, 256, 256).to("cuda:3")

    sam_post_embed = torch.randn(2, 256*4, 64, 64).to("cuda:3")
    hidden_states_out_sam = [torch.randn(2, 768, 64, 64).to("cuda:3") for i in range(12)]

    flops, params = profile(model, inputs=(input,sam_post_embed,hidden_states_out_sam, ))
    print(flops, params)
    macs, params = clever_format([flops, params], "%.3f")
    print(macs, params)
    print("UNet have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))


    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_params = clever_format([trainable_params], "%.3f")
    print(f"Trainable parameters: {trainable_params}")

    trainable_params = sum(p.numel() for (n, p) in model.named_parameters() if 'cross_attn' in n)
    trainable_params = clever_format([trainable_params], "%.3f")
    print(f"Cross Attention parameters: {trainable_params}")

    trainable_params = sum(p.numel() for (n, p) in model.named_parameters() if 'proj' in n)
    trainable_params = clever_format([trainable_params], "%.3f")
    print(f"Proj parameters: {trainable_params}")

    trainable_params = sum(p.numel() for (n, p) in model.named_parameters() if 'proj_align' in n)
    trainable_params = clever_format([trainable_params], "%.3f")
    print(f"Proj Align parameters: {trainable_params}")

    trainable_params = sum(p.numel() for (n, p) in model.named_parameters() if 's_prior_encoder' in n)
    trainable_params = clever_format([trainable_params], "%.3f")
    print(f"Student Prior Encoder parameters: {trainable_params}")