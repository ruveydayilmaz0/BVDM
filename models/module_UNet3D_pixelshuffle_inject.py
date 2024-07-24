# -*- coding: utf-8 -*-
"""
Implementation of the 3D UNet architecture with PixelShuffle upsampling.
https://arxiv.org/pdf/1609.05158v2.pdf
"""

import math
import torch
import torch.nn as nn

from utils.layers import PixelShuffle3d


# code from https://huggingface.co/blog/annotated-diffusion
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# adapted from:
# https://medium.com/@vedantjumle/class-conditioned-diffusion-models-using-keras-and-\tensorflow-9997fa6d958c
class ClassConditioning(nn.Module):
    def __init__(self, num_classes=3, channels=16):
        super().__init__()
        self.num_classes = num_classes
        self.emb_size = channels
        self.sin_emb = SinusoidalPositionEmbeddings(self.emb_size)

    def forward(self, sizes):
        phases = sizes[0]
        patch_size = sizes[1]
        emb = []
        for phase in phases:
            emb.append(
                self.sin_emb(2 * torch.pi * torch.tensor([phase]) / self.num_classes)
            )

        # emb = torch.FloatTensor(emb)##
        emb = torch.cat(emb, dim=0).type("torch.FloatTensor")
        emb = emb[..., None].permute(2, 1, 0)
        pad_width = (patch_size - emb.shape[2], 0, 0, 0, 0, 0)
        emb = nn.functional.pad(emb, pad_width, mode="constant")
        return emb.to("cuda")
        # return emb.repeat(1,self.emb_size//2,1).to(device='cuda')


class module_UNet3D_pixelshuffle_inject(nn.Module):
    """Implementation of the 3D U-Net architecture."""

    def __init__(
        self,
        patch_size,
        in_channels,
        out_channels,
        feat_channels=16,
        t_channels=128,
        c_channels=64,
        out_activation="sigmoid",
        layer_norm="none",
        **kwargs
    ):
        super(module_UNet3D_pixelshuffle_inject, self).__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feat_channels = feat_channels
        self.t_channels = t_channels
        self.layer_norm = layer_norm  # instance | batch | none
        self.out_activation = (
            out_activation  # relu | leakyrelu | sigmoid | tanh | hardtanh | none
        )

        self.norm_methods = {
            "instance": nn.InstanceNorm3d,
            "batch": nn.BatchNorm3d,
            "none": nn.Identity,
        }

        self.out_activations = nn.ModuleDict(
            {
                "relu": nn.ReLU(),
                "leakyrelu": nn.LeakyReLU(negative_slope=0.2, inplace=True),
                "sigmoid": nn.Sigmoid(),
                "tanh": nn.Tanh(),
                "hardtanh": nn.Hardtanh(0, 1),
                "none": nn.Identity(),
            }
        )

        # Add class conditioning for mitosis stages num_classes=3, class_emb_dim=in_channels
        # self.class_embeddings = nn.Embedding(3, 64)
        # self.cond = ClassConditioning()

        # Define layer instances
        self.t1 = nn.Sequential(
            SinusoidalPositionEmbeddings(t_channels),
            nn.Linear(t_channels, feat_channels),
            nn.PReLU(feat_channels),
            nn.Linear(feat_channels, feat_channels),
        )
        self.cond1 = nn.Sequential(ClassConditioning(channels=feat_channels))
        self.c1 = nn.Sequential(
            nn.Conv3d(in_channels, feat_channels // 2, kernel_size=3, padding=1),
            nn.PReLU(feat_channels // 2),
            self.norm_methods[self.layer_norm](feat_channels // 2),
            nn.Conv3d(feat_channels // 2, feat_channels, kernel_size=3, padding=1),
            nn.PReLU(feat_channels),
            self.norm_methods[self.layer_norm](feat_channels),
        )
        self.d1 = nn.Sequential(
            nn.Conv3d(feat_channels, feat_channels, kernel_size=4, stride=2, padding=1),
            nn.PReLU(feat_channels),
            self.norm_methods[self.layer_norm](feat_channels),
        )

        self.t2 = nn.Sequential(
            SinusoidalPositionEmbeddings(t_channels),
            nn.Linear(t_channels, feat_channels * 2),
            nn.PReLU(feat_channels * 2),
            nn.Linear(feat_channels * 2, feat_channels * 2),
        )
        self.cond2 = nn.Sequential(ClassConditioning(channels=feat_channels * 2))
        self.c2 = nn.Sequential(
            nn.Conv3d(feat_channels, feat_channels, kernel_size=3, padding=1),
            nn.PReLU(feat_channels),
            self.norm_methods[self.layer_norm](feat_channels),
            nn.Conv3d(feat_channels, feat_channels * 2, kernel_size=3, padding=1),
            nn.PReLU(feat_channels * 2),
            self.norm_methods[self.layer_norm](feat_channels * 2),
        )
        self.d2 = nn.Sequential(
            nn.Conv3d(
                feat_channels * 2, feat_channels * 2, kernel_size=4, stride=2, padding=1
            ),
            nn.PReLU(feat_channels * 2),
            self.norm_methods[self.layer_norm](feat_channels * 2),
        )

        self.t3 = nn.Sequential(
            SinusoidalPositionEmbeddings(t_channels),
            nn.Linear(t_channels, feat_channels * 4),
            nn.PReLU(feat_channels * 4),
            nn.Linear(feat_channels * 4, feat_channels * 4),
        )
        self.cond3 = nn.Sequential(ClassConditioning(channels=feat_channels * 4))
        self.c3 = nn.Sequential(
            nn.Conv3d(feat_channels * 2, feat_channels * 2, kernel_size=3, padding=1),
            nn.PReLU(feat_channels * 2),
            self.norm_methods[self.layer_norm](feat_channels * 2),
            nn.Conv3d(feat_channels * 2, feat_channels * 4, kernel_size=3, padding=1),
            nn.PReLU(feat_channels * 4),
            self.norm_methods[self.layer_norm](feat_channels * 4),
        )
        self.d3 = nn.Sequential(
            nn.Conv3d(
                feat_channels * 4, feat_channels * 4, kernel_size=4, stride=2, padding=1
            ),
            nn.PReLU(feat_channels * 4),
            self.norm_methods[self.layer_norm](feat_channels * 4),
        )

        self.t4 = nn.Sequential(
            SinusoidalPositionEmbeddings(t_channels),
            nn.Linear(t_channels, feat_channels * 8),
            nn.PReLU(feat_channels * 8),
            nn.Linear(feat_channels * 8, feat_channels * 8),
        )
        self.cond4 = nn.Sequential(ClassConditioning(channels=feat_channels * 8))
        self.c4 = nn.Sequential(
            nn.Conv3d(feat_channels * 4, feat_channels * 4, kernel_size=3, padding=1),
            nn.PReLU(feat_channels * 4),
            self.norm_methods[self.layer_norm](feat_channels * 4),
            nn.Conv3d(feat_channels * 4, feat_channels * 8, kernel_size=3, padding=1),
            nn.PReLU(feat_channels * 8),
            self.norm_methods[self.layer_norm](feat_channels * 8),
        )

        self.u1 = nn.Sequential(
            nn.Conv3d(feat_channels * 8, feat_channels * 8, kernel_size=1),
            nn.PReLU(feat_channels * 8),
            self.norm_methods[self.layer_norm](feat_channels * 8),
            PixelShuffle3d(2),
        )
        self.t5 = nn.Sequential(
            SinusoidalPositionEmbeddings(t_channels),
            nn.Linear(t_channels, feat_channels * 8),
            nn.PReLU(feat_channels * 8),
            nn.Linear(feat_channels * 8, feat_channels * 8),
        )
        self.cond5 = nn.Sequential(ClassConditioning(channels=feat_channels * 8))
        self.c5 = nn.Sequential(
            nn.Conv3d(feat_channels * 5, feat_channels * 8, kernel_size=3, padding=1),
            nn.PReLU(feat_channels * 8),
            self.norm_methods[self.layer_norm](feat_channels * 8),
            nn.Conv3d(feat_channels * 8, feat_channels * 8, kernel_size=3, padding=1),
            nn.PReLU(feat_channels * 8),
            self.norm_methods[self.layer_norm](feat_channels * 8),
        )

        self.u2 = nn.Sequential(
            nn.Conv3d(feat_channels * 8, feat_channels * 8, kernel_size=1),
            nn.PReLU(feat_channels * 8),
            self.norm_methods[self.layer_norm](feat_channels * 8),
            PixelShuffle3d(2),
        )
        self.t6 = nn.Sequential(
            SinusoidalPositionEmbeddings(t_channels),
            nn.Linear(t_channels, feat_channels * 8),
            nn.PReLU(feat_channels * 8),
            nn.Linear(feat_channels * 8, feat_channels * 8),
        )
        self.cond6 = nn.Sequential(ClassConditioning(channels=feat_channels * 8))
        self.c6 = nn.Sequential(
            nn.Conv3d(feat_channels * 3, feat_channels * 8, kernel_size=3, padding=1),
            nn.PReLU(feat_channels * 8),
            self.norm_methods[self.layer_norm](feat_channels * 8),
            nn.Conv3d(feat_channels * 8, feat_channels * 8, kernel_size=3, padding=1),
            nn.PReLU(feat_channels * 8),
            self.norm_methods[self.layer_norm](feat_channels * 8),
        )

        self.u3 = nn.Sequential(
            nn.Conv3d(feat_channels * 8, feat_channels * 8, kernel_size=1),
            nn.PReLU(feat_channels * 8),
            self.norm_methods[self.layer_norm](feat_channels * 8),
            PixelShuffle3d(2),
        )
        self.t7 = nn.Sequential(
            SinusoidalPositionEmbeddings(t_channels),
            nn.Linear(t_channels, feat_channels),
            nn.PReLU(feat_channels),
            nn.Linear(feat_channels, feat_channels),
        )
        self.cond7 = nn.Sequential(ClassConditioning(channels=feat_channels))
        self.c7 = nn.Sequential(
            nn.Conv3d(feat_channels * 2, feat_channels, kernel_size=3, padding=1),
            nn.PReLU(feat_channels),
            self.norm_methods[self.layer_norm](feat_channels),
            nn.Conv3d(feat_channels, feat_channels, kernel_size=3, padding=1),
            nn.PReLU(feat_channels),
            self.norm_methods[self.layer_norm](feat_channels),
        )

        self.out = nn.Sequential(
            nn.Conv3d(feat_channels, feat_channels, kernel_size=3, padding=1),
            nn.PReLU(feat_channels),
            self.norm_methods[self.layer_norm](feat_channels),
            nn.Conv3d(feat_channels, feat_channels, kernel_size=3, padding=1),
            nn.PReLU(feat_channels),
            self.norm_methods[self.layer_norm](feat_channels),
            nn.Conv3d(feat_channels, out_channels, kernel_size=1),
            self.out_activations[self.out_activation],
        )

    def forward(self, img, t, phases):

        x = img
        t1 = self.t1(t)
        cond1 = self.cond1((phases, x.shape[-3]))
        c1 = self.c1(x) + t1[..., None, None, None] + 10 * cond1[..., None, None]
        # c1 = self.c1(x)+t1[...,None,None,None]+cond1[...,None,None]
        d1 = self.d1(c1)

        t2 = self.t2(t)
        cond2 = self.cond2((phases, d1.shape[-3]))
        c2 = self.c2(d1) + t2[..., None, None, None] + 10 * cond2[..., None, None]
        # c2 = self.c2(d1)+t2[...,None,None,None]+cond2[...,None,None]
        d2 = self.d2(c2)

        t3 = self.t3(t)
        cond3 = self.cond3((phases, d2.shape[-3]))
        # c3 = self.c3(d2)+t3[...,None,None,None]+cond3[...,None,None]
        c3 = self.c3(d2) + t3[..., None, None, None] + 10 * cond3[..., None, None]
        d3 = self.d3(c3)

        t4 = self.t4(t)
        cond4 = self.cond4((phases, d3.shape[-3]))
        # c4 = self.c4(d3)+t4[...,None,None,None]+cond4[...,None,None]
        c4 = self.c4(d3) + t4[..., None, None, None] + 10 * cond4[..., None, None]

        u1 = self.u1(c4)
        t5 = self.t5(t)
        cond5 = self.cond5((phases, u1.shape[-3]))
        # c5 = self.c5(torch.cat((u1,c3),1))+t5[...,None,None,None]+cond5[...,None,None]
        c5 = (
            self.c5(torch.cat((u1, c3), 1))
            + t5[..., None, None, None]
            + 10 * cond5[..., None, None]
        )

        u2 = self.u2(c5)
        t6 = self.t6(t)
        cond6 = self.cond6((phases, u2.shape[-3]))
        # c6 = self.c6(torch.cat((u2,c2),1))+t6[...,None,None,None]+cond6[...,None,None]
        c6 = (
            self.c6(torch.cat((u2, c2), 1))
            + t6[..., None, None, None]
            + 10 * cond6[..., None, None]
        )

        u3 = self.u3(c6)
        t7 = self.t7(t)
        cond7 = self.cond7((phases, u3.shape[-3]))
        # c7 = self.c7(torch.cat((u3,c1),1))+t7[...,None,None,None]+cond7[...,None,None]
        c7 = (
            self.c7(torch.cat((u3, c1), 1))
            + t7[..., None, None, None]
            + 10 * cond7[..., None, None]
        )

        out = self.out(c7)

        return out
