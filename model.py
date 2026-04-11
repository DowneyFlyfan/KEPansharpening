from models.common import ConvNext, Permute
from models.UNet import *
from MTF import MTF_MS
from config import margs

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- Main Nets ----
@torch.compile
class MainNet(nn.Module):
    def __init__(self, hrms2pan):
        super().__init__()
        self.pan_pred_net = hrms2pan
        self.register_buffer("kernel", MTF_MS(ratio=4, N=41))

        self.backbone = UNet(in_ch=margs.channel, out_ch=margs.channel)
        self.to(margs.device, margs._dtype)

    def forward(self, x, kernel=None):
        gt_pred = self.backbone(x)
        pan_pred = self.pan_pred_net(gt_pred)
        ms_pred = F.conv2d(
            F.pad(gt_pred, (20, 20, 20, 20), mode="replicate"),
            weight=kernel if isinstance(kernel, torch.Tensor) else self.kernel,
            bias=None,
            stride=1,
            padding=0,
            groups=margs.channel,
        )[:, :, 2:-1:4, 2:-1:4]

        return gt_pred, ms_pred, pan_pred


@torch.compile
class HRMS2PAN(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(
                margs.channel,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            ConvNext(in_dim=hidden_dim, out_dim=hidden_dim),
            Permute((0, 2, 3, 1)),
            nn.LayerNorm(hidden_dim),
            Permute((0, 3, 1, 2)),
            nn.Conv2d(
                hidden_dim,
                1,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

        self.to(margs.device, margs._dtype)

    def forward(self, x):
        return self.backbone(x)


@torch.compile
class MRANet(nn.Module):
    def __init__(self, hrms2pan, hidden_dim=32):
        super().__init__()
        self.mranet = nn.Sequential(
            nn.Conv2d(
                margs.channel,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            ConvNext(in_dim=hidden_dim, out_dim=hidden_dim),
            Permute((0, 2, 3, 1)),
            nn.LayerNorm(hidden_dim),
            Permute((0, 3, 1, 2)),
            nn.Conv2d(
                hidden_dim,
                margs.channel,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

        self.pan_pred_net = hrms2pan
        self.register_buffer("kernel", MTF_MS(ratio=4, N=41))

        self.ms_conv = nn.Conv2d(
            margs.channel,
            margs.channel,
            kernel_size=41,
            stride=1,
            padding=0,
            groups=margs.channel,
            bias=False,
        )
        # Assign the pre-computed kernel and make it non-trainable
        self.ms_conv.weight.data = self.kernel
        self.ms_conv.weight.requires_grad = False

        self.to(margs.device, margs._dtype)

    def forward(self, lms, g, pan_hp, ret_g=False):
        gout = self.mranet(g)
        gt_pred = lms + gout * pan_hp
        pan_pred = self.pan_pred_net(gt_pred)
        ms_pred = self.ms_conv(F.pad(gt_pred, (20, 20, 20, 20), mode="replicate"))[
            :, :, 2:-1:4, 2:-1:4
        ]

        if not ret_g:
            return gt_pred, ms_pred, pan_pred
        else:
            return gt_pred, ms_pred, pan_pred, gout


