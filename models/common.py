import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    r"""
    3x3 Conv -> Norm -> Act (RNN Update Available)
    """

    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
        dilation=1,
        activation=nn.LeakyReLU(),
        padding_mode="replicate",
        res=False,
        res_ratio=0.1,
    ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
                padding_mode=padding_mode,
                dilation=dilation,
            ),
            nn.BatchNorm2d(out_ch),
            activation,
        )

        self.res = in_ch == out_ch and stride == 1 and res
        self.res_ratio = res_ratio

    def forward(self, x):
        if self.res:
            return self.conv(x) * (1 - self.res_ratio) + x * self.res_ratio

        return self.conv(x)


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBlock(in_ch, out_ch),
            ConvBlock(out_ch, out_ch, 1, 1, 0),
        )

    def forward(self, x, y=None):
        return self.up(x)


class ConvNext(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_factor=2,
        layer_scale_init_value=1e-6,
    ):
        super().__init__()

        self.dwconv = nn.Conv2d(
            in_dim,
            in_dim,
            kernel_size=7,
            padding=3,
            groups=in_dim,
        )

        self.pwconv = nn.Sequential(
            Permute((0, 2, 3, 1)),
            nn.LayerNorm(in_dim, eps=layer_scale_init_value),
            nn.Linear(in_dim, hidden_factor * in_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_factor * in_dim, out_dim),
        )
        self.gamma = (
            nn.Parameter(
                layer_scale_init_value * torch.ones((out_dim)), requires_grad=True
            )
            if layer_scale_init_value > 0
            else None
        )
        self.shortcut = nn.Sequential(
            Permute((0, 2, 3, 1)),
            nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity(),
        )

    def forward(self, x):
        return (
            (self.gamma * (self.pwconv((self.dwconv(x)))) + self.shortcut(x))
            .permute(0, 3, 1, 2)
            .contiguous()
        )


class DownNext(nn.Module):
    def __init__(
        self, in_ch, out_ch, is_first=False, first_kernel_size=2, first_stride=2
    ):
        super(DownNext, self).__init__()

        if is_first:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, first_kernel_size, first_stride),
                nn.BatchNorm2d(out_ch),
                ConvNext(out_ch, out_ch),
            )
        else:
            self.down = nn.Sequential(
                nn.BatchNorm2d(in_ch),
                nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2),
                ConvNext(out_ch, out_ch),
            )

    def forward(self, x):
        return self.down(x)


class UpNext(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpNext, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            Permute((0, 2, 3, 1)),
            nn.LayerNorm(in_ch),
            Permute((0, 3, 1, 2)),
            ConvNext(in_ch, out_ch),
        )

    def forward(self, x):
        return self.up(x)


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims).contiguous()
