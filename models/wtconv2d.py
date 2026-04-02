import torch
import torch.nn as nn
from timm.models.layers import DropPath
import torch.nn.functional as F
import pywt

from .common import ConvBlock, UpConv


# ---- Utils ----
# 创建小波对象, 获取分解滤波器(low, high), -1是用来反转的(因为卷积不是那个卷积)
def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)  # decomposition
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack(
        [
            dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
            dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
            dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
            dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1),
        ],
        dim=0,
    )

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])  # reconstruction
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack(
        [
            rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
            rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
            rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
            rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1),
        ],
        dim=0,
    )

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


# ---- Conv ----
class WTConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=5,
        stride=1,
        bias=True,
        wt_levels=1,
        wt_type="db1",
    ):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(
            wt_type, in_channels, in_channels, torch.float
        )
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.base_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            padding="same",
            stride=1,
            dilation=1,
            groups=in_channels,
            bias=bias,
        )  # Channel-Wise Conv

        self.base_scale = _ScaleModule(
            [1, in_channels, 1, 1]
        )  # 缩放每个channel的元素用的

        self.wavelet_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels * 4,
                    in_channels * 4,
                    kernel_size,
                    padding="same",
                    stride=1,
                    dilation=1,
                    groups=in_channels * 4,
                    bias=False,
                )  # Channel-Wise Conv
                for _ in range(self.wt_levels)
            ]
        )
        self.wavelet_scale = nn.ModuleList(
            [
                _ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1)
                for _ in range(self.wt_levels)
            ]
        )

    def forward(self, x):
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        # 每一级卷一次, 并把每一级结果存在列表里
        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = wavelet_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = (
                self.wavelet_scale[i](
                    self.wavelet_convs[i](
                        curr_x.reshape(
                            shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4]
                        ).contiguous()
                    )
                )
                .reshape(shape_x)
                .contiguous()
            )

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        # 根据卷积结果重构图像
        next_x_ll = 0
        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = inverse_wavelet_transform(curr_x, self.iwt_filter)

            next_x_ll = next_x_ll[:, :, : curr_shape[2], : curr_shape[3]]

        assert len(x_ll_in_levels) == 0
        return self.base_scale(self.base_conv(next_x_ll)) + next_x_ll


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


class WTConvNext(nn.Module):
    r"""WTConvNext Block.
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, in_ch, out_ch, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = WTConv2d(in_ch, out_ch)
        self.norm = nn.LayerNorm(out_ch, eps=1e-6)
        self.pwconv1 = nn.Linear(
            out_ch, 4 * out_ch
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * out_ch, out_ch)
        self.gamma = (
            nn.Parameter(
                layer_scale_init_value * torch.ones((out_ch)), requires_grad=True
            )
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


# ---- UNet ----
class DownConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownConv, self).__init__()
        self.down = nn.Sequential(
            ConvBlock(in_ch, out_ch, 2, 2, 0),
            WTConvNext(out_ch, out_ch),
        )

    def forward(self, x):
        return self.down(x)


class WTUNet(nn.Module):
    def __init__(
        self,
        in_ch=8,
        out_ch=8,
        depth=5,
        hidden_channels=64,
        upsample_mode="nearest",
        use_sigmoid=True,
    ):
        super(WTUNet, self).__init__()
        self.depth = depth

        # Initialize Downsampling layers
        self.down_convs = nn.ModuleList([DownConv(in_ch, hidden_channels)])
        for _ in range(depth - 1):
            self.down_convs.append(DownConv(hidden_channels, hidden_channels))

        # Initialize Upsampling layers
        self.up_convs = nn.ModuleList()
        for i in range(1, self.depth):
            self.up_convs.append(
                UpConv(
                    2 * hidden_channels,
                    hidden_channels,
                    upsample_mode,
                )
            )

        self.up_convs.append(UpConv(hidden_channels, hidden_channels, upsample_mode))

        self.final_conv = nn.Conv2d(hidden_channels, out_ch, kernel_size=1)
        self.last_act = nn.Sigmoid() if use_sigmoid else nn.Identity()

    def forward(self, x):
        down = []

        # Encoding
        out = x
        for i in range(self.depth):
            out = self.down_convs[i](out)
            down.append(out)

        # Decoding
        out = self.up_convs[i](out)
        for i in reversed(range(self.depth - 1)):
            out = self.up_convs[i](torch.cat([down[i], out], dim=1))

        out = self.last_act(self.final_conv(out))
        return out
