import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange


# ---- Basic Blocks ----
class Resblock(nn.Module):
    def __init__(self, hidden_dim):
        super(Resblock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.conv2 = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(hidden_dim)

    def forward(self, x):
        res = self.conv2(F.leaky_relu_(self.conv1(x)))
        out = F.leaky_relu_(self.bn(x + res))
        return out


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


# ---- ConvNext Block ----
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


class mHC_ConvNextBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_lanes=4,
        hidden_factor=4,
        layer_scale_init_value=1e-6,
    ):
        super().__init__()
        assert dim % num_lanes == 0
        self.dim = dim
        self.num_lanes = num_lanes
        self.lane_dim = dim // num_lanes

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

        self.pwconv = nn.Sequential(
            Permute((0, 2, 3, 1)),
            nn.LayerNorm(dim, eps=1e-6),
            nn.Linear(dim, hidden_factor * dim),
            nn.GELU(),
            nn.Linear(hidden_factor * dim, dim),
            Permute((0, 3, 1, 2)),
        )

        self.mhc_logits = nn.Parameter(torch.randn(num_lanes, num_lanes) * 0.02)

        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True
        )

    def _get_manifold_matrix(self, iterations=3):
        M = torch.exp(self.mhc_logits)
        for _ in range(iterations):
            M = F.normalize(F.normalize(M, p=1, dim=-1), p=1, dim=-2)
        return M

    def forward(self, x):
        B, C, H, W = x.shape
        lanes = x.view(B, self.num_lanes, -1)
        M = self._get_manifold_matrix()
        x_mhc = torch.matmul(M, lanes).view(B, C, H, W)

        return self.pwconv(self.dwconv(x)) * self.gamma + x_mhc


class ResidualBlock(nn.Module):
    def __init__(
        self,
        InputChannels,
        Cardinality,
        ExpansionFactor,
        KernelSize,
        VarianceScalingParameter,
    ):
        super(ResidualBlock, self).__init__()

        NumberOfLinearLayers = 3
        ExpandedChannels = InputChannels * ExpansionFactor
        ActivationGain = VarianceScalingParameter ** (
            -1 / (2 * NumberOfLinearLayers - 2)
        )

        self.LinearLayer1 = Convolution(
            InputChannels, ExpandedChannels, KernelSize=1, ActivationGain=ActivationGain
        )
        self.LinearLayer2 = Convolution(
            ExpandedChannels,
            ExpandedChannels,
            KernelSize=KernelSize,
            Groups=Cardinality,
            ActivationGain=ActivationGain,
        )
        self.LinearLayer3 = Convolution(
            ExpandedChannels, InputChannels, KernelSize=1, ActivationGain=0
        )

        self.NonLinearity1 = nn.LeakyReLU()
        self.NonLinearity2 = nn.LeakyReLU()

    def forward(self, x):
        y = self.LinearLayer1(x)
        y = self.LinearLayer2(self.NonLinearity1(y))
        y = self.LinearLayer3(self.NonLinearity2(y))

        return x + y


class ConvNextV2(nn.Module):
    """
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.pwconv = nn.Sequential(
            nn.LayerNorm(dim, eps=1e-6),
            nn.Linear(dim, 4 * dim),
            nn.LeakyReLU(),
            GRN(4 * dim),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x):
        out = self.pwconv(self.dwconv(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2) + x

        return out


class DownNext(nn.Module):
    def __init__(
        self, in_ch, out_ch, is_first=False, first_kernel_size=2, first_stride=2
    ):
        super(DownNext, self).__init__()

        # Important to distinguish the 1st one
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


# ---- FusionNet ----
class FusionNet(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        # ---- ResBlock ----
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(),
        )

        self.resnet = nn.Sequential(
            Resblock(hidden_dim),
            Resblock(hidden_dim),
            Resblock(hidden_dim),
            Resblock(hidden_dim),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=8,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
        )

    def forward(self, pan, lms):
        res_input = pan.repeat(1, 8, 1, 1) - lms
        res_out = self.conv2(self.resnet(self.conv1(res_input)))
        gt_pred = res_out + lms

        return gt_pred


# ---- tcnn configs ----
"""
mlp_config = {
    "otype": "FullyFusedMLP",  # CUTLASS_MLP for more layers and neurons
    "activation": "LeakyReLu",
    "output_activation": "None",
    "n_neurons": 128,  # 32, 64, 128 Supported for FullyFusedMLP
    "n_hidden_layers": 1,  # 1- 5 Layers Supported for FullyFusedMLP
}

dense_encode_config = {  # Multi-Resolution densegrid Encoding
    "otype": "densegrid",
    "n_levels": 3,
    "n_features_per_level": 8,
    "base_resolution": 64,
    "per_level_scale": 2,
}

hash_encode_config = (
    {  # Multi-Resolution Hash Encoding (More Computation, Better Quality)
        "otype": "Grid",
        "type": "Hash",
        "n_levels": 3,
        "n_features_per_level": 8,
        "log2_hashmap_size": 22,  # 加到22效果比densegrid好，但算力大得多
        "base_resolution": 64,
        "per_level_scale": 2,
    }
)

encoding = tcnn.Encoding(n_input_dims, config["encoding"])
network = tcnn.Network(encoding.n_output_dims, n_output_dims, config["network"])
model = torch.nn.Sequential(encoding, network)
"""


# ---- Init, Norm, Activation ----
def MSRInitializer(Layer, ActivationGain=1):
    FanIn = Layer.weight.data.size(1) * Layer.weight.data[0][0].numel()
    Layer.weight.data.normal_(0, ActivationGain / math.sqrt(FanIn))

    if Layer.bias is not None:
        Layer.bias.data.zero_()

    return Layer


class Convolution(nn.Module):
    def __init__(
        self, InputChannels, OutputChannels, KernelSize, Groups=1, ActivationGain=1
    ):
        super(Convolution, self).__init__()

        self.Layer = MSRInitializer(
            nn.Conv2d(
                InputChannels,
                OutputChannels,
                kernel_size=KernelSize,
                stride=1,
                padding=(KernelSize - 1) // 2,
                groups=Groups,
                bias=False,
            ),
            ActivationGain=ActivationGain,
        )

    def forward(self, x):
        return nn.functional.conv2d(
            x,
            self.Layer.weight.to(x.dtype),
            padding=self.Layer.padding,
            groups=self.Layer.groups,
        )


class RMSNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, alpha_init_value=0.5):
        """
        normalized_shape: (c,1,1) for (b,c,h,w); (1,1,c) for (b,h,w,c)
        """
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        return self.weight * torch.tanh(self.alpha * x) + self.bias


class GRN(nn.Module):
    """GRN (Global Response Normalization) layer
    Weight each channel to avoid the homogenity of different features
    spatial norm -> channel norm -> scale + shift -> res
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


# ---- Others ----
class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.C = in_channels

        self.norm = nn.BatchNorm2d(in_channels)
        self.qkv = torch.nn.Conv2d(
            in_channels, 3 * in_channels, kernel_size=1, stride=1, padding=0
        )
        self.w_ratio = int(in_channels) ** (-0.5)
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        qkv = self.qkv(self.norm(x))
        B, _, H, W = qkv.shape  # should be B,3C,H,W
        C = self.C
        q, k, v = qkv.reshape(B, 3, C, H, W).unbind(1)

        # Positional Embedding (有可能是one-shot的原因，这个效果还差一些; 也有可能是代码写的不好)
        # q = einops.rearrange(q, "b (head c) h w -> b head (h w) c", head=1).contiguous()
        # k = einops.rearrange(k, "b (head c) h w -> b head (h w) c", head=1).contiguous()
        # q = self.pos_embed.rotate_queries_or_keys(q)
        # k = self.pos_embed.rotate_queries_or_keys(k)

        q = q.view(B, C, H * W).contiguous()
        q = q.permute(0, 2, 1).contiguous()  # B,HW,C
        k = k.view(B, C, H * W).contiguous()  # B,C,HW
        w = torch.bmm(q, k).mul_(self.w_ratio)
        w = F.softmax(w, dim=2)

        # attend to values
        v = v.view(B, C, H * W).contiguous()
        w = w.permute(0, 2, 1).contiguous()  # B,HW,HW (first HW of k, second of q)
        h = torch.bmm(v, w)  # B, C,HW (HW of q) h[B,C,j] = sum_i v[B,C,i] w[B,i,j]
        h = h.view(B, C, H, W).contiguous()

        return x + self.proj_out(h)


class Self_Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=1,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims).contiguous()


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
