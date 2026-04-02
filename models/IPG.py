import math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rotary_embedding_torch import RotaryEmbedding as RoPE
import einops

from .common import UpConv, DownConv, ConvBlock, AttnBlock


# ---- Utils ----


def get_mask(idx, array):  # 比按顺序排列的小，就打码 (好奇怪的方式啊???)
    """
    idx: (b, pp, kk)
    array: b pp # 本质上是细节图归一化、缩放再调整后的结果
    """
    b, m, n = idx.shape
    A = (
        torch.arange(n, dtype=idx.dtype, device=idx.device)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(b, m, n)
    )  # 1 1 n -> b m n
    mask = A > array.unsqueeze(-1)
    return mask


def alloc(var, rest, budget, maximum, times=0, fast=False):
    """
    var: (b m) variance of each pixel POSITIVE VALUE
    rest: (b m) list of already allocated budgets
    budget: (b) remaining to be allocated
    maximum: maximum budget for each pixel (kk)
    """
    b, m = var.shape  # (bnn, pp)
    var_p = var * (rest < maximum)
    var_sum = var_p.sum(dim=-1, keepdim=True)  # b 1
    proportion = var_p / var_sum  # b m, 细节归一化

    allocation = torch.round(
        proportion * budget.unsqueeze(1)
    )  # b m, 缩放budget倍 = 分配
    new_rest = torch.clamp(
        rest + allocation, 0, maximum
    )  # b m, 新的分配 = 上一次分配 + 新增的分配
    remain_budget = budget - (new_rest - rest).sum(
        dim=-1
    )  # b m, 剩下的上限 = 上一次剩下的上限 - budget
    negative_remain = remain_budget < 0

    while negative_remain.sum() > 0:  # 分配过头, 砍预算
        offset = torch.eye(m, device=rest.device)[
            torch.randint(m, (negative_remain.sum().int().item(),), device=rest.device)
        ]  # 随机挑几个减少offset, (小于0的总数, m)
        new_rest[negative_remain] = torch.clamp(
            new_rest[negative_remain] - offset, 1, maximum
        )  # reduce by one

        # update remain budget
        remain_budget = budget - (new_rest - rest).sum(dim=-1)  # b m allocated
        negative_remain = remain_budget < 0

    if (remain_budget > 0).sum() > 0:  # 分配不够,继续分配
        if times < 3:
            new_rest[remain_budget > 0] = alloc(
                var[remain_budget > 0],
                new_rest[remain_budget > 0],
                remain_budget[remain_budget > 0],
                maximum,
                times + 1,
                fast=fast,
            )
        elif not fast:  # precise budget allocation
            positive_remain = remain_budget > 0  # 分配不够,继续offset加预算
            while positive_remain.sum() > 0:
                offset = torch.eye(m, device=rest.device)[
                    torch.randint(
                        m, (positive_remain.sum().int().item(),), device=rest.device
                    )
                ]
                new_rest[positive_remain] = torch.clamp(
                    new_rest[positive_remain] + offset, 1, maximum
                )  # add by one
                # update remain budget
                remain_budget = budget - (new_rest - rest).sum(dim=-1)  # b m allocated
                positive_remain = remain_budget > 0
    return new_rest


def flex(
    idx: torch.Tensor,
    topk,
    X_diff,
    return_maskarray=False,
):
    """
    idx: (b m n) sorted index of D
    out: (b m n) Binary mask
    """
    b = idx.shape[0]
    rest = torch.ones_like(X_diff)  # (bnn, ww)个1
    budget = (
        torch.ones(b, dtype=torch.int, device=idx.device)
        * (topk - 1)
        * idx.size(1)  # b个 (topk-1)*kk
    )
    mask_array = alloc(X_diff, rest, budget, maximum=idx.size(-1))  # 范围:[0,kk]

    if return_maskarray:
        return mask_array

    mask = get_mask(idx, mask_array)  # (b, pp, kk)

    return mask


def cossim(X_sample, Y_sample, graph=None):
    """
    b: batch
    h: num_heads
    m: pp
    n: kk
    c: num_channels for samples
    """
    if graph is not None:
        return torch.einsum(
            "b h m c, b h n c-> b h m n",
            F.normalize(X_sample, dim=-1),
            F.normalize(Y_sample, dim=-1),
        ) + (-100.0) * (
            ~graph
        )  # after softmax, irrelavant points in the graph have almost 0 weights
    return torch.einsum(
        "b h m c, b h n c-> b h m n",  # 260MB挺大的 (256 * 4 * 256 * 1024)
        F.normalize(X_sample, dim=-1),
        F.normalize(Y_sample, dim=-1),
    )


def local_sampling(
    x, group_size, unfold_dict, output=0
):  # 0/1/2: grouped, sampled, both
    if isnstance(group_size, int):
        group_size = (group_size,) * 2

    if output != 1:
        x_grouped = einops.rearrange(
            x,
            "b c (numh sh) (numw sw)-> (b numh numw) (sh sw) c",
            sh=group_size[0],
            sw=group_size[1],
        )

        if output == 0:
            return x_grouped  # (bnn, pp, c), p = 16

    x_sampled = einops.rearrange(
        F.unfold(x, **unfold_dict),
        "b (c k0 k1) l -> (b l) (k0 k1) c",  # l = num_kernels
        k0=unfold_dict["kernel_size"][0],
        k1=unfold_dict["kernel_size"][1],
    )  # (bl, kk, c), k = 32

    if output == 1:
        return x_sampled

    assert x_grouped.size(0) == x_sampled.size(0)
    return x_grouped, x_sampled


def global_sampling(x, group_size, sample_size, output=0):
    if isinstance(group_size, int):
        group_size = (group_size, group_size)
    if isinstance(sample_size, int):
        sample_size = (sample_size, sample_size)

    if output != 1:
        x_grouped = einops.rearrange(
            x,
            "b c (sh numh) (sw numw) -> (b numh numw) (sh sw) c",
            sh=group_size[0],
            sw=group_size[1],
        )  # (bNN, c, pp), p=4

        if output == 0:
            return x_grouped

        x_sampled = einops.rearrange(
            x,
            "b c (sh extrah numh) (sw extraw numw) -> b extrah numh extraw numw c sh sw",
            sh=sample_size[0],
            sw=sample_size[1],
            extrah=1,
            extraw=1,
        )

    b_y, _, numh, _, numw, c_y, sh_y, sw_y = x_sampled.shape
    ratio_h, ratio_w = (
        sample_size[0] // group_size[0],
        sample_size[1] // group_size[1],
    )  # r=8
    x_sampled = (
        x_sampled.expand(b_y, ratio_h, numh, ratio_w, numw, c_y, sh_y, sw_y)
        .reshape(-1, c_y, sh_y * sw_y)
        .permute(0, 2, 1)
    )  # (brrnn, c, kk)

    if output == 1:
        return x_sampled

    assert x_grouped.size(0) == x_sampled.size(0)
    return x_grouped, x_sampled


# ---- Backbones ----
class IPG_Grapher(nn.Module):  # Attention - ~Graph
    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        unfold_dict,
    ):

        super(IPG_Grapher, self).__init__()
        self.dim = dim
        self.group_size = window_size
        self.num_heads = num_heads

        head_dim = int(self.dim / num_heads)
        self.scale = head_dim ** (-0.5)

        # graph_related
        self.unfold_dict = unfold_dict
        self.sample_size = unfold_dict["kernel_size"]

        self.proj_group = nn.Conv1d(dim, dim, 1)
        self.proj_sample = nn.Conv2d(dim, 2 * dim, 1)

        self.proj = nn.Linear(dim, dim)

        # Positional Bias
        self.rope = RoPE(dim=int(dim / num_heads))

    def get_correlation(self, x1, x2, graph):  # 相似度 -> 缩放 -> 偏置 -> softmax
        assert (x1.size(-2) == graph.size(-2)) and (x2.size(-2) == graph.size(-1))

        x1 = self.rope.rotate_queries_or_keys(x1)  # (b,h,n,c)
        x2 = self.rope.rotate_queries_or_keys(x2)
        sim = F.softmax(cossim(x1, x2, graph=graph) * self.scale, dim=-1)

        return sim

    def forward(self, x_complete, graph=None, sampling_method=0):
        if sampling_method == 0:
            x = local_sampling(
                x_complete,
                group_size=self.group_size,
                unfold_dict=None,
                output=0,
            )
        else:
            x = global_sampling(
                x_complete,
                group_size=self.group_size,
                sample_size=None,
                output=0,
            )

        b_, n, c = x.shape  # (bnn, pp, c)
        x1 = einops.rearrange(
            self.proj_group(x.permute(0, 2, 1)),
            "b (h c) n-> b h n c",
            b=b_,
            n=n,
            h=self.num_heads,
        )

        if sampling_method == 0:
            x_sampled = local_sampling(
                self.proj_sample(x_complete),
                group_size=self.group_size,
                unfold_dict=self.unfold_dict,
                output=1,
            )
        else:
            x_sampled = global_sampling(
                self.proj_sample(x_complete),
                group_size=self.group_size,
                sample_size=self.sample_size,
                output=1,
            )

        x2, feat = einops.rearrange(
            x_sampled,
            "b n (div h c) -> div b h n c",
            div=2,
            h=self.num_heads,
            c=c // self.num_heads,
        )

        corr = self.get_correlation(x1, x2, graph)

        x = (corr @ feat).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)

        return x


class GAL(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        in_ch=8,
        window_size=16,  # 在UNet中注意调整
        sampling_method=0,
        norm_layer=nn.LayerNorm,
        down: bool = True,
    ):
        super(GAL, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.sampling_method = sampling_method
        self.down = down
        self.sample_size = window_size * 2
        self.topk = window_size * 16

        padding_size = (self.sample_size - window_size) // 2
        self.unfold_dict = dict(
            kernel_size=(self.sample_size, self.sample_size),
            stride=(window_size, window_size),
            padding=(padding_size, padding_size),
        )

        self.grapher = IPG_Grapher(
            dim=in_ch,
            window_size=window_size,
            num_heads=num_heads,
            unfold_dict=self.unfold_dict,
        )

        self.norm = norm_layer(dim)
        self.cnn = DownConv(in_ch, dim) if down else UpConv(2 * dim, dim)

    def diff(self, x, scale=2):  # Detail_Map
        H, W = x.shape[-2:]
        return (
            (
                x
                - F.interpolate(
                    F.interpolate(
                        x,
                        (H // scale, W // scale),
                        mode="bilinear",
                        align_corners=False,
                    ),
                    (H, W),
                    mode="bilinear",
                    align_corners=False,
                )
            )
            .abs()
            .sum(dim=1)
        )

    @torch.no_grad()
    def calc_graph(self, x, X_diff):  # x:(b,c,h,w)
        X_diff = self.diff(x)
        X_diff = einops.rearrange(
            X_diff,
            "b (numh wh) (numw ww)-> (b numh numw) (wh ww)",
            wh=self.window_size,
            ww=self.window_size,
        )  # (bnn, pp), p = 16

        graph0 = self.calc_graph_(x, X_diff, sampling_method=0)
        graph1 = self.calc_graph_(x, X_diff, sampling_method=1)
        return (graph0, graph1)

    @torch.no_grad()
    def calc_graph_(self, x, X_diff, sampling_method):
        if sampling_method:  # sparse global
            X_sample, Y_sample = global_sampling(
                x,
                group_size=self.window_size,
                sample_size=self.sample_size,
                output=2,
            )
        else:  # dense local
            X_sample, Y_sample = local_sampling(
                x,
                group_size=self.window_size,
                unfold_dict=self.unfold_dict,
                output=2,
            )  # X是patch, Y是周围32*32的范围的像素

        assert X_sample.size(0) == Y_sample.size(0)
        D = cossim(X_sample.unsqueeze(1), Y_sample.unsqueeze(1)).squeeze(1)

        val, idx = D.sort(dim=-1, descending=True)  # (b, pp, kk)
        b, m, n = idx.shape
        mask = flex(idx, self.topk, X_diff)
        b_coord = torch.arange(b, device=idx.device).contiguous().view(-1, 1, 1) * m * n
        m_coord = torch.arange(m, device=idx.device).contiguous().view(1, -1, 1) * n
        overall_coord = (
            b_coord + m_coord + idx
        )  # 得到唯一坐标 (一张图已经被拆成很多批次了，需要区分它们的位置)

        selected_coord = torch.masked_select(overall_coord, mask)
        graph = torch.ones_like(idx).bool()
        graph.view(-1)[selected_coord] = False  # turned off connections

        return graph

    def forward(self, x, X_diff=None, skip=None):  # x: (b,c,h,w)
        graph = self.calc_graph(x, X_diff)
        H, W = x.shape[-2:]
        x = self.grapher(
            x,
            graph=graph[0] if self.sampling_method == 0 else graph[1],
            sampling_method=self.sampling_method,
        )

        x = einops.rearrange(
            x,
            "(b numh numw) (sh sw) c -> b c (numh sh) (numw sw)",
            numh=H // self.window_size,
            numw=W // self.window_size,
            sh=self.window_size,
            sw=self.window_size,
        )  # (b,c,h,w)

        if not skip:
            return x + self.norm(self.cnn(x))
        else:
            return self.norm(self.cnn(torch.cat([x, skip], dim=1)))


class GNN_UNet(nn.Module):
    def __init__(
        self,
        in_ch=8,
        out_ch=8,
        hidden_channels=64,
        depth=5,
        upsample_mode="nearest",
        use_sigmoid=True,
    ):
        super(GNN_UNet, self).__init__()
        self.depth = depth

        # Initialize Downsampling layers + AttnBlock
        self.down_convs = nn.ModuleList([DownConv(in_ch, hidden_channels)])
        for i in range(1, depth + 1):
            if i <= depth - 1:
                self.down_convs.append(DownConv(hidden_channels, hidden_channels))
            else:
                self.down_convs.append(
                    nn.Sequential(
                        DownConv(hidden_channels, hidden_channels),
                        AttnBlock(hidden_channels),
                        AttnBlock(hidden_channels),
                    )
                )

        # Initialize Upsampling layers + AttnBlock
        self.up_convs = nn.ModuleList()
        for i in range(1, depth):
            self.up_convs.append(
                UpConv(
                    hidden_channels * 2,
                    hidden_channels,
                    upsample_mode,
                )
            )

        self.up_convs.append(
            nn.Sequential(
                UpConv(
                    hidden_channels,
                    hidden_channels,
                    upsample_mode,
                ),
                AttnBlock(hidden_channels),
                AttnBlock(hidden_channels),
            )
        )

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
        out = self.up_convs[-1](out)
        for i in reversed(range(self.depth - 1)):
            out = self.up_convs[i](torch.cat([down[i], out], dim=1))

        out = self.last_act(self.final_conv(out))
        return out


"""
network_g:
  upscale: 4
  in_chans: 8
  img_size: 256
  window_size: 16
  img_range: 1.
  depths: [6, 6, 6, 6, 6, 6]
  embed_dim: 180
  num_heads: [6, 6, 6, 6, 6, 6]
  mlp_ratio: 4
  upsampler: 'pixelshuffle'
  resi_connection: '1conv'
  graph_flags: [1, 1, 1, 1, 1, 1] # 每一层都重算一遍graph
  stage_spec: [['GN','GS','GN','GS','GN','GS'],['GN','GS','GN','GS','GN','GS'],['GN','GS','GN','GS','GN','GS'],['GN','GS','GN','GS','GN','GS'],['GN','GS','GN','GS','GN','GS'],['GN','GS','GN','GS','GN','GS']] # 全局/局部 交替进行
  dist_type: 'cossim'
  top_k: 256
  head_wise: 0 # BOOL
  sample_size: 32
  graph_switch: 1 # BOOL
  flex_type: 'interdiff_plain'
  FFNtype: 'basic-dwconv3'
  conv_scale: 0
  conv_type: 'dwconv3-gelu-conv1-ca'
  diff_scales: [10,1.5,1.5,1.5,1.5,1.5]
  fast_graph: 1 # set as 1 for testing
"""
