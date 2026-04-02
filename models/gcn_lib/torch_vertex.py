import numpy as np
import torch
from torch import nn
from einops import rearrange
from timm.models.layers import DropPath

from .torch_nn import BasicConv, batched_index_select, act_layer
from .torch_edge import DenseDilatedKnnGraph
from .pos_embed import get_2d_relative_pos_embed
import torch.nn.functional as F


# ---- Common GNN Convs ----
class MRConv2d(nn.Module):
    r"""
    cat(x, 离x最远的邻居的距离) -> 分组卷积
    """

    def __init__(
        self, in_channels, out_channels, act="leakyrelu", norm=None, bias=True
    ):
        super(MRConv2d, self).__init__()
        self.nn = BasicConv(
            [in_channels * 2, out_channels], act, norm, bias
        )  # 分组卷积

    def forward(self, x, edge_index, y=None):  # (b,d,n,1), (b, n, k)
        x_i = batched_index_select(x, edge_index[1])  # (b,d,n,k), k个neighbors全是自己
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])  # (b,d,n,k), k个neighbors
        x_j, _ = torch.max(
            x_j - x_i, -1, keepdim=True
        )  # 离每个patch最远的邻居(b,d,n,1)
        b, c, n, _ = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(
            b, 2 * c, n, _
        )  # (b, 2d, n, 1)
        return self.nn(x)


class EdgeConv2d(nn.Module):
    r"""
    cat(x复制k次, x离所有邻居的距离) -> 分组卷积 -> 取最大
    """

    def __init__(self, in_channels, out_channels, act="relu", norm=None, bias=True):
        super(EdgeConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])  # (b,d,n,k)
        max_value, _ = torch.max(
            self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True
        )
        return max_value


class GraphSAGE(nn.Module):
    """
    cat(x, NN[x最大的邻居]) -> 分组卷积
    """

    def __init__(self, in_channels, out_channels, act="relu", norm=None, bias=True):
        super(GraphSAGE, self).__init__()
        self.nn1 = BasicConv([in_channels, in_channels], act, norm, bias)
        self.nn2 = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(self.nn1(x_j), -1, keepdim=True)
        return self.nn2(torch.cat([x, x_j], dim=1))


class GINConv2d(nn.Module):
    """
    GIN Graph Convolution (Paper: https://arxiv.org/abs/1810.00826) for dense data type
    """

    def __init__(self, in_channels, out_channels, act="relu", norm=None, bias=True):
        super(GINConv2d, self).__init__()
        self.nn = BasicConv([in_channels, out_channels], act, norm, bias)
        eps_init = 0.0
        self.eps = nn.Parameter(torch.Tensor([eps_init]))

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j = torch.sum(x_j, -1, keepdim=True)
        return self.nn((1 + self.eps) * x + x_j)


# ---- My GNN Convs ----
class KNNConv(nn.Module):
    r"""
    cat(x, x的邻居) -> 1x1 Conv
    """

    def __init__(self, in_ch, out_ch, k):
        super(KNNConv, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv1d(in_ch * (k + 1), out_ch, 1, 1, 0),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x, edge_index, y=None):  # (b,d,n,1)
        x_i = batched_index_select(x, edge_index[1])  # (b,d,n,k)
        return self.nn(
            rearrange(torch.cat((x, x_i), dim=-1), "b d n kplus -> b (kplus d) n")
        ).unsqueeze(-1)


# ---- Grapher ----
class GraphConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        conv="edge",
        act="relu",
        norm=None,
        bias=True,
    ):
        super(GraphConv2d, self).__init__()
        if conv == "edge":
            self.gconv = EdgeConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == "mr":
            self.gconv = MRConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == "sage":
            self.gconv = GraphSAGE(in_channels, out_channels, act, norm, bias)
        elif conv == "gin":
            self.gconv = GINConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == "knn":
            self.gconv = KNNConv(in_channels, out_channels, kernel_size)
        else:
            raise NotImplementedError("conv:{} is not supported".format(conv))

    def forward(self, x, edge_index, y=None):
        return self.gconv(x, edge_index, y)


class DyGraphConv2d(GraphConv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=9,
        dilation=1,
        conv="mr",
        act="leakyrelu",
        norm=None,
        bias=True,
        stochastic=False,
        epsilon=0.0,
    ):
        super(DyGraphConv2d, self).__init__(
            in_channels, out_channels, kernel_size, conv, act, norm, bias
        )
        self.k = kernel_size
        self.d = dilation
        self.dilated_knn_graph = DenseDilatedKnnGraph(
            kernel_size, dilation, stochastic, epsilon
        )

    def forward(self, x, relative_pos=None):
        B, C, H, W = x.shape
        y = None
        x = x.reshape(B, C, -1, 1).contiguous()
        edge_index = self.dilated_knn_graph(x, y, relative_pos)
        x = super(DyGraphConv2d, self).forward(x, edge_index, y)
        return x.reshape(B, -1, H, W).contiguous()


class Grapher(nn.Module):
    """
    Input -> Stem -> 特征 -> knn后cat(特征, 跟特征最近的neighbor的距离) -> ResNet -> Out
    """

    def __init__(
        self,
        in_channels,
        kernel_size=9,
        dilation=1,
        conv="knn",
        act="leakyrelu",
        norm=None,
        bias=True,
        stochastic=False,
        epsilon=0.0,
        n=196,
        drop_path=0.0,
        relative_pos=False,
    ):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.n = n
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = DyGraphConv2d(
            in_channels,
            in_channels * 2,
            kernel_size,
            dilation,
            conv,
            act,
            norm,
            bias,
            stochastic,
            epsilon,
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.relative_pos = None
        if relative_pos:  # TODO
            relative_pos_tensor = (
                torch.from_numpy(
                    np.float32(get_2d_relative_pos_embed(in_channels, int(n**0.5)))
                )
                .unsqueeze(0)
                .unsqueeze(1)
            )
            relative_pos_tensor = F.interpolate(
                relative_pos_tensor,
                size=(n, n),
                mode="bicubic",
                align_corners=False,
            )
            self.relative_pos = nn.Parameter(
                -relative_pos_tensor.squeeze(1), requires_grad=False
            )

    def _get_relative_pos(self, relative_pos, H, W):
        if relative_pos is None or H * W == self.n:
            return relative_pos
        else:
            N = H * W
            N_reduced = N // (self.r * self.r)
            return F.interpolate(
                relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic"
            ).squeeze(0)

    def forward(self, x):
        _tmp = x
        x = self.fc1(x)
        B, C, H, W = x.shape
        relative_pos = self._get_relative_pos(self.relative_pos, H, W)
        x = self.graph_conv(x, relative_pos)
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp
        return x
