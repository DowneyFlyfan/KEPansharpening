import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath

from .common import DownConv, UpConv, DownNext
from .gcn_lib import Grapher, act_layer


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    "gnn_patch16_224": _cfg(
        crop_pct=0.9,
        input_size=(3, 224, 224),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
}


class FFN(nn.Module):
    r"""
    Resblock (Linear版)
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act="leakyrelu",
        drop_path=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        # self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x))) + x


class Stem(nn.Module):
    r"""
    Conv2d下采样5次
    """

    def __init__(self, in_dim=8, out_dim=128, act="leakyrelu"):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim // 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim // 8),
            act_layer(act),
            nn.Conv2d(out_dim // 8, out_dim // 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim // 4),
            act_layer(act),
            nn.Conv2d(out_dim // 4, out_dim // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim // 2),
            act_layer(act),
            nn.Conv2d(out_dim // 2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x


class DeepGCN(nn.Module):
    """
    Input -> Stem -> 特征 -> [knn后cat(特征, 跟特征最近的neighbor的距离) -> 分组卷积ResNet -> Out] *n
    """

    def __init__(self, opt):
        super(DeepGCN, self).__init__()
        channels = opt.n_filters
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        conv = opt.conv
        self.n_blocks = opt.n_blocks
        drop_path = opt.drop_path

        self.stem = Stem(out_dim=channels, act=act)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path, self.n_blocks)
        ]  # stochastic depth decay rule
        print("dpr", dpr)
        num_knn = [
            int(x.item()) for x in torch.linspace(k, 2 * k, self.n_blocks)
        ]  # number of knn's k
        print("num_knn", num_knn)
        max_dilation = 196 // max(num_knn)

        self.pos_embed = nn.Parameter(torch.zeros(1, channels, 14, 14))

        if opt.use_dilation:
            self.backbone = Seq(
                *[
                    Seq(
                        Grapher(
                            channels,
                            num_knn[i],
                            min(i // 4 + 1, max_dilation),
                            conv,
                            act,
                            norm,
                            bias,
                            stochastic,
                            epsilon,
                            1,
                            drop_path=dpr[i],
                        ),
                        FFN(channels, channels * 4, act=act, drop_path=dpr[i]),
                    )
                    for i in range(self.n_blocks)
                ]
            )
        else:
            self.backbone = Seq(
                *[
                    Seq(
                        Grapher(
                            channels,
                            num_knn[i],
                            1,
                            conv,
                            act,
                            norm,
                            bias,
                            stochastic,
                            epsilon,
                            1,
                            drop_path=dpr[i],
                        ),
                        FFN(channels, channels * 4, act=act, drop_path=dpr[i]),
                    )
                    for i in range(self.n_blocks)
                ]
            )

        self.prediction = Seq(
            nn.Conv2d(channels, 1024, 1, bias=True),
            nn.BatchNorm2d(1024),
            act_layer(act),
            nn.Dropout(opt.dropout),
            nn.Conv2d(1024, opt.n_classes, 1, bias=True),
        )
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        x = self.stem(inputs) + self.pos_embed
        B, C, H, W = x.shape

        for i in range(self.n_blocks):
            x = self.backbone[i](x)

        x = F.adaptive_avg_pool2d(x, 1)
        return self.prediction(x).squeeze(-1).squeeze(-1)


class UGCN(nn.Module):
    def __init__(
        self,
        in_ch=8,
        out_ch=8,
        hidden_channels=64,
        n_blocks=1,
        k=16,  # number of neighbors
        depth=4,
        upsample_mode="nearest",
        use_sigmoid=True,
    ):
        super(UGCN, self).__init__()
        self.depth = depth

        # Initialize Downsampling layers
        self.down_convs = nn.ModuleList([DownNext(in_ch, hidden_channels)])
        for _ in range(depth - 1):
            self.down_convs.append(DownNext(hidden_channels, hidden_channels))

        # Initialize GCN Block
        num_knn = [
            int(x.item()) for x in torch.linspace(k, 2 * k, n_blocks)
        ]  # number of knn's k

        self.GCN = nn.Sequential(
            *[
                nn.Sequential(
                    Grapher(
                        hidden_channels,
                        kernel_size=num_knn[i],  # TODO
                        dilation=1,
                        conv="knn",
                        act="leakyrelu",
                        drop_path=0,
                    ),
                )
                for i in range(n_blocks)
            ]
        )

        # Initialize Upsampling layers
        self.up_convs = nn.ModuleList()
        for _ in range(depth - 1):
            self.up_convs.append(
                UpConv(
                    hidden_channels * 2,
                    hidden_channels,
                    upsample_mode,
                )
            )

        self.up_convs.append(
            UpConv(
                hidden_channels,
                hidden_channels,
                upsample_mode,
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

        # GCN Block
        out = self.GCN(out)

        # Decoding
        out = self.up_convs[-1](out)
        for i in reversed(range(self.depth - 1)):
            out = self.up_convs[i](torch.cat([down[i], out], dim=1))

        out = self.last_act(self.final_conv(out))
        return out
