import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import UpConv, DownNext, ConvNext, UpNext


class UNet(nn.Module):
    def __init__(
        self,
        in_ch: int = 8,
        out_ch: int = 8,
        depth: int = 5,
        hidden_channels: int = 32,
        use_sigmoid: bool = True,
    ):
        super().__init__()
        self.depth = depth

        # Initialize Downsampling layers
        self.down_convs = nn.ModuleList([DownNext(in_ch, hidden_channels, True)])
        for _ in range(depth - 1):
            self.down_convs.append(DownNext(hidden_channels, hidden_channels, False))

        # Initialize Upsampling layers
        self.up_convs = nn.ModuleList()
        for _ in range(1, depth):
            self.up_convs.append(
                UpConv(
                    2 * hidden_channels,
                    hidden_channels,
                )
            )

        self.up_convs.append(UpConv(hidden_channels, hidden_channels))

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


class AddUNet(nn.Module):
    def __init__(
        self,
        in_ch=8,
        out_ch=8,
        depth=5,
        hidden_channels=64,
        use_sigmoid=True,
    ):
        super(AddUNet, self).__init__()
        self.depth = depth

        # Initialize Downsampling layers
        self.down_convs = nn.ModuleList([DownNext(in_ch, hidden_channels, True)])
        for _ in range(depth - 1):
            self.down_convs.append(DownNext(hidden_channels, hidden_channels, False))

        # Initialize Upsampling layers
        self.up_convs = nn.ModuleList([UpConv(hidden_channels, out_ch, inp_mode="add")])
        for _ in range(depth - 1):
            self.up_convs.append(
                UpConv(
                    hidden_channels,
                    hidden_channels,
                    inp_mode="add",
                )
            )

        self.final_conv = nn.Conv2d(out_ch, out_ch, kernel_size=1)
        self.last_act = nn.Sigmoid() if use_sigmoid else nn.Identity()

    def forward(self, x):
        down = []

        # Encoding
        out = x
        for i in range(self.depth):
            down.append(out)
            out = self.down_convs[i](out)

        # Decoding
        for i in reversed(range(self.depth)):
            out = self.up_convs[i](out, down[i])

        out = self.last_act(self.final_conv(out))
        return out


# ---- Mixed ----
class CANDown(nn.Module):
    """
    可以考虑下采样后的Resblock
    """

    def __init__(
        self,
        in_ch,
        out_ch,
        cluster_num,
        filter_threshold,
        cluster_source="channel",
    ):
        super().__init__()

        self.conv1 = CANConv(
            in_ch,
            out_ch,
            cluster_num=cluster_num,
            cluster_source=cluster_source,
            filter_threshold=filter_threshold,
            kernel_size=2,
            stride=2,
        )
        self.conv2 = ConvNext(out_ch, 0.01)

    def forward(self, x, cache_indice=None, cluster_override=None, ret_all=False):
        res, idx = self.conv1(x, cache_indice, cluster_override)
        res = self.conv2(F.leaky_relu(res))
        if ret_all:
            return res, idx
        else:
            return res


class UNetMixed(nn.Module):
    def __init__(
        self,
        in_ch=8,
        out_ch=8,
        depth=4,
        hidden_channels=64,
        upsample_mode="nearest",
        use_sigmoid=True,
    ):
        super(UNetMixed, self).__init__()
        self.depth = depth

        # Initialize Downsampling layers
        self.down_convs = nn.ModuleList(
            [
                CANDown(
                    in_ch=in_ch,
                    out_ch=hidden_channels,
                    cluster_num=4,
                    filter_threshold=0.01,
                )
            ]
        )
        for _ in range(depth - 1):
            self.down_convs.append(DownNext(hidden_channels, hidden_channels))

        # Initialize Upsampling layers
        self.up_convs = nn.ModuleList()
        for _ in range(1, self.depth):
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

    def forward(self, x, cluster_override=None):
        down = []

        # Encoding
        out, idx = self.down_convs[0](
            x, cluster_override=cluster_override, ret_all=True
        )
        down.append(out)

        for i in range(1, self.depth):
            out = self.down_convs[i](out)
            down.append(out)

        # Decoding
        out = self.up_convs[i](out)
        for i in reversed(range(self.depth - 1)):
            out = self.up_convs[i](torch.cat([down[i], out], dim=1))

        out = self.last_act(self.final_conv(out))

        if cluster_override == None:
            return out, idx
        return out
