import torch
import torch.nn as nn

from .common import UpConv, DownNext


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
