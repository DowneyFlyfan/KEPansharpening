import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import DownNext, UpConv


# ---- Initialization ----
def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode="fan_in")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


# ---- Utils ----
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or in_dim * 2
        out_dim = out_dim or in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim=512,
        heads=8,
        dim_head=64,
        sliding_window_size=2,
        compress_block_size=4,
        selection_block_size=4,
        num_selected_blocks=2,
    ):
        super().__init__()

        self.attn = SparseAttention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            sliding_window_size=sliding_window_size,
            compress_block_size=compress_block_size,
            selection_block_size=selection_block_size,
            num_selected_blocks=num_selected_blocks,
        )

    def forward(self, x):
        return self.attn(x)


class Transformer_E(nn.Module):
    def __init__(
        self,
        dim,
        depth=2,
        heads=3,
        dim_head=16,
        mlp_dim=48,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Residual(
                            PreNorm(
                                dim,
                                Attention(dim, heads=heads, dim_head=dim_head),
                            )
                        ),
                        Residual(PreNorm(dim, FeedForward(dim, mlp_dim))),
                    ]
                )
            )

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class Transformer_D(nn.Module):
    def __init__(
        self,
        dim,
        depth=2,
        heads=3,
        dim_head=16,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Residual(
                            PreNorm(
                                dim,
                                Attention(dim, heads=heads, dim_head=dim_head),
                            )
                        ),
                        Residual(
                            PreNorm(
                                dim,
                                Attention(dim, heads=heads, dim_head=dim_head),
                            )
                        ),
                        Residual(PreNorm(dim, FeedForward(dim, dim * 2, dim))),
                    ]
                )
            )

    def forward(self, x, mask=None):
        for attn1, attn2, ff in self.layers:
            x = attn1(x)
            x = attn2(x)
            x = ff(x)
        return x


class NSA_Fusion(nn.Module):
    def __init__(
        self,
        in_ch=8,
        out_ch=8,
        depth=2,
        patch_size=8,
        hidden_channels=32,
        upsample_mode="nearest",
        use_sigmoid=True,
    ):
        super(NSA_Fusion, self).__init__()
        self.depth = depth
        self.patch_size = patch_size

        # Initialize Downsampling layers
        self.down_convs = nn.ModuleList([DownNext(in_ch, hidden_channels, True)])
        for _ in range(depth - 1):
            self.down_convs.append(DownNext(hidden_channels, hidden_channels))

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

        # Initialize the backbone
        self.backbone = nn.Sequential(
            nn.LazyLinear(512),
            nn.LayerNorm(512),
            nn.LeakyReLU(inplace=True),
            Transformer_D(
                dim=512,
                depth=2,
                heads=8,
                dim_head=64,
            ),
            nn.LazyLinear(hidden_channels * patch_size**2),
            nn.LayerNorm(hidden_channels * patch_size**2),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        down = []

        # Encoding
        out = x
        for i in range(self.depth):
            out = self.down_convs[i](out)
            down.append(out)

        # Transformer
        height, width = out.shape[-2:]
        height = height // self.patch_size
        width = width // self.patch_size

        out = rearrange(
            self.backbone(
                rearrange(
                    F.unfold(
                        input=out,
                        kernel_size=self.patch_size,
                        stride=self.patch_size,
                    ),
                    "b d n -> b n d",
                )
            ),
            "b (h w) (ph pw c) -> b c (ph h) (pw w)",
            ph=self.patch_size,
            pw=self.patch_size,
            h=height,
            w=width,
        )

        # Decoding
        out = self.up_convs[i](out)
        for i in reversed(range(self.depth - 1)):
            out = self.up_convs[i](torch.cat([down[i], out], dim=1))

        out = self.last_act(self.final_conv(out))
        return out
