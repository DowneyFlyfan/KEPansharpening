import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
import torch.nn.functional as F
import numpy.random as random
from timm.models.layers import trunc_normal_
from einops import rearrange
import scipy.stats as stats

from MinkowskiEngine import (
    MinkowskiConvolution,
    MinkowskiChannelwiseConvolution,
    MinkowskiLinear,
    MinkowskiGELU,
    SparseTensor,
)

from MinkowskiOps import (
    to_sparse,
)

from .common import GRN, Permute, UpConv, DownNext
from loss import patchified_mseloss, patchified_samloss
from config import margs


# ---- Utils ----
class MinkowskiGRN(nn.Module):
    """GRN layer for sparse tensors."""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim))
        self.beta = nn.Parameter(torch.zeros(1, dim))

    def forward(self, x):
        cm = x.coordinate_manager
        in_key = x.coordinate_map_key

        Gx = torch.norm(x.F, p=2, dim=0, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return SparseTensor(
            self.gamma * (x.F * Nx) + self.beta + x.F,
            coordinate_map_key=in_key,
            coordinate_manager=cm,
        )


class MinkowskiDropPath(nn.Module):
    """Drop Path for sparse tensors."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(MinkowskiDropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        cm = x.coordinate_manager
        in_key = x.coordinate_map_key
        keep_prob = 1 - self.drop_prob
        mask = (
            torch.cat(
                [
                    (
                        torch.ones(len(_))
                        if random.uniform(0, 1) > self.drop_prob
                        else torch.zeros(len(_))
                    )
                    for _ in x.decomposed_coordinates
                ]
            )
            .view(-1, 1)
            .to(x.device)
        )
        if keep_prob > 0.0 and self.scale_by_keep:
            mask.div_(keep_prob)
        return SparseTensor(
            x.F * mask, coordinate_map_key=in_key, coordinate_manager=cm
        )


class MinkowskiLayerNorm(nn.Module):
    """Channel-wise layer normalization for sparse tensors."""

    def __init__(
        self,
        normalized_shape,
        eps=1e-6,
    ):
        super(MinkowskiLayerNorm, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, input):
        output = self.ln(input.F)
        return SparseTensor(
            output,
            coordinate_map_key=input.coordinate_map_key,
            coordinate_manager=input.coordinate_manager,
        )


# ---- Blocks ----
class Block(nn.Module):
    """ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.pwconv = nn.Sequential(
            nn.LayerNorm(dim, eps=1e-6),
            nn.Linear(dim, 4 * dim),
            nn.LeakyReLU(),
            GRN(4 * dim),
            nn.Linear(4 * dim, dim),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        out = (
            self.drop_path(self.pwconv(self.dwconv(x).permute(0, 2, 3, 1))).permute(
                0, 3, 1, 2
            )
            + x
        )

        return out


class SparseBlock(nn.Module):
    """Sparse ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0.0, D=3):
        super().__init__()
        self.dwconv = MinkowskiChannelwiseConvolution(
            dim, kernel_size=7, bias=True, dimension=D
        )
        self.pwconv = nn.Sequential(
            MinkowskiLayerNorm(dim, eps=1e-6),
            MinkowskiLinear(dim, 4 * dim),
            MinkowskiGELU(),
            MinkowskiGRN(4 * dim),
            MinkowskiLinear(4 * dim, dim),
            MinkowskiDropPath(drop_path),
        )

    def forward(self, x):
        out = (self.pwconv(self.dwconv(x))) + x
        return out


class SparseConvNeXtV2(nn.Module):
    """Sparse ConvNeXtV2.

    Args:
        in_chans (int): Number of input image channels. Default: 3
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
        self,
        in_chans=3,
        depths=[1, 1, 1, 1],
        dims=[32, 32, 32, 32],
        drop_path_rate=0.0,
        patch_size=32,
        D=3,
    ):
        super().__init__()
        self.depths = depths
        self.patch_size = patch_size
        self.out_dim = dims[-1]
        self.downsample_layers = (
            nn.ModuleList()
        )  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            Permute((0, 2, 3, 1)),
            nn.LayerNorm(dims[0], eps=1e-6),
            Permute((0, 3, 1, 2)),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                MinkowskiLayerNorm(dims[i], eps=1e-6),
                MinkowskiConvolution(
                    dims[i],
                    dims[i + 1],
                    kernel_size=2,
                    stride=2,
                    bias=True,
                    dimension=D,
                ),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = (
            nn.ModuleList()
        )  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    SparseBlock(dim=dims[i], drop_path=dp_rates[cur + j], D=D)
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, MinkowskiConvolution):
            trunc_normal_(m.kernel, std=0.02)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, MinkowskiChannelwiseConvolution):
            trunc_normal_(m.kernel, std=0.02)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, MinkowskiLinear):
            trunc_normal_(m.linear.weight, std=0.02)
            nn.init.constant_(m.linear.bias, 0)

    def forward(self, x, mask):
        """
        x: (b,c,h,w)
        mask: (1,1,h,w)
        """
        x = self.downsample_layers[0](x)  # (b, dim, h*scale, w*scale)
        x *= 1.0 - mask

        # sparse encoding
        x = to_sparse(x)
        for i in range(4):
            x = self.downsample_layers[i](x) if i > 0 else x
            x = self.stages[i](x)

        # densify
        x = x.dense(
            shape=torch.Size(
                [
                    1,
                    self.out_dim,
                    margs.test_sidelen // self.patch_size,
                    margs.test_sidelen // self.patch_size,
                ]
            )
        )[0]
        return x


# ---- FCMAE ----
class FCMAE(nn.Module):
    """Fully Convolutional Masked Autoencoder with ConvNeXtV2 backbone
    Only Supported For zero-shot task Now!!!!!!!
    pe: patch_size of encoder output: 32
    pm: middle patch_size: 4
    pl: loss patch_size: 4
    """

    def __init__(
        self,
        img_size=256,
        in_chans=8,
        out_chans=8,
        depths=[1, 1, 1, 1],
        dims=[32, 32, 32, 32],
        decoder_depth=5,
        decoder_embed_dim=32,
        patch_size=32,
        loss_size=4,
        mask_ratio_min=0.2,
    ):
        super().__init__()

        # configs

        assert (
            patch_size % loss_size == 0
        ), "patch_size has to be the integear times of loss_size"

        self.img_size = img_size
        self.depths = depths
        self.dims = dims
        self.feature_to_1st_downsample_scale = patch_size // 4
        self.feature_to_ms_patch_scale = (img_size // margs.ratio // loss_size) // (
            img_size // patch_size
        )

        # patch_size
        self.patch_size = patch_size
        self.loss_size = loss_size
        self.encoding_patch_sidelen = img_size // patch_size
        self.num_patches_encoder = self.encoding_patch_sidelen**2

        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth

        # generate the binary mask: 0 is keep 1 is remove
        self.mask_ratio_generator = stats.truncnorm(
            (mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25
        )
        self.mask = torch.ones(
            [1, self.num_patches_encoder],
            device=margs.device,
            dtype=margs._dtype,
        )

        # Loss
        self.loss_patch_sidelen = img_size // loss_size
        self.num_patches_loss = self.loss_patch_sidelen**2
        self.pwsam = patchified_samloss(self.patch_size)
        self.pwmse = patchified_mseloss(self.patch_size)

        self.encoder_to_loss_scale = patch_size // loss_size

        # encoder
        self.encoder = SparseConvNeXtV2(
            in_chans=in_chans,
            depths=depths,
            dims=dims,
            D=2,
            patch_size=patch_size,
        )

        # mask tokens & decoder
        self.mask_token = nn.Parameter(torch.zeros(1, decoder_embed_dim, 1, 1))
        decoder = [UpConv(in_ch=32, out_ch=32) for _ in range(decoder_depth)]
        self.decoder = nn.Sequential(
            *decoder,
            nn.Conv2d(
                in_channels=decoder_embed_dim, out_channels=out_chans, kernel_size=1
            ),
            nn.Sigmoid(),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, MinkowskiConvolution):
            trunc_normal_(m.kernel, std=0.02)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, MinkowskiChannelwiseConvolution):
            trunc_normal_(m.kernel)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, MinkowskiLinear):
            trunc_normal_(m.linear.weight)
            nn.init.constant_(m.linear.bias, 0)
        if isinstance(m, nn.Conv2d):
            w = m.weight.data
            trunc_normal_(w.view([w.shape[0], -1]))
            nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        if hasattr(self, "mask_token"):
            torch.nn.init.normal_(self.mask_token, std=0.02)

    def gen_random_mask(self, inference=False):
        """
        For a (1,8,256,256) input,
        encoder mask (1,1,64,64)
        decoder_mask (1,1,8,8)
        ms_loss_mask (256)
        pan_loss_mask (4096)
        """
        ids_shuffle = torch.argsort(
            torch.randn(
                [self.bsz, self.num_patches_encoder],
                device=margs.device,
                dtype=margs._dtype,
            ),
            dim=1,
        )

        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # unshuffle to get the binary mask
        if not inference:
            self.mask[
                :, : int(self.mask_ratio_generator.rvs(1)[0] * self.num_patches_encoder)
            ] = 0
        mask = torch.gather(self.mask, dim=1, index=ids_restore)  # (B, HW//pe**2)

        decoder_mask = mask.reshape(
            -1, self.encoding_patch_sidelen, self.encoding_patch_sidelen
        ).unsqueeze(
            1
        )  # (b,1,H // pe, W // pe)

        encoder_mask = (
            self.upsample_mask(
                decoder_mask.squeeze(), self.feature_to_1st_downsample_scale
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )  # (b,1, H//p, W//p)

        ms_loss_mask = (
            self.upsample_mask(decoder_mask.squeeze(), self.feature_to_ms_patch_scale)
            .contiguous()
            .view(-1)
        )  # (hw // pl**2)

        pan_loss_mask = (
            self.upsample_mask(decoder_mask.squeeze(), self.encoder_to_loss_scale)
            .contiguous()
            .view(-1)
        )  # (HW // pl**2)

        return encoder_mask, decoder_mask, pan_loss_mask, ms_loss_mask

    def upsample_mask(self, mask, scale):
        assert len(mask.shape) == 2, "mask must be in shape (h,w)"
        return (
            mask.repeat_interleave(scale, axis=0)
            .repeat_interleave(scale, axis=1)
            .contiguous()
        )  # (H // pe, W // pe) -> (H // p, W // p)

    def unpatchify(self, imgs):
        return (
            rearrange(
                imgs,
                "b (c peh pew) npeh npew -> b c (peh npeh) (pew npew)",
                c=margs.channel,
                peh=self.patch_size,
                pew=self.patch_size,
                npeh=self.encoding_patch_sidelen,
                npew=self.encoding_patch_sidelen,
            ),
        )

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, c*pe**2, H/pe, W/pe]
        mask: [H // pe * W // pe]
        """
        pred = rearrange(
            pred,
            "b (c peh pew) npeh npew -> b c (peh npeh) (pew npew)",
            c=margs.channel,
            peh=self.patch_size,
            pew=self.patch_size,
            npeh=self.encoding_patch_sidelen,
            npew=self.encoding_patch_sidelen,
        )  # (B,C,H,W)

        pred_patches = rearrange(
            pred,
            "b c (nplh plh) (nplw plw) -> (b nplh nplw) c plh plw",
            nplh=self.loss_patch_sidelen,
            nplw=self.loss_patch_sidelen,
            plh=self.loss_size,
            plw=self.loss_size,
        )  # (L, c, p, p)

        gt_patches = (
            imgs.unfold(2, self.loss_size, self.loss_size)
            .unfold(3, self.loss_size, self.loss_size)
            .contiguous()
            .view(-1, margs.channel, self.loss_size, self.loss_size)
        )  # (L,c,p,p)

        loss = self.pwmse(
            gt_patches,
            pred_patches,
            mask.unsqueeze(-1).repeat(1, margs.channel),
        ) + self.pwsam(gt_patches, pred_patches, mask)

        return loss, pred

    def forward(self, imgs):
        self.bsz = imgs.shape[0]
        encoder_mask, decoder_mask, pan_loss_mask, ms_loss_mask = (
            self.gen_random_mask()
        )  # (b,1, H // pm, W // pm), (b,1,H // pe, W // pe)

        mask_token = self.mask_token.repeat(
            self.bsz, 1, self.encoding_patch_sidelen, self.encoding_patch_sidelen
        )  # (b,c,H//pe, W//pe)

        pred = self.decoder(
            self.encoder(imgs, encoder_mask) * (1 - decoder_mask)
            + mask_token * (decoder_mask)
        )
        return (
            pred,
            pan_loss_mask,
            ms_loss_mask,
        )

    def inference(self, imgs):
        with torch.no_grad():
            self.bsz = imgs.shape[0]
            encoder_mask, decoder_mask, _, _ = self.gen_random_mask(
                True
            )  # (b,1, H // pm, W // pm), (b,1,H // pe, W // pe)

            mask_token = self.mask_token.repeat(
                self.bsz, 1, self.encoding_patch_sidelen, self.encoding_patch_sidelen
            )  # (b,c,H//pe, W//pe)

            pred = self.decoder(
                self.encoder(imgs, encoder_mask) * (1 - decoder_mask)
                + mask_token * (decoder_mask)
            )
            return pred


class UMAE(nn.Module):
    def __init__(
        self,
        img_size=256,
        in_ch=8,
        out_ch=8,
        depth=5,
        hidden_channels=32,
        use_sigmoid=True,
        loss_size=4,
        mask_ratio_min=0.7,
    ):
        super().__init__()

        # configs
        self.depth = depth
        self.patch_size = loss_size * margs.ratio
        self.patch_len = img_size // (self.patch_size)
        self.num_patches = self.patch_size**2

        # generate the binary mask: 0 is keep 1 is remove
        self.mask_ratio_generator = stats.truncnorm(
            (mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25
        )
        self.mask = torch.ones(
            [1, self.num_patches],
            device=margs.device,
            dtype=margs._dtype,
        )

        # Initialize Downsampling layers
        self.down_convs = nn.ModuleList([DownNext(in_ch, hidden_channels, True)])
        for _ in range(depth - 1):
            self.down_convs.append(DownNext(hidden_channels, hidden_channels, False))

        # Initialize Upsampling layers
        self.up_convs = nn.ModuleList()
        for i in range(1, depth):
            self.up_convs.append(
                UpConv(
                    2 * hidden_channels,
                    hidden_channels,
                )
            )

        self.up_convs.append(UpConv(hidden_channels, hidden_channels))

        self.final_conv = nn.Conv2d(hidden_channels, out_ch, kernel_size=1)
        self.last_act = nn.Sigmoid() if use_sigmoid else nn.Identity()

    def upsample_mask(self, mask, scale):
        assert len(mask.shape) == 2, "mask must be in shape (h,w)"
        return (
            mask.repeat_interleave(scale, axis=0)
            .repeat_interleave(scale, axis=1)
            .contiguous()
        )  # (H // pe, W // pe) -> (H // p, W // p)

    def gen_random_mask(self, inference=False):
        """
        For a (1,8,256,256) input,
        mask (1,1,16,16)
        encoder_mask (1,1,256,256)
        ms_loss_mask (256)
        pan_loss_mask (4096)
        """
        ids_shuffle = torch.argsort(
            torch.randn(
                [1, self.num_patches],
                device=margs.device,
                dtype=margs._dtype,
            ),
            dim=1,
        )

        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # unshuffle to get the binary mask
        if not inference:
            self.mask[
                :, : int(self.mask_ratio_generator.rvs(1)[0] * self.num_patches)
            ] = 0
        mask = torch.gather(self.mask, dim=1, index=ids_restore)  # (B, HW//pe**2)
        mask = mask.reshape(-1, self.patch_len, self.patch_len).unsqueeze(1)

        encoder_mask = self.upsample_mask(mask.squeeze(), self.patch_size)
        pan_loss_mask = self.upsample_mask(mask.squeeze(), margs.ratio).reshape(-1)
        ms_loss_mask = mask.squeeze().reshape(-1)

        return encoder_mask, pan_loss_mask, ms_loss_mask

    def forward(self, x, inference=False):
        down = []
        mask, pan_loss_mask, ms_loss_mask = self.gen_random_mask(inference)

        # Encoding
        out = x * (1 - mask)
        for i in range(self.depth):
            out = self.down_convs[i](out)
            down.append(out)

        # Decoding
        out = self.up_convs[-1](out)
        for i in reversed(range(self.depth - 1)):
            out = self.up_convs[i](torch.cat([down[i], out], dim=1))

        out = self.last_act(self.final_conv(out))
        if inference:
            return out
        return out, pan_loss_mask, ms_loss_mask
