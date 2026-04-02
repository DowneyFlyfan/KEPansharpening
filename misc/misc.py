import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import gaussian_blur

from config import bargs


# ---- Coords ----
def make_coord(shape, ranges=None, flatten=True):
    """Make normalized coords
    coords range = [-1,1], one cell length = 2/h, half a cell length = 1/h
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = (
            v0 + r + (2 * r) * torch.arange(n).float()
        )  # [-1 + 1/h, ... 1 + 1/h]，共h个数字
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def pixel_samples(img):  # img = (c,h,w), Return Coords in [-1,1]
    coord = make_coord(img.shape[-2:])
    value = img.view(img.shape[0], -1).permute(1, 0)
    return coord, value  # (hw, 2) , (hw, c)


def make_intcoord(sidelen, linear=True):  # (N,) -> (N**2, 2) or (1, 2, N, N)
    tensors = tuple(2 * [torch.linspace(-1, 1, steps=sidelen)])
    if linear:
        mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1).reshape(-1, 2)
    else:
        mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1).reshape(
            1, 2, sidelen, sidelen
        )

    return mgrid


# ---- gradients ----
def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):  # Sum of all partial derivatives
    div = 0.0
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(
            y[..., i], x, torch.ones_like(y[..., i]), create_graph=True
        )[0][..., i : i + 1]
        print(div)
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    # 2nd element of grad is grad_fn = <Mulbackwardl>
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


# ---- highpass & lowpass ----
class Highpass:
    def __init__(self):
        self.laplacian_kernel = torch.tensor(
            [[1.0, 1.0, 1.0], [1.0, -8.0, 1.0], [1.0, 1.0, 1.0]], requires_grad=False
        )

        self.sharpen_kernel = torch.tensor(
            [[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]], requires_grad=False
        )

        self.scharr_x = torch.tensor(
            [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]],
            requires_grad=False,
        )

        self.scharr_y = torch.tensor(
            [[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]],
            requires_grad=False,
        )

    def laplacian_filter(self, data):  # Partial Edge
        dev = data.device
        c = data.shape[1]
        return F.conv2d(
            inp=data,
            weight=self.laplacian_kernel.expand(c, c, 3, 3).to(dev),
            padding=1,
        )

    def sharpen_filter(self, data):  # Sharpening
        dev = data.device
        c = data.shape[1]
        return F.conv2d(
            inp=data, weight=self.sharpen_kernel.expand(c, c, 3, 3).to(dev), padding=1
        )

    def edge_filter(self, data, approach="mix"):
        dev = data.device
        c = data.shape[1]

        if approach == "sharpen":
            rs = F.conv2d(
                inp=data,
                weight=self.sharpen_kernel.expand(c, c, 3, 3).to(dev),
                padding=1,
            )
            rs = rs - data

        elif approach == "blur":
            rs = F.avg_pool2d(data, kernel_size=5, stride=1, padding=2)
            rs = data - rs

        elif approach == "mix":
            rs = F.conv2d(
                inp=F.conv2d(
                    inp=data,
                    weight=self.sharpen_kernel.expand(c, c, 3, 3).to(dev),
                    padding=1,
                ),
                weight=self.laplacian_kernel.expand(c, c, 3, 3).to(dev),
                padding=1,
            )

        return rs


def get_lowpass(data, kernel_size=5):
    rs = gaussian_blur(data, kernel_size=[kernel_size, kernel_size])
    return rs


# ---- Image Functions----
def image_display(img_dict: dict):
    num_images = len(img_dict)
    cols = int(np.ceil(np.sqrt(num_images)))
    rows = int(np.ceil(num_images / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()
    for i, (name, tensor) in enumerate(img_dict.items()):
        tensor = np.array(tensor)
        if tensor.shape[0] == 1:
            image = tensor[0]
        elif tensor.shape[0] == 3:
            image = np.transpose(tensor, (1, 2, 0))
        else:
            raise ValueError(f"Unsupported channel size: {tensor.shape[0]}")
        axes[i].imshow(image, cmap="gray" if tensor.shape[0] == 1 else None)
        axes[i].set_title(name)
        axes[i].axis("off")
    for ax in axes[len(img_dict) :]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


class PatchEmbed(nn.Module):
    def __init__(
        self, img_size=256, patch_size=4, in_chans=8, embed_dim=96, norm_layer=None
    ):
        super().__init__()
        img_size = (img_size,) * 2
        patch_size = (patch_size,) * 2
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(
        self, img_size=256, patch_size=4, in_chans=8, embed_dim=96, norm_layer=None
    ):
        super().__init__()
        img_size = (img_size,) * 2
        patch_size = (patch_size,) * 2
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(
            x.shape[0], self.embed_dim, x_size[0], x_size[1]
        )  # b Ph*Pw c
        return x


# ---- Histogram Matching ----
def histogram_matching(
    inp: torch.Tensor,
    target: torch.Tensor,
    blocksize: list = [4, 4],
):
    """
    Performs histogram matching from inp (PAN) to target (MS).
    Can operate globally or block-wise.
    inp: (B, 1, H, W)
    target: (B, C, h, w)
    Output: (B, C, H, W)
    """
    C_target = target.shape[1]

    def block_mapping(inp_block, target_block):
        """
        Performs histogram matching for a batch of blocks.
        inp_block: (N, 1, H_pan_block, W_pan_block)
        target_block: (N, C_target, H_ms_block, W_ms_block)
        Output: (N, C_target, H_pan_block, W_pan_block)
        """
        mean_I = torch.mean(inp_block, dim=(-1, -2), keepdim=True)  # (N, 1, 1, 1)
        std_I = torch.std(inp_block, dim=(-1, -2), keepdim=True)  # (N, 1, 1, 1)

        mean_T_channelwise = torch.mean(
            target_block, dim=(-1, -2), keepdim=True
        )  # (N, C_target, 1, 1)
        std_T_channelwise = torch.std(
            target_block, dim=(-1, -2), keepdim=True
        )  # (N, C_target, 1, 1)

        matched = (inp_block - mean_I) * (
            std_T_channelwise / (std_I + 1e-6)
        ) + mean_T_channelwise
        return matched

    if not blocksize[0]:
        mean_I_global = torch.mean(inp, dim=(-1, -2), keepdim=True)  # (B, 1, 1, 1)
        std_I_global = torch.std(inp, dim=(-1, -2), keepdim=True)  # (B, 1, 1, 1)

        mean_T_global_channelwise = torch.mean(
            target, dim=(-1, -2), keepdim=True
        )  # (B, C_target, 1, 1)
        std_T_global_channelwise = torch.std(
            target, dim=(-1, -2), keepdim=True
        )  # (B, C_target, 1, 1)

        pan_match = (inp - mean_I_global) * (
            std_T_global_channelwise / (std_I_global + 1e-6)
        ) + mean_T_global_channelwise

    else:
        ratio = int(inp.shape[-1] / target.shape[-1])
        h_ms, w_ms = target.shape[-2:]
        ph, pw = blocksize[0], blocksize[1]

        num_h_blocks = h_ms // ph
        num_w_blocks = w_ms // pw

        inp_blocks = (
            inp.unfold(2, ph * ratio, ph * ratio)
            .unfold(3, pw * ratio, pw * ratio)
            .permute(0, 2, 3, 1, 4, 5)
            .contiguous()
            .view(-1, inp.shape[1], ph * ratio, pw * ratio)
        )

        target_blocks = (
            target.unfold(2, ph, ph)
            .unfold(3, pw, pw)
            .permute(0, 2, 3, 1, 4, 5)
            .contiguous()
            .view(-1, C_target, ph, pw)
        )

        matched_blocks = block_mapping(inp_blocks, target_blocks)

        pan_match = matched_blocks.reshape(
            inp.shape[0], num_h_blocks, num_w_blocks, C_target, ph * ratio, pw * ratio
        )
        pan_match = pan_match.permute(0, 3, 1, 4, 2, 5)
        pan_match = pan_match.reshape(
            inp.shape[0], C_target, h_ms * ratio, w_ms * ratio
        )

    return pan_match


# ---- Displacement ----
def warp(
    source,
    offset_value,
    angle,
    inverse=False,
    ret_full=False,
):
    """
    Forward Warp:
    offset_value, angle -> warp

    Reverse Warp:
    -offset_value, angle -> warp(inverse)
    """
    device = source.device

    # Step 1: Set Displacement Matrix & Rotation Matrix
    B, C, H, W = source.shape
    id_h, id_w = torch.meshgrid(
        [
            torch.linspace(-1, 1, H, device=device, dtype=source.dtype),
            torch.linspace(-1, 1, W, device=device, dtype=source.dtype),
        ]
    )

    # Initialize displacement matrix
    displacement = torch.zeros(B, 2, H, W, device=device, dtype=torch.float32)
    displacement[:, 0, ...] = offset_value[0]  # Horizontal displacement
    displacement[:, 1, ...] = offset_value[1]  # Vertical displacement

    # Calculate rotation matrix components
    angle_rad = math.radians(angle)  # Convert angle to radians
    cos_theta = torch.tensor(math.cos(angle_rad), device=device, dtype=source.dtype)
    sin_theta = torch.tensor(math.sin(angle_rad), device=device, dtype=source.dtype)

    # Step 2: Apply displacement and rotation to the mesh grid
    if not inverse:
        # For forward warp: rotate the mesh grid and apply displacement
        rotated_h = id_h * cos_theta - id_w * sin_theta
        rotated_w = id_h * sin_theta + id_w * cos_theta

        deform_h = rotated_h + displacement[:, 0] * 2 / H
        deform_w = rotated_w + displacement[:, 1] * 2 / W
    else:
        # For inverse warp: first apply displacement and then rotate in the opposite direction
        offset_h = id_h + displacement[:, 0] * 2 / H
        offset_w = id_w + displacement[:, 1] * 2 / W

        deform_h = offset_h * cos_theta + offset_w * sin_theta
        deform_w = -offset_h * sin_theta + offset_w * cos_theta

    # Create a deformation grid (N, H, W, 2) for grid_sample
    deformation = torch.stack((deform_w, deform_h), -1)

    # Step 3: Use grid_sample to warp the source image
    warped_image = torch.clamp(
        F.grid_sample(
            source,
            deformation,
            mode="bilinear",
            padding_mode="reflection",
            align_corners=True,
        ),
        min=0,
        max=1,
    )

    if ret_full:
        return (
            warped_image,
            warped_image[:, :, H // 4 : H // 4 * 3, W // 4 : W // 4 * 3],
        )
    return warped_image[:, :, H // 4 : H // 4 * 3, W // 4 : W // 4 * 3]


# ---- Others ----
def conti_clamp(inp, min, max):
    return inp.clamp(min=min, max=max).detach() + inp - inp.detach()


def save_checkpoint(model, epoch, name):
    model_path = "Weights" + "/" + "{}_".format(name) + "{}.pth".format(epoch)
    torch.save(model.state_dict(), model_path)
    print(f"==== Model successfully saved at {model_path} ====")


if __name__ == "__main__":
    coord_2d = torch.randn(41 * 41, 2)
    coord_4d = coord_2d.reshape(1, 2, 41, 41)
    out1 = F.softmax(coord_2d.transpose(0, 1))
    sm = nn.Softmax2d()
    out2 = sm(coord_4d).reshape(2, 41 * 41)

    print(out1[0, 0:10])
    print(out2[0, 0:10])
