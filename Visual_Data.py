from config import dargs
from main import PansharpeningTrainer
from MTF import MtfConv, MTF_MS, wald_protocol
from misc.misc import histogram_matching
from model import HRMS2PAN, MRANet

from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import torch
import h5py
from itertools import islice
import seaborn as sns
import warnings
import os

import logging

logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(
    logging.ERROR
)

warnings.filterwarnings("ignore")


class DatasetPan(Dataset):
    def __init__(self, file_path):
        super(DatasetPan, self).__init__()
        dataset = h5py.File(file_path)

        self.ms = torch.from_numpy(np.array(dataset["ms"][...]) / 2047.0)
        self.lms = torch.from_numpy(np.array(dataset["lms"][...]) / 2047.0)
        self.pan = torch.from_numpy(np.array(dataset["pan"][...]) / 2047.0)
        self.exists_gt = "gt" in dataset

        if self.exists_gt:
            self.gt = torch.from_numpy(np.array(dataset["gt"][...]) / 2047.0)

    def __getitem__(self, index):
        if self.exists_gt:
            return {
                "ms": self.ms[index, :, :, :],
                "pan": self.pan[index, :, :, :],
                "gt": self.gt[index, :, :, :],
                "lms": self.lms[index, :, :, :],
            }
        else:
            return {
                "ms": self.ms[index, :, :, :],
                "pan": self.pan[index, :, :, :],
                "lms": self.lms[index, :, :, :],
            }

    def __len__(self):
        return self.lms.shape[0]

    class Data_Visualization(PansharpeningTrainer):
        def __init__(self, img_num, path):
            """
            Initializes the Data_Visualization class, loading and preparing image data
            for visualization and analysis.
            Args:
                img_num (int): The index of the image to load from the dataset (1-indexed).
            """
            super().__init__()
            self.blur_conv = MtfConv(kernel="pan")
            self.img_num = img_num
            self.gnyq_0 = MTF_MS(dargs.ratio, gain_only=True)

            # The original file is in (b, h, w, c) format.
            # After extraction with h5py, it becomes (c, w, h, b), so permute(3, 0, 2, 1).
            # With MATLAB, it would be permute(0, 3, 1, 2).

            if dargs.sensor == "QB":
                mat_data = sio.loadmat(path)
                self.gt_pred = (
                    torch.from_numpy(np.array(mat_data["HRMS"]))
                    .permute(0, 3, 1, 2)[self.img_num - 1]
                    .unsqueeze(0)
                )
            elif dargs.sensor == "WV3":
                with h5py.File(path, "r") as h5_data:
                    self.gt_pred = torch.from_numpy(np.array(h5_data["HRMS"][...]))
                    self.gt_pred = self.gt_pred.permute(3, 0, 2, 1)[
                        self.img_num - 1
                    ].unsqueeze(0)

            dataloader = self.get_dataloader(one_shot=True)

            batch = next(islice(dataloader, img_num - 1, img_num))
            self.ms = batch["ms"].to(dargs.device, dargs._dtype)
            self.lms = batch["lms"].to(dargs.device, dargs._dtype)
            self.pan = batch["pan"].to(dargs.device, dargs._dtype)
            self.pan_down = self.blur_conv(self.pan, 4)
            self.g_init, self.pan_hp = self.MRAm(self.pan, self.ms)

            B, C, H, W = self.pan.shape
            new_H = H // 4
            new_W = W // 4
            self.pan_patches = self.pan.contiguous().view(B, C, 4, new_H, 4, new_W)
            # Permute to bring block dimensions to the front: (4, 4, B, C, new_H, new_W)
            self.pan_patches = self.pan_patches.permute(2, 4, 0, 1, 3, 5)
            # Reshape to (16, B, C, new_H, new_W)
            self.pan_patches = self.pan_patches.reshape(16, B, C, new_H, new_W)

            if "gt" in batch:
                self.has_gt = True
                self.gt = batch["gt"].to(dargs.device, dargs._dtype)
                B, C, H, W = self.gt.shape
                self.gt_patches = self.gt.contiguous().view(B, C, 4, new_H, 4, new_W)
                self.gt_patches = self.gt_patches.permute(2, 4, 0, 1, 3, 5)
                self.gt_patches = self.gt_patches.reshape(16, B, C, new_H, new_W)

                self.gt_pred_patches = self.gt.contiguous().view(
                    B, C, 4, new_H, 4, new_W
                )
                self.gt_pred_patches = self.gt_pred_patches.permute(2, 4, 0, 1, 3, 5)
                self.gt_pred_patches = self.gt_pred_patches.reshape(
                    16, B, C, new_H, new_W
                )
            else:
                self.has_gt = False

        def get_dataloader(self, one_shot=True):
            if one_shot:
                dataset = DatasetPan(dargs.test_path)

                dataloader = DataLoader(
                    dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=dargs.workers,
                    pin_memory=True,
                    drop_last=True,
                )
                return dataloader

            else:
                train_dataset = DatasetPan(dargs.train_path)

                train_dataloader = DataLoader(
                    train_dataset,
                    batch_size=dargs.train_batch_size,
                    shuffle=True,
                    num_workers=dargs.workers,
                    pin_memory=True,
                    drop_last=True,
                )

                validate_dataset = DatasetPan(dargs.validate_path)

                validate_dataloader = DataLoader(
                    validate_dataset,
                    batch_size=dargs.validate_batch_size,
                    shuffle=False,
                    num_workers=dargs.workers,
                    pin_memory=True,
                    drop_last=True,
                )

                return train_dataloader, validate_dataloader

        def save_img(self, img, name: str):
            """
            Saves an image using torchvision.utils.save_image.

            Args:
                img (Union[torch.Tensor, np.ndarray]): Input image. Supported formats:
                    - torch.Tensor: (b, c, h, w), (c, h, w), or (h, w, c)
                    - np.ndarray: (h, w, c)
                name (str): Base name for the saved image.
            """
            # Type Process
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img).permute(2, 0, 1)  # (c, h, w)

            elif isinstance(img, torch.Tensor):
                img = img.cpu()
                if img.ndim == 4:
                    # (b, c, h, w) -> (c, h, w)
                    img = img.squeeze(0)
                elif img.ndim == 3 and img.shape[0] not in [1, 3]:
                    img = img.permute(2, 0, 1)  # (c, h, w)

            else:
                raise TypeError(f"Unsupported input type: {type(img)}")

            # 2. Handle channel logic
            num_channels = img.shape[0]

            if num_channels > 3:
                if dargs.sensor in ["WV3", "QB"]:
                    img = torch.stack([img[0, :, :], img[1, :, :], img[2, :, :]], dim=0)
                else:
                    print(
                        f"Warning: Input has {num_channels} channels. Defaulting to save the first 3 channels as RGB."
                    )
                    img = img[:3, :, :]

            filepath = f"./Paper/Primitive_Images/{name}_{self.img_num}.png"
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            save_image(img, filepath)

        def enhance_contrast(self, img_tensor):
            img_tensor = self.channel_extract(img_tensor).unsqueeze(0)
            img_np = img_tensor.squeeze().cpu().numpy()
            if len(img_np.shape) == 3:
                img_np = np.transpose(img_np, (1, 2, 0))
            img_8bit = cv2.normalize(
                img_np,
                None,
                alpha=0,
                beta=255,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_8U,
            )
            img_yuv = cv2.cvtColor(img_8bit, cv2.COLOR_BGR2YUV)
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
            img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            return (
                torch.from_numpy(img_output)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(dargs.device, dargs._dtype)
                / 255.0
            )

        def save_imgs(self, save_gout=False):
            if save_gout:
                gnet = MRANet(HRMS2PAN())
                self.optimizer_reset(gnet, 50)
                for _ in range(50):
                    unet_inp, ms_pred, pan_pred, gout = gnet(
                        self.lms, self.g_init, self.pan_hp, True
                    )
                    loss = (
                        self.pwsam(self.ms, ms_pred)
                        + self.pw_mse(self.ms, ms_pred)
                        + self.pw_mse(self.pan, pan_pred) * 8
                    )
                    self.epoch_optimize(loss, 5)

                self.save_img(
                    self.cube_proj(unet_inp.detach()), "cube/stacked_unet_inp"
                )
                self.save_img(
                    self.cube_proj(gout.detach(), brightness=0.5), "cube/stacked_gout"
                )
                self.save_img(unet_inp.detach(), "unet_inp")
                gout_enhanced = self.enhance_contrast(gout.detach())
                self.save_img(gout_enhanced, "gout")

            if self.has_gt:
                self.save_img(self.gt, "gt")
                self.save_img(
                    self.add_grid_to_image(self.gt).unsqueeze(0),
                    "grided_gt",
                )

            # MS
            self.save_img(self.ms, "ms")
            self.save_img(self.lms, "lms")
            self.save_img(self.cube_proj(self.ms), "cube/stacked_ms")
            self.save_img(
                self.cube_proj(self.lms, up_down_offset=1), "cube/stacked_lms"
            )

            # PAN
            self.save_img(self.pan_down, "pan_down")
            self.save_img(self.pan, "pan")
            self.save_img(self.pan_hp, "pan_hp")
            self.save_img(
                self.cube_proj(self.pan_hp, up_down_offset=1), "cube/stacked_pan_hp"
            )

            self.save_img(
                self.add_grid_to_image(self.pan).unsqueeze(0),
                "grided_pan",
            )
            for idx in range(8):
                self.save_img(
                    self.sloped_proj(self.pan_patches[idx], brightness=1.0),
                    "patches/pan_patch_" + str(idx),
                )

            # GT Fused
            self.save_img(self.g_init, "g_init")
            self.save_img(self.gt_pred, "gt_pred")
            self.save_img(
                self.cube_proj(self.g_init, up_down_offset=1), "cube/stacked_g_init"
            )
            self.save_img(
                self.cube_proj(self.gt_pred, up_down_offset=1), "cube/stacked_gt_pred"
            )
            self.save_img(
                self.add_grid_to_image(self.gt_pred).unsqueeze(0),
                "grided_gt_pred",
            )

            for idx in range(8):
                self.save_img(
                    self.cube_proj(self.gt_patches[idx]),
                    "patches/gt_patch_" + str(idx),
                )
            for idx in range(8):
                self.save_img(
                    self.cube_proj(self.gt_pred_patches[idx]),
                    "patches/gt_pred_patch_" + str(idx),
                )

        def channel_extract(self, img):
            c = img.shape[1]
            if c == 8:  # WV3
                rgb_tensor_stacked = torch.stack(
                    [
                        img[:, 4, :, :],
                        img[:, 2, :, :],
                        img[:, 1, :, :],
                    ],
                    dim=1,
                ).squeeze(
                    0
                )  # (BGR) Format
            elif c == 4:  # QB
                rgb_tensor_stacked = torch.stack(
                    [
                        img[:, 2, :, :],
                        img[:, 1, :, :],
                        img[:, 0, :, :],
                    ],
                    dim=1,
                ).squeeze(0)
            elif c == 3 or c == 1:  # RGB & Single-Channel
                rgb_tensor_stacked = img.squeeze(0)
            else:
                raise ValueError("Wrong Number of Channels!!")

            return rgb_tensor_stacked

        def cube_proj(
            self,
            img,
            offset_ratio=0.4,
            side_ratio=0.25,
            channel_repeat_ratio=0.25,
            brightness=1.5,
            up_down_offset=0,
        ):
            """
            Args:
                img (torch.Tensor): Input tensor of shape (1, c, h, w).
                brightness (float): Adjusts the brightness of the top face.

            Returns:
                torch.Tensor: A tensor of shape (1, 3, h, w) representing the final image.
            """
            _, c, h, w = img.shape
            C = int(h * channel_repeat_ratio)
            rgb_tensor_stacked = self.channel_extract(img)

            rgb_tensor = (
                rgb_tensor_stacked / rgb_tensor_stacked.max() * brightness * 255
            )
            rgb_tensor = torch.clamp(rgb_tensor, 0, 255)  # (C, H, W)

            # Get side and top views from the front view
            side_tensor = torch.full(size=(3, h, C), fill_value=255)
            side_tensor[:, :, :] = rgb_tensor[:, :, 0].unsqueeze(-1)

            up_tensor = torch.full(size=(3, C, w), fill_value=255)
            up_tensor[:, :, :] = rgb_tensor[:, 0, :].unsqueeze(1)

            rgb_np = rgb_tensor.byte().cpu().permute(1, 2, 0).numpy()  # (H, W, C)
            side_np = side_tensor.byte().cpu().permute(1, 2, 0).numpy()  # (H, W, C)
            up_np = up_tensor.byte().cpu().permute(1, 2, 0).numpy()  # (H, W, C)

            # Warp front view
            points_src = np.array([[0, 0], [w - 1, 0], [0, h - 1]], dtype=np.float32)
            points_dst = np.array(
                [
                    [0, offset_ratio * h],
                    [w * (1 - offset_ratio), 0],
                    [0, h * (1 + offset_ratio)],
                ],
                dtype=np.float32,
            )
            mat_trans = cv2.getAffineTransform(points_src, points_dst)
            rgb_np = cv2.warpAffine(
                rgb_np,
                mat_trans,
                dsize=(int(w * (1 - offset_ratio)), int(h * (1 + offset_ratio))),
                borderValue=(255, 255, 255),
            )
            rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).to(img.device)

            # Warp side view
            points_src = np.array([[0, 0], [C - 1, 0], [0, h - 1]], dtype=np.float32)
            points_dst = np.array(
                [
                    [0, 0],
                    [C - 1, int(C * side_ratio) - 1],
                    [0, h - 1],
                ],
                dtype=np.float32,
            )
            mat_trans = cv2.getAffineTransform(points_src, points_dst)
            side_np = cv2.warpAffine(
                side_np,
                mat_trans,
                dsize=(C, int(h + C * side_ratio)),
                borderValue=(255, 255, 255),
            )
            side_tensor = torch.from_numpy(side_np).permute(2, 0, 1).to(img.device)

            # Warp top view
            points_src = np.array([[0, 0], [0, C - 1], [w - 1, 0]], dtype=np.float32)
            points_dst = np.array(
                [
                    [0, int(h * offset_ratio) - 1],
                    [C - 1, int(C * side_ratio + h * offset_ratio) - 1],
                    [int(w * (1 - offset_ratio)) - 1, 0],
                ],
                dtype=np.float32,
            )
            mat_trans = cv2.getAffineTransform(points_src, points_dst)
            up_np = cv2.warpAffine(
                up_np,
                mat_trans,
                dsize=(
                    int(C + w * (1 - offset_ratio)),
                    int(C * side_ratio + h * offset_ratio),
                ),
                borderValue=(255, 255, 255),
            )
            up_tensor = torch.from_numpy(up_np).permute(2, 0, 1).to(img.device)

            # Merge the three warped images: top -> partial side and front -> triangular side and front
            out = torch.full(
                size=(
                    3,
                    int(C * side_ratio + h * (1 + offset_ratio)),
                    int(C + w * (1 - offset_ratio)),
                ),
                fill_value=255,
            )

            # Partial merge
            out[
                :,
                up_down_offset : int(C * side_ratio + h * offset_ratio)
                + up_down_offset,
                :,
            ] = up_tensor
            out[:, int(C * side_ratio + h * offset_ratio) - 1 :, :C] = side_tensor[
                :, int(C * side_ratio) - 1 :, :
            ]
            out[:, int(C * side_ratio + h * offset_ratio) - 1 :, C:] = rgb_tensor[
                :, int(h * offset_ratio) - 1 :, :
            ]

            # Triangular merge
            # TODO: There is an offset on the top
            tan_side = 1 / side_ratio
            for i in range(int(C * side_ratio)):
                y_coord = i * tan_side
                out[:, int(h * offset_ratio + i), : int(y_coord)] = side_tensor[
                    :, i, : int(y_coord)
                ]

            tan_up = (1 - offset_ratio) / offset_ratio
            for j in range(int(h * offset_ratio)):
                y_coord = h * (1 - offset_ratio) - tan_up * j - 1
                out[:, int(C * side_ratio + j), int(C + y_coord) :] = rgb_tensor[
                    :, j + 1, int(y_coord) :
                ]
            return out / 255.0

        def add_grid_to_image(self, img, brightness=1.5, grid_size=64):
            rgb_tensor_stacked = self.channel_extract(img)
            _, c, h, w = img.shape
            grid_color = (255, 255, 255)
            rgb_tensor = (
                rgb_tensor_stacked / rgb_tensor_stacked.max() * brightness * 255
            )
            rgb_tensor = torch.clamp(rgb_tensor, 0, 255)  # (C, H, W)

            img_np = rgb_tensor.permute(1, 2, 0).cpu().numpy()
            img_np = np.ascontiguousarray(img_np.astype(np.uint8))

            for i in range(grid_size, h, grid_size):
                cv2.line(img_np, (0, i), (w, i), grid_color, 1, lineType=cv2.LINE_AA)
            for i in range(grid_size, w, grid_size):
                cv2.line(img_np, (i, 0), (i, h), grid_color, 1, lineType=cv2.LINE_AA)

            img_with_grid = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
            return img_with_grid

        def sloped_proj(
            self,
            img,
            brightness=1.5,
            offset_ratio=0.4,
        ):
            b, c, h, w = img.shape
            assert b == 1 and c == 1, "Input shape must be (1, 1, h, w)"

            rgb_tensor = img.squeeze(0) / img.max() * brightness * 255
            rgb_tensor = torch.clamp(rgb_tensor, 0, 255).repeat(3, 1, 1)  # (C, H, W)
            rgb_np = rgb_tensor.byte().cpu().permute(1, 2, 0).numpy()  # (H, W, C)

            # warp
            points_src = np.array([[0, 0], [w - 1, 0], [0, h - 1]], dtype=np.float32)
            points_dst = np.array(
                [
                    [0, offset_ratio * h],
                    [w * (1 - offset_ratio), 0],
                    [0, h * (1 + offset_ratio)],
                ],
                dtype=np.float32,
            )
            mat_trans = cv2.getAffineTransform(points_src, points_dst)
            rgb_np = cv2.warpAffine(
                rgb_np,
                mat_trans,
                dsize=(int(w * (1 - offset_ratio)), int(h * (1 + offset_ratio))),
                borderValue=[255, 255, 255],
            )
            rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).to(img.device)

            return rgb_tensor / 255.0

        def channel_stack(self, img, brightness=1.0, stack_offset=0.05):
            h, w = img.shape[-2:]
            if (dargs.sensor == "WV3") and (img.shape[1] > 4):
                img = (
                    torch.stack(
                        [
                            img[:, 4, :, :],
                            img[:, 2, :, :],
                            img[:, 1, :, :],
                        ],
                        dim=1,
                    )
                    * 255.0
                    * brightness
                )

            max_offset = int(w * stack_offset * 4)
            canvas_h = h * 2
            canvas_w = w + max_offset * 2
            canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

            initial_x = (canvas_w - w) // 2
            initial_y = (canvas_h - h) // 2

            dx = -int(w * stack_offset)
            dy = int(h * stack_offset)

            for i in range(4):
                current_dx = initial_x + dx * i
                current_dy = initial_y + dy * i

                M_translate = np.float32([[1, 0, current_dx], [0, 1, current_dy]])
                translated = cv2.warpAffine(
                    img.squeeze()
                    .permute(1, 2, 0)
                    .to(device="cpu", dtype=torch.uint8)
                    .numpy(),
                    M_translate,
                    (canvas_w, canvas_h),
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(255, 255, 255),
                )

                mask = cv2.cvtColor(translated, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY_INV)
                cv2.copyTo(translated, mask=mask, dst=canvas)

            gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            # TODO: Make threshold values (250, 255) configurable
            _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
            x, y, w, h = cv2.boundingRect(thresh)
            cropped = canvas[y : y + h, x : x + w]
            return cropped

        def kernel_display(self):
            kernel = MTF_MS(dargs.ratio).squeeze().cpu().numpy()  # (c,h,w)

            Z = kernel[0]
            x = np.arange(Z.shape[1])
            y = np.arange(Z.shape[0])
            X, Y = np.meshgrid(x, y)

            fig = plt.figure(figsize=(10, 8), dpi=120, facecolor="black")
            ax = fig.add_subplot(111, projection="3d")

            # Plot the surface (using winter colormap and transparency)
            surf = ax.plot_surface(
                X,
                Y,
                Z,
                cmap="winter",
                rstride=1,
                cstride=1,
                alpha=0.9,
                linewidth=0,
                antialiased=True,
            )

            # Hide all axes and borders
            ax.set_axis_off()
            ax.grid(False)
            ax.view_init(elev=30, azim=45)  # Elevation and azimuth angles

            cbar = fig.colorbar(surf, pad=0.01, shrink=0.7)
            cbar.outline.set_edgecolor("white")
            fig.patch.set_facecolor("black")
            ax.set_facecolor("black")

            plt.savefig(
                "./Paper/Primitive_Images/Kernel/"
                + str(self.img_num)
                + "th_kernel_WV3.png",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0,
                transparent=True,
            )

        def tensor_visualize(self):
            data = self.gnyq_0.cpu().numpy().reshape(1, -1)
            plt.figure(figsize=(8, 1))

            ax = sns.heatmap(
                data,
                cmap="viridis",  # Colormap
                annot=False,  # Do not display values
                cbar=False,  # Do not display the color bar
                square=True,  # Force square cells
                linewidths=0.5,  # Width of the lines between cells
                linecolor="white",
            )

            ax.set(xticks=[], yticks=[])
            ax.set_title("1D Tensor Heatmap (8 elements)", pad=10)
            ax.set_aspect("equal")

            plt.tight_layout()

        def generate_heatmap(self, save_path="./Paper/Primitive_Images/g_heatmap.png"):
            """
            Generates and saves an 8x8 heatmap with random green shades.
            """
            data = np.random.rand(8, 8)
            plt.figure(figsize=(6, 6))
            ax = sns.heatmap(
                data,
                cmap="Greens",  # Use the green color map
                annot=False,  # Do not show values on the heatmap cells
                cbar=False,  # Do not show the color bar
                square=True,  # Make cells square
                linewidths=0.5,  # Add lines between cells
                linecolor="lightgray",  # Color of the lines
            )

            ax.set_xticks([])
            ax.set_yticks([])

            self.save_img(
                self.channel_stack(
                    torch.from_numpy(np.array(Image.open(save_path)))
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                ),
                "g_heatmap",
            )


if __name__ == "__main__":
    process = Data_Visualization(
        img_num=14, path="./Paper/Results/" + dargs.sensor + "_Reduced" + "/Ours.mat"
    )
    process.save_imgs(True)
