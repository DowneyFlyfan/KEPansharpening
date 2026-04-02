from config import dargs
from MTF import wald_protocol, MtfConv

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import TenCrop, Lambda, Compose
import matplotlib.pyplot as plt
import numpy as np
import torch
import h5py
import os
import scipy.io as sio


""" Dataset Info
WorldView2:
Full Resolution Testing Dataset: lms = (20, 8, 512, 512), ms = (20, 8, 128, 128), pan = (20, 1, 512, 512)
Reduced Resolution Testing Dataset: lms = (20, 8, 256, 256), ms = (20, 8, 64, 64), pan = (20, 1, 256, 256), gt = (20, 8, 256, 256)

WorldView3:
Full Resolution Testing Dataset: Same as WV2 (No GT)
Reduced Resolution Testing Dataset: lms = (20, 8, 256, 256), ms = (20, 8, 64, 64), pan = (20, 1, 256, 256), gt = (20, 8, 256, 256)

Training Dataset: lms = (9714, 8, 64, 64), ms = (9714, 8, 16, 16), gt = (9714, 8, 64, 64), pan = (9714, 1, 64, 64)
Validation Dataset: lms = (1080, 8, 64, 64), ms = (1080, 8, 16, 16), gt = (1080, 8, 64, 64), pan = (1080, 1, 64, 64)

Warped_WV3
lms = (20, 8, 256, 256), warped_lms = (20, 8, 512, 512), ms = (20, 8, 64, 64), pan = (20, 1, 256, 256)
"""


# ---- Archive ----
"""
class Dataset_Extract(Dataset):
    def __init__(self, file_path, no_gt=False):
        super(Dataset_Extract, self).__init__()
        dataset = h5py.File(file_path)

        self.no_gt = no_gt

        ms = np.array(dataset["ms"][...])
        lms = np.array(dataset["lms"][...])
        pan = np.array(dataset["pan"][...])

        if not no_gt:
            gt = np.array(dataset["gt"][...])
            self.gt_int = torch.from_numpy(gt)
            self.gt = self.gt_int / 2047.0

        self.pan_int = torch.from_numpy(pan)
        self.ms_int = torch.from_numpy(ms)
        self.lms_int = torch.from_numpy(lms)

        self.pan = self.pan_int / 2047.0
        self.lms = self.lms_int / 2047.0
        self.ms = self.ms_int / 2047.0

    def __getitem__(self, index):
        if not self.no_gt:
            return {
                "ms": self.ms[index, :, :, :],
                "pan": self.pan[index, :, :, :],
                "gt": self.gt[index, :, :, :],
                "lms": self.lms[index, :, :, :],
                "ms_int": self.ms_int[index, :, :, :],
                "pan_int": self.pan_int[index, :, :, :],
                "gt_int": self.gt_int[index, :, :, :],
                "lms_int": self.lms_int[index, :, :, :],
            }
        else:
            return {
                "ms": self.ms[index, :, :, :],
                "pan": self.pan[index, :, :, :],
                "lms": self.lms[index, :, :, :],
                "ms_int": self.ms_int[index, :, :, :],
                "pan_int": self.pan_int[index, :, :, :],
                "lms_int": self.lms_int[index, :, :, :],
            }

    def __len__(self):
        return self.lms.shape[0]


class Dataset_warped(Dataset):
    def __init__(self, file_path, no_gt=False):
        super(Dataset_warped, self).__init__()
        dataset = h5py.File(file_path)

        self.no_gt = no_gt

        self.ms = torch.from_numpy(np.array(dataset["ms"][...]) / 2047.0)
        self.lms = torch.from_numpy(np.array(dataset["lms"][...]) / 2047.0)
        self.pan = torch.from_numpy(np.array(dataset["pan"][...]) / 2047.0)

        self.warped_ms = torch.from_numpy(np.array(dataset["warped_ms"][...]) / 2047.0)
        self.warped_lms = torch.from_numpy(
            np.array(dataset["warped_lms"][...]) / 2047.0
        )
        self.warped_pan = torch.from_numpy(
            np.array(dataset["warped_pan"][...]) / 2047.0
        )

        if not no_gt:
            self.warped_gt = torch.from_numpy(
                np.array(dataset["warped_gt"][...]) / 2047.0
            )
            self.gt = torch.from_numpy(np.array(dataset["gt"][...]) / 2047.0)

    def __getitem__(self, index):
        if not self.no_gt:
            return {
                "ms": self.ms[index, :, :, :],
                "pan": self.pan[index, :, :, :],
                "gt": self.gt[index, :, :, :],
                "lms": self.lms[index, :, :, :],
                "warped_ms": self.warped_ms[index, :, :, :],
                "warped_pan": self.warped_pan[index, :, :, :],
                "warped_gt": self.warped_gt[index, :, :, :],
                "warped_lms": self.warped_lms[index, :, :, :],
            }

        else:
            return {
                "ms": self.ms[index, :, :, :],
                "pan": self.pan[index, :, :, :],
                "lms": self.lms[index, :, :, :],
                "warped_ms": self.warped_ms[index, :, :, :],
                "warped_pan": self.warped_pan[index, :, :, :],
                "warped_lms": self.warped_lms[index, :, :, :],
            }

    def __len__(self):
        return self.lms.shape[0]


# ---- New Dataset Construction & Test ----
# For constructing man-made warping data,
# with orginal dataset rotate and shift (theta, x, y)


def create_newdataset(dargs=Datadargs(data="wv3_OrigScale")):
    device = "cpu"
    dtype = torch.float64

    dataset = Dataset_Extract("./test_data/test_wv3_OrigScale.h5", True)
    dataloader = DataLoader(dataset, batch_size=20, shuffle=False, num_workers=8)

    H, W = dargs.test_sidelen, dargs.test_sidelen

    # ---- HyperParams ----
    offset_value = (4, -5)  # 小一些
    angle = -3.96  # ±5以内

    with h5py.File("./test_data/warped_wv3_Reduced.h5", "w") as f:
        for batch in dataloader:
            pan = batch["pan"].to(device, dtype)
            ms = batch["ms"].to(device, dtype)
            lms = batch["lms"].to(device, dtype)
            H, W = pan.shape[-2:]
            h, w = ms.shape[-2:]

            pan_int = (
                batch["pan_int"][:, :, H // 4 : H // 4 * 3, W // 4 : W // 4 * 3]
                .cpu()
                .numpy()
            )

            lms_int = (
                batch["lms_int"][:, :, H // 4 : H // 4 * 3, W // 4 : W // 4 * 3]
                .cpu()
                .numpy()
            )

            ms_int = (
                batch["ms_int"][:, :, h // 4 : h // 4 * 3, w // 4 : w // 4 * 3]
                .cpu()
                .numpy()
            )

            warped_pan, _ = warp(pan, offset_value, angle, ret_full=True)
            warped_ms, _ = warp(ms, offset_value, angle, ret_full=True)
            warped_lms, _ = warp(lms, offset_value, angle, ret_full=True)

            warped_ms = warped_ms.cpu().numpy() * 2047.0
            warped_ms = warped_ms.astype(int)

            warped_lms = warped_lms.cpu().numpy() * 2047.0
            warped_lms = warped_lms.astype(int)

            warped_pan = warped_pan.cpu().numpy() * 2047.0
            warped_pan = warped_pan.astype(int)

            f.create_dataset("warped_pan", data=warped_pan)
            f.create_dataset("warped_ms", data=warped_ms)
            f.create_dataset("warped_lms", data=warped_lms)
            f.create_dataset("ms", data=ms_int)
            f.create_dataset("lms", data=lms_int)
            f.create_dataset("pan", data=pan_int)

    print("新的 HDF5 数据集已创建：./test_data/warped_wv3_Reduced.h5")


def image_compare(device="cpu", dtype=torch.float64):
    file_path = "./test_data/warped_wv3_Reduced.h5"

    dataset = Dataset_warped(file_path, True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

    for batch in dataloader:
        lms = batch["lms"].to(device, dtype)
        warped_lms = batch["warped_lms"].to(device, dtype)
        ms = batch["ms"].to(device, dtype)
        warped_ms = batch["warped_ms"].to(device, dtype)
        warped_pan = batch["warped_pan"].to(device, dtype)
        pan = batch["pan"].to(device, dtype)

        break

    # warped = [warped_pan[0], warped_lms[0], warped_ms[0]]
    # origin = [pan[0], lms[0], ms[0]]

    ms_edge = ms - wald_protocol(ms, 1, "WV3")
    pan_edge = pan - wald_protocol(pan, 1, "WV3")
    show_paired_images([ms_edge[0, 0].unsqueeze(0)], [pan_edge[0, :, 2:-1:4, 2:-1:4]])
"""


# ---- Utils ----
def show_paired_images(list1, list2):
    assert len(list1) == len(list2), "Length of 2 lists must be the same!"

    num_pairs = len(list1)
    fig, axes = plt.subplots(2, num_pairs, figsize=(5 * num_pairs, 8))

    if num_pairs == 1:
        axes = axes.reshape(2, 1)  # 将 axes 从 (2,) 转换为 (2, 1)

    for i, (tensor1, tensor2) in enumerate(zip(list1, list2)):
        assert len(tensor1.shape) == 3, "Channel number of tensors must be 3!"
        assert len(tensor2.shape) == 3, "Channel number of tensors must be 3!"

        if tensor1.shape[0] > 3:
            img1_channels = tensor1[1:4].cpu().detach().numpy().transpose(1, 2, 0)
            img2_channels = tensor2[1:4].cpu().detach().numpy().transpose(1, 2, 0)

        elif tensor1.shape[0] == 1:
            img1_channels = tensor1.cpu().detach().numpy().transpose(1, 2, 0)
            img2_channels = tensor2.cpu().detach().numpy().transpose(1, 2, 0)

        axes[0, i].imshow(img1_channels)
        axes[0, i].set_title(f"Pair {i+1}: List1")
        axes[0, i].axis("off")

        axes[1, i].imshow(img2_channels)
        axes[1, i].set_title(f"Pair {i+1}: List2")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()


# ---- Dataset Preprocessing ----
class PanSharp_Dataset(Dataset):
    def __init__(self, file_path, aug=True):
        super(PanSharp_Dataset, self).__init__()
        dataset = h5py.File(file_path)

        self.ms_int = torch.from_numpy(np.array(dataset["ms"][...]))
        self.lms_int = torch.from_numpy(np.array(dataset["lms"][...]))
        self.pan_int = torch.from_numpy(np.array(dataset["pan"][...]))

        self.ms = self.ms_int / 2047.0
        self.lms = self.lms_int / 2047.0
        self.pan = self.pan_int / 2047.0

        self.exists_gt = "gt" in dataset
        self.aug = aug

        if self.exists_gt:
            self.gt_int = torch.from_numpy(np.array(dataset["gt"][...]))
            self.gt = self.gt_int / 2047.0

        if self.aug:
            self.big_transform = Compose(
                [
                    TenCrop(dargs.aug_size),
                    Lambda(lambda crops: torch.stack([crop for crop in crops])),
                ]
            )
            self.small_transform = Compose(
                [
                    TenCrop(dargs.aug_size // dargs.ratio),
                    Lambda(lambda crops: torch.stack([crop for crop in crops])),
                ]
            )

    def __getitem__(self, index):
        item = {
            "ms": self.ms[index, :, :, :],
            "lms": self.lms[index, :, :, :],
            "pan": self.pan[index, :, :, :],
            "ms_int": self.ms_int[index, :, :, :],
            "pan_int": self.pan_int[index, :, :, :],
            "lms_int": self.lms_int[index, :, :, :],
        }

        if self.exists_gt:
            item["gt"] = self.gt[index, :, :, :]
            item["gt_int"] = self.gt_int[index, :, :, :]

        if self.aug:
            item["aug_ms"] = self.small_transform(self.ms[index, :, :, :])
            item["aug_lms"] = self.big_transform(self.lms[index, :, :, :])
            item["aug_pan"] = self.big_transform(self.pan[index, :, :, :])
            if self.exists_gt:
                item["aug_gt"] = self.big_transform(self.gt[index, :, :, :])

        return item

    def __len__(self):
        return self.lms.shape[0]


def get_dataloader(one_shot: bool = True) -> DataLoader | tuple[DataLoader, DataLoader]:
    if one_shot:
        dataset = PanSharp_Dataset(dargs.test_path)

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
        train_dataset = PanSharp_Dataset(dargs.train_path)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=dargs.train_batch_size,
            shuffle=True,
            num_workers=dargs.workers,
            pin_memory=True,
            drop_last=True,
        )

        validate_dataset = PanSharp_Dataset(dargs.validate_path)

        validate_dataloader = DataLoader(
            validate_dataset,
            batch_size=dargs.validate_batch_size,
            shuffle=False,
            num_workers=dargs.workers,
            pin_memory=True,
            drop_last=True,
        )

        return train_dataloader, validate_dataloader


def save_as_mat(output_dir):
    dataloader = get_dataloader()
    os.makedirs(output_dir, exist_ok=True)

    merged_data = {"ms": [], "lms": [], "pan": [], "gt": []}

    for batch in dataloader:
        merged_data["ms"].append(batch["ms"].permute(0, 2, 3, 1).numpy())
        merged_data["lms"].append(batch["lms"].permute(0, 2, 3, 1).numpy())
        merged_data["pan"].append(batch["pan"].squeeze(1).numpy())

        if "gt" in batch:
            merged_data["gt"].append(batch["gt"].permute(0, 2, 3, 1).numpy())

    final_data = {
        "ms": np.concatenate(merged_data["ms"], axis=0),
        "lms": np.concatenate(merged_data["lms"], axis=0),
        "pan": np.concatenate(merged_data["pan"], axis=0),
    }

    filename = os.path.join(output_dir, "test_" + dargs.data + ".mat")
    sio.savemat(filename, final_data)
    print(f"Saved merged data to {filename} with shapes:")
    for k, v in final_data.items():
        print(f"{k}: {v.shape}")


if __name__ == "__main__":
    save_as_mat("./test_data/")
