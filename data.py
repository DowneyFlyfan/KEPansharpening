from config import dargs

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import TenCrop, Lambda, Compose
import numpy as np
import torch
import h5py


""" Dataset Info
WorldView2:
Full Resolution Testing Dataset: lms = (20, 8, 512, 512), ms = (20, 8, 128, 128), pan = (20, 1, 512, 512)
Reduced Resolution Testing Dataset: lms = (20, 8, 256, 256), ms = (20, 8, 64, 64), pan = (20, 1, 256, 256), gt = (20, 8, 256, 256)

WorldView3:
Full Resolution Testing Dataset: Same as WV2 (No GT)
Reduced Resolution Testing Dataset: lms = (20, 8, 256, 256), ms = (20, 8, 64, 64), pan = (20, 1, 256, 256), gt = (20, 8, 256, 256)

Training Dataset: lms = (9714, 8, 64, 64), ms = (9714, 8, 16, 16), gt = (9714, 8, 64, 64), pan = (9714, 1, 64, 64)
Validation Dataset: lms = (1080, 8, 64, 64), ms = (1080, 8, 16, 16), gt = (1080, 8, 64, 64), pan = (1080, 1, 64, 64)
"""


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
        dataset = PanSharp_Dataset(dargs.test_path, aug=False)

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
