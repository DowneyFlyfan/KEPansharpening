import torch
import torch.nn as nn
import torch.nn.functional as F

from MTF import wald_protocol
from misc.misc import histogram_matching
from config import margs
from einops import rearrange, repeat
from models.common import ConvNext, Permute


@torch.compile
class Compressive_Sensing(nn.Module):
    def __init__(
        self,
        pan,
        ms,
        lms,
        tile_size=7,
        overlap=4,
        n_atoms=10,
    ):
        super().__init__()

        # Detail Extraction
        self.pan_match = histogram_matching(pan, lms, [0, 0])
        self.pan_hp = self.pan_match - lms

        self.lr_pan_hp = wald_protocol(self.pan_hp, 4)[:, :, 2:-1:4, 2:-1:4]
        # self.ms_hp = ms - wald_protocol(ms, 1, N=11)
        self.ms_hp = (lms - wald_protocol(lms, 4))[:, :, 2:-1:4, 2:-1:4]

        self.lms = lms

        # Hyper Params
        self.stride = tile_size - overlap
        if (ms.shape[-1] - tile_size) % self.stride == 0:
            self.pad_size = 0
        else:
            self.pad_size = int(
                (self.stride - (ms.shape[-1] - tile_size) % self.stride) / 2
            )
        self.patchified_size = (
            int((ms.shape[-1] - tile_size + 2 * self.pad_size) / self.stride) + 1
        )

        self.num_patches = self.patchified_size**2
        self.TS = tile_size
        self.n_atoms = n_atoms

        # Patchified Matrices
        self.hr_dict = torch.empty(
            tile_size**2 * margs.ratio**2 * margs.channel, self.num_patches
        )

        self.lr_dict = torch.empty(tile_size**2 * margs.channel, self.num_patches)
        self.ms_patches = torch.empty(tile_size**2 * margs.channel, self.num_patches)

        self.divisor = F.fold(
            torch.ones(
                1, margs.channel * (self.TS * margs.ratio) ** 2, self.num_patches
            ),
            output_size=(margs.test_sidelen, margs.test_sidelen),
            kernel_size=self.TS * margs.ratio,
            stride=self.stride * margs.ratio,
        )
        self.divisor[self.divisor == 0] = 1.0

        self.index = torch.full(
            size=(self.num_patches, n_atoms),
            fill_value=-1,
            device=margs.device,
            dtype=torch.int,
        )
        self.index[:, 0] = torch.arange(0, self.num_patches)
        self.omp()

    def _patchify(self):
        self.lr_dict = (
            F.unfold(
                self.lr_pan_hp,
                kernel_size=self.TS,
                stride=self.stride,
                padding=self.pad_size,
            )
            .squeeze(0)
            .permute(1, 0)
        )

        self.ms_patches = rearrange(
            F.unfold(
                self.ms_hp,
                kernel_size=self.TS,
                stride=self.stride,
                padding=self.pad_size,
            ),
            "1 (c hw) n -> (n c) hw",
            c=margs.channel,
        )

        self.ms_patches_cube = rearrange(
            self.ms_patches, "(n c) hw -> n (c hw)", c=margs.channel
        )

        self.hr_dict = (
            F.unfold(
                self.pan_hp,
                kernel_size=self.TS * margs.ratio,
                stride=self.stride * margs.ratio,
                padding=self.pad_size * margs.ratio,
            )
            .squeeze(0)
            .permute(1, 0)
        )

    def omp(self):
        self._patchify()

        res = self.ms_patches_cube
        for i in range(self.n_atoms):
            if i != 0:
                self.index[:, i] = torch.argmax(
                    torch.abs_(torch.einsum("mp, np -> mn", res, self.lr_dict)),
                    dim=-1,
                )

            indexed_lr_dict = rearrange(
                self.lr_dict[self.index[:, : i + 1]],
                "n i (c hw) -> (n c) hw i",
                c=margs.channel,
            )

            alpha = torch.linalg.lstsq(indexed_lr_dict, self.ms_patches)[0]
            res = self.ms_patches - torch.einsum(
                "ni, npi -> np", alpha, indexed_lr_dict
            )

            res = rearrange(
                res,
                "(n c) hw -> n (c hw)",
                c=margs.channel,
            )

        self.alpha = alpha
        self.hrms_hp_patches = torch.einsum(
            "ni, nip -> np",
            self.alpha,
            rearrange(
                self.hr_dict[self.index], "n i (c hw) -> (n c) i hw", c=margs.channel
            ),
        )

        hrms_hp = F.fold(
            rearrange(self.hrms_hp_patches, "(n c) hw -> 1 (c hw) n", c=margs.channel),
            output_size=(margs.test_sidelen, margs.test_sidelen),
            kernel_size=self.TS * margs.ratio,
            stride=self.stride * margs.ratio,
        )

        self.hrms_hp = hrms_hp / (self.divisor.to(margs.device, margs._dtype) + 1e-8)

    def forward(self):
        gt_pred = self.hrms_hp + self.lms
        return gt_pred


# @torch.compile
class Compressive_Representation(nn.Module):
    def __init__(
        self,
        pan,
        ms,
        lms,
        tile_size=7,
        overlap=4,
        n_atoms=10,
    ):
        super().__init__()

        # Detail Extraction
        # self.pan_match = histogram_matching(pan, lms, [0, 0])
        # self.pan_hp = self.pan_match - wald_protocol(self.pan_match, 1)
        #
        # self.lr_pan_hp = wald_protocol(self.pan_hp, 4)[:, :, 2:-1:4, 2:-1:4]
        # self.ms_hp = ms - wald_protocol(ms, 1)

        self.pan_match = histogram_matching(pan, lms, [0, 0])
        self.pan_hp = self.pan_match - wald_protocol(self.pan_match, 4)

        self.lr_pan_hp = wald_protocol(self.pan_hp, 4)[:, :, 2:-1:4, 2:-1:4]
        self.ms_hp = ms - wald_protocol(ms, 1, N=11)

        self.lms = lms

        # Hyper Params
        self.stride = tile_size - overlap
        if (ms.shape[-1] - tile_size) % self.stride == 0:
            self.pad_size = 0
        else:
            self.pad_size = int(
                (self.stride - (ms.shape[-1] - tile_size) % self.stride) / 2
            )
        self.patchified_size = (
            int((ms.shape[-1] - tile_size + 2 * self.pad_size) / self.stride) + 1
        )

        self.num_patches = self.patchified_size**2
        self.TS = tile_size
        self.n_atoms = n_atoms

        # Patchified Matrices
        self.hr_dict = torch.empty(
            tile_size**2 * margs.ratio**2 * margs.channel, self.num_patches
        )

        self.lr_dict = torch.empty(tile_size**2 * margs.channel, self.num_patches)
        self.ms_patches = torch.empty(tile_size**2 * margs.channel, self.num_patches)

        self.divisor = F.fold(
            torch.ones(
                1, margs.channel * (self.TS * margs.ratio) ** 2, self.num_patches
            ),
            output_size=(margs.test_sidelen, margs.test_sidelen),
            kernel_size=self.TS * margs.ratio,
            stride=self.stride * margs.ratio,
        )
        self.divisor[self.divisor == 0] = 1.0

        self.index = torch.full(
            size=(self.num_patches, n_atoms),
            fill_value=-1,
            device=margs.device,
            dtype=torch.int,
        )
        self.index[:, 0] = torch.arange(0, self.num_patches)

    def _patchify(self, hrms_hp):
        self.lr_dict = (
            F.unfold(
                self.lr_pan_hp,
                kernel_size=self.TS,
                stride=self.stride,
                padding=self.pad_size,
            )
            .squeeze(0)
            .permute(1, 0)
        )

        self.res_ms_patches = rearrange(
            F.unfold(
                self.ms_hp - wald_protocol(hrms_hp, 4)[:, :, 2:-1:4, 2:-1:4],
                kernel_size=self.TS,
                stride=self.stride,
                padding=self.pad_size,
            ),
            "1 (c hw) n -> (n c) hw",
            c=margs.channel,
        )

        self.res_ms_patches_cube = rearrange(
            self.res_ms_patches, "(n c) hw -> n (c hw)", c=margs.channel
        )

        self.hr_dict = (
            F.unfold(
                self.pan_hp,
                kernel_size=self.TS * margs.ratio,
                stride=self.stride * margs.ratio,
                padding=self.pad_size * margs.ratio,
            )
            .squeeze(0)
            .permute(1, 0)
        )

    def omp(self, hrms_hp):
        self._patchify(hrms_hp)

        res = self.res_ms_patches_cube
        for i in range(self.n_atoms):
            if i != 0:
                self.index[:, i] = torch.argmax(
                    torch.abs_(torch.einsum("mp, np -> mn", res, self.lr_dict)),
                    dim=-1,
                )

            indexed_lr_dict = rearrange(
                self.lr_dict[self.index[:, : i + 1]],
                "n i (c hw) -> (n c) hw i",
                c=margs.channel,
            )

            alpha = torch.linalg.lstsq(indexed_lr_dict, self.res_ms_patches)[0]
            res = self.res_ms_patches - torch.einsum(
                "ni, npi -> np", alpha, indexed_lr_dict
            )

            res = rearrange(
                res,
                "(n c) hw -> n (c hw)",
                c=margs.channel,
            )

        self.hrms_hp = (
            F.fold(
                rearrange(
                    torch.einsum(
                        "ni, nip -> np",
                        alpha,
                        rearrange(
                            self.hr_dict[self.index],
                            "n i (c hw) -> (n c) i hw",
                            c=margs.channel,
                        ),
                    ),
                    "(n c) hw -> 1 (c hw) n",
                    c=margs.channel,
                ),
                output_size=(margs.test_sidelen, margs.test_sidelen),
                kernel_size=self.TS * margs.ratio,
                stride=self.stride * margs.ratio,
            )
            / (self.divisor.to(margs.device, margs._dtype) + 1e-8)
            + hrms_hp
        )

        return self.hrms_hp + self.lms

    def forward(self, hrms_hp):
        return self.omp(hrms_hp)
