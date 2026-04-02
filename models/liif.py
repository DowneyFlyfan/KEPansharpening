from misc.misc import make_coord, pixel_samples
from .UNet import UNet

import torch
import torch.nn as nn

# import tinycudann as tcnn
import torch.nn.functional as F


# ---- Decoder == CNN ----
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.unfold = nn.Unfold(3, 1, 1)
        self.backbone = nn.Sequential(
            nn.LazyConv2d(128, 3, 1, 2, 2),
            nn.LeakyReLU(),
            nn.LazyConv2d(128, 3, 1, 1),
            nn.LeakyReLU(),
            nn.LazyConv2d(1, 3, 1, 1),
        )

    def forward(self, input_):
        b, c, h, w = input_.shape
        feat = self.unfold(input_).view(b, 9 * c, h, w)
        pan_pred = self.backbone(feat)
        return pan_pred


# ---- LIIF ----
"""
class LIIF(nn.Module):
    def __init__(
        self, sidelen, local_ensemble=True, feat_unfold=True, cell_decode=True
    ):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        self.sidelen = sidelen

        decoder_in_dim = 8

        if self.feat_unfold:
            decoder_in_dim *= 9
        decoder_in_dim += 2
        if self.cell_decode:
            decoder_in_dim += 2

        self.decoder_in_dim = decoder_in_dim

        self.decoder = Decoder()

    def forward(
        self, feat, coord, cell
    ):  # coord = (b,q = H*W,2), coords range = (-1,1)

        if self.decoder is None:
            ret = F.grid_sample(
                feat, coord.flip(-1).unsqueeze(1), mode="nearest", align_corners=False
            )[:, :, 0, :].permute(
                0, 2, 1
            )  # (b,q,d)
            return ret

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3]
            )

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2  # 1/w
        ry = 2 / feat.shape[-1] / 2  # 1/h

        feat_coord = (
            make_coord(feat.shape[-2:], flatten=False)
            .cuda()
            .permute(2, 0, 1)
            .unsqueeze(0)
            .expand(feat.shape[0], 2, *feat.shape[-2:])
        )  # M Coords = (b,2,h,w)

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()  # coord_是用来移位采点的
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat,
                    coord_.flip(-1).unsqueeze(1),  # flip: w,h -> h,w (row, column)
                    mode="nearest",
                    align_corners=False,
                )[:, :, 0, :].permute(
                    0, 2, 1
                )  # z = (b,q,d)

                q_coord = F.grid_sample(
                    feat_coord,
                    coord_.flip(-1).unsqueeze(1),
                    mode="nearest",
                    align_corners=False,
                )[:, :, 0, :].permute(
                    0, 2, 1
                )  # v = (b,q,2)

                rel_coord = coord - q_coord  # x-v
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                bs, q = coord.shape[:2]

                inp = torch.cat([q_feat, coord], dim=-1)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.decoder(
                    inp.contiguous().view(
                        bs, self.decoder_in_dim, self.sidelen, self.sidelen
                    )
                ).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:  # Exchange the area of diagonal areas
            t = areas[0]
            areas[0] = areas[3]
            areas[3] = t
            t = areas[1]
            areas[1] = areas[2]
            areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)

        return ret

    def batched_predict(self, model, coord, cell, bsize):
        with torch.no_grad():
            n = coord.shape[1]
            ql = 0
            preds = []
            while ql < n:
                qr = min(ql + bsize, n)
                pred = model.query_img(coord[:, ql:qr, :], cell[:, ql:qr, :])
                preds.append(pred)
                ql = qr
            pred = torch.cat(preds, dim=1)
        return pred  # (b,hw,1)
"""


class LIIF(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = Decoder()

        self.v_lst = [-1, 1]

    def forward(self, feat, coord=None):
        """
        coord = (b,q = H*W,2), coords range = (-1,1)
        feat = (b, c, h, w)
        """
        feat = self.decoder(feat)
        b, c, h, w = feat.shape
        feat_coord = (
            make_coord((h, w), flatten=False)
            .cuda()
            .permute(2, 0, 1)
            .unsqueeze(0)
            .expand(b, 2, h, w)
        )  # M Coords = (1,2,h,w)

        coord = feat_coord.reshape(b, -1, 2)

        rx = 2 / feat.shape[-2] / 2  # 1/w
        ry = 2 / feat.shape[-1] / 2  # 1/h

        preds = []
        areas = []
        for vx in self.v_lst:
            for vy in self.v_lst:
                coord_ = coord.clone()  # coord_是用来移位采点的
                coord_[:, :, 0] += vx * rx + 1e-6
                coord_[:, :, 1] += vy * ry + 1e-6
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                pred = F.grid_sample(
                    feat,
                    coord_.flip(-1).unsqueeze(1),  # flip: w,h -> h,w (row, column)
                    mode="nearest",
                    align_corners=False,
                )[:, :, 0, :].permute(
                    0, 2, 1
                )  # z = (b,q,d)

                q_coord = F.grid_sample(
                    feat_coord,
                    coord_.flip(-1).unsqueeze(1),
                    mode="nearest",
                    align_corners=False,
                )[:, :, 0, :].permute(
                    0, 2, 1
                )  # v = (b,q,2)

                rel_coord = coord - q_coord  # x-v
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]

                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        # Local Ensemble
        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]
        areas[0] = areas[3]
        areas[3] = t
        t = areas[1]
        areas[1] = areas[2]
        areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)

        return ret.contiguous().view(b, c, h, w)
