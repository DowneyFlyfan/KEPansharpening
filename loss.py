import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt

from config import bargs


class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    @torch.jit.script
    def forward(self, img):  # (b,c,h,w)
        w_variance = torch.sum(torch.pow(img[:, :, :, :-1] - img[:, :, :, 1:], 2))
        h_variance = torch.sum(torch.pow(img[:, :, :-1, :] - img[:, :, 1:, :], 2))
        loss = (h_variance + w_variance) / torch.prod(torch.tensor(img.shape[1:]))
        return loss


class multi_batch_mse(nn.Module):
    """a double qsrt Loss for zero-shot pansharpening
    gt: (L, c, p, p)
    pred: (L, c, p, p)
    """

    def __init__(self):
        super().__init__()

    def forward(self, gt, pred):
        loss = (pred - gt).pow(2).mean(dim=(-1, -2)).sqrt().sum(dim=0).sqrt().sum()

        return loss


class SAMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gt, pred):
        t = (torch.sum(gt * gt, 1) * torch.sum(pred * pred, 1)) ** 0.5
        num = torch.sum(torch.gt(t, 0))
        angle = torch.acos(
            torch.clamp(torch.sum(gt * pred, 1) / t, min=-1, max=0.99999997)
        )
        return angle.sum() / num


class multi_batch_sam(nn.Module):
    """a patch-wise SAM Loss for zero-shot pansharpening
    gt: (bhw, c, p, p)
    pred: (bhw, c, p, p)
    """

    def __init__(self):
        super().__init__()
        self.eps = 1e-8

    def forward(self, gt, pred):
        angle = torch.acos(
            torch.clamp(
                torch.sum(gt * pred, 1)
                / (torch.sum(gt * gt, 1) * torch.sum(pred * pred, 1)) ** 0.5,
                min=-1,
                max=0.99999997,  # 是1的话角度为0.还是会出现nan
            ).mean(dim=(-1, -2))
        )

        return angle.sum()


class RegLoss(nn.Module):
    def __init__(self, lambda_reg):
        super(RegLoss, self).__init__()
        self.lambda_reg = lambda_reg

    def forward(self, residual):
        residual_flatten = residual.reshape(-1)
        residual_zoom = residual_flatten * self.lambda_reg
        loss = torch.norm(residual_zoom, p=2, dim=0)
        return loss


class Wavelet_Loss(nn.Module):
    def __init__(self, wavelet="haar", level=2):
        super(Wavelet_Loss, self).__init__()
        self.loss_fn = nn.MSELoss()
        self.wavelet = wavelet
        self.level = level

    def forward(self, pan_pred, pan):
        pan_pred_np = pan_pred.squeeze().cpu().detach().numpy()
        pan_np = pan.squeeze().cpu().detach().numpy()

        coeffs_pred = pywt.wavedec2(pan_pred_np, wavelet=self.wavelet, level=self.level)
        coeffs_true = pywt.wavedec2(pan_np, wavelet=self.wavelet, level=self.level)

        total_loss = 0.0

        for coeff_p, coeff_t in zip(coeffs_pred, coeffs_true):
            if isinstance(coeff_p, tuple):
                for cp, ct in zip(coeff_p, coeff_t):
                    cp_tensor = torch.from_numpy(cp).float().to(pan_pred.device)
                    ct_tensor = torch.from_numpy(ct).float().to(pan.device)

                    total_loss += self.loss_fn(
                        cp_tensor.unsqueeze(0), ct_tensor.unsqueeze(0)
                    )
            else:
                cp_tensor = torch.from_numpy(coeff_p).float().to(pan_pred.device)
                ct_tensor = torch.from_numpy(coeff_t).float().to(pan.device)
                total_loss += self.loss_fn(
                    cp_tensor.unsqueeze(0), ct_tensor.unsqueeze(0)
                )

        return total_loss


if __name__ == "__main__":
    loss_fn = SAMLoss()
    gt = torch.randn(100, 8, 256, 256)
    pred = torch.rand(100, 8, 256, 256)


"""Archive
class patchified_mseloss(nn.Module):
    double sqrted mse loss for patchified inputs (zero-shot only)
    batch_size = 1 supported for now !!!

    gt: (L, c, p, p)
    pred: (L, c, p, p)
    mask: (L,)

    def __init__(self, patch_size=4):
        super().__init__()
        self.num_patches = (bargs.test_sidelen // patch_size) ** 2
        self.patch_size = patch_size

    def forward(self, gt, pred, mask):
        # print(gt.shape)
        # print(pred.shape)
        # print(mask.shape)
        loss = (
            (torch.mean((pred - gt) ** 2, dim=(-1, -2)).sqrt() * mask.unsqueeze(-1))
            .sum(dim=0)
            .sqrt()
            .sum()
        )
        return loss


class patchified_samloss(nn.Module):
    a patch-wise SAM Loss for patchified inputs (zero-shot only)
    batch_size = 1 supported for now !!!

    gt: (L, c, p, p)
    pred: (L, c, p, p)
    mask: (L,)

    def __init__(self, patch_size=4):
        super().__init__()
        self.num_patches = (bargs.test_sidelen // patch_size) ** 2
        self.patch_size = patch_size
        self.eps = 1e-8

    def forward(self, gt, pred, mask):
        angle = torch.acos(
            torch.clamp(
                torch.sum(gt * pred, 1)
                / (torch.sum(gt * gt, 1) * torch.sum(pred * pred, 1)) ** 0.5,
                min=-1,
                max=0.99999997,  # 是1的话角度为0.还是会出现nan
            ).mean(dim=(-1, -2))
        )  # (L, )

        return (angle * mask).sum() / mask.sum()
        # return angle.sum()
"""
