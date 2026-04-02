import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import DownNext


class Discriminator(nn.Module):
    def __init__(
        self,
        in_ch=8,
        out_ch=1,
        hidden_ch=64,
        depth=3,
    ):
        super(Discriminator, self).__init__()
        self.depth = depth

        self.backbone = nn.Sequential(
            DownNext(in_ch, hidden_ch),
            *(DownNext(hidden_ch, hidden_ch) for _ in range(depth - 1)),
            nn.LazyConv2d(out_ch, 1, 1),
        )

    def forward(self, x):
        return self.backbone(x)


class GANLoss:
    def __init__(self, Dms, Dpan):
        self.Dms = Dms
        self.Dpan = Dpan

    @staticmethod
    def ZeroCenteredGradientPenalty(Samples, Critics):  # 0-GP
        """
        Critics 关于 Samples 求导，获得的的梯度的平方(在各个维度上求和)
        """
        (Gradient,) = torch.autograd.grad(
            outputs=Critics.sum(), inputs=Samples, create_graph=True
        )
        return Gradient.square().sum([1, 2, 3])

    def Generator_Loss(self, ms, pan, ms_pred, pan_pred, Scale=1, _lambda=2):
        ms = ms.detach()
        pan = pan.detach()
        Fakepan = self.Dpan(pan_pred)
        Realpan = self.Dpan(pan)

        Fakems = self.Dms(ms_pred)
        Realms = self.Dms(ms)

        RelativisticLogits = _lambda * (Fakepan - Realpan) + Fakems - Realms
        AdversarialLoss = F.softplus(-RelativisticLogits)

        return Scale * AdversarialLoss.mean()

    def Discriminator_Loss(
        self, ms, pan, ms_pred, pan_pred, Scale=1, _lambda=2, Gamma=1
    ):
        ms = ms.detach().requires_grad_(True)
        ms_pred = ms_pred.detach().requires_grad_(True)

        pan = pan.detach().requires_grad_(True)
        pan_pred = pan_pred.detach().requires_grad_(True)

        Fakepan = self.Dpan(pan_pred)
        Realpan = self.Dpan(pan)

        Fakems = self.Dms(ms_pred)
        Realms = self.Dms(ms)

        R1Penalty_ms = GANLoss.ZeroCenteredGradientPenalty(ms, Realms)
        R2Penalty_ms = GANLoss.ZeroCenteredGradientPenalty(ms_pred, Fakems)

        R1Penalty_pan = GANLoss.ZeroCenteredGradientPenalty(pan, Realpan)
        R2Penalty_pan = GANLoss.ZeroCenteredGradientPenalty(pan_pred, Fakepan)

        RelativisticLogits = _lambda * (Fakepan - Realpan) + Fakems - Realms
        AdversarialLoss = F.softplus(-RelativisticLogits)

        DiscriminatorLoss = AdversarialLoss + (Gamma / 2) * (
            R1Penalty_ms + R2Penalty_ms + R1Penalty_pan + R2Penalty_pan
        )

        return Scale * DiscriminatorLoss.mean()
