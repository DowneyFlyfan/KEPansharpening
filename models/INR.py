"""
INR不管做并行还是串行，输入形状一定是(bhw,c)
1x1Conv 等效于 并行的MLP
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SineLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        is_first=False,
        omega_0=30,
        act="siren",
        parallel=False,
        groups=1,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.act = act

        self.in_features = in_features
        if parallel:
            self.net = nn.Conv2d(in_features, out_features, 1, groups=groups)
        else:
            self.net = nn.Linear(in_features, out_features)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if isinstance(self.net, nn.Conv2d):
                groups = self.net.groups
                effective_in_features = self.in_features // groups
            else:
                effective_in_features = self.in_features

            if self.is_first:
                self.net.weight.uniform_(
                    -1 / effective_in_features, 1 / effective_in_features
                )
            else:
                self.net.weight.uniform_(
                    -np.sqrt(6 / effective_in_features) / self.omega_0,
                    np.sqrt(6 / effective_in_features) / self.omega_0,
                )

            if self.act == "finer" and self.net.bias is not None:
                self.net.bias.uniform_(
                    -1 / np.sqrt(effective_in_features),
                    1 / np.sqrt(effective_in_features),
                )

    def forward(self, x):
        if self.act == "siren":
            return torch.sin(self.omega_0 * self.net(x))
        elif self.act == "hsiren":
            if self.is_first:
                return torch.sin(self.omega_0 * torch.sinh(2 * self.net(x)))
            else:
                return torch.sin(self.omega_0 * self.net(x))
        elif self.act == "finer":
            out = self.net(x)
            return torch.sin(self.omega_0 * out * (1 + torch.abs_(out)))
        else:
            raise ValueError("Wrong type of activation function")


class INR(nn.Module):
    """
    Instance:
    INR(
        in_features=16,
        hidden_features=512,
        hidden_layers=3,
        out_features=margs.channel,
        outermost_linear=True,
        parallel=True,
        act="hsiren",
        groups=8,
        softmax=True,
    )
    """

    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        outermost_linear=True,
        first_omega_0=30,
        hidden_omega_0=30,
        parallel=False,
        act="siren",
        groups=1,
        softmax=True,
    ):
        super().__init__()

        self.softmax = softmax

        net = []
        net.append(
            SineLayer(
                in_features,
                hidden_features,
                is_first=True,
                omega_0=first_omega_0,
                parallel=parallel,
                act=act,
                groups=groups,
            )
        )

        for _ in range(hidden_layers):
            net.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                    parallel=parallel,
                    act=act,
                    groups=groups,
                )
            )

        if outermost_linear:
            if parallel:
                final_layer = nn.Conv2d(hidden_features, out_features, 1, groups=groups)
            else:
                final_layer = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_layer.weight.uniform_(
                    -np.sqrt(6 / (hidden_features // groups)) / hidden_omega_0,
                    np.sqrt(6 / (hidden_features // groups)) / hidden_omega_0,
                )

            net.append(final_layer)
        else:
            net.append(
                SineLayer(
                    hidden_features,
                    out_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                    act=act,
                    parallel=parallel,
                    groups=groups,
                )
            )

        self.net = nn.Sequential(*net)

    def forward(self, coords):
        if self.softmax:
            # return F.softmax(self.net(coords).transpose(0, 1)).view(1, 8, 41, 41)
            return rearrange(
                F.softmax(self.net(coords).squeeze(-1).squeeze(-1).transpose(0, 1)),
                "c (b h w) -> b c h w",
                b=1,
                h=41,
                w=41,
            )
        else:
            return self.net(coords)


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
