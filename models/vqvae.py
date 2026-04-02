from .common import DownNext, UpConv, Permute

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# ---- Utils ----
class Phi(nn.Conv2d):
    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=ks,
            stride=1,
            padding=ks // 2,
        )
        self.resi_ratio = abs(quant_resi)

    def forward(self, h_BChw):
        return h_BChw.mul(1 - self.resi_ratio) + super().forward(h_BChw).mul_(
            self.resi_ratio
        )


class PhiShared(nn.Module):
    def __init__(self, qresi: Phi):
        super().__init__()
        self.qresi: Phi = qresi

    def __getitem__(self, _) -> Phi:
        return self.qresi


class PhiPartiallyShared(nn.Module):
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = (
            np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K)  # (1/12, 11/12, K等分)
            if K == 4
            else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)
        )

    def __getitem__(
        self, at_from_0_to_1: float
    ) -> Phi:  # 输入的可能取值: [0, 1/n, 2/n, ... 1]
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]

    def extra_repr(self) -> str:
        return f"ticks={self.ticks}"


class PhiNonShared(nn.ModuleList):
    def __init__(self, qresi: list):
        super().__init__(qresi)
        K = len(qresi)
        self.ticks = (
            np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K)
            if K == 4
            else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)
        )

    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return super().__getitem__(
            np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()
        )

    def extra_repr(self) -> str:
        return f"ticks={self.ticks}"


# ---- Main Network ----
class VectorQuantizer(nn.Module):
    # VQGAN originally use beta=1.0, never tried 0.25; SD seems using 0.25
    def __init__(
        self,
        vocab_size,
        Cvae,
        using_znorm,
        v_patch_nums,
        beta: float = 0.25,
        default_qresi_counts=0,
        quant_resi=0.5,
        share_quant_resi=4,  # share_quant_resi: args.qsr
    ):
        super().__init__()
        self.vocab_size: int = vocab_size
        self.Cvae: int = Cvae
        self.using_znorm: bool = using_znorm
        self.v_patch_nums = v_patch_nums

        # Step 1: 生成share_quant_resi个3x3卷积
        self.quant_resi_ratio = quant_resi
        if share_quant_resi == 0:  # non-shared: \phi_{1 to K} for K scales
            self.quant_resi = PhiNonShared(  # 不卷
                [
                    (Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity())
                    for _ in range(default_qresi_counts or len(self.v_patch_nums))
                ]
            )
        elif share_quant_resi == 1:  # fully shared: only a single \phi for K scales
            self.quant_resi = PhiShared(  # 只卷
                Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()
            )
        else:  # partially shared: \phi_{1 to share_quant_resi} for K scales
            self.quant_resi = PhiPartiallyShared(
                nn.ModuleList(
                    [
                        (
                            Phi(Cvae, quant_resi)
                            if abs(quant_resi) > 1e-6
                            else nn.Identity()
                        )
                        for _ in range(share_quant_resi)
                    ]
                )
            )

        # Step 2: 注册每一层字典(不进入计算图) 和一个要更新的字典
        self.register_buffer(
            "ema_vocab_hit_SV",
            torch.full((len(self.v_patch_nums), self.vocab_size), fill_value=0.0),
        )
        self.record_hit = 0

        self.beta: float = beta
        self.embedding = nn.Embedding(self.vocab_size, self.Cvae)

        self.prog_si = -1  # progressive training: not supported yet, prog_si always -1

    def forward(self, f_BChw: torch.Tensor):
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()

        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        with torch.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16):
            mean_vq_loss: torch.Tensor = 0.0

            SN = len(self.v_patch_nums)  # scale numbers
            for si, pn in enumerate(self.v_patch_nums):  # from small to large
                # find the nearest embedding
                if self.using_znorm:  # 插值 -> 归一化 -> 找词
                    rest_NC = (
                        F.interpolate(f_rest, size=(pn, pn), mode="area")
                        .permute(0, 2, 3, 1)
                        .reshape(-1, C)
                        if (si != SN - 1)
                        else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                    )
                    rest_NC = F.normalize(rest_NC, dim=-1)  # (bpp, c)
                    idx_N = torch.argmax(
                        rest_NC
                        @ F.normalize(
                            self.embedding.weight.data.T, dim=0
                        ),  # (bpp, vocab_size)
                        dim=1,
                    )
                else:  # 插值 -> x**2 + y**2 - 2xy (计算欧几里得距离)
                    rest_NC = (
                        F.interpolate(f_rest, size=(pn, pn), mode="area")
                        .permute(0, 2, 3, 1)
                        .reshape(-1, C)
                        if (si != SN - 1)
                        else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                    )
                    d_no_grad = torch.sum(
                        rest_NC.square(), dim=1, keepdim=True  # (bpp, 1)
                    ) + torch.sum(
                        self.embedding.weight.data.square(),
                        dim=1,
                        keepdim=False,  # (vocab_size)
                    )  # (bpp, vocab_size)
                    d_no_grad.addmm_(
                        rest_NC,
                        self.embedding.weight.data.T,
                        alpha=-2,
                        beta=1,  # (beta*input + alpha(mat1*mat2))
                    )  # (bpp, vocab_size)

                    idx_N = torch.argmin(d_no_grad, dim=1)  # 找到距离最小的

                # calc loss
                idx_Bhw = idx_N.view(B, pn, pn)
                h_BChw = (
                    F.interpolate(
                        self.embedding(idx_Bhw).permute(
                            0, 3, 1, 2
                        ),  # bhw -> (取词) -> bhwc -> (变形) -> bchw
                        size=(H, W),
                        mode="bicubic",
                    ).contiguous()
                    if (si != SN - 1)
                    else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
                )  # 从字典里取词然后插值回原形状
                h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)  # 按index看选第几个Conv
                f_hat = f_hat + h_BChw  # = h_BChw (Grad)
                f_rest -= h_BChw  # = input - h_BChw (No_Grad)

                mean_vq_loss += F.mse_loss(f_hat.data, f_BChw).mul_(
                    self.beta
                ) + F.mse_loss(f_hat, f_no_grad)

            mean_vq_loss *= 1.0 / SN
            f_hat = (f_hat.data - f_no_grad).add_(f_BChw)

        return f_hat, mean_vq_loss

    def extra_repr(self):  # extra representaiton
        return f"{self.v_patch_nums}, znorm={self.using_znorm}, beta={self.beta}  |  S={len(self.v_patch_nums)}, quant_resi={self.quant_resi_ratio}"

    def embed_to_fhat(self, ms_h_BChw, last_one=False):
        ls_f_hat_BChw = []
        B = ms_h_BChw[0].shape[0]
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)
        f_hat = ms_h_BChw[0].new_zeros(B, self.Cvae, H, W, dtype=torch.float32)
        for si, pn in enumerate(self.v_patch_nums):  # from small to large
            h_BChw = ms_h_BChw[si]
            if si < len(self.v_patch_nums) - 1:
                h_BChw = F.interpolate(h_BChw, size=(H, W), mode="bicubic")
            h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
            f_hat.add_(h_BChw)
            if last_one:
                ls_f_hat_BChw = f_hat
            else:
                ls_f_hat_BChw.append(f_hat.clone())

        return ls_f_hat_BChw

    def f_to_idxBl_or_fhat(
        self,
        f_BChw: torch.Tensor,
        to_fhat: bool,
        v_patch_nums=[1, 2, 3, 4, 8, 10, 12, 14, 16],
    ):  # z_BChw is the feature from inp_img_no_grad
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        f_hat_or_idx_Bl = []

        patch_hws = [
            (pn, pn) if isinstance(pn, int) else (pn[0], pn[1])
            for pn in (v_patch_nums or self.v_patch_nums)
        ]  # from small to large
        assert (
            patch_hws[-1][0] == H and patch_hws[-1][1] == W
        ), f"{patch_hws[-1]=} != ({H=}, {W=})"

        SN = len(patch_hws)
        for si, (ph, pw) in enumerate(patch_hws):  # from small to large
            if 0 <= self.prog_si < si:
                break  # progressive training: not supported yet, prog_si always -1
            # find the nearest embedding
            z_NC = (
                F.interpolate(f_rest, size=(ph, pw), mode="area")
                .permute(0, 2, 3, 1)
                .reshape(-1, C)
                if (si != SN - 1)
                else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
            )
            if self.using_znorm:
                z_NC = F.normalize(z_NC, dim=-1)
                idx_N = torch.argmax(
                    z_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1
                )
            else:
                d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + torch.sum(
                    self.embedding.weight.data.square(), dim=1, keepdim=False
                )
                d_no_grad.addmm_(
                    z_NC, self.embedding.weight.data.T, alpha=-2, beta=1
                )  # (B*h*w, vocab_size)
                idx_N = torch.argmin(d_no_grad, dim=1)

            idx_Bhw = idx_N.view(B, ph, pw)
            h_BChw = (
                F.interpolate(
                    self.embedding(idx_Bhw).permute(0, 3, 1, 2),
                    size=(H, W),
                    mode="bicubic",
                ).contiguous()
                if (si != SN - 1)
                else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
            )
            h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
            f_hat.add_(h_BChw)
            f_rest.sub_(h_BChw)
            f_hat_or_idx_Bl.append(
                f_hat.clone() if to_fhat else idx_N.reshape(B, ph * pw)
            )

        return f_hat_or_idx_Bl

    # ===================== idxBl_to_var_input: only used in VAR training, for getting teacher-forcing input =====================
    def idxBl_to_var_input(self, gt_ms_idx_Bl):
        next_scales = []
        B = gt_ms_idx_Bl[0].shape[0]
        C = self.Cvae
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)

        f_hat = gt_ms_idx_Bl[0].new_zeros(B, C, H, W, dtype=torch.float32)
        pn_next: int = self.v_patch_nums[0]
        for si in range(SN - 1):
            if self.prog_si == 0 or (0 <= self.prog_si - 1 < si):
                break  # progressive training: not supported yet, prog_si always -1
            h_BChw = F.interpolate(
                self.embedding(gt_ms_idx_Bl[si])
                .transpose_(1, 2)
                .view(B, C, pn_next, pn_next),
                size=(H, W),
                mode="bicubic",
            )
            f_hat.add_(self.quant_resi[si / (SN - 1)](h_BChw))
            pn_next = self.v_patch_nums[si + 1]
            next_scales.append(
                F.interpolate(f_hat, size=(pn_next, pn_next), mode="area")
                .view(B, C, -1)
                .transpose(1, 2)
            )
        return (
            torch.cat(next_scales, dim=1) if len(next_scales) else None
        )  # cat BlCs to BLC, this should be float32

    # ===================== get_next_autoregressive_input: only used in VAR inference, for getting next step's input =====================
    def get_next_autoregressive_input(
        self, si: int, SN: int, f_hat: torch.Tensor, h_BChw: torch.Tensor
    ):  # only used in VAR inference
        HW = self.v_patch_nums[-1]
        if si != SN - 1:
            h = self.quant_resi[si / (SN - 1)](
                F.interpolate(h_BChw, size=(HW, HW), mode="bicubic")
            )  # conv after upsample
            f_hat.add_(h)
            return f_hat, F.interpolate(
                f_hat,
                size=(self.v_patch_nums[si + 1], self.v_patch_nums[si + 1]),
                mode="area",
            )
        else:
            h = self.quant_resi[si / (SN - 1)](h_BChw)
            f_hat.add_(h)
            return f_hat, f_hat


class VQVAE(nn.Module):
    def __init__(
        self,
        in_ch=8,
        out_ch=8,
        depth=3,
        upsample_mode="nearest",
        hidden_channels=64,
        vocab_size=256,
        embed_dim=256,
        beta=0.25,  # commitment loss weight
        using_znorm=False,  # whether to normalize when computing the nearest neighbors
        quant_resi=0.5,  # 0.5 means \phi(x) = 0.5conv(x) + (1-0.5)x
        share_quant_resi=4,  # use 4 \phi layers for K scales: partially-shared \phi
        default_qresi_counts=0,  # if is 0: automatically set to len(v_patch_nums)
        v_patch_nums=(
            1,
            2,
            3,
            4,
            5,
            6,
            8,
            10,
            13,
            16,
        ),
    ):
        super().__init__()

        # ---- Encoder & Decoder ----

        ## Decoder 比 Encoder 深一层
        self.depth = depth

        encoder = [DownNext(in_ch, hidden_channels, True, 4, 4)]
        for _ in range(depth - 1):
            encoder.append(DownNext(hidden_channels, embed_dim))
        self.encoder = nn.Sequential(*encoder)

        decoder = [UpConv(embed_dim, hidden_channels, upsample_mode)]
        for _ in range(depth):
            decoder.append(UpConv(hidden_channels, hidden_channels, upsample_mode))

        self.decoder = nn.Sequential(*decoder)

        ## Conv 改 Linear
        self.final_conv = nn.Sequential(
            Permute((0, 2, 3, 1)),
            nn.Linear(out_ch, out_ch),
            Permute((0, 3, 1, 2)),
            nn.Sigmoid(),
        )

        # ---- Quantizer ----
        self.vocab_size, self.Cvae = vocab_size, embed_dim
        self.quantize = VectorQuantizer(
            vocab_size=vocab_size,
            Cvae=self.Cvae,
            using_znorm=using_znorm,
            beta=beta,
            default_qresi_counts=default_qresi_counts,
            v_patch_nums=v_patch_nums,
            quant_resi=quant_resi,
            share_quant_resi=share_quant_resi,
        )
        self.quant_conv = nn.Conv2d(self.Cvae, self.Cvae, 3, 1, 1)
        self.post_quant_conv = nn.Conv2d(self.Cvae, self.Cvae, 3, 1, 1)

    def forward(self, inp):  # -> rec_B3HW, idx_N, loss
        f_hat, vq_loss = self.quantize(self.quant_conv(self.encoder(inp)))
        return self.final_conv(self.decoder(self.post_quant_conv(f_hat))), vq_loss

    def img_to_reconstructed_img(
        self,
        x,
        v_patch_nums=[1, 2, 3, 4, 8, 16],
        last_one=False,
    ):
        f = self.quant_conv(self.encoder(x))
        ls_f_hat_BChw = self.quantize.f_to_idxBl_or_fhat(
            f, to_fhat=True, v_patch_nums=v_patch_nums
        )
        if last_one:
            return self.decoder(self.post_quant_conv(ls_f_hat_BChw[-1])).clamp_(0, 1)
        else:
            return [
                self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1)
                for f_hat in ls_f_hat_BChw
            ]
