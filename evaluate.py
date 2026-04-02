from __future__ import print_function
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import h5py
import numpy as np
from math import ceil
from data import get_dataloader

from MTF import smoothe_interpolation, MtfConv
from config import bargs


# ---- Utils ----
tap_intp = smoothe_interpolation(1)
blur_conv = MtfConv()


def create_window(window_size, sigma, channel):
    gauss = torch.Tensor(
        [
            math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )

    gauss = gauss / gauss.sum()

    _1D_window = gauss.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size)

    return window.to(bargs._dtype)


def norm_blocco(x, eps=1e-8):
    a = x.mean()
    c = x.std()
    if c == 0:
        c = eps
    return (x - a) / c + 1, a, c


def deriveSizeFromScale(img_shape, scale):
    output_shape = []
    for k in range(2):
        output_shape.append(int(ceil(scale[k] * img_shape[k])))
    return output_shape


def deriveScaleFromSize(img_shape_in, img_shape_out):
    scale = []
    for k in range(2):
        scale.append(1.0 * img_shape_out[k] / img_shape_in[k])
    return scale


def triangle(x):
    x = np.array(x).astype(np.float64)
    lessthanzero = np.logical_and((x >= -1), x < 0)
    greaterthanzero = np.logical_and((x <= 1), x >= 0)
    f = np.multiply((x + 1), lessthanzero) + np.multiply((1 - x), greaterthanzero)
    return f


def cubic(x):
    x = np.array(x).astype(np.float64)
    absx = np.absolute(x)
    absx2 = np.multiply(absx, absx)
    absx3 = np.multiply(absx2, absx)
    f = np.multiply(1.5 * absx3 - 2.5 * absx2 + 1, absx <= 1) + np.multiply(
        -0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2, (1 < absx) & (absx <= 2)
    )
    return f


def contributions(in_length, out_length, scale, kernel, k_width):
    if scale < 1:
        h = lambda x: scale * kernel(scale * x)
        kernel_width = 1.0 * k_width / scale
    else:
        h = kernel
        kernel_width = k_width
    x = np.arange(1, out_length + 1).astype(np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    P = int(ceil(kernel_width)) + 2
    ind = np.expand_dims(left, axis=1) + np.arange(P) - 1  # -1 because indexing from 0
    indices = ind.astype(np.int32)
    weights = h(np.expand_dims(u, axis=1) - indices - 1)  # -1 because indexing from 0
    weights = np.divide(weights, np.expand_dims(np.sum(weights, axis=1), axis=1))
    aux = np.concatenate(
        (np.arange(in_length), np.arange(in_length - 1, -1, step=-1))
    ).astype(np.int32)
    indices = aux[np.mod(indices, aux.size)]
    ind2store = np.nonzero(np.any(weights, axis=0))
    weights = weights[:, ind2store]
    indices = indices[:, ind2store]
    return weights, indices


def imresizemex(inimg, weights, indices, dim):
    in_shape = inimg.shape
    w_shape = weights.shape
    out_shape = list(in_shape)
    out_shape[dim] = w_shape[0]
    outimg = np.zeros(out_shape)
    if dim == 0:
        for i_img in range(in_shape[1]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[ind, i_img].astype(np.float64)
                outimg[i_w, i_img] = np.sum(
                    np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0
                )
    elif dim == 1:
        for i_img in range(in_shape[0]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[i_img, ind].astype(np.float64)
                outimg[i_img, i_w] = np.sum(
                    np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0
                )
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg


def imresizevec(inimg, weights, indices, dim):
    wshape = weights.shape
    if dim == 0:
        weights = weights.reshape((wshape[0], wshape[2], 1, 1))
        outimg = np.sum(
            weights * ((inimg[indices].squeeze(axis=1)).astype(np.float64)), axis=1
        )
    elif dim == 1:
        weights = weights.reshape((1, wshape[0], wshape[2], 1))
        outimg = np.sum(
            weights * ((inimg[:, indices].squeeze(axis=2)).astype(np.float64)), axis=2
        )
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg


def resizeAlongDim(A, dim, weights, indices, mode="vec"):
    if mode == "org":
        out = imresizemex(A, weights, indices, dim)
    else:
        out = imresizevec(A, weights, indices, dim)
    return out


def imresize(I, scalar_scale=None, method="bicubic", output_shape=None, mode="vec"):
    if method == "bicubic":
        kernel = cubic
    elif method == "bilinear":
        kernel = triangle
    else:
        raise ValueError("unidentified kernel method supplied")

    kernel_width = 4.0
    # Fill scale and output_size
    if scalar_scale is not None and output_shape is not None:
        raise ValueError("either scalar_scale OR output_shape should be defined")
    if scalar_scale is not None:
        scalar_scale = float(scalar_scale)
        scale = [scalar_scale, scalar_scale]
        output_size = deriveSizeFromScale(I.shape, scale)
    elif output_shape is not None:
        scale = deriveScaleFromSize(I.shape, output_shape)
        output_size = list(output_shape)
    else:
        raise ValueError("either scalar_scale OR output_shape should be defined")
    scale_np = np.array(scale)
    order = np.argsort(scale_np)
    weights = []
    indices = []
    for k in range(2):
        w, ind = contributions(
            I.shape[k], output_size[k], scale[k], kernel, kernel_width
        )
        weights.append(w)
        indices.append(ind)
    B = np.copy(I)
    flag2D = False
    if B.ndim == 2:
        B = np.expand_dims(B, axis=2)
        flag2D = True
    for k in range(2):
        dim = order[k]
        B = resizeAlongDim(B, dim, weights[dim], indices[dim], mode)
    if flag2D:
        B = np.squeeze(B, axis=2)
    return B


def convertDouble2Byte(I):
    B = np.clip(I, 0.0, 1.0)
    B = 255 * B
    return np.around(B).astype(np.uint8)


# ---- Supervised Metrics ----
def q2n(I_GT, I_F, Q_blocks_size=32, Q_shift=32):
    def onion_mult2D(img1, img2):
        channel = img1.shape[1]

        if channel > 1:
            L = channel // 2
            a = img1[:, :L, :, :]
            b = img1[:, L:, :, :]
            b = torch.cat((b[:, 0, :, :].unsqueeze(1), -b[:, 1:, :, :]), dim=1)

            c = img2[:, :L, :, :]
            d = img2[:, L:, :, :]
            d = torch.cat((d[:, 0, :, :].unsqueeze(1), -d[:, 1:, :, :]), dim=1)

            if channel == 2:
                ris = torch.cat((a * c - d * b, a * d + c * b), dim=1)
            else:
                ris1 = onion_mult2D(a, c)
                ris2 = onion_mult2D(
                    d, torch.cat((b[:, 0, :, :].unsqueeze(1), -b[:, 1:, :, :]), dim=1)
                )
                ris3 = onion_mult2D(
                    torch.cat((a[:, 0, :, :].unsqueeze(1), -a[:, 1:, :, :]), dim=1), d
                )
                ris4 = onion_mult2D(c, b)

                ris = torch.cat((ris1 - ris2, ris3 + ris4), dim=1)
        else:
            ris = img1 * img2
        return ris

    def onion_mult(m1, m2):
        channel = m1.shape[1]

        if channel > 1:
            L = channel // 2
            a = m1[:, :L]
            b = m1[:, L:]
            b = torch.cat((b[:, 0].unsqueeze(1), -b[:, 1:]), dim=1)

            c = m2[:, :L]
            d = m2[:, L:]
            d = torch.cat((d[:, 0].unsqueeze(1), -d[:, 1:]), dim=1)

            if channel == 2:
                ris = torch.cat((a * c - d * b, a * d + c * b), dim=1)
            else:
                ris1 = onion_mult(a, c)
                ris2 = onion_mult(
                    d, torch.cat((b[:, 0].unsqueeze(1), -b[:, 1:]), dim=1)
                )
                ris3 = onion_mult(
                    torch.cat((a[:, 0].unsqueeze(1), -a[:, 1:]), dim=1), d
                )
                ris4 = onion_mult(c, b)

                ris = torch.cat((ris1 - ris2, ris3 + ris4), dim=1)
        else:
            ris = m1 * m2
        return ris

    def onions_quality(img1, img2, size_):
        # Fix sign for channels after first one
        img2 = torch.cat((img2[:, :1, :, :], -img2[:, 1:, :, :]), dim=1)
        channel = img1.shape[1]

        # Block normalization
        for i in range(channel):
            a1, s, t = norm_blocco(img1[0, i, :, :])
            img1[0, i, :, :] = a1

            if s == 0:
                if i == 0:  # Note: Python uses 0-based indexing
                    img2[:, i, :, :] = img2[:, i, :, :] - s + 1
                else:
                    img2[:, i, :, :] = -(-img2[:, i, :, :] - s + 1)
            else:
                if i == 0:
                    img2[:, i, :, :] = ((img2[:, i, :, :] - s) / t) + 1
                else:
                    img2[:, i, :, :] = -(((-img2[:, i, :, :] - s) / t) + 1)

        # Initialize means
        m1 = torch.zeros(channel, device=img1.device)
        m2 = torch.zeros(channel, device=img1.device)

        mod_q1m = 0
        mod_q2m = 0
        mod_q1 = torch.zeros_like(img1[0, 0, :, :])
        mod_q2 = torch.zeros_like(img2[0, 0, :, :])

        # Calculate means and moduli
        for i in range(channel):
            m1[i] = torch.mean(img1[:, i, :, :])
            m2[i] = torch.mean(img2[:, i, :, :])
            mod_q1m = mod_q1m + (m1[i] ** 2)
            mod_q2m = mod_q2m + (m2[i] ** 2)
            mod_q1 = mod_q1 + (img1[0, i, :, :] ** 2)
            mod_q2 = mod_q2 + (img2[0, i, :, :] ** 2)

        mod_q1m = torch.sqrt(mod_q1m)
        mod_q2m = torch.sqrt(mod_q2m)
        mod_q1 = torch.sqrt(mod_q1)
        mod_q2 = torch.sqrt(mod_q2)

        # Calculate terms
        termine2 = mod_q1m * mod_q2m
        termine4 = (mod_q1m**2) + (mod_q2m**2)
        int1 = (size_**2 / (size_**2 - 1)) * torch.mean(mod_q1**2)
        int2 = (size_**2 / (size_**2 - 1)) * torch.mean(mod_q2**2)
        termine3 = (
            int1 + int2 - (size_**2 / (size_**2 - 1)) * ((mod_q1m**2) + (mod_q2m**2))
        )
        mean_bias = 2 * termine2 / termine4

        if termine3 == 0:
            q = torch.zeros(1, 1, channel)
            q[:, :, -1] = mean_bias
        else:
            cbm = 2 / termine3
            qu = onion_mult2D(img1, img2)
            qm = onion_mult(m1.unsqueeze(0), m2.unsqueeze(0))
            qv = torch.zeros(channel, device=img1.device)

            for i in range(channel):
                qv[i] = (size_**2 / (size_**2 - 1)) * torch.mean(qu[0, i, :, :])

            q = qv - (size_**2 / (size_**2 - 1)) * qm
            q = q * mean_bias * cbm

        return q

    I_GT = I_GT.squeeze()
    I_F = I_F.squeeze()
    c, h, w = I_GT.shape

    stepx = math.ceil(h / Q_shift)
    stepy = math.ceil(w / Q_shift)

    if stepy <= 0:
        stepy = 1
        stepx = 1

    est1 = (stepx - 1) * Q_shift + Q_blocks_size - h
    est2 = (stepy - 1) * Q_shift + Q_blocks_size - w

    if est1 != 0 or est2 != 0:
        refref = torch.Tensor()
        fusfus = torch.Tensor()

        for i in range(c):
            a1 = I_GT[0, :, :].squeeze()
            ia1 = torch.zeros(h + est1, w + est2)
            ia1[:h, :w] = a1
            ia1[:, w:] = ia1[:, w - 1 : w - est2 + 1 : -1]  # ???
            ia1[h:, :] = ia1[h - 1 : h - est1 + 1 : -1, :]
            if refref.numel() == 0:
                refref = ia1.unsqueeze(0)
            else:
                refref = torch.cat((refref, ia1.unsqueeze(0)), dim=0)

            if i < c - 1:
                I_GT = I_GT[1:, :, :]
        I_GT = refref

        for i in range(c):
            a2 = I_F[0, :, :].squeeze()
            ia2 = torch.zeros(h + est1, w + est2)
            ia2[:h, :w] = a2
            ia2[:, w:] = ia2[:, w - 1 : w - est2 + 1 : -1]  # ???
            ia2[h:, :] = ia2[h - 1 : h - est1 + 1 : -1, :]
            if refref.numel() == 0:
                fusfus = ia2.unsqueeze(0)
            else:
                fusfus = torch.cat((refref, ia2.unsqueeze(0)), dim=0)

            if i < c - 1:
                I_F = I_F[1:, :, :]
        I_F = fusfus

        del a1, a2, ia1, ia2, refref, fusfus

    c, h, w = I_GT.shape
    if ((math.ceil(math.log2(c))) - math.log2(c)) != 0:
        Ndif = (2 ** (math.ceil(math.log2(c)))) - c
        dif = torch.zeros(Ndif, h, w)
        I_GT = torch.cat((I_GT, dif), dim=0)
        I_F = torch.cat((I_F, dif), dim=0)

    c = I_GT.shape[0]
    valori = torch.zeros(c, stepx, stepy)

    for j in range(stepx):
        for i in range(stepy):
            valori[:, j, i] = onions_quality(
                img1=I_GT[
                    :,
                    (j * Q_shift) : (j * Q_shift) + Q_blocks_size,
                    (i * Q_shift) : (i * Q_shift) + Q_blocks_size,
                ].unsqueeze(0),
                img2=I_F[
                    :,
                    (j * Q_shift) : (j * Q_shift) + Q_blocks_size,
                    (i * Q_shift) : (i * Q_shift) + Q_blocks_size,
                ].unsqueeze(0),
                size_=Q_blocks_size,
            )

    Q2n_index_map = torch.sqrt(torch.sum(valori**2, dim=0))
    Q2n_index = torch.mean(Q2n_index_map)

    return Q2n_index


def SSIM(img1, img2):
    C1, C2 = (0.01**2, 0.03**2)
    _, c, h, w = img1.shape
    wsize = min(h, w, 11)
    window = create_window(wsize, 1.5 * wsize / 11, c).to(img1.device)

    img1 = F.pad(img1, [wsize // 2] * 4, mode="replicate")
    img2 = F.pad(img2, [wsize // 2] * 4, mode="replicate")

    mu1 = F.conv2d(img1, window, groups=c)
    mu2 = F.conv2d(img2, window, groups=c)

    sigma1 = F.conv2d(img1 * img1, window, groups=c) - mu1**2
    sigma2 = F.conv2d(img2 * img2, window, groups=c) - mu2**2
    sigma12 = F.conv2d(img1 * img2, window, groups=c) - mu1 * mu2

    return (
        (
            ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2))
            / ((mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2))
        )
        .mean(dim=(2, 3))
        .mean()
        .float()
    )


def SAM(gt, pred, n_digits=6):
    t = (torch.sum(gt * gt, 1) * torch.sum(pred * pred, 1)) ** 0.5
    num = torch.sum(torch.gt(t, 0))
    angle = torch.acos(torch.sum(gt * pred, 1) / t)
    sumangle = torch.where(torch.isnan(angle), torch.full_like(angle, 0), angle).sum()

    if num == 0:
        averangle = sumangle
    else:
        averangle = sumangle / num
    averangle = (averangle * 10**n_digits).round() / (10**n_digits)

    return float(averangle * 180 / torch.pi)


def ERGAS(gt, pred, ratio=4):
    channel = gt.shape[1]
    sum = torch.sum(
        torch.mean((gt - pred) ** 2, dim=(-1, -2)) / torch.mean(gt, dim=(-1, -2)) ** 2,
        dim=-1,  # Spatial -> Spectral
    )
    ergas = 100 / ratio * ((sum.mean() / channel) ** 0.5)

    return float(ergas)


def CC(gt, pred):
    h, w = gt.shape[-2:]
    C1 = torch.sum(gt * pred, dim=(-1, -2)) - h * w * torch.mean(
        gt, dim=(-1, -2)
    ) * torch.mean(pred, dim=(-1, -2))

    C2 = torch.sum(pred**2, dim=(-1, -2)) - h * w * torch.mean(pred, dim=(-1, -2)) ** 2
    C3 = torch.sum(gt**2, dim=(-1, -2)) - h * w * torch.mean(gt, dim=(-1, -2)) ** 2

    cc = C1 / ((C2 * C3) ** 0.5)
    return float(torch.mean(cc))


def PSNR(gt, pred):
    psnr = 10 * torch.log10(1.0 / torch.mean((pred - gt) ** 2, [-1, -2]))
    psnr = float(psnr.mean(-1).mean(-1))
    return psnr


def metrics_compute(gt, pred, ratio=4):
    with torch.no_grad():
        psnr = PSNR(gt, pred)
        sam = SAM(gt, pred)
        ergas = ERGAS(gt, pred, ratio)
        ssim = SSIM(gt, pred)
        cc = CC(gt, pred)
        q2n_index = q2n(gt * 2047.0, pred * 2047.0)

    return {
        "PSNR": psnr,
        "SAM": sam,
        "SSIM": ssim,
        "CC": cc,
        "ERGAS": ergas,
        "Q2N": q2n_index,
    }


# ---- Unsupervised Metrics ----
def uqi_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    PyTorch实现UQI质量指数计算
    输入形状：(N, S*S)的二维张量
    """
    x = x.to(torch.float64)  # 使用双精度保证数值稳定性
    y = y.to(torch.float64)

    mu_x = x.mean(dim=1, keepdim=True)  # (N,1)
    mu_y = y.mean(dim=1, keepdim=True)  # (N,1)

    cov = ((x - mu_x) * (y - mu_y)).sum(dim=1) / (x.shape[1] - 1)  # (N,)
    var_x = x.var(dim=1, unbiased=True)  # (N,)
    var_y = y.var(dim=1, unbiased=True)  # (N,)

    numerator = 4 * cov * mu_x.squeeze() * mu_y.squeeze()  # (N,)
    denominator = (var_x + var_y) * (mu_x.squeeze() ** 2 + mu_y.squeeze() ** 2)
    return numerator / denominator  # (N,)


def block_processing(tensor: torch.Tensor, S: int) -> torch.Tensor:
    """
    将张量分块为SxS的非重叠块
    输入形状：(b,c,h,w)
    输出形状：(b,c,num_blocks_h,num_blocks_w,S,S)
    """
    return tensor.unfold(2, S, S).unfold(3, S, S).contiguous()


def compute_q_map(
    feat_blocks: torch.Tensor, pan_blocks: torch.Tensor, S: int
) -> torch.Tensor:
    """
    计算Q index map
    feat_blocks/pan_blocks形状：(b,c,num_h,num_w,S,S)
    """
    b, c, num_h, num_w, _, _ = feat_blocks.shape
    feat_flat = (
        feat_blocks.view(b, c, num_h * num_w, S * S).permute(2, 0, 1, 3).flatten(0, 2)
    )  # (total_blocks, S*S)
    pan_flat = (
        pan_blocks.view(b, 1, num_h * num_w, S * S)
        .expand(-1, c, -1, -1)
        .permute(2, 0, 1, 3)
        .flatten(0, 2)
    )

    q_values = uqi_torch(feat_flat, pan_flat)  # (total_blocks,)
    return q_values.view(b, c, num_h, num_w).mean(dim=(-1, -2))  # (b,c)


def D_s_torch(
    fused: torch.Tensor,
    lms: torch.Tensor,
    pan: torch.Tensor,
    S: int = 32,
    q: int = 1,
) -> torch.Tensor:
    """
    PyTorch实现的D_s空间失真指数
    输入形状要求：
    fused: (b,c,h,w) 融合后的图像
    lms: (b,c,h,w) 上采样后的多光谱图像
    pan: (b,1,h,w) 全色图像
    """
    b, c, h, w = fused.shape
    assert h % S == 0 and w % S == 0, "图像尺寸必须能被块大小整除"

    pan_down = (
        torch.from_numpy(
            imresize(np.array(pan.squeeze(0).permute(1, 2, 0)), output_shape=(128, 128))
        )
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(torch.float64)
    )
    pan_filt = tap_intp(pan_down)  # (b,1,h,w)

    # Step 3: 分块处理
    feat_blocks = block_processing(fused, S)  # (b,c,num_h,num_w,S,S)
    pan_blocks = block_processing(pan, S)  # (b,1,num_h,num_w,S,S)
    q_high = compute_q_map(feat_blocks, pan_blocks, S)  # (b,c)

    ms_blocks = block_processing(lms, S)  # (b,c,num_h,num_w,S,S)
    pan_filt_blocks = block_processing(pan_filt, S)  # (b,1,num_h,num_w,S,S)
    q_low = compute_q_map(ms_blocks, pan_filt_blocks, S)  # (b,c)

    # Step 4: 计算D_s指数
    delta = (q_high - q_low).abs().pow(q)  # (b,c)
    return delta.mean(dim=1).pow(1 / q)  # (b,)


def HQNR(fused, lms, pan, block_size=32):
    with torch.no_grad():
        d_s = D_s_torch(fused, lms, pan)

        Q2n_index = q2n(lms.clone(), blur_conv(fused), block_size, block_size)
        d_lambda_K = 1 - Q2n_index

        hqnr = (1 - d_lambda_K) * (1 - d_s)
        return float(hqnr), float(d_lambda_K), float(d_s)


# ---- Main Function ----
if __name__ == "__main__":
    # HRMS = (
    #     torch.from_numpy(sio.loadmat("./Paper/Results/WV3_Full/Ours.mat")["HRMS"][0])
    #     .permute(2, 0, 1)
    #     .unsqueeze(0)
    # ) * 2047.0
    #
    # I_fused_BDSD = (
    #     torch.from_numpy(
    #         sio.loadmat("./Paper/Results/WV3_Full/I_fused_BDSD.mat")["I_fused_BDSD"][0]
    #     )
    #     .permute(2, 0, 1)
    #     .unsqueeze(0)
    # )

    I_fused_SR_D = (
        torch.from_numpy(
            sio.loadmat("./Paper/Results/WV3_Full/I_fused_SR_D.mat")["I_fused_SR_D"][0]
        )
        .permute(2, 0, 1)
        .unsqueeze(0)
    )

    dataset = h5py.File("./test_data/test_wv3_OrigScale.h5")
    LMS = torch.from_numpy(np.array(dataset["lms"][...]))[0]
    PAN = torch.from_numpy(np.array(dataset["pan"][...]))[0]

    print("\n--- Testing I_fused_SR_D ---")
    hqnr_srd, d_lambda_srd, d_s_srd = HQNR(I_fused_SR_D, LMS, PAN)
    print(f"HQNR: {hqnr_srd:.4f}")
    print(f"D_lambda_K: {d_lambda_srd:.4f}")
    print(f"D_s: {d_s_srd:.4f}")
