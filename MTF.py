import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.ndimage.filters as ft
import matplotlib.pyplot as plt

from config import bargs

class MTFGenrator_np:
    def __init__(self, ratio, kernel_size=41):
        """
        Initialize the Nyquist filter generator with a scaling ratio and kernel size.

        Parameters:
            kernel_size (int): Size of the square kernel (N x N).
        """
        self.kernel_size = kernel_size
        self.window = np.kaiser(kernel_size, 0.5)
        self.window = np.outer(self.window, self.window)

        sidelen = (kernel_size - 1) // 2
        y, x = np.ogrid[-sidelen : sidelen + 1, -sidelen : sidelen + 1]

        self.x = x[np.newaxis, :, :].astype(np.float64)  # Shape (1, N, N)
        self.y = y[np.newaxis, :, :].astype(np.float64)  # Shape (1, N, N)

        self.numerator = -((kernel_size - 1) ** 2) / (8 * np.double(ratio) ** 2)

    def gaussian_kernel(self, sigma):
        """
        Generate Gaussian filter kernels using vectorized operations.

        bargs:
            sigma (np.ndarray): Array of standard deviations (nbands,).

        Returns:
            np.ndarray: Gaussian kernels of shape (nbands, kernel_size, kernel_size).
        """

        sigma = sigma[:, np.newaxis, np.newaxis]  # Shape (nbands, 1, 1)
        h = np.exp(-(self.x**2 + self.y**2) / (2.0 * sigma**2))

        # Cut & Normalize the kernel
        h[h < np.finfo(h.dtype).eps * h.max(axis=(-1, -2), keepdims=True)] = 0
        h = h / h.sum(axis=(1, 2), keepdims=True)
        return h

    def fir_filter_wind(self, Hd):
        """
        Apply FIR filter windowing to the frequency response.

        Parameters:
            Hd (np.ndarray): Gaussian frequency response (nbands, kernel_size, kernel_size).

        Returns:
            np.ndarray: Spatial response after windowing, shaped (nbands, kernel_size, kernel_size).
        """
        # Rotate and FFT operations to convert to spatial domain
        hd = np.rot90(
            np.fft.fftshift(np.rot90(Hd, 2, axes=(1, 2)), axes=(1, 2)), 2, axes=(1, 2)
        )
        h = np.fft.fftshift(np.fft.ifft2(hd, axes=(1, 2)), axes=(1, 2))
        h = np.rot90(h, 2, axes=(1, 2))
        h = h * self.window

        h = np.clip(h, a_min=0, a_max=np.max(h, axis=(1, 2), keepdims=True))
        h = h / h.sum(axis=(1, 2), keepdims=True)

        return h

    @classmethod
    def generate(cls, Gnyq, ratio, kernel_size=41):
        """
        Generate the Nyquist filter kernel for given Nyquist gains.

        Parameters:
            Gnyq (np.ndarray, torch.Tensor, list): Nyquist gains for each band (nbands, ).

        Returns:
            torch.Tensor: Filter kernels shaped (nbands, 1, kernel_size, kernel_size).
        """
        # Convert Gnyq to numpy array if necessary
        instance = cls(ratio, kernel_size)

        if isinstance(Gnyq, torch.Tensor):
            Gnyq = Gnyq.cpu().numpy().astype(np.float64)
        if isinstance(Gnyq, list):
            Gnyq = np.array(Gnyq).astype(np.float64)

        H = instance.gaussian_kernel(sigma=np.sqrt(instance.numerator / (np.log(Gnyq))))

        # Convert to PyTorch tensor and adjust dimensions
        kernel = torch.from_numpy(
            np.real(instance.fir_filter_wind(H / np.max(H, axis=(1, 2), keepdims=True)))
        ).unsqueeze(
            1
        )  # Shape (nbands, 1, N, N)
        return kernel


def MTF_MS(ratio, N=41, gain_only=False):
    """
    input:
        N: kernel size
        gain_only: return Nyquist Gain only or not

    return:
        Gnyq(nbands,): gain_only = True
        MTF kernel(N,N): gain_only = False
    """
    sensor = bargs.sensor
    if sensor == "QB":
        GNyq = np.asarray([0.34, 0.32, 0.30, 0.22])  # Bands Order: B,G,R,NIR
    elif (sensor == "Ikonos") or (sensor == "IKONOS"):
        GNyq = np.asarray([0.26, 0.28, 0.29, 0.28])  # Bands Order: B,G,R,NIR
    elif (sensor == "GeoEye1") or (sensor == "WV4"):
        GNyq = np.asarray([0.23, 0.23, 0.23, 0.23])  # Bands Order: B, G, R, NIR
    elif sensor == "WV2":
        GNyq = 0.35 * np.ones((1, 7))
        GNyq = np.append(GNyq, 0.27)
    elif sensor == "WV3":
        # Corresponding Sigma: 10.03 - 10.75
        GNyq = [0.325, 0.355, 0.360, 0.350, 0.365, 0.360, 0.335, 0.315]
    else:
        GNyq = np.asarray([0.3, 0.3, 0.3, 0.3])

    if gain_only:
        return torch.Tensor(GNyq)
    return MTFGenrator_np.generate(GNyq, ratio, N)
    # Th = MTFGenrator_torch(ratio, N)
    # return Th(GNyq)


def MTF_PAN(ratio, N=41):
    """
    input:
        N: kernel size

    return:
        MTF kernel(N,N)
    """
    sensor = bargs.sensor
    if sensor == "QB":
        GNyq = np.array([0.15])
    elif (sensor == "Ikonos") or (sensor == "IKONOS"):
        GNyq = np.array([0.17])
    elif (sensor == "GeoEye1") or (sensor == "WV4"):
        GNyq = np.array([0.16])
    elif sensor == "WV2":
        GNyq = np.array([0.11])
    elif sensor == "WV3":
        GNyq = np.array([0.14])
    else:
        GNyq = np.array([0.15])
    return MTFGenrator_np.generate(GNyq, ratio, N)


class smoothe_interpolation(nn.Module):
    """
    Attributes:
        ratio (int): Upsampling ratio (must be a power of 2)
        channel (int): Number of channels in the input image
        cdfconv (nn.Conv2d): Convolutional layer with fixed CDF 2/3 wavelet coefficients
    """

    def __init__(self, channel=bargs.channel):
        super().__init__()

        CDF23 = (
            torch.tensor(
                [
                    0.5,
                    0.305334091185,
                    0,
                    -0.072698593239,
                    0,
                    0.021809577942,
                    0,
                    -0.005192756653,
                    0,
                    0.000807762146,
                    0,
                    -0.000060081482,
                ]
            )
            * 2
        )

        BaseCoeff = (
            torch.cat(([torch.flip(CDF23[1:], dims=[0]), CDF23]), dim=0)
            .view(1, 1, -1, 1)
            .repeat(channel, 1, 1, 1)
        ).to(
            torch.float64
        )  # (c,1,23,1)

        self.cdfconv = nn.Conv2d(
            in_channels=channel,
            out_channels=channel,
            padding=(11, 0),
            kernel_size=(23, 1),
            groups=channel,
            bias=False,
            padding_mode="circular",
        )

        self.cdfconv.weight.data = BaseCoeff
        self.cdfconv.weight.requires_grad = False

    def forward(self, img):
        b, c, h, w = img.shape
        for z in range(int(bargs.ratio / 2)):
            I1LRU = torch.zeros(b, c, (2 ** (z + 1)) * h, (2 ** (z + 1)) * w).to(
                img.device, img.dtype
            )

            if z == 0:
                I1LRU[:, :, 1::2, 1::2] = img
            else:
                I1LRU[:, :, ::2, ::2] = img

            img = self.cdfconv(self.cdfconv(I1LRU.transpose(2, 3)).transpose(2, 3))

        return img


class MtfConv(nn.Module):
    """
    args:
        ratio: int
        sensor: str
        size_: int, kernel size
        kernel: str (ms | pan)
        img: (b,c,h,w)

    return:
        Downsampled and Blured image
    """

    def __init__(self, ratio=4, size_=41, kernel="ms"):
        super().__init__()

        if kernel == "ms":
            self.blur_conv = nn.Conv2d(
                in_channels=bargs.channel,
                out_channels=bargs.channel,
                kernel_size=size_,
                padding=int((size_ - 1) / 2),
                groups=bargs.channel,
                bias=False,
                padding_mode="replicate",
            )
            self.blur_conv.weight.data = MTF_MS(ratio, size_)

        elif kernel == "pan":
            self.blur_conv = nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=size_,
                padding=int((size_ - 1) / 2),
                groups=1,
                bias=False,
                padding_mode="replicate",
            )
            self.blur_conv.weight.data = MTF_PAN(ratio, size_)
        else:
            raise ValueError("Only support ms and pan kernels !!!")

        self.blur_conv.weight.requires_grad = False

    def forward(self, img):
        return self.blur_conv(img)


def wald_protocol(img, ratio, N=41, kernel="ms"):
    channels = img.shape[1]
    psz = int((N - 1) / 2)

    dev, _dtype = img.device, img.dtype
    if channels == 1:
        mtf_kernel = MTF_PAN(ratio, N).to(dev, _dtype)
    else:
        if kernel == "ms":
            mtf_kernel = MTF_MS(ratio, N).to(dev, _dtype)
        elif kernel == "pan":
            mtf_kernel = (
                MTF_PAN(ratio, N).repeat(1, bargs.channel, 1, 1).to(dev, _dtype)
            )
        else:
            raise ValueError("Wrong Kernel Type, Has to be 'ms' or 'pan'")

    # Blur -> Downsample
    img_blur = F.conv2d(
        F.pad(img, (psz, psz, psz, psz), mode="replicate"),
        weight=mtf_kernel,
        bias=None,
        stride=1,
        padding=0,
        groups=channels,
    )

    return img_blur  # [:, :, 2:-1:4, 2:-1:4]

