import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.ndimage.filters as ft
import matplotlib.pyplot as plt

from config import bargs


class MBFE(nn.Module):
    def __init__(self, lms, edge_sigma=1.0):
        super().__init__()
        gv = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=bargs._dtype,
            device=bargs.device,
        )

        gh = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=bargs._dtype,
            device=bargs.device,
        )

        gvf = torch.fft.fft2(gv, s=(bargs.test_sidelen, bargs.test_sidelen))
        ghf = torch.fft.fft2(gh, s=(bargs.test_sidelen, bargs.test_sidelen))

        gvf2 = torch.conj(gvf) * gvf
        ghf2 = torch.conj(ghf) * ghf

        self.edge_sigma = edge_sigma
        self.kernel_size = int(6 * edge_sigma + 1)

        self.grad = gvf2 + ghf2

        # Edge Blur
        self.edge_kernel = self._get_edge_kernel().to(bargs.device)
        self.pad = self.kernel_size // 2
        self.alpha = torch.ones(
            1, bargs.channel, bargs.test_sidelen, bargs.test_sidelen
        ).to(bargs.device, bargs._dtype)
        self.border_width = bargs.test_sidelen // 6

        self.lms_fft = torch.fft.fft2(self.edge_blur(lms))
        self.PSF_l = torch.zeros(bargs.channel, 1, 41, 41)

        self.to(bargs.device, bargs._dtype)

    def _get_edge_kernel(self):
        coords = (
            torch.arange(self.kernel_size, dtype=torch.float32)
            - (self.kernel_size - 1.0) / 2.0
        )
        gauss_1d = torch.exp(-(coords**2) / (2.0 * self.edge_sigma**2))
        gauss_1d /= gauss_1d.sum()

        gauss_2d = torch.outer(gauss_1d, gauss_1d)
        gauss_2d = gauss_2d.unsqueeze(0).unsqueeze(0)
        return gauss_2d.repeat(bargs.channel, 1, 1, 1)

    def edge_blur(self, img):
        blurred_img = F.conv2d(
            img, weight=self.edge_kernel, padding=self.pad, groups=bargs.channel
        )
        for i in range(self.border_width):
            weight = i / self.border_width
            self.alpha[:, :, i, :] = weight
            self.alpha[:, :, bargs.test_sidelen - 1 - i, :] = weight
            self.alpha[:, :, :, i] = weight
            self.alpha[:, :, :, bargs.test_sidelen - 1 - i] = weight

        return self.alpha * img + (1 - self.alpha) * blurred_img

    def forward(self, fuse, _lambda=5, _mu=5, th=1e-4):
        fuse_fft = torch.fft.fft2(self.edge_blur(fuse))
        conj_fuse_fft = torch.conj(fuse_fft)

        denominator = torch.abs(fuse_fft) ** 2 + _lambda + _mu * self.grad
        numerator = conj_fuse_fft * self.lms_fft

        PSF = torch.fft.fftshift(
            torch.real(torch.fft.ifft2(numerator / denominator)), dim=(-1, -2)
        )
        PSF = torch.where(PSF < th, torch.zeros_like(PSF), PSF)

        for i in range(bargs.channel):
            max_psf_index = torch.unravel_index(
                torch.argmax(PSF[:, i]), PSF[:, i].shape
            )
            self.PSF_l[i, :] = PSF[
                :,
                i,
                max_psf_index[-2] - 20 : max_psf_index[-1] + 21,
                max_psf_index[-2] - 20 : max_psf_index[-1] + 21,
            ]
            self.PSF_l[i, :] /= torch.sum(self.PSF_l[i])
        print(f"Max value of PSF is {PSF.max()}, min is {PSF.min()}")

        return self.PSF_l.to(bargs.device, bargs._dtype)


class MTFGenrator_torch(nn.Module):
    def __init__(self, GNyq, ratio, kernel_size=41, hidden_dim=128):
        super().__init__()
        """
        Initialize the MTF generator with specified parameters.

        bargs:
            kernel_size (int): Spatial size of the filter kernel
        """

        # Params
        self.kernel_size = kernel_size
        self.sidelen = (kernel_size - 1.0) // 2
        self.numerator = -((kernel_size - 1) ** 2) / (8 * ratio**2)
        window = np.kaiser(kernel_size, 0.5)
        self.window = torch.Tensor(np.outer(window, window)).to(
            bargs.device, bargs._dtype
        )

        self.register_buffer("GNyq", GNyq)

        # Coords
        y, x = torch.meshgrid(
            torch.arange(
                -self.sidelen, self.sidelen + 1, device=bargs.device, dtype=bargs._dtype
            ),
            torch.arange(
                -self.sidelen, self.sidelen + 1, device=bargs.device, dtype=bargs._dtype
            ),
            indexing="ij",
        )

        self.x = x.unsqueeze(0)
        self.y = y.unsqueeze(0)

        self.mtfnet = nn.Sequential(
            nn.Linear(bargs.channel, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, bargs.channel * 2),
            nn.Sigmoid(),
        )
        self.to(bargs.device, bargs._dtype)

    def gaussian_kernel(self, sigmas):
        """
        Generate Gaussian kernels using PyTorch tensor operations.

        eps * max: A scale of absolute min to adapt it to data

        args:
            sigma (torch.Tensor): Standard deviations for each band (nbands,1,1)

        Returns:
            torch.Tensor: Sum Normed Gaussian kernels (nbands, kernel_size, kernel_size)
        """
        # h = torch.exp(-(self.x**2 + self.y**2) / (2.0 * sigma**2)).to(
        #     bargs.device, bargs._dtype
        # )

        h = torch.exp(
            -self.x**2 / (2.0 * sigmas[: bargs.channel] ** 2)
            - self.y**2 / (2.0 * sigmas[bargs.channel :] ** 2)
        ).to(bargs.device, bargs._dtype)

        # WARN: Max for each channel
        h = torch.where(
            h >= torch.finfo(h.dtype).eps * h.detach().amax(dim=(1, 2), keepdim=True),
            h,
            torch.zeros_like(h),
        )
        return h / h.sum(dim=(1, 2), keepdim=True)

    def fir_filter_wind(self, Hd):
        """
        Apply FIR windowing in the frequency domain.

        args:
            Hd (torch.Tensor): Frequency response (nbands, kernel_size, kernel_size)

        Returns:
            torch.Tensor: Spatial domain filter kernels
        """
        hd = torch.rot90(
            torch.fft.fftshift(torch.rot90(Hd, 2, dims=(1, 2)), dim=(1, 2)),
            2,
            dims=(1, 2),
        )
        h = torch.fft.fftshift(torch.fft.ifft2(hd, dim=(1, 2)), dim=(1, 2))
        h = torch.rot90(h, 2, dims=(1, 2))

        # Apply window and normalize
        h = h * self.window
        hreal = h.real
        h.real[hreal < 0] = 0
        h.imag[hreal < 0] = 0

        h = h / (h.sum(dim=(1, 2), keepdim=True))

        return torch.real(h)

    def forward(self):
        """
        Generate MTF filter kernels for given Nyquist gains.

        bargs:
            Gnyq (torch.Tensor): Nyquist gains for each band (nbands,)

        Returns:
            torch.Tensor: Filter kernels shaped (nbands, 1, kernel_size, kernel_size)
        """
        # Generate and process filter
        H = self.gaussian_kernel(
            torch.sqrt(
                self.numerator / torch.log(self.mtfnet(self.GNyq).view(-1, 1, 1))
            )
        )
        h = self.fir_filter_wind(H / H.amax(dim=(-1, -2), keepdim=True))

        return h.unsqueeze(1)


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


if __name__ == "__main__":
    inp = torch.randn(1, 1, 512, 512)
    conv = smoothe_interpolation(1)

    pan_filt = conv(
        F.interpolate(inp, scale_factor=1 / 4, mode="bicubic", align_corners=False)
    ).squeeze(
        1
    )  # (N,H,W)
    print(pan_filt.shape)
