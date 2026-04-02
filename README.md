<h1 align="center">KZ-PAN</h1>

<h3 align="center">Three-Stage Learning with Kernel Estimation and Hierarchical Loss Functions for Unsupervised Zero-shot Pansharpening</h3>

<p align="center">
  <a href="#overview">Overview</a> &bull;
  <a href="#architecture">Architecture</a> &bull;
  <a href="#getting-started">Getting Started</a> &bull;
  <a href="#results">Results</a> &bull;
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/pytorch-2.7+-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
</p>

---

## Overview

**KZ-PAN** (KZ-PanNet) is a novel unsupervised zero-shot pansharpening framework that fuses **Low-Resolution Multispectral (LRMS)** images with **High-Resolution Panchromatic (PAN)** images to produce **High-Resolution Multispectral (HRMS)** images — using only a single image pair, without any training data.

Our key contributions:

- **Nyquist Gain Learning** — a data-driven kernel estimation technique that adaptively learns Nyquist gains for each spectral band via a lightweight **KENet**, replacing fixed or manufacturer-provided MTF (Modulation Transfer Function) specifications.
- **Hierarchical Loss Functions** — **QSE (Quadratic Spatial Error)** and **LSAE (Local Spectral Angle Error)** losses that operate at the patch level, preserving fine-grained spatial details and spectral fidelity far better than conventional MSE/SAM losses.
- **Three-Stage Training Strategy** — a progressive optimization pipeline consisting of pseudo-pretraining, adaptive refinement (kernel estimation), and final enhancement stages.

<p align="center">
  <img src="Paper/Imgs/Overall.pdf" width="95%" alt="KZ-PAN Architecture Overview">
</p>

## Architecture

KZ-PAN consists of four network components:

| Component | Role |
|---|---|
| **GNet** | Generates initial fused HRMS via histogram matching coefficient estimation |
| **UNext** | Main fusion backbone (UNet + ConvNeXt/ConvNeXtV2 encoder) for high-fidelity reconstruction |
| **OmniBlend Net** | Models the multispectral-to-panchromatic relationship with patch-wise processing |
| **KENet** | Estimates per-band blur kernels by learning Nyquist gains through a lightweight MLP |

### Training Stages

```
Stage 1: Prior Learning
  ├── Pre-train OmniBlend Net (MS → PAN mapping)
  └── Pre-train GNet (initial coefficient g estimation)

Stage 2: Adaptive Refinement
  ├── Train UNext with QSE + LSAE losses
  └── Train KENet for Nyquist gain estimation → Refined kernel

Stage 3: Kernel Estimation (Final Enhancement)
  └── Fine-tune UNext + OmniBlend Net using refined kernel
```

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.7+
- CUDA-compatible GPU (recommended) or Apple Silicon (MPS supported)

### Installation

```bash
git clone https://github.com/DowneyFlyfan/KEPansharpening.git
cd KEPansharpening
pip install torch numpy h5py scipy matplotlib
```

### Dataset Preparation

Place `.h5` test data files in the `test_data/` directory. Supported datasets:

| Dataset | Reduced Resolution | Full Resolution |
|---|---|---|
| WorldView-3 (WV3) | `test_wv3_Reduced.h5` | `test_wv3_OrigScale.h5` |
| WorldView-2 (WV2) | `test_wv2_Reduced.h5` | `test_wv2_OrigScale.h5` |
| QuickBird (QB) | `test_qb_Reduced.h5` | `test_qb_OrigScale.h5` |
| GaoFen-2 (GF2) | `test_gf2_Reduced.h5` | `test_gf2_OrigScale.h5` |

### Training & Testing

```bash
# Train on WV3 reduced-resolution dataset (default)
python main.py

# Test with a pretrained model
python test.py
```

Configure dataset and hyperparameters in `config.py`:

```python
data: str = "wv3_reduced"  # Options: wv3_reduced, qb_reduced, gf2_reduced, etc.
```

## Results

### Reduced-Resolution WV3 Dataset

| Method | PSNR | SAM | SSIM | SCC | ERGAS |
|---|---|---|---|---|---|
| BDSD-PC | 32.210 | 6.090 | 0.757 | 0.911 | 5.067 |
| SR-D | 33.095 | 5.078 | 0.762 | 0.921 | 4.502 |
| Ps-DIP | 34.368 | 4.475 | 0.815 | 0.946 | 3.815 |
| **KZ-PanNet** | **35.479** | **4.351** | **0.826** | **0.964** | **3.286** |

### Reduced-Resolution QB Dataset

| Method | PSNR | SAM | SSIM | SCC | ERGAS |
|---|---|---|---|---|---|
| BDSD-PC | 32.152 | 8.559 | 0.685 | 0.905 | 7.739 |
| SR-D | 32.160 | 7.648 | 0.666 | 0.899 | 7.723 |
| Ps-DIP | 33.902 | 5.801 | 0.741 | 0.939 | 6.325 |
| **KZ-PanNet** | **35.171** | **6.149** | **0.799** | **0.959** | **5.337** |

### Visual Comparisons

<p align="center">
  <img src="Paper/Imgs/WV3_Reduced.png" width="95%" alt="WV3 Reduced Resolution Results">
  <br><em>Qualitative comparison on WV3 reduced-resolution dataset</em>
</p>

<p align="center">
  <img src="Paper/Imgs/QB_Reduced.png" width="95%" alt="QB Reduced Resolution Results">
  <br><em>Qualitative comparison on QB reduced-resolution dataset</em>
</p>

## Project Structure

```
KZPansharpening/
├── main.py           # Training pipeline (three-stage)
├── model.py          # KZ-PAN model definition
├── config.py         # Hyperparameters and dataset configuration
├── data.py           # Data loading and preprocessing
├── loss.py           # QSE and LSAE loss functions
├── evaluate.py       # Evaluation metrics (PSNR, SAM, SSIM, SCC, ERGAS, Q2n)
├── MTF.py            # Modulation Transfer Function utilities
├── test.py           # Testing script
├── Visual_Data.py    # Visualization utilities
├── models/           # Network architectures (UNet, GNet, OmniBlend, KENet, etc.)
├── misc/             # Miscellaneous utilities
├── test_data/        # Test datasets (.h5)
└── Paper/            # Paper source and figures
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{kzpannet2025,
  title     = {KZ-PAN: Three-Stage Learning with Kernel Estimation and 
               Hierarchical Loss Functions for Unsupervised Zero-shot Pansharpening},
  year      = {2025},
  note      = {Submitted to IEEE}
}
```

## License

This project is licensed under the MIT License.
