import os
import warnings
import random
import torch
import scipy.io as sio
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim
import torch.nn.functional as F
import math

warnings.filterwarnings("ignore")
os.environ.update({"CUDA_VISIBLE_DEVICES": "0"})

from model import *
from data import get_dataloader
from loss import *
from MTF import wald_protocol
from evaluate import metrics_compute, HQNR, SAM
from config import targs
from models.SR_D import Compressive_Sensing


class PansharpeningTrainer:
    def __init__(self):
        self.gt_pred_total = []
        self.ms_sam_psz = 2
        self.ms_patch_size = 4
        self.pan_patch_size = 4

        # Initialize loss functions
        self.pw_mse = multi_batch_mse()
        self.pwsam = multi_batch_sam()
        self.sam = SAMLoss()

        # Initialize results dictionary
        self.results = {
            key: [] for key in ["SAM", "ERGAS", "PSNR", "CC", "SSIM", "Q2N", "time"]
        }

        self.unsupervised_results = {key: [] for key in ["HQNR", "D_lambda_K", "D_s"]}

        # Set random seed for reproducibility
        SEED = 2025
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # torch.use_deterministic_algorithms(True)

    def psnr_calculator(self, pred, gt, eps=1e-6):
        psnr = torch.mean(
            10
            * torch.log10(
                1.0
                / torch.mean(
                    torch.clamp(
                        (pred.to(torch.float32) - gt.to(torch.float32)) ** 2, min=eps
                    ),
                    [-1, -2],
                )
            )
        )
        return float(psnr)

    def loss_shaped(self, img: torch.Tensor, size: int):
        """
        img(b,c,H,W) -> (bhw,c,size,size)
        """
        return (
            img.unfold(2, size, size)
            .unfold(3, size, size)
            .contiguous()
            .view(-1, img.shape[1], size, size)
        )

    def MRAm(self, pan, ms, lms):
        pan_match = (pan - torch.mean(pan, dim=(-1, -2), keepdim=True)) / torch.std(
            pan, dim=(-1, -2), keepdim=True
        ) * torch.std(lms, dim=(-1, -2), keepdim=True) + torch.mean(
            lms, dim=(-1, -2), keepdim=True
        )
        pan_hp = pan_match - wald_protocol(pan_match, 1)

        X = wald_protocol(pan_hp, 4)[:, :, 2:-1:4, 2:-1:4]
        y = ms - wald_protocol(ms, 1, N=11)

        g_down = y * X / torch.norm(X, p=2, dim=[-1, -2], keepdim=True)
        g = F.interpolate(g_down, scale_factor=4, mode="nearest")

        return g, pan_hp

    def lr_lambda(self, epoch, config):
        if config["initial_lr"] == 0:
            return 0

        eta_ratio = config["eta_min"] / config["initial_lr"]

        def cosine_progress(progress):
            return eta_ratio + 0.5 * (1 - eta_ratio) * (
                1 + math.cos(math.pi * progress)
            )

        if config["type"] == "cosine":
            progress = epoch / config["T_max"]
            return cosine_progress(progress)

        elif config["type"] == "flat_down":
            switch_epoch = int(targs.cosine_point * config["T_max"])
            if epoch < switch_epoch:
                return 1.0
            progress = (epoch - switch_epoch) / (config["T_max"] - switch_epoch)
            return cosine_progress(progress)

        elif config["type"] == "up_flat_down":
            flat_epoch = int(targs.flat_point * config["T_max"])
            cosine_epoch = int(targs.cosine_point * config["T_max"])

            if epoch <= flat_epoch:
                return 4.0
            elif epoch >= cosine_epoch:
                progress = (epoch - cosine_epoch) / (config["T_max"] - cosine_epoch)
                return cosine_progress(progress)
            else:
                return 1.0

        else:
            raise ValueError("Unsupported config type: {}".format(config["type"]))

    def optimizer_reset(self, model, left_epochs: int):
        for attr in ("optimizer", "lr_schedulers"):
            if hasattr(self, attr):
                delattr(self, attr)
        torch.cuda.empty_cache()

        param_groups = []
        for module_name, lr in targs.lr.items():
            if hasattr(model, module_name):
                params = getattr(model, module_name).parameters()
                param_groups.append(
                    {
                        "params": params,
                        "lr": lr,
                        "module_name": module_name,
                        "scheduler_config": {
                            "type": targs.scheduler_type[module_name],
                            "T_max": left_epochs,
                            "eta_min": targs.min_lr[module_name],
                            "initial_lr": lr,
                        },
                    }
                )

        self.optimizer = optim.Adam(param_groups, betas=(0.9, 0.999))

        # Initialize schedulers
        self.lr_schedulers = []
        for group in param_groups:
            scheduler = LambdaLR(
                self.optimizer,
                lr_lambda=lambda epoch, c=group["scheduler_config"]: self.lr_lambda(
                    epoch, c
                ),
                last_epoch=-1,
            )
            self.lr_schedulers.append(scheduler)
        self.scaler = torch.GradScaler(device=targs.device, enabled=targs.mixed)

    def epoch_optimize(self, loss, steps, schedule=True):
        with torch.autocast(targs.device, dtype=torch.bfloat16, enabled=targs.mixed):
            self.scaler.scale(loss).backward()

            if steps % targs.update_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            if schedule:
                for scheduler in self.lr_schedulers:
                    scheduler.step()

    def Oneshot_Preparation(self, index: int = 0):
        self.loop = (
            tqdm(
                range(1, targs.epoch["total"] + 1),
                desc=f"Training {index}th image",
                total=targs.epoch["total"],
            )
            if index
            else ...
        )

        self.dataloader = get_dataloader(one_shot=True) if type else ...

    def metrics_check(
        self,
        gt=None,
        gt_pred=None,
        ms=None,
        ms_pred=None,
        pan=None,
        pan_pred=None,
        unsupervised=False,
    ):
        has_gt = isinstance(gt, torch.Tensor)
        has_ms = isinstance(ms, torch.Tensor)
        has_pan = isinstance(pan, torch.Tensor)

        if not unsupervised:
            if has_gt and has_ms and has_pan:  # MS + PAN + GT
                metrics = metrics_compute(gt, gt_pred)
                metrics["pan_psnr"] = self.psnr_calculator(pan, pan_pred)
                metrics["ms_psnr"] = self.psnr_calculator(ms, ms_pred)

                self.loop.set_postfix(
                    pan_psnr=f"{metrics['pan_psnr']:.2f}",
                    ms_psnr=f"{metrics['ms_psnr']:.2f}",
                    psnr=f"{metrics['PSNR']:.2f}",
                    sam=f"{metrics['SAM']:.2f}",
                    ergas=f"{metrics['ERGAS']:.2f}",
                )

            elif has_ms and has_pan and not has_gt:  # MS + PAN
                self.loop.set_postfix(
                    pan_psnr=f"{self.psnr_calculator(pan, pan_pred):.2f}",
                    ms_psnr=f"{self.psnr_calculator(ms, ms_pred):.2f}",
                )

            elif has_pan and not has_ms and not has_gt:  # PAN
                self.loop.set_postfix(
                    pan_psnr=f"{self.psnr_calculator(pan, pan_pred):.2f}",
                )

            elif has_gt and not has_ms and not has_pan:  # GT
                self.loop.set_postfix(
                    psnr=f"{self.psnr_calculator(gt, gt_pred):.2f}",
                )

            else:
                raise ValueError(
                    "Invalid combination of inputs: at least one of GT, MS, or PAN must be provided."
                )
        else:
            assert (
                has_ms and has_pan and isinstance(gt_pred, torch.Tensor)
            ), "lms and pan are needed to compute HQNR!!"
            hqnr, d_lambda_K, d_s = HQNR(
                (gt_pred * 2047.0).to("cpu", torch.float32), ms, pan
            )
            pan_psnr = self.psnr_calculator(
                pan_pred.to("cpu", torch.float32), pan / 2047.0
            )
            self.loop.set_postfix(
                pan_psnr=f"{pan_psnr:.3f}",
                HQNR=f"{hqnr:.3f}",
                D_lambda_K=f"{d_lambda_K:.3f}",
                D_s=f"{d_s:.3f}",
            )

    def metrics_save(self, gt_pred, gt=None, lms_int=None, pan_int=None, add=True):
        supervised = isinstance(gt, torch.Tensor)

        if add:
            with torch.no_grad():
                self.gt_pred_total.append(gt_pred)

                if supervised:
                    result_dict = {
                        **metrics_compute(gt, gt_pred),
                        "time": self.loop.format_dict.get("elapsed", 0),
                    }
                    for k, v in result_dict.items():
                        if isinstance(v, torch.Tensor):
                            v = v.cpu().numpy()
                        self.results.setdefault(k, []).append(v)
                else:
                    hqnr, d_lambda_K, d_s = HQNR(
                        (gt_pred * 2047.0).to("cpu", torch.float64), lms_int, pan_int
                    )
                    self.unsupervised_results["HQNR"].append(hqnr)
                    self.unsupervised_results["D_lambda_K"].append(d_lambda_K)
                    self.unsupervised_results["D_s"].append(d_s)
        else:
            os.makedirs(targs.result_path, exist_ok=True)

            metrics = (
                ["SAM", "ERGAS", "PSNR", "CC", "SSIM", "Q2N"]
                if supervised
                else ["HQNR", "D_lambda_K", "D_s"]
            )

            if not supervised:
                for metric in metrics:
                    mean, std = np.mean(self.unsupervised_results[metric]), np.std(
                        self.unsupervised_results[metric]
                    )
                    self.unsupervised_results[metric].append(f"{mean:.3f} ± {std:.3f}")
                df_metrics = pd.DataFrame(self.unsupervised_results)

            else:
                for metric in metrics:
                    mean, std = np.mean(self.results[metric]), np.std(
                        self.results[metric]
                    )
                    self.results[metric].append(f"{mean:.3f} ± {std:.3f}")
                self.results["time"].append(np.mean(self.results["time"]))
                print(self.results)
                df_metrics = pd.DataFrame(self.results)

            print("\n==== Here is the testing results ====\n")
            print(df_metrics.to_string(index=False))
            df_metrics.to_csv(
                os.path.join(targs.result_path, "KEPansharpening.csv"), index=False
            )

            sio.savemat(
                os.path.join(targs.result_path, "HRMS_Prediction.mat"),
                {
                    "HRMS": torch.cat(self.gt_pred_total, dim=0)
                    .permute(0, 2, 3, 1)
                    .cpu()
                    .detach()
                    .numpy()
                },
            )

            print("\n==== Results successfully saved! ====")


@torch.compile
class MainTrainer(PansharpeningTrainer):
    def __init__(self):
        super().__init__()

    def SS_Train(self):
        hrms2pan = HRMS2PAN().to(targs.device)
        self.Oneshot_Preparation()

        for index, batch in enumerate(self.dataloader, 1):
            ms = batch["ms"].to(targs.device, dtype=targs._dtype)
            lms = batch["lms"].to(targs.device, dtype=targs._dtype)
            pan = batch["pan"].to(targs.device, dtype=targs._dtype)
            gt = batch["gt"].to(targs.device, dtype=targs._dtype)
            pan_down = wald_protocol(pan, 4)[:, :, 2:-1:4, 2:-1:4]

            ms_patches = self.loss_shaped(ms, self.ms_patch_size)
            ms_sam_patches = self.loss_shaped(ms, self.ms_sam_psz)
            pan_patches = self.loss_shaped(pan, self.pan_patch_size)
            pan_down_patches = self.loss_shaped(pan_down, self.pan_patch_size)

            self.Oneshot_Preparation(index=index)
            self.optimizer_reset(hrms2pan, targs.epoch["stageI"])

            g, pan_hp = self.MRAm(pan, ms, lms)
            executed = False

            for epoch in self.loop:
                if epoch <= targs.epoch["csum"][0]:
                    pan_pred = hrms2pan(ms)
                    loss = self.pw_mse(
                        pan_down_patches,
                        self.loss_shaped(pan_pred, self.pan_patch_size),
                    )
                elif targs.epoch["csum"][0] < epoch <= targs.epoch["csum"][1]:
                    if not executed:
                        gnet = MRANet(hrms2pan).to(targs.device)
                        self.optimizer_reset(gnet, targs.epoch["stageII"])
                        executed = True

                    gt_pred, ms_pred, pan_pred = gnet(lms, g, pan_hp)
                    ms_pred_patches = self.loss_shaped(ms_pred, self.ms_patch_size)
                    loss = (
                        self.pwsam(
                            self.loss_shaped(ms_pred, self.ms_sam_psz), ms_sam_patches
                        )
                        * targs.lambda_mssam
                        + self.pw_mse(ms_patches, ms_pred_patches) * targs.lambda_msmse
                        + self.pw_mse(
                            pan_patches,
                            self.loss_shaped(pan_pred, self.pan_patch_size),
                        )
                        * targs.lambda_panmse
                    )
                else:
                    if executed:
                        model = MainNet(hrms2pan).to(targs.device)
                        self.optimizer_reset(model, targs.epoch["stageIII"])
                        hrms = gt_pred.detach().clone()
                        executed = False

                    gt_pred, ms_pred, pan_pred = model(hrms)
                    ms_pred_patches = self.loss_shaped(ms_pred, self.ms_patch_size)
                    loss = (
                        self.pwsam(
                            self.loss_shaped(ms_pred, self.ms_sam_psz), ms_sam_patches
                        )
                        * targs.lambda_mssam
                        + self.pw_mse(ms_patches, ms_pred_patches) * targs.lambda_msmse
                        + self.pw_mse(
                            pan_patches,
                            self.loss_shaped(pan_pred, self.pan_patch_size),
                        )
                        * targs.lambda_panmse
                    )

                self.epoch_optimize(loss, targs.update_steps)
                if epoch % targs.epoch["check"] == 0:
                    with torch.no_grad():
                        if epoch <= targs.epoch["csum"][0]:
                            self.metrics_check(
                                pan=pan_down.to(torch.float32),
                                pan_pred=pan_pred.to(torch.float32),
                            )
                        else:
                            self.metrics_check(
                                gt=gt.to(torch.float32),
                                gt_pred=gt_pred.to(torch.float32),
                                ms=ms.to(torch.float32),
                                ms_pred=ms_pred.to(torch.float32),
                                pan=pan.to(torch.float32),
                                pan_pred=pan_pred.to(torch.float32),
                            )

            self.metrics_save(
                gt_pred=gt_pred.to(torch.float32), gt=gt.to(torch.float32), add=True
            )
            break
        self.metrics_save(gt_pred=gt_pred, gt=gt, add=False)

    def SS_Full_Train(self):
        hrms2pan = HRMS2PAN().to(targs.device, targs._dtype)
        self.Oneshot_Preparation()

        for index, batch in enumerate(self.dataloader, 1):
            with torch.inference_mode(False):
                ms = batch["ms"].to(targs.device, dtype=targs._dtype)
                lms = batch["lms"].to(targs.device, dtype=targs._dtype)
                pan = batch["pan"].to(targs.device, dtype=targs._dtype)
                pan_down = wald_protocol(pan, 4)[:, :, 2:-1:4, 2:-1:4]

                lms_int = batch["lms_int"].to("cpu", torch.float64)
                pan_int = batch["pan_int"].to("cpu", torch.float64)

            ms_patches = self.loss_shaped(ms, self.ms_patch_size)
            ms_sam_patches = self.loss_shaped(ms, self.ms_sam_psz)
            pan_patches = self.loss_shaped(pan, self.pan_patch_size)
            pan_down_patches = self.loss_shaped(pan_down, self.pan_patch_size)

            self.Oneshot_Preparation(index=index)
            self.optimizer_reset(hrms2pan, targs.epoch["stageI"])

            g, pan_hp = self.MRAm(pan, ms, lms)
            executed = False
            gt_pred = None

            for epoch in self.loop:
                if epoch <= targs.epoch["csum"][0]:
                    pan_pred = hrms2pan(ms)
                    loss = self.pw_mse(
                        pan_down_patches,
                        self.loss_shaped(pan_pred, self.pan_patch_size),
                    )
                elif targs.epoch["csum"][0] < epoch <= targs.epoch["csum"][1]:
                    if not executed:
                        gnet = MRANet(hrms2pan).to(targs.device)
                        self.optimizer_reset(gnet, targs.epoch["stageII"])
                        executed = True

                    gt_pred, ms_pred, pan_pred = gnet(lms, g, pan_hp)
                    ms_pred_patches = self.loss_shaped(ms_pred, self.ms_patch_size)
                    loss = (
                        self.pwsam(
                            self.loss_shaped(ms_pred, self.ms_sam_psz), ms_sam_patches
                        )
                        * targs.lambda_mssam
                        + self.pw_mse(ms_patches, ms_pred_patches) * targs.lambda_msmse
                        + self.pw_mse(
                            pan_patches,
                            self.loss_shaped(pan_pred, self.pan_patch_size),
                        )
                        * targs.lambda_panmse
                    )
                else:
                    if executed:
                        model = MainNet(hrms2pan).to(targs.device)
                        self.optimizer_reset(model, targs.epoch["stageIII"])
                        hrms = (
                            gt_pred.detach().clone()
                            if gt_pred is not None
                            else torch.zeros_like(gt)
                        )
                        executed = False
                    gt_pred, ms_pred, pan_pred = model(hrms)
                    ms_pred_patches = self.loss_shaped(ms_pred, self.ms_patch_size)
                    # loss = (
                    #     self.pwsam(ms_patches, ms_pred_patches) * targs.lambda_mssam
                    #     + self.pwmse(ms_patches, ms_pred_patches) * targs.lambda_msmse
                    #     + self.pwmse(
                    #         pan_patches,
                    #         self.loss_shaped(pan_pred, self.pan_patch_size),
                    #     )
                    #     * targs.lambda_panmse
                    # )
                    loss = (
                        self.pwsam(
                            self.loss_shaped(ms_pred, self.ms_sam_psz), ms_sam_patches
                        )
                        * targs.lambda_mssam
                        + self.pw_mse(ms_patches, ms_pred_patches) * targs.lambda_msmse
                        + self.pw_mse(
                            pan_patches,
                            self.loss_shaped(pan_pred, self.pan_patch_size),
                        )
                        * targs.lambda_panmse
                    )

                if epoch % targs.epoch["check"] == 0:
                    with torch.no_grad():
                        if epoch <= targs.epoch["csum"][0]:
                            self.metrics_check(
                                pan=pan_down.to(torch.float32),
                                pan_pred=pan_pred.to(torch.float32),
                            )

                self.epoch_optimize(loss, targs.update_steps)
            CS = Compressive_Sensing(
                pan, ms, gt_pred.detach().clone(), tile_size=8, overlap=0
            )
            gt_pred = CS()
            self.metrics_save(gt_pred, lms_int=lms_int, pan_int=pan_int, add=True)
            torch.cuda.empty_cache()
            break
        self.metrics_save(gt_pred, lms_int=lms_int, pan_int=pan_int, add=False)

    def Origin_CS_Train(self):
        self.Oneshot_Preparation()

        for index, batch in enumerate(self.dataloader, 1):
            self.Oneshot_Preparation(index=index)
            with torch.inference_mode(False):
                ms = batch["ms"].to(targs.device, dtype=targs._dtype)
                lms = batch["lms"].to(targs.device, dtype=targs._dtype)
                pan = batch["pan"].to(targs.device, dtype=targs._dtype)
                gt = batch["gt"].to(targs.device, dtype=targs._dtype)

            CS = Compressive_Sensing(pan, ms, lms).to(margs.device, margs._dtype)
            gt_pred = CS()
            self.metrics_save(gt_pred=gt_pred, gt=gt, add=True)
        self.metrics_save(gt_pred=gt_pred, gt=gt, add=False)


if __name__ == "__main__":
    if targs.device == "cuda":
        device_id = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_id)
        print(f"GPU: {device_id} ({device_name})")
    trainer = MainTrainer()
    trainer.SS_Train()
