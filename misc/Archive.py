import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---- Trainer ----
class VQVAETrainer(PansharpeningTrainer):
    def __init__(self, args=TrainerArgs(data=cargs.data)):
        super().__init__()

    def Pretrain(self):
        model = VQVAE_CNN(self.args)
        model = nn.DataParallel(model).to(self.args.device)

        g_net = MRANet(ModelArgs(data=CommonArgs.data)).to(self.args.device)

        train_dataloader, validate_dataloader = self.Pretrain_Preparation(
            "tradition", model
        )

        for batch in validate_dataloader:
            panv = batch["pan"].to(self.args.device, self.args.dtype)
            msv = batch["ms"].to(self.args.device, self.args.dtype)
            lmsv = batch["lms"].to(self.args.device, self.args.dtype)

            g, pan_hp = self.MRAm(panv, msv)
            with torch.no_grad():
                inpv = g_net(lmsv, g, pan_hp, True)

            gtv = batch["gt"].to(self.args.device, self.args.dtype)

            del panv, msv, lmsv, g, pan_hp, batch
            break
        print("=== Validation Data Extracted ===")

        for epoch in self.loop:
            for step, batch in enumerate(train_dataloader, 1):
                ms = batch["ms"].to(self.args.device, self.args.dtype)
                lms = batch["lms"].to(self.args.device, self.args.dtype)
                pan = batch["pan"].to(self.args.device, self.args.dtype)
                del batch

                g, pan_hp = self.MRAm(pan, ms)
                with torch.no_grad():
                    hrms = g_net(lms, g, pan_hp, True)

                gt_pred, ms_pred, pan_pred, vq_loss = model(hrms)

                l_images = (
                    self.criterion_SAM(ms, ms_pred)
                    + self.PWMSE_loss4(ms, ms_pred)
                    + self.PWMSE_loss4(pan_pred, pan) * self.args.lambda_fusion
                )

                loss = l_images + vq_loss
                self.epoch_optimize(loss, step, False)
                del ms, pan, ms_pred, pan_pred, gt_pred
            self.lr_scheduler.step()

            if epoch % self.args.check_epoch == 0:
                with torch.no_grad():
                    gt_pred, _, _, _ = model(inpv)
                    self.metrics_check(gtv, gt_pred)

            if epoch % self.args.save_epoch == 0:
                save_checkpoint(model, epoch, "VQVAE")


class UNet_Registration_Trainer(PansharpeningTrainer):
    def __init__(self, args=TrainerArgs(data=cargs.data)):
        super().__init__()

    def optimizer_reset(self, model=None, siren=None):
        if model != None:
            self.main_optimizer = optim.AdamW(
                [
                    {
                        "params": model.backbone.parameters(),
                        "lr": self.args.backbone_lr,
                    },
                    {
                        "params": model.pan_pred_net.parameters(),
                        "lr": self.args.pan_pred_lr,
                    },
                ],
                weight_decay=self.args.weight_decay,
            )

            self.main_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=self.main_optimizer,
                T_max=self.args.epochs,
                eta_min=self.args.eta_min,
            )

            self.main_scaler = GradScaler()

        if siren != None:
            self.siren_optimizer = optim.AdamW(
                [
                    {
                        "params": siren.backbone.parameters(),
                        "lr": self.args.backbone_lr,
                    },
                    {
                        "params": siren.ms2pan.parameters(),
                        "lr": 0,
                    },
                ],
                weight_decay=self.args.weight_decay,
            )

            self.siren_scaler = GradScaler()

    def epoch_optimize(self, loss, stage=1, schedule=False):
        with torch.autocast(self.args.device):
            if stage == 1:
                scaled_loss = self.siren_scaler.scale(loss)
                scaled_loss.backward()
                self.siren_scaler.step(self.siren_optimizer)
                self.siren_scaler.update()
                self.siren_optimizer.zero_grad()

            else:
                scaled_loss = self.main_scaler.scale(loss)
                scaled_loss.backward()
                self.main_scaler.step(self.main_optimizer)
                self.main_scaler.update()
                self.main_optimizer.zero_grad()
                if schedule:
                    self.main_lr_scheduler.step()

    def Warp_test(self):
        dataset = Dataset_warped("./test_data/warped_wv3_Reduced.h5", True)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        siren = OffsetNet(self.args.device).to(self.args.device)

        ms2pan = PanNet().to(self.args.device)
        weights = torch.load("./Weights/PanNet.pth", map_location=self.args.device)
        ms2pan.load_state_dict(weights)

        filter = Highpass()

        for index, batch in enumerate(dataloader, 1):
            warped_ms = batch["warped_ms"].to(self.args.device, self.args.dtype)
            warped_lms = batch["warped_lms"].to(self.args.device, self.args.dtype)
            lms = batch["lms"].to(self.args.device, self.args.dtype)
            ms = batch["ms"].to(self.args.device, self.args.dtype)

            pan = batch["pan"].to(self.args.device, self.args.dtype)
            pan_down = wald_protocol(pan, 4, self.args.sensor)
            pan_down_struct = filter.edge_filter(
                pan_down,
            )

            self.Oneshot_Preparation(index=index)
            self.optimizer_reset(siren=siren)

            show_paired_images([pan_down[0]], [pan_down_struct[0]])

            """
            for epoch in self.loop:
                pan_pred, x, y, angle = siren(warped_ms)
                loss = F.l1_loss(filter.edge_filter(pan_pred), pan_down_struct)
                self.epoch_optimize(loss, 1)

                if epoch % self.args.check_epoch == 0:
                    self.metrics_check(pan_pred, pan_down)
                    # 正确答案: -4, 5, -3.96
                    print(f"offset = {x*10}, {y*10}, angle = {angle*10}")
            break
            # show_paired_images([registered_lms[0]], [lms[0]])
            """

    def Train(self):
        model = UNet_MLP(ModelArgs(data=cargs.data)).to(self.args.device)
        warpnet = OffsetNet(self.args.device).to(self.args.device)
        dataset = Dataset_warped("./test_data/warped_wv3_Reduced.h5")
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        for index, batch in enumerate(self.dataloader, 1):
            ms = batch["ms"].to(self.args.device, self.args.dtype)
            gt = batch["gt"].to(self.args.device, self.args.dtype)
            lms = batch["warped_lms"].to(self.args.device, self.args.dtype)
            pan = batch["pan"].to(self.args.device, self.args.dtype)

            self.Oneshot_Preparation(index=index)
            self.optimizer_reset(model, warpnet)

            for epoch in self.loop:
                if epoch <= self.args.stageI_epochs:
                    ms_pred, registered_lms = warpnet(lms)
                    loss = F.l1_loss(ms, ms_pred)
                    self.epoch_optimize(loss, 1)

                elif epoch == self.args.stageI_epochs + 1:
                    unet_input = torch.cat(
                        [registered_lms.detach().clone(), pan], dim=1
                    )

                else:
                    gt_pred, ms_pred, pan_pred = model(unet_input)
                    loss = (
                        self.criterion_SAM(ms, ms_pred)
                        + self.PWMSE_loss4(ms, ms_pred)
                        + self.PWMSE_loss4(pan_pred, pan) * self.args.lambda_fusion
                    )
                    self.epoch_optimize(loss, 2)
                    if epoch % self.args.check_epoch == 0:
                        self.metrics_check(gt, gt_pred, ms, ms_pred, pan, pan_pred)

            del model
            torch.cuda.empty_cache()

            self.metrics(
                gt, gt_pred, self.loop.format_dict["elapsed"], add_results=True
            )
            self.gt_pred_total.append(gt_pred)
            break

        self.metrics(
            torch.stack(self.gt_pred_total, 0), gt, add_results=False, store=True
        )


class GAN_Trainer(PansharpeningTrainer):  # Saved
    def __init__(self, args=TrainerArgs(data=cargs.data)):
        super().__init__()

    def epoch_optimize(self, loss, optimizer, scaler, lr_scheduler=None):
        with torch.autocast(self.args.device, dtype=torch.bfloat16):
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if lr_scheduler != None:
            lr_scheduler.step()

    def Train(self):
        model = UNet_CNN(ModelArgs(data=cargs.data)).to(self.args.device)
        g_net = MRANet(args=ModelArgs(data=cargs.data)).to(self.args.device)
        D_ms = Discriminator(in_ch=8, out_ch=1, hidden_ch=64, depth=2).to(
            self.args.device
        )
        D_pan = Discriminator(in_ch=1, out_ch=1, hidden_ch=32, depth=4).to(
            self.args.device
        )

        model_optimizer, model_lr_scheduler, model_scaler = self.optimizer_init(
            list(model.parameters()), self.args.stageII_epochs
        )

        gnet_optimizer, gnet_lr_scheduler, gnet_scaler = self.optimizer_init(
            list(g_net.parameters()), self.args.stageI_epochs
        )

        D_optimizer, D_lr_scheduler, D_scaler = self.optimizer_init(
            list(D_ms.parameters()) + list(D_pan.parameters()), self.args.epochs
        )

        L = GANLoss(D_ms, D_pan)

        executed = False

        self.Oneshot_Preparation(type="tradition")

        for index, batch in enumerate(self.dataloader, 1):
            ms = batch["ms"].to(self.args.device, self.args.dtype)
            gt = batch["gt"].to(self.args.device, self.args.dtype)
            lms = batch["lms"].to(self.args.device, self.args.dtype)
            pan = batch["pan"].to(self.args.device, self.args.dtype)

            self.Oneshot_Preparation(index=index)

            self.optimizer_reset(g_net, self.args.stageI_epochs)
            g, pan_hp = self.MRAm(pan, ms)

            for epoch in self.loop:
                if epoch <= self.args.stageI_epochs:  # gnet + Discriminator 更新
                    gt_pred, ms_pred, pan_pred = g_net(lms, g, pan_hp)

                    if epoch % 2 == 0:  # 更新gnet
                        ganloss = L.Generator_Loss(ms, pan, ms_pred, pan_pred)
                        loss = (
                            self.criterion_SAM(ms, ms_pred)
                            + self.PWMSE_loss4(ms, ms_pred)
                            + self.PWMSE_loss4(pan_pred, pan) * self.args.lambda_fusion
                            + ganloss
                        )

                        self.epoch_optimize(
                            loss, gnet_optimizer, gnet_scaler, gnet_lr_scheduler
                        )

                    else:  # 更新Discriminator
                        loss = L.Discriminator_Loss(ms, pan, ms_pred, pan_pred)
                        self.epoch_optimize(loss, D_optimizer, D_scaler, D_lr_scheduler)

                else:
                    if not executed:
                        hrms = gt_pred.detach().clone()
                        executed = True

                    gt_pred, ms_pred, pan_pred = model(hrms)

                    if epoch % 2 == 0:  # 更新model
                        ganloss = L.Generator_Loss(ms, pan, ms_pred, pan_pred)
                        loss = (
                            self.criterion_SAM(ms, ms_pred)
                            + self.PWMSE_loss4(ms, ms_pred)
                            + self.PWMSE_loss4(pan_pred, pan) * self.args.lambda_fusion
                            + ganloss
                        )

                        self.epoch_optimize(
                            loss, model_optimizer, model_scaler, model_lr_scheduler
                        )

                    else:  # 更新Discriminator
                        loss = L.Discriminator_Loss(ms, pan, ms_pred, pan_pred)
                        self.epoch_optimize(loss, D_optimizer, D_scaler, D_lr_scheduler)

                if epoch % self.args.check_epoch == 0:
                    self.metrics_check(gt, gt_pred, ms, ms_pred, pan, pan_pred)

            self.metrics(
                gt, gt_pred, self.loop.format_dict["elapsed"], add_results=True
            )
            self.gt_pred_total.append(gt_pred)

        self.metrics(
            torch.stack(self.gt_pred_total, 0), gt, add_results=False, store=True
        )


class OffsetNet(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.backbone = INR(
            in_features=3,
            hidden_features=48,
            hidden_layers=2,
            out_features=3,
            outermost_linear=True,
            parallel=False,
            act="hsiren",
        )

        self.ms2pan = PanNet()
        weights = torch.load("./Weights/PanNet.pth", map_location=device)
        self.ms2pan.load_state_dict(weights)

        self.siren_input = torch.ones(3).to(device)

    def forward(self, ms, ret_real=False):
        x, y, angle = self.backbone(self.siren_input)

        pan_pred = self.ms2pan(warp(ms, [x * 10, y * 10], angle * 10, inverse=True))
        if ret_real:
            pan_pred_real = self.ms2pan(warp(ms, [-4, 5], -3.96, inverse=True))
            return pan_pred, pan_pred_real, x, y, angle
        else:
            return pan_pred, x, y, angle

        """
        lms_registrated = warp(lms, [x * 10, y * 10], angle * 10, inverse=True)
        return lms_registrated, x, y, angle
        """


class Local_ensemble:
    def __init__(self, size):
        super().__init__()
        self.size = size

    def query_img(self, inp, coord):
        feat = inp
        sidelen = inp.shape[-1]

        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

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

                b, q, _ = q_coord.shape
                rel_coord = coord - q_coord  # x-v
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

                preds.append(q_feat)

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

        return ret.reshape(b, 8, self.size, self.size)

    def Low_High_Train(self):
        model = UNet_CNN(ModelArgs(data=cargs.data)).to(self.args.device)
        g_net = MRANet(args=ModelArgs(data=cargs.data)).to(self.args.device)

        self.Oneshot_Preparation(type="tradition")

        for index, batch in enumerate(self.dataloader, 1):
            ms = batch["ms"].to(self.args.device, self.args.dtype)
            gt = batch["gt"].to(self.args.device, self.args.dtype)
            lms = batch["lms"].to(self.args.device, self.args.dtype)
            pan = batch["pan"].to(self.args.device, self.args.dtype)

            pan_down = wald_protocol(pan, 4, self.args.sensor)
            ms_down = wald_protocol(ms, 4, self.args.sensor)
            lms_down = wald_protocol(ms, 1, self.args.sensor)

            self.Oneshot_Preparation(index=index)

            self.optimizer_reset(g_net, 20)
            g_down, pan_hp_down = self.MRAm(pan_down, ms_down)
            g, pan_hp = self.MRAm(pan, ms)

            executed = False

            for epoch in self.loop:
                if epoch <= self.args.stageI_epochs:
                    if epoch <= 20:
                        ms_pred_init = g_net(lms_down, g_down, pan_hp_down, None, True)
                        loss = self.criterion_SAM(ms, ms_pred_init) + self.PWMSE_loss4(
                            ms, ms_pred_init
                        )

                    else:
                        if not executed:
                            self.optimizer_reset(model, self.args.stageI_epochs - 30)
                            ms_pred_init = ms_pred_init.detach().clone()
                            executed = True
                        ms_pred = model(ms_pred_init, None, True)
                        loss = self.criterion_SAM(ms, ms_pred) + self.PWMSE_loss4(
                            ms, ms_pred
                        )

                else:
                    if epoch == self.args.stageI_epochs + 1:  # 更新g_net
                        self.optimizer_reset(g_net, 30)
                        hrms, ms_pred, pan_pred = g_net(lms, g, pan_hp)
                    elif epoch <= self.args.stageI_epochs + 30:
                        hrms, ms_pred, pan_pred = g_net(lms, g, pan_hp)

                    else:
                        if executed:
                            hrms = hrms.detach().clone()
                            self.optimizer_reset(model, self.args.stageII_epochs - 30)
                            executed = False
                        gt_pred, ms_pred, pan_pred = model(hrms)

                    loss = (
                        self.criterion_SAM(ms, ms_pred)
                        + self.PWMSE_loss4(ms, ms_pred)
                        + self.PWMSE_loss4(pan_pred, pan) * self.args.lambda_fusion
                    )

                self.self_optimize(loss, self.args.update_steps, True)

                if (epoch % self.args.check_epoch == 0) and (epoch <= 200):
                    self.metrics_check(ms, ms_pred)
                if (epoch % self.args.check_epoch == 0) and (epoch > 200):
                    self.metrics_check(gt, gt_pred, ms, ms_pred, pan, pan_pred)

            self.metrics(
                gt, gt_pred, self.loop.format_dict["elapsed"], add_results=True
            )
            self.gt_pred_total.append(gt_pred)

        self.metrics(
            torch.stack(self.gt_pred_total, 0), gt, add_results=False, store=True
        )

    def HRMS2PANPretrain(self):
        model = HRMS2PAN().to(targs.device)
        train_dataloader, validate_dataloader = self.Pretrain_Preparation("tradition")
        self.optimizer_reset(model, targs.epoch["total"])

        for batch in validate_dataloader:
            gt_v = batch["gt"].to(targs.device, targs._dtype)
            pan_v = batch["pan"].to(targs.device, targs._dtype)

        for epoch in self.loop:
            for step, batch in enumerate(train_dataloader, 1):
                pan = batch["pan"].to(targs.device, targs._dtype)
                ms = batch["ms"].to(targs.device, targs._dtype)
                pan_down = wald_protocol(pan, 4, targs.sensor)
                del pan

                pan_pred = model(ms)

                loss = self.pwmse(pan_pred, pan_down)  # Loss再看看！！！！！！！！！
                self.epoch_optimize(loss, step, False)

            for scheduler in self.lr_schedulers:
                scheduler.step()

            if epoch % targs.epoch["check"] == 0:
                with torch.no_grad():
                    pan_pred = model(gt_v)
                    self.metrics_check(pan_v, pan_pred)

            if epoch % targs.epoch["save"] == 0:
                save_checkpoint(model, epoch, "HRMS2PAN")

        self.Oneshot_Preparation(type="tradition")

        for index, batch in enumerate(self.dataloader, 1):
            ms = batch["ms"].to(targs.device, targs._dtype)
            gt = batch["gt"].to(targs.device, targs._dtype)
            lms = batch["lms"].to(targs.device, targs._dtype)
            pan = batch["pan"].to(targs.device, targs._dtype)
            pan_down = wald_protocol(pan, 4, targs.sensor)

            self.Oneshot_Preparation(index=index)
            self.optimizer_reset(gnet, targs.epoch["stageI"])

            g, pan_hp = self.MRAm(pan, ms)
            executed = False

            for epoch in self.loop:
                if epoch <= targs.epoch["stageI"]:
                    pan_pred = hrms2pan(ms)
                    loss = self.pwmse(pan_down, pan_pred)

                elif targs.epoch["stageI"] < epoch <= targs.epoch["stageII"]:
                    gt_pred, ms_pred = model(lms, g, pan_hp, False)
                    loss = (
                        self.pwsam(ms, ms_pred)
                        + self.pwmse(ms, ms_pred)
                        + self.pwmse(pan, pan_pred) * targs.lambda_fusion
                    )

                else:
                    if not executed:
                        self.optimizer_reset(model, targs.epoch["stageII"])
                        hrms = gt_pred.detach().clone()
                        executed = True

                    gt_pred, ms_pred = model(hrms, False)
                    loss = (
                        self.pwsam(ms, ms_pred)
                        + self.pwmse(ms, ms_pred)
                        + self.pwmse(pan, pan_pred) * targs.lambda_fusion
                    )

                self.epoch_optimize(loss, targs.update_steps)
                if epoch % targs.epoch["check"] == 0:
                    if epoch <= targs.epoch["stageI"]:
                        self.metrics_check(pan, pan_pred)
                    else:
                        self.metrics_check(gt, gt_pred, ms, ms_pred, pan, pan_pred)

    def MRAPretrain(self):
        model = MRANet().to(targs.device)
        train_dataloader, validate_dataloader = self.Pretrain_Preparation("tradition")
        self.optimizer_reset(model, targs.epoch["total"])

        for batch in validate_dataloader:
            pan = batch["pan"].to(targs.device, targs._dtype)
            ms = batch["ms"].to(targs.device, targs._dtype)
            lms_v = batch["lms"].to(targs.device, targs._dtype)
            g_v, pan_hp_v = self.MRAm(pan, ms)

            del pan, ms
            gt_v = batch["gt"].to(targs.device, targs._dtype)

            break

        for epoch in self.loop:
            for step, batch in enumerate(train_dataloader, 1):
                ms = batch["ms"].to(targs.device, targs._dtype)
                lms = batch["lms"].to(targs.device, targs._dtype)
                pan = batch["pan"].to(targs.device, targs._dtype)
                g, pan_hp = self.MRAm(pan, ms)

                gt_pred, ms_pred, pan_pred = model(lms, g, pan_hp)

                loss = (
                    self.pwsam(ms, ms_pred)
                    + self.PWMSE_loss4(ms, ms_pred)
                    + self.PWMSE_loss4(pan_pred, pan) * targs.lambda_fusion
                )

                self.epoch_optimize(loss, step, False)

            self.lr_scheduler.step()

            if epoch % targs.epoch["check"] == 0:
                with torch.no_grad():
                    gt_pred = model(lms_v, g_v, pan_hp_v, True)
                    self.metrics_check(gt_pred, gt_v)

            if epoch % targs.epoch["save"] == 0:
                save_checkpoint(model.backbone, epoch, "MRA")

    def MTF_Pretrain(self):
        model = MTFNet()
        # weights = torch.load("./Weights/MTF_2000.pth")
        # model.load_state_dict(weights)

        coord = make_intcoord(41, True).to(targs.device, targs._dtype)
        coord_list = [coord] * 8
        coords = torch.cat(coord_list, dim=1).unsqueeze(-1).unsqueeze(-1)

        self.optimizer_reset(model, targs.epoch["pretrain"])
        _, _ = self.Pretrain_Preparation("tradition")
        orig_kernel = (
            MTF_MS(4, targs.sensor, 41)
            .permute(1, 0, 2, 3)
            .to(targs.device, targs._dtype)
        )

        for epoch in self.loop:
            kernel = model(coords)
            loss = F.huber_loss(kernel, orig_kernel)
            self.epoch_optimize(loss, targs.update_steps, True)

            if epoch % targs.epoch["check"] == 0:
                print(f"orgi kernel: {orig_kernel[0,0,18:22,18:22]}")
                print(f"predicted kernel: {kernel[0,0,18:22,18:22]}")
                self.metrics_check(orig_kernel, kernel)

            if epoch % targs.epoch["save"] == 0:
                save_checkpoint(model, epoch, "MTF")

    def Train(self):
        model = UNet_CNN()
        gnet = MRANet()

        self.Oneshot_Preparation(type="tradition")

        for index, batch in enumerate(self.dataloader, 1):
            ms = batch["ms"].to(targs.device, targs._dtype)
            gt = batch["gt"].to(targs.device, targs._dtype)
            lms = batch["lms"].to(targs.device, targs._dtype)
            pan = batch["pan"].to(targs.device, targs._dtype)

            self.Oneshot_Preparation(index=index)
            self.optimizer_reset(gnet, targs.epoch["stageI"])

            g, pan_hp = self.MRAm(pan, ms)
            executed = False

            for epoch in self.loop:
                if epoch <= targs.epoch["stageI"]:
                    gt_pred, ms_pred, pan_pred = gnet(lms, g, pan_hp)

                else:
                    if not executed:
                        self.optimizer_reset(model, targs.epoch["stageII"])
                        hrms = gt_pred.detach().clone()
                        executed = True

                    gt_pred, ms_pred, pan_pred = model(hrms)

                loss = (
                    self.pwsam(ms, ms_pred)
                    + self.pwmse(ms, ms_pred)
                    + self.pwmse(pan, pan_pred) * targs.lambda_fusion
                )

                self.epoch_optimize(loss, targs.update_steps)
                if epoch % targs.epoch["check"] == 0:
                    self.metrics_check(gt, gt_pred, ms, ms_pred, pan, pan_pred)

            self.metrics_save(gt_pred, gt, self.loop.format_dict["elapsed"], True)
            break

        self.metrics_save(gt_pred, gt, self.loop.format_dict["elapsed"], False)


# ---- RDB ----
class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1), nn.LeakyReLU()
        )

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class RDN(nn.Module):
    def __init__(self, args):
        super(RDN, self).__init__()
        self.args = args
        r = args.scale[0]
        G0 = args.G0
        kSize = args.RDNkSize

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            "A": (2, 2, 64),
            "B": (2, 2, 8),
        }[args.RDNconfig]

        # Shallow feature extraction net
        self.SFENet1 = nn.LazyConv2d(G0, kSize, padding=(kSize - 1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(RDB(growRate0=G0, growRate=G, nConvLayers=C))

        # Global Feature Fusion
        self.GFF = nn.Sequential(
            *[
                nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
                nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1),
            ]
        )

        if args.no_upsampling:
            self.out_dim = G0
        else:
            self.out_dim = args.n_colors
            # Up-sampling net
            if r == 2 or r == 3:
                self.UPNet = nn.Sequential(
                    *[
                        nn.Conv2d(
                            G0, G * r * r, kSize, padding=(kSize - 1) // 2, stride=1
                        ),
                        nn.PixelShuffle(r),
                        nn.Conv2d(
                            G, args.n_colors, kSize, padding=(kSize - 1) // 2, stride=1
                        ),
                    ]
                )
            elif r == 4:
                self.UPNet = nn.Sequential(
                    *[
                        nn.Conv2d(G0, G * 4, kSize, padding=(kSize - 1) // 2, stride=1),
                        nn.PixelShuffle(2),
                        nn.Conv2d(G, G * 4, kSize, padding=(kSize - 1) // 2, stride=1),
                        nn.PixelShuffle(2),
                        nn.Conv2d(
                            G, args.n_colors, kSize, padding=(kSize - 1) // 2, stride=1
                        ),
                    ]
                )
            else:
                raise ValueError("scale must be 2 or 3 or 4.")

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1

        if self.args.no_upsampling:
            return x
        else:
            return self.UPNet(x)


def make_rdn(G0=64, RDNkSize=3, RDNconfig="A", scale=4, no_upsampling=True):
    args = Namespace()
    args.G0 = G0
    args.RDNkSize = RDNkSize
    args.RDNconfig = RDNconfig

    args.scale = [scale]
    args.no_upsampling = no_upsampling

    args.n_colors = 8
    args.input_dim = 9
    return RDN(args)


# ---- BID-INR ----
class BID_INR(nn.Module):  # For Image Debluring!!!! Not SR!!!
    def __init__(self, args):
        super().__init__()
        self.args = args
        ratio = self.args.ratio

        # ---- Spectral Response Function ----
        self.srf = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=args.spectral_num,
                out_channels=ratio * args.spectral_num,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                in_channels=args.spectral_num * ratio,
                out_channels=args.spectral_num,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Softmax(dim=-1),
        )
        init_weights(self.srf)

        # ---- SIREN to generate kernels ----
        self.k_INR = Siren(
            in_features=2,
            hidden_features=128,
            hidden_layers=3,
            out_features=self.args.spectral_num,
            first_omega_0=args.first_omega_0,
            hidden_omega_0=args.hidden_omega_0,
            outermost_linear=False,
        )

        # ---- backbone to generate image ----
        self.x_INR = get_net(
            input_depth=10,
            NET_TYPE="skip",
            pad=self.args.pad_type,
            upsample_mode="bilinear",
            n_channels=8,
            act_fun="LeakyReLU",
            skip_n33d=128,
            skip_n33u=128,
            skip_n11=4,
            num_scales=5,
            downsample_mode="stride",
        )

    def kernel_centering(self, kernel):
        H, W = kernel.shape[-1], kernel.shape[-2]

        i_coords = torch.arange(H, device=kernel.device, dtype=kernel.dtype).view(-1, 1)
        j_coords = torch.arange(W, device=kernel.device, dtype=kernel.dtype).view(1, -1)

        c_y = torch.sum(i_coords * kernel) / torch.sum(kernel)
        c_x = torch.sum(j_coords * kernel) / torch.sum(kernel)

        shift_y = (H // 2) - c_y
        shift_x = (W // 2) - c_x

        kernel_centered = torch.roll(
            kernel, shifts=(int(shift_y), int(shift_x)), dims=(0, 1)
        )

        return kernel_centered

    def forward(self, kernel_coords, img_coords, cross_scale=False):
        kernel = self.k_INR(kernel_coords).reshape(
            self.args.spectral_num, self.args.spectral_num, -1
        )
        kernel_size = int(math.sqrt(kernel.shape[2]))
        kernel = F.softmax(kernel, dim=-1).reshape(
            self.args.spectral_num, self.args.spectral_num, kernel_size, kernel_size
        )
        kernel = self.kernel_centering(kernel)

        gt_pred = self.x_INR(img_coords)

        gt_pred_blur = F.conv2d(gt_pred, weight=kernel, padding=2, stride=4)
        ms_pred = F.interpolate(gt_pred_blur, scale_factor=0.25, mode="area")

        pan_pred = torch.sum(self.srf(gt_pred) * gt_pred, dim=1)

        if cross_scale:
            with torch.no_grad():
                m_indices = (-1) ** torch.arange(kernel_size).reshape(
                    kernel_size, 1
                ).repeat(1, kernel_size).to(self.args.device, torch.float32)
                n_indices = (-1) ** torch.arange(kernel_size).reshape(
                    1, kernel_size
                ).repeat(kernel_size, 1).to(self.args.device, torch.float32)
                mn_indices = (-1) ** torch.arange(kernel_size**2).reshape(
                    kernel_size, kernel_size
                ).to(self.args.device, torch.float32)

                g1_kernel = m_indices * kernel
                g2_kernel = n_indices * kernel
                g3_kernel = mn_indices * kernel

                gt_pred_blur_down = F.interpolate(
                    gt_pred_blur, scale_factor=0.5, mode="area"
                )
                qmf_conv_down = (
                    F.interpolate(
                        F.conv2d(gt_pred, weight=g1_kernel, padding="same"),
                        scale_factor=0.5,
                        mode="area",
                    )
                    + F.interpolate(
                        F.conv2d(gt_pred, weight=g2_kernel, padding="same"),
                        scale_factor=0.5,
                        mode="area",
                    )
                    + F.interpolate(
                        F.conv2d(gt_pred, weight=g3_kernel, padding="same"),
                        scale_factor=0.5,
                        mode="area",
                    )
                )

            gt_pred_blur_decompose = qmf_conv_down + gt_pred_blur_down

            return gt_pred, gt_pred_blur_decompose, ms_pred, pan_pred

        return gt_pred, ms_pred, pan_pred



# ---- Others ----
def q2n(I_GT, I_F, Q_blocks_size=32, Q_shift=32):
    b,c,h,w = I_F.shape
    stepx=math.ceil(h/Q_shift)
    stepy=math.ceil(w/Q_shift)

    if stepy<=0:
        stepy=1
        stepx=1

    est1=(stepx-1)*Q_shift+Q_blocks_size-h
    est2=(stepy-1)*Q_shift+Q_blocks_size-w

    if (est1 != 0) or (est2 != 0):
        refref=[]
        fusfus=[]
        for i in range(c):
            a1=I_GT[:,0,:,:]
            ia1=torch.zeros(h+est1,w+est2)
            ia1[:,:,:h,:w]=a1;
            ia1[:, :, :, w:] = ia1[:, :, :, w-1:w-est2-1:-1]
            ia1[:, :, h:] = ia1[:, :, :, w-1:w-est2-1:-1]
      ia1(N1+1:N1+est1,:)=ia1(N1:-1:N1-est1+1,:);
            refref = torch.cat((refref, ia1), dim=1)
      
        if i<c:
            I_GT=I_GT[:,1:,:,:]

    I_GT=refref;

        for i in range(c):
            a2=I_GT[:,0,:,:]
            ia2=torch.zeros(h+est1,w+est2)
            ia2[:,:,:h,:w]=a2;
            ia2[:, :, :, w:w+est2] = ia2[:, :, :, w-1:w-est2-1:-1]
            refref = torch.cat((refref, ia1), dim=1)
      
        if i<c:
            I_GT=I_GT[:,1:,:,:]

    I_GT=refref;
  

I_F=uint16(I_F);
I_GT=uint16(I_GT);

[N1,N2,N3]=size(I_GT);

if ((ceil(log2(N3)))-log2(N3))~=0
    Ndif=(2^(ceil(log2(N3))))-N3;
    dif=zeros(N1,N2,Ndif);
    dif=uint16(dif);
    I_GT=cat(3,I_GT,dif);
    I_F=cat(3,I_F,dif);
end
[~,~,N3]=size(I_GT);

valori=zeros(stepx,stepy,N3);

for j=1:stepx
    for i=1:stepy
        o=onions_quality(I_GT(((j-1)*Q_shift)+1:((j-1)*Q_shift)+Q_blocks_size,((i-1)*Q_shift)+1:((i-1)*Q_shift)+size2,:),I_F(((j-1)*Q_shift)+1:((j-1)*Q_shift)+Q_blocks_size,((i-1)*Q_shift)+1:((i-1)*Q_shift)+size2,:),Q_blocks_size);
        valori(j,i,:)=o;    
    end
end

Q2n_index_map=sqrt(sum((valori.^2),3));

Q2n_index=mean2(Q2n_index_map);

end
