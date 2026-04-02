import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import einops
import cv2

# ---- Model ----
class VerticalMaskConv2d(nn.Module):
    def __init__(self, *args, **kwags):
        super().__init__()
        self.conv = nn.Conv2d(*args, **kwags)
        H, W = self.conv.weight.shape[-2:]
        mask = torch.zeros((H, W), dtype=torch.float32)
        mask[0:H // 2 + 1] = 1
        mask = mask.reshape((1, 1, H, W))
        self.register_buffer('mask', mask, False)

    def forward(self, x):
        self.conv.weight.data *= self.mask
        conv_res = self.conv(x)
        return conv_res

class HorizontalMaskConv2d(nn.Module):
    def __init__(self, conv_type, *args, **kwags):
        super().__init__()
        assert conv_type in ('A', 'B')
        self.conv = nn.Conv2d(*args, **kwags)
        H, W = self.conv.weight.shape[-2:]
        mask = torch.zeros((H, W), dtype=torch.float32)
        mask[H // 2, 0:W // 2] = 1
        if conv_type == 'B':
            mask[H // 2, W // 2] = 1
        mask = mask.reshape((1, 1, H, W))
        self.register_buffer('mask', mask, False)

    def forward(self, x):
        self.conv.weight.data *= self.mask
        conv_res = self.conv(x)
        return conv_res

class GatedBlock(nn.Module):
    def __init__(self, conv_type, in_channels, p, bn=True):
        super().__init__()
        self.conv_type = conv_type
        self.p = p
        self.v_conv = VerticalMaskConv2d(in_channels, 2 * p, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(2 * p) if bn else nn.Identity()
        self.v_to_h_conv = nn.Conv2d(2 * p, 2 * p, 1)
        self.bn2 = nn.BatchNorm2d(2 * p) if bn else nn.Identity()
        self.h_conv = HorizontalMaskConv2d(conv_type, in_channels, 2 * p, 3, 1,
                                           1)
        self.bn3 = nn.BatchNorm2d(2 * p) if bn else nn.Identity()
        self.h_output_conv = nn.Conv2d(p, p, 1)
        self.bn4 = nn.BatchNorm2d(p) if bn else nn.Identity()

    def forward(self, v_input, h_input):
        v = self.v_conv(v_input)
        v = self.bn1(v)
        v_to_h = v[:, :, 0:-1]
        v_to_h = F.pad(v_to_h, (0, 0, 1, 0))
        v_to_h = self.v_to_h_conv(v_to_h)
        v_to_h = self.bn2(v_to_h)

        v1, v2 = v[:, :self.p], v[:, self.p:]
        v1 = torch.tanh(v1)
        v2 = torch.sigmoid(v2)
        v = v1 * v2

        h = self.h_conv(h_input)
        h = self.bn3(h)
        h = h + v_to_h
        h1, h2 = h[:, :self.p], h[:, self.p:]
        h1 = torch.tanh(h1)
        h2 = torch.sigmoid(h2)
        h = h1 * h2
        h = self.h_output_conv(h)
        h = self.bn4(h)
        if self.conv_type == 'B':
            h = h + h_input
        return v, h

class GatedPixelCNN(nn.Module):
    def __init__(self, n_blocks, p, linear_dim, bn=True, color_level=2048):
        super().__init__()
        self.block1 = GatedBlock('A', 1, p, bn)
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(GatedBlock('B', p, p, bn))
        self.linear1 = nn.Conv2d(p, linear_dim, 1)
        self.linear2 = nn.Conv2d(linear_dim, linear_dim, 1)
        self.out = nn.Conv2d(linear_dim, color_level, 1)

    def forward(self, x):
        v, h = self.block1(x, x)
        for block in self.blocks:
            v, h = block(v, h)
        x = F.leaky_relu_(h)
        x = self.linear1(x)
        x = F.leaky_relu_(x)
        x = self.linear2(x)
        x = self.out(x)
        return x

class PixelCNNWithEmbedding(GatedPixelCNN):
    def __init__(self, n_blocks, p, linear_dim, bn=True, color_level = 2048):
        super().__init__(n_blocks, p, linear_dim, bn, color_level)
        self.embedding = nn.Embedding(color_level, p)
        self.block1 = GatedBlock('A', p, p, bn)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return super().forward(x)

# ---- Training (Reference-Only) ----
def train(model, device, model_path):
    dataloader = get_dataloader(batch_size)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    loss_fn = nn.CrossEntropyLoss()
    n_epochs = 40
    tic = time.time()
    for e in range(n_epochs):
        total_loss = 0
        for x, _ in dataloader:
            current_batch_size = x.shape[0]
            x = x.to(device)
            y = torch.ceil(x * (color_level - 1)).long()
            y = y.squeeze(1)
            predict_y = model(x)
            loss = loss_fn(predict_y, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * current_batch_size
        total_loss /= len(dataloader.dataset)
        toc = time.time()
        torch.save(model.state_dict(), model_path)
        print(f'epoch {e} loss: {total_loss} elapsed {(toc - tic):.2f}s')
    print('Done')

def sample(args, model, device, model_path, output_path, n_sample=81):
    model.eval()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    C, H, W = 8, args.test_sidelen, args.test_sidelen
    x = torch.zeros((n_sample, C, H, W)).to(device)
    with torch.no_grad():
        for i in range(H):
            for j in range(W):
                output = model(x)
                prob_dist = F.softmax(output[:, :, i, j], -1)
                pixel = torch.multinomial(prob_dist,
                                          1).float() / (color_level - 1)
                x[:, :, i, j] = pixel

    imgs = x * 255
    imgs = imgs.clamp(0, 255)
    imgs = einops.rearrange(imgs,
                            '(b1 b2) c h w -> (b1 h) (b2 w) c',
                            b1=int(n_sample**0.5))

    imgs = imgs.detach().cpu().numpy().astype(np.uint8)

    cv2.imwrite(output_path, imgs)
