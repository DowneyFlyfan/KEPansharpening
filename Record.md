# Experiment Records

## Early Test (RTX 3090, 40轮 LRTCF + 500轮 UNet)

| UNet 变体                                       | PSNR (dB)   |
| :---------------------------------------------- | :---------- |
| 原始配置                                        | 35.6  ±  0.2  |
| k=2, s=2 下采样                                 | 35.5        |
| WTConvNext + k=2, s=2 下采样，原始上采样        | 35.3        |
| ConvNext + k=2, s=2 下采样，原始上采样          | 36.0        |
| UGCN (多次调参)                                 | 最高 35.9   |
| 4层 + Attn                                      | 35.6        |
| ConvNextV2                                      | 36.0        |
| ConvNextV2 + DynamicTanh                        | 35.0        |

## UNext上采样和Norm改进(500 + 50 + 600), WV3 Reduced

| Model Variant | Configuration | Normalization | Learning Rate (pan) | PSNR (dB) | SAM |
| :------------ | :------------ | :------------ | :------------------ | :-------- | :-- |
| ConvNext      | 1x1 Conv Shortcut | Default       | 4e-3 -> 1e-3        | 34.35     | 3.03 |
| ConvNext      | FC + Shared dwconv Shortcut | Default | 2.5e-3 -> 1e-3      | 34.71     | 3.13 |
| Ordinary Upsampling | -             | Default       | 2.5e-3 -> 1e-3      | 36.18     | 2.76 |
| ConvNext      | hidden_factor = 2, Ordinary Upsampling | Default | 2.5e-3 -> 1e-3      | 36.17     | 2.82 |
| ConvNext      | hidden_factor = 2, Ordinary Upsampling | BN            | 2.5e-3 -> 1e-3      | 36.38     | 2.86 |
| ConvNext      | hidden_factor = 2, nn.LayerNorm, Ordinary Upsampling | LayerNorm | 2.5e-3 -> 1e-3      | 35.98     | 2.89 |

- `ConvNext`用`LayerNorm`, 其他部分用BN最好!!!

## Precision

- 500 + 50 + 600, M1 Pro

| 类型       | PSNR (dB) | SAM  | 时间 (包括指标计算) |
| :--------- | :-------- | :--- | :------------------ |
| float32    | 36.52     | 2.73 | 82                  |
| bfloat16   | 36.61     | 2.69 | 98                  |
| float16    | 36.55     | 2.69 | 92                  |

## MS2PAN 组件测试

### S: 单图500轮, A: 整个模型(WV3-Reduced)

| Configuration                                            | PSNR (dB) (Single Image) | PSNR (dB) (Full Model) |
| :------------------------------------------------------- | :----------------------- | :--------------------- |
| Right MTF                                                     |  51 | 35.43                  |
| Wrong MTF                                                     |  51.77 |                   |
| 5层 H-SIREN                                              | 25                       | 34                     |
| Unfold(3x3) -> 5层 INR (hsiren / siren)                  | 26 (未收敛)              | N/A                    |

### 25.8实验 (WV3-Reduced)

- 500 + 50 + 600

| 结构 | PSNR | SAM |
|:-|:-|:-|
| Conv3x3(dilation = 2) -> BN -> ReLu -> Conv3x3 -> BN -> ReLu -> Conv3x3, channel=32| **36.59** | 2.73 |
| Conv3x3(dilation = 2) -> BN -> ReLu -> Conv3x3 -> BN -> ReLu -> Conv1x1, channel=32| 36.19 | 2.85 |
| Conv5x5(dilation = 2) -> BN -> ReLu -> Conv3x3 -> BN -> ReLu -> Conv3x3, channel=32| 36.15 | 2.84 |
| Conv3x3(dilation = 2) -> BN -> ReLu -> ConvNext -> Conv3x3, channel=32| 35.91 | 2.75 |
|  BN -> Conv3x3 -> ConvNext -> Conv3x3, channel=32 | 35.19 | 2.88 |
|  BN -> Conv3x3 -> ConvNext -> BN -> Conv3x3, channel=32 | 34.25 | 2.86 |
| Conv3x3 -> BN -> ConvNext -> BN -> Conv3x3, channel=32 | 36.46 | 2.82 |
| Conv3x3 -> BN -> ReLu -> ConvNext -> BN -> ReLu -> Conv3x3, channel=32 | 36.12 | 2.71 |
| Conv3x3 ->  ConvNext -> BN -> Conv3x3, channel=32 | 36.36 | 2.86 |
| Conv3x3 ->  ConvNext -> LN(eps=1e-6) -> Conv3x3, channel=32 | 36.47 | 2.77 |
| Conv3x3 ->  ConvNext -> Conv3x3, channel=16 | 36.45 | *2.69* |
| **目前采用的** Conv3x3 ->  ConvNext -> LN -> Conv3x3, channel=16 | *36.53* | **2.65**|
| Conv3x3 ->  ConvNext -> LN(eps=1e-6) -> Conv3x3, channel=16 | 36.51 | 2.71 |
| Conv3x3 ->  ConvNext(hidden_factor=1) -> BN -> Conv3x3, channel=32 | 36.11 | 2.88 |
| ConvNext -> LN -> Conv3x3, channel=32 | 36.03 | 2.84 |
| ConvNext -> ConvNext, channel=32 | 34.77 | 2.93 |
| Conv3x3 -> ConvNext -> ConvNext, channel=32 | 36.04 | 2.70 |

### 结论

- ConvNext后跟LN更稳定

### WV3-Full

| Configuration             | Norm/Act                         | LR                 | Scheduler   | Epochs | PSNR (dB) |
| :------------------------------- | :------------ | :------------ | :------------------ | :-------- | :-- |
| Original                  | -                                | -                  | -           | 500    | 40.35     |
| Original                  | -                                | -                  | -           | 1000   | 42.98     |
| -                         | -                                | 1e-3 -> 5e-4       | -           | (1000) | 42.976    |
| psz=8                     | BN -> LeakyReLU (except last)  | 4e-3 -> 5e-4       | flat_down  | 1000   | 54.92     |
| psz=8                     | BN -> LeakyReLU (except last)  | 4e-3 -> 5e-4       | flat_down  | 1500   | 57.26     |
| psz=2                     | BN -> LeakyReLU (except last)  | 4e-3 -> 5e-4       | flat_down  | 1000   | 52.77     |
| psz=8                     | BN -> LeakyReLU (except last) + Sigmoid (last) | 4e-3 -> 5e-4 | flat_down  | 1000   | 53.50     |
| psz=8, ConvNext in middle | BN -> LeakyReLU (except last)  | 4e-3 -> 5e-4       | flat_down  | 1000   | 50        |
| -                         | LN -> LeakyReLU (except last)  | 4e-3 -> 5e-4       | flat_down  | 1000   | 39        |
| -                         | BN -> LeakyReLU (except last)  | 2e-3 -> 5e-4       | cosine      | 1000   | 41.37     |
| -                         | BN -> LeakyReLU (except last)  | 4e-3 -> 5e-4       | cosine      | 1000   | 50.68     |
| dim=64                    | BN -> LeakyReLU (except last)  | 4e-3 -> 5e-4       | cosine      | 1000   | 47        |

## Prior Learning \& Traditional Methods

### GNet (4090)

1. Results Only for GNet

| Configuration | PSNR (dB) | SAM  | Notes |
| :------------------------- | :-------- | :--- | :---- |
| 4x4                        | 33.13     | -    |       |
| 8x8                        | 33.30     | -    |       |
| 16x16                      | 33.38     | -    |       |
| 32x32                      | 33.42     | -    |       |
| No Network                 | 30.29     | -    |       |
| g = y / X                  | 33.56     | 3.56 |       |
| pan to lms, ms = ms - mtf(ms, ratio=1, N=11)                  | 33.57     | 3.47 |       |
| 直接学pan_hp                | 28 | - | 多种mranet 和 pan_pred_net 的学习率下训练都很不稳定|
| 逐通道学习`g` | 33.37   | -    | hidden_dim = 256, 3个3x3Conv |

2. Impacts of GNet on overall results

| Configuration | PSNR (dB) | SAM  | Notes |
| :------------------------- | :-------- | :--- | :---- |
|Original|36.40|2.72|-|
|N=11|36.40|2.71|-|
|g = y * X / Xnorm**2 | 36.23 | 2.71 | - |
|pan to ms|36.36|2.77|-|
|lstsq|36.36|2.75|-|

| Conclusion                                                                 | Observation                                   |
| :------------------------------------------------------------------------- | :-------------------------------------------- |
| Global matching works better                                               | GNet's performance is not strongly related to MS2PAN |
| $g_{init}$                                                                 | is not important                              |
| Per-channel learning                                                       | does not improve performance                  |
| The original **LRTCFPAN** is one of the few traditional methods that **requires normalized input** | otherwise the result is wrong                 |
| After **MTF is corrected**                                                 | Performance **remains the same**              |

### SR-D Early Test (Failed because of Wrong MTF!!)

1. Early Version

| Method                      | PSNR (dB) | SAM  | 备注        |
| :-------------------------- | :-------- | :--- | :---------- |
| MATLAB bicubic | 30.29     | 3.90 | -           |
| 不用我的函数，直接直方图匹配 | 30.29     | 3.90 | -           |
| 低分辨率的PAN直接从高分辨率PAN下采样得来, 不模糊 | 30.29     | 3.90 | -           |
| 15个atoms | 30.29     | 3.90 | -           |
| `F.interpolate`   | 30.29     | 3.90 | -           |
|使用Conv3d训练上采样4倍的`alpha`(for each patch) | 30.19 | 4.74 | 越训练越差! 试了多种学习率也是如此 |
|使用MLP训练`alpha` | 30.29 |3.90 | |

2. Closer to MATLAB

| Method                      | PSNR (dB) | SAM  | 备注        |
| :--------------- | :-------- | :--- | :---------- |
| Mimic the original MATLAB Code|30.28|3.96|Using det to decide whether to calculate it or not|
| 16个atoms | 30.28     | 3.96 | -           |
| imresize instead of Pytorch upsample | 30.28     | 3.95 | -           |
| imresize instead of Pytorch upsample + (delta_limit = 0) | 30.28     | 3.94 | -           |
| No Divisors|30.18|4.37|-|
| No Overlap, tile size=4|24.98|6.09|-|
| Overlap=4, tile size=8|28.48|4.37|-|
| Overlap=0, tile size=8|30.24|4.03|-|
| alpha upsampled to (ni, c, h, w) -> Conv2d Net(double, dim=32) |29.34 |7.46 | Converged |
| alpha upsampled to (ni, c, h, w) -> Conv2d Net(trible, dim=128) | 30.12 | 6.03 | Converged |
| alpha upsampled to (1, c, i, h, w) -> Conv3d Net(trible, dim=32) | 17.81 | 7.09 | Converged |

3. Now Starts with **imresize + delta_limit = 0**

| Method                      | PSNR (dB) | SAM  | 备注        |
| :--------------- | :-------- | :--- | :---------- |
| manual lstsq |30.29|3.91|-|
| alpha(i,c,nhw) -> (i,c,h',w') -> CNN |10.58|27.17|-|

4. int input, max-normalized output

| Method                      | PSNR (dB) | SAM  | 备注        |
| :--------------- | :-------- | :--- | :---------- |
| $A^TA x = A^T B$ |30.27|4.00|-|
|Orignial |30.27 | 4.00 |-|
|cpu + float64|30.27|4.00|-|
|index计算max换一个dim(index all the same)|30.21|3.92|**Wrong!!**|
|全部换成MTF的下采样方式|30.81|3.88|**Progress**!!|
|MTF的下采样方式 + float64 + cpu|31.03|3.81|**Progress**!!|
|MTF的下采样方式 + float64 + cpu + 去掉pan_hp的繁琐上下采样|31.41|3.80|**Progress**!!|
|MTF的下采样方式 + float64 + cpu + 去掉pan_hp的繁琐上下采样 + 直接直方图匹配 |31.41|3.80|**histogram_matching 没问题**|
|cv2下采样 + float64 + cpu|30.53|3.93|-|

5. MATLAB Code

| Method                      | PSNR (dB) | SAM  | 备注        |
| :-------------------------- | :-------- | :--- | :---------- |
| Original (**1st Image**)|33.10|-|-|
| 去掉繁琐的上下采样 pan_hp (**1st Image**)|33.55|-|-|
| Float Input (**Overall**) |15.51|5.838|可以说结果是错误的|
| Float Input (**Overall**, det=1e-12) |15.51|5.838|-|
| int input, TS=8,OL=0 (**Overall**) |32.27|5.48|-|
| int input, TS=4,OL=0 (**Overall**) |31.9|5.79|-|

- **奇异值检查对结果根本没有影响!!**

### SR-D Network

1. Networks

| Method                      | PSNR (dB) | SAM  | 备注        |
| :-------------------------- | :-------- | :--- | :---------- |
|float32|33.56|3.90|Same|
|TS=8,OL=0|33.21|4.43|-|
|TS=8,OL=0, Non-int input|33.21|4.43|-|
|Conv2d(i,c,h,w alpha x4 upsampled input, hidden_dim=64) |29.99|4.68|-|
|Conv2d(c,i,h,w alpha x4 upsampled input, hidden_dim=64) |30.21|4.26|-|
|Conv3d(i,c,h,w alpha x4 upsampled input, hidden_dim=32) |27...|4...|-|

2. Original Improvement (TS = 8, ol=0)

| Method                      | PSNR (dB) | SAM  | 备注        |
| :-------------------------- | :-------- | :--- | :---------- |
|ms_hp from ms |28...|-|-|
|ms_hp from ms, pan match to ms |27.89|9.16|-|
|n_atoms = 20|33.44|-|-|
|n_atoms = 15|33.50|-|-|
|n_atoms = 8|33.55|-|-|
|n_atoms = 12|33.54|-|-|
|ms = ms - mtf(ms, ratio=1)|33.74|3.80|-|
|ms = ms - mtf(ms, ratio=1, N=11)|33.78|3.74|-|

### GNet + SR_D

| Method                      | PSNR (dB) | SAM  | 备注        |
| :-------------------------- | :-------- | :--- | :---------- |
|GNet -> SR_D for residual part|33.58 -> 33.78|3.49 -> 3.75| Better Spatial fidelity, less Spectral info |

## Overall Performace

### Netwoek Improvement

1. ConvNext: 总体采用MS2PAN中选用的最佳SAM的方法

| 结构 | PSNR | SAM |
|:-|:-|:-|
| 原版 | 36.53 | 2.65 |
| LeakyReLU, 改成PReLu |35.36 | 2.76 |
| LeakyReLU 改成GeLu |36.34 | 2.72 |
|hidden_factor=4|36.07|2.78|

2. Initialization

| Config | PSNR | SAM |
|:-|:-|:-|
| Original |36.38 | 2.75 |
| ConvNext changed to ResidualNet|36.45|2.78|
|psz: 4,8|36.17|2.76|
|psz: 4,4|35.44|2.85|
|lambda: 4,1,8|36.52|2.71|
|lambda: 4,2,8|36.48|2.70|
|lambda: 4,0,8|Sucks!!!|-|
|lambda: 4,4,8|36.32|2.77|
|lambda: 2,1,3|36.18|2.72|
|lambda: 4,1,4|36.19|2.73|
|1000 + 100 + 500|36.48|2.72|

### 预训练效果

| 模型/配置                                | 训练阶段/轮数         | PSNR (dB) |
| :--------------------------------------- | :-------------------- | :-------- |
| UNet(4层) + NSA                          | 14轮                  | 32.2      |
| 纯 UNet(5层)                             | 14轮                  | 32.5      |
| MRA + 纯 UNet(5层) (UNet阶段)            | 15轮 (UNet 2轮)       | 30.39     |
| MRA + 纯 UNet(5层) (NSA阶段)             | 15轮 (NSA 5轮)        | 30.38     |
| UNet(5层) SS                             | 3阶段 (300+40+600)    | 36.5      |

### MRA 效果

| 配置                                                | PSNR (dB) (UNet后) | PSNR (dB) (UNet前) | 备注                             |
| :-------------------------------------------------- | :----------------- | :----------------- | :------------------------------- |
| Conv3x3(64) -> ConvNext(64, 2) -> Conv3x3(64)       | 最高 35.8          | 最高 33.72         | 约100轮收敛到 33.5dB 附近        |
| g 换 bicubic 上采样                                 | -                  | -                  | 改变了上采样方式                 |

### ConvNext V1 vs V2 对比

| 模型        | 性能      | Zero-shot 性能 | 速度       |
| :---------- | :-------- | :------------- | :--------- |
| ConvNext V1 | 接近 V2   | 更好           | 比 V2 快   |
| ConvNext V2 | 接近 V1   | 稍差           | 比 V1 慢   |

### 其他观察

| 观察项                                  | 备注/影响                          |
| :-------------------------------------- | :--------------------------------- |
| 4x4 下采样                              | 效果不佳                           |
| DynamicTanh 替代 ConvNeXt 的 LayerNorm | 效果稍差 (带 weight+bias 表现更好) |

### 网络初始化 (WV3 Reduced)

| 配置                                        | GPU      | PSNR (dB) | SAM  |
| :------------------------------------------ | :------- | :-------- | :--- |
| SS (1000+40+600)                            | 2080Ti   | 35.95     | 2.92 |
| SS (1000+40+600) + gnet 输出拼接 PAN        | 2080Ti   | 35.52     | 2.83 |

### Start with right MTF!!!

| Model        | PSNR | SAM | Others |
| :---------- | :-------- | :------------- | :--------- |
|main lr = 2e-3|35.23|2.88|-|
|main lr = 4e-3|35.39|2.82|-|
|pan_pred_net lr = 2e-3|35.46|2.83|-|
|pan_pred_net(dim=32)|35.14|2.82|-|

## ideas

### Data Aug - Tencrop

| 结构 | PSNR | SAM |
|:-|:-|:-|
|-|35.98|2.86|
|MS2PAN (dim=32)|36.04|2.90|
|MS2PAN (dim=32), UNet(dim=64)|36.00|2.87|

### RAO

| 结构 | PSNR | SAM |
|:-|:-|:-|
|p=0.8|35.83|2.79|

## MTF

### Early Experiment

| Experiment Configuration                                 | Result/Observation                 | Details                    |
| :------------------------------------------------------- | :--------------------------------- | :------------------------- |
| Gnyq -> INR -> Gnyq -> Kernel                            | Highest ~35dB                      | Trained with gnet or UNet  |
| Kernel -> INR -> Kernel                                  | Negative effect                    | -                          |
| coord -> INR + kernel_center -> kernel                   | Overall training skewed            | -                          |
| coord -> 8*INR -> kernel                                 | Pre-training fine                  | -                          |
| (bhw,c) -> 1x1conv mimicking parallel INR                | Pre-training fine, overall training skewed | -                          |
| Added Kernel_centering layer                             | Worse effect                       | -                          |
| Gynq -> Linear -> Sigmoid -> Gnyq -> Kernel              | Effective                          | Trained separately         |

### 训练策略对比 (850轮, 3090) & 交替训练 (4090)

| 训练策略                                              | PSNR (dB) | SAM  | 备注             |
| :---------------------------------------------------- | :-------- | :--- | :--------------- |
| 分阶段训练                                            | 35.23     | -    | -                |
| 后期联合训练                                          | 34.98     | -    | -                |
| 全程联合训练                                          | 33.64     | -    | -                |
| repeat(backbone -> backbone + kern_estimate), 低 lr, 850轮 | 35.50     | -    | 后期进展缓慢     |
| repeat(kern_estimate(90) -> backbone(10)), 500轮        | 35.84     | 2.74 | 最终核与原核几乎一样 |
| repeat(kern_estimate(90) -> backbone(10)), 500轮        | 35.92     | 2.74 | 最终核与原核几乎一样 |

### SS vs MTF 训练流程对比

1. 4090

| 训练流程 | 轮数 | Lambda | PSNR (dB) | SAM | 备注 |
|:---|:---|:---|:---|:---|:---|
| MTF_Train (5阶段) | 500 | - | 35.95, 35.70, 35.87 | - | - |
| MTF_Train (4阶段, 不训MTF) | 500 | - | 35.68, 35.81, 35.75 | - | - |
| SS (3阶段) | 600 | - | 36.09, 36.08 | 2.73, 2.71 | - |
| MTF_Train (5阶段) | 600 | - | 36.00, 36.14 | 2.75, 2.80 | - |
| SS (3阶段) | 600 | 2 | 36.15, 36.32 | 2.72, 2.72 | - |
| MTF_Train (5阶段) | 600 | 2 | 36.48, 36.17 | 2.70, 2.73 | - |
| SS (300+40+600, 3090) | - | 8 (fp16) | 35.41 | 2.92 | - |
| SS (300+40+600, 3090) | - | 8 (fp32) | 35.23 | 2.93 | - |
| SS_Train (有 mtf optimizer) | - | - | 35.53, 35.43, 35.48 | - | lr: 1e-3 -> 5e-4, mtfnet 的存在会影响主网络优化 |
| SS_Train (无 mtf optimizer, 无 mtfnet) | - | - | 35.95, 35.71, 35.87 | - | lr: 1e-3 -> 5e-4, mtfnet 的存在会影响主网络优化 |
| SS_Train (无 mtf optimizer, 有 mtfnet) | - | - | 35.24, 35.40 | - | lr: 1e-3 -> 5e-4, mtfnet 的存在会影响主网络优化 |
| SS_Train (无 mtf optimizer, 无 mtfnet) | 500 | - | 36.18, 35.89, 35.84 | - | backbone_lr: 3e-3 -> 5e-4, pan_pred_net: 2.5e-3 -> 5e-4, 调整学习率后问题解决 |
| SS_Train (无 mtf optimizer, 有 mtfnet) | 500 | - | 35.75, 35.85, 35.82 | - | backbone_lr: 3e-3 -> 5e-4, pan_pred_net: 2.5e-3 -> 5e-4, 调整学习率后问题解决 |
| MTF_Train (4阶段, 不训 MTF) | 500 | - | 35.68, 35.81, 35.75 | - | backbone_lr: 3e-3 -> 5e-4, pan_pred_net: 2.5e-3 -> 5e-4, 调整学习率后问题解决 |

2. Ellipse (500+100+600+1000+100)

- GNyq -> Net -> GNyq -> sigmax, sigma_y

| Model | PSNR (dB) | SAM  | Notes |
| :--------- | :-------- | :--- | :------------------ |
|SS|36.40|2.69|-|
|linear->LeakyReLU->linear->relu,dim=128|36.61|2.78|ms psnr = 57.72, **Very Unstable**|

### MBFE和我们的方法的比较

| 方法       | 数据集/图像             | 指标     | 值 (dB) | 参数             |
| :--------- | :---------------------- | :------- | :------ | :--------------- |
| 我们的方法 | WV3 Reduced 第一张图 MS | MS PSNR  | 60.42   | -                |
| MBFE       | WV3 Reduced 第一张图 MS | MS PSNR  | 42.64   | lambda, mu = 1e1, sigma=5.0 |
| MBFE       | WV3 Reduced 第一张图 MS | MS PSNR  | 43.90   | lambda, mu = 5.0, sigma=1.0   |
| MBFE       | WV3 Reduced 第一张图 MS | MS PSNR  | **44.24**   | **lambda, mu = 2.0, sigma=1.0**   |
| MBFE       | WV3 Reduced 第一张图 MS | MS PSNR  | 44.14   | lambda, mu = 1.2, sigma=1.0   |
| MBFE       | WV3 Reduced 第一张图 MS | MS PSNR  | 43.71   | lambda, mu = 1.2, sigma=5.0   |
| MBFE       | WV3 Reduced 第一张图 MS | MS PSNR  | 43.65   | lambda, mu = 1, sigma=5.0   |
| MBFE       | WV3 Reduced 第一张图 MS | MS PSNR  | 43.56   | lambda, mu = 0.8, sigma=5.0   |
| MBFE       | WV3 Reduced 第一张图 MS | MS PSNR  | 42.06   | lambda, mu = 1e-1, sigma=5.0 |
| MBFE       | WV3 Reduced 第一张图 MS | MS PSNR  | 38.25   | lambda, mu = 1e-4, sigma=5.0 |

## Loss 函数测试

- **组合效果 (540轮)**

| Loss Combination          | Patch Size | PSNR (dB) |
| :------------------------ | :--------- | :-------- |
| pwmse + patch sam + ergas | (4, 4)     | 35.92     |
| pwmse + patch sam        | (4, 2)     | 36.14     |
| pwmse + sam               | (4, 4)     | 35.70     |

- pwmse 不使用归一化会好一些

- 要求和，不要取平均

## 训练策略与超参数

*   **Epoch 测试 (MainNet 3阶段, 4090, 600轮)**

    | Backbone LR       | Pan Pred LR       | PSNR (dB) | 备注      |
    | :---------------- | :---------------- | :-------- | :-------- |
    | 1e-3 -> 5e-4      | 1e-3 -> 5e-4      | 35.46     |           |
    | 1e-3 -> 2e-4      | 1e-3 -> 2e-4      | 33.34     |           |
    | 1e-3 -> 8e-4      | 1e-3 -> 8e-4      | 35.61, 35.34 | SAM 很高 |
    | 1e-3 -> 9e-4      | 1e-3 -> 9e-4      | 35.40     |           |
    *   *结论: 600轮 + 8e-4 较好, 但轮数少 pan 差*

*   **调度器策略**
    *   Backbone: 前期不变，后期 Cosine 最好 (4e-3 -> 5e-4)
    *   Pan_pred_net: 预训练快一些，联合训练用 cosine (2.5e-3 -> 1e-3)
    *   Pan_pred_net lr (-> 1e-3 好一点):
        *   -> 5e-4: 35.99dB, 36.03dB, 35.96dB
        *   -> 1e-3: 35.86dB, 35.94dB, 36.17dB, 36.05dB

## WV3 FR

-  **Gnet**: ms=40dB, pan=35dB (原版), HQNR 总是 0.949 (与第一阶段无关)

### UNet (SS 训练)

    | Epochs        | GPU    | UNet Config | MS2PAN | Cosine Pt | Lambda | MS PSNR | PAN PSNR |
    | :------------ | :----- | :---------- | :----- | :-------- | :----- | :------ | :------- |
    | 300+100+600   | 2080Ti | 5层, dim=32 | -      | -         | -      | 47.93   | 36.41    |
    | 300+100+600   | 2080Ti | 6层, dim=32 | -      | -         | -      | 46.91   | 36.53    |
    | 300+100+600   | 2080Ti | 6层, dim=64 | -      | -         | -      | 49.82   | 35.45    |
    | 300+100+600   | 2080Ti | 6层, dim=32 | (2)    | -         | -      | 47.90   | 38.56    |
    | 1000+100+600  | 2080Ti | 6层, dim=64 | (1)    | -         | -      | 49.50   | 40.82    |
    | 1000+100+700  | 2080Ti | 6层, dim=64 | (1)    | -         | -      | 50.23   | 41.61    |
    | 1000+100+1000 | 2080Ti | 6层, dim=64 | (1)    | -         | -      | 52.53   | 43.04    |
    | 1500+100+2000 | 2080Ti | 6层, dim=64 | (1)    | 0.65      | -      | 55.96   | 46.18    |
    | 1500+100+3000 | 2080Ti | 6层, dim=64 | (1)    | 0.65      | -      | 57.45   | 48.73    |
    | 1500+100+2000 | 2080Ti | 6层, dim=64 | (1)    | 0.65      | 8      | 50.53   | 54.4     |
    | 1500+100+2000 | 2080Ti | 6层, dim=64 | (1)    | 0.65      | 16     | 46.87   | 56.64    |
    | 1500+100+2000 | 2080Ti | 6层, dim=64 | (1)    | 0.65      | 8 (no sam) | 44.04 | 57.61 |

### UNet (MTF 训练)

    | Epochs              | GPU    | UNet Config | MS2PAN | Cosine Pt | Lambda | MS PSNR | PAN PSNR |
    | :------------------ | :----- | :---------- | :----- | :-------- | :----- | :------ | :------- |
    | 1000+100+900+300+100 | 2080Ti | 6层, dim=64 | (1)    | -         | -      | 52.67   | 42.75    |
    | 1500+100+2900+300+100| 2080Ti | 6层, dim=64 | (1)    | 0.65      | -      | 57.79   | 49.15    |
    | 1500+100+2900+300+100| 4090   | 6层, dim=64 | (1)    | 0.65      | -      | 53.74   | 55.87    |

### New Test (2025.10)

- 1st image (**Wrong Python Indication**)

| Model       | HQNR | D_lambda_K  | D_s |
| :--------- | :-------- | :--- | :------------------ |
|Original|0.954|0.007|0.040|
|500 + 300 + 600|0.952|0.007|0.042|
|300 + 300 + 600|0.949|0.007|0.044|
|1000 + 100 + 600|0.953|0.006|0.041|
|ms_sam_psz=4, ms_l2_psz=2, pan_psz=4|0.950|0.009|0.042|
|ms_psz=4, pan_psz=8|0.950|0.008|0.043|
|MS2PAN(dim=32)|0.951|0.006|0.043|
|SR_D init|0.931 | 0.01 | 0.06 |
|SR_D post|0.952 | 0.007 | 0.041|
|SR_D post (gt_pred as lms)|0.875|0.021|0.106|
|SR_D post (gt_pred as lms, ms_hp obtained from gt_pred MTF downsample)|0.828|0.037|0.140|

- 1st image (**Wrong Python Indication**, 1000+100+600)

| Model       | HQNR | D_lambda_K  | D_s |
| :--------- | :-------- | :--- | :------------------ |
|Original|0.954|0.007|0.040|
|lambda=2,2,4|0.954|0.006|0.039|
|lambda=2,3,4|0.955|0.006|0.039|
|lambda=3,3,4|0.955|0.006|0.040|
|lambda=1,3,4|0.953|0.007|0.041|
|lambda=2,4,4|0.952|0.007|0.041|

- Full WV3 (**Right MATLAB Indication**)

| Model       | HQNR | D_lambda_K  | D_s |
| :--------- | :-------- | :--- | :------------------ |
|Original | 0.952 | 0.006 | 0.043 |

## Lambda 和 Patch Size 测试 (Full Resolution, HQNR 指标)

*   **Lambda 对比 (SS, 4090, UNet5层, dim=32, 500+100+300 epochs)**

    | Lambda | HQNR  | $D_{\lambda}$ | $D_s$  |
    | :----- | :---- | :----------- | :---- |
    | 32     | 0.942 | 0.041        | 0.018 |
    | 20     | 0.937 | 0.052        | 0.012 |
    | 16     | 0.971 | 0.018        | 0.012 |
    | 12     | 0.968 | 0.016        | 0.017 |
    | 8      | 0.967 | 0.013        | 0.021 |
    | 4      | 0.957 | 0.013        | 0.030 |
    | 2      | 0.958 | 0.010        | 0.032 |
    *   *结论: lambda=16 最好*

*   **Patch Size 对比 (SS, 4090, UNet5层, dim=32, 500+100+300 epochs, lambda=1,4,16)**

    | MS Size | PAN Size | HQNR  | $D_{\lambda}$ | $D_s$  |
    | :------ | :------- | :---- | :----------- | :---- |
    | 4       | 8        | 0.973 | 0.014        | 0.013 |
    | 2       | 8        | 0.965 | 0.011        | 0.025 |
    | 2       | 4        | 0.969 | 0.011        | 0.021 |
    | 8       | 8        | 0.966 | 0.026        | 0.009 |
    *   *结论: ms_size=4, pan_size=8 最好*

*   **Lambda 组合对比 (SS, 4090, UNet5层, dim=32, 500+100+300 epochs)**

    | Lambda (ms_sam, ms_mse, pan_mse) | HQNR  | $D_{\lambda}$ | $D_s$  |
    | :---------------------------------- | :---- | :----------- | :---- |
    | 1, 2, 16                            | 0.969 | 0.016        | 0.015 |
    | 1, 4, 16                            | 0.973 | 0.014        | 0.013 |
    | 1, 8, 16                            | 0.969 | 0.013        | 0.018 |
    | 4, 1, 16                            | 0.962 | 0.013        | 0.026 |
    | 1, 4, 16 -> 1, 4, 32                | 0.970 | 0.015        | 0.015 |
    *   *结论: lambda=1,4,16 最好*

## Epoch 测试 (Full Resolution, HQNR 指标)

1. **SS 训练 (4090, UNet5层, dim=32)**

- 本次实验的**HQNR**计算方式是错误的

| Epochs (S1+S2+S3) | Lambda        | HQNR  | $D_{\lambda}$ | $D_s$  |
| :---------------- | :------------ | :---- | :----------- | :---- |
| 500 + 100 + 200   | 1, 4, 16      | 0.975 | 0.018        | 0.007 |
| 500 + 100 + 250   | 1, 4, 16      | 0.975 | 0.015        | 0.010 |
| 500 + 100 + 300   | 16            | 0.971 | 0.018        | 0.012 |
| 500 + 100 + 400   | 16            | 0.966 | 0.015        | 0.019 |
| 500 + 100 + 500   | 16            | 0.964 | 0.013        | 0.023 |
| 500 + 100 + 600   | 16            | 0.967 | 0.012        | 0.022 |
| 500 + 100 + 600   | 1,4,16 -> 32  | 0.965 (1st), 0.942 ± 0.013  | 0.014 (1st), 0.010 ± 0.003  | 0.021 (1st), 0.048 ± 0.011  |
| 1500 + 100 + 600  | 1,4,16 -> 32  | 0.943 ± 0.012  | 0.010 ± 0.003  | 0.048 ± 0.011  |
| 1500 + 100 + 400  | 1,1,8 -> 16   | 0.950 ± 0.011 | 0.009 ± 0.004 | 0.041 ± 0.00   |

2. **MTF 训练 (4090, UNet5层, dim=32)**

| Epochs (S1+S2+S3+S4+S5) | Lambda        | HQNR  | $D_{\lambda}$ | $D_s$  |
| :---------------------- | :------------ | :---- | :----------- | :---- |
| 500+100+200+300+50      | 1,4,16 -> 32  | 0.973 | 0.017        | 0.010 |
| 500+100+250+300+50      | 1,4,16 -> 32  | 0.973 | 0.014        | 0.012 |
| 500+100+200+300+100     | 16            | 0.968 | 0.016        | 0.016 |
| 500+100+200+300+100     | 16 -> 32      | 0.972 | 0.020        | 0.007 |
| 500+100+300+300+100     | 16            | 0.965 | 0.013        | 0.021 |
| 500+100+500+300+100     | 16            | 0.961 | 0.011        | 0.028 |
| 1500+100+200+300+100    | 1, 1, 8 -> 16    | 0.950 ± 0.012 |0.011 ± 0.004 |0.040 ± 0.011 |
| **1500+100+300+300+100**    | 1, 1, 8       | 0.951 ± 0.010  |  0.008 ± 0.003  | 0.042 ± 0.009  |
| 1500+100+300+300+100    | 1, 1, 12      | 0.948 ± 0.012 | 0.011 ± 0.004 | 0.042 ± 0.011 |
| 1500+100+300+300+100    | 1, 1, 8 -> 16 | 0.949 ± 0.012 | 0.009 ± 0.003 | 0.042 ± 0.011 |
| 1500+100+400+300+100    | 1,1,16 -> 32  | 0.943 ± 0.015  | 0.013 ± 0.004  | 0.045 ± 0.012  |
| 1500+100+500+300+100    | 1,4,16 -> 32  | 0.942 ± 0.014  | 0.010 ± 0.003  | 0.048 ± 0.012  |

-   **MTF 训练 (4090, UNet6层, dim=64)**

    | Epochs (S1+S2+S3+S4+S5) | Lambda        | HQNR  | $D_{\lambda}$ | $D_s$  |
    | :---------------------- | :------------ | :---- | :----------- | :---- |
    | 1500+100+500+300+100    | 1,1,8  | 

# Overall Conclusion

1. INT metrics and Float metrics make no difference in **pytorch, except PSNR**

2. Ratio for MTF is an important factor

- If you **want to downsample in the end**, use the ratio > 1; or take it as 1

# Best Hyper-Params

## WV3-Reduced

| 参数类别              | 参数项                               | 值                             |
|-----------------------|--------------------------------------|--------------------------------|
| 训练参数 (TrainArgs)  | 学习率 (lr) - backbone             | `4e-3`                         |
| 训练参数 (TrainArgs)  | 学习率 (lr) - pan_pred_net         | `2.5e-3`                       |
| 训练参数 (TrainArgs)  | 学习率 (lr) - mtfnet               | `1e-3`                         |
| 训练参数 (TrainArgs)  | 学习率 (lr) - mranet               | `2e-3`                         |
| 训练参数 (TrainArgs)  | 最小学习率 (min_lr) - backbone     | `5e-4`                         |
| 训练参数 (TrainArgs)  | 最小学习率 (min_lr) - pan_pred_net | `1e-3`                         |
| 训练参数 (TrainArgs)  | 最小学习率 (min_lr) - mranet       | `1e-3`                         |
| 训练参数 (TrainArgs)  | 调度器类型 (scheduler_type) - backbone | `"flat_down"`                  |
| 训练参数 (TrainArgs)  | 调度器类型 (scheduler_type) - pan_pred_net | `"cosine"`                     |
| 训练参数 (TrainArgs)  | 调度器类型 (scheduler_type) - mranet | `"flat_down"`                  |
| 训练参数 (TrainArgs)  | Epochs (epoch) - stageI              | `1000`                         |
| 训练参数 (TrainArgs)  | Epochs (epoch) - stageII             | `40`                           |
| 训练参数 (TrainArgs)  | Epochs (epoch) - stageIII            | `800`                          |
| 训练参数 (TrainArgs)  | Epochs (epoch) - stageIV             | `500`                          |
| 训练参数 (TrainArgs)  | Epochs (epoch) - stageV              | `100`                          |
| 训练参数 (TrainArgs)  | Cosine 调度点 (cosine_point)       | `0.8`                          |
| 训练参数 (TrainArgs)  | Flat 调度点 (flat_point)           | `0.2`                          |
| 训练参数 (TrainArgs)  | 混合精度 (mixed)                     | `False`                        |
| 模型参数 | UNet dim| 32|
| Loss参数  | lambda_mssam, lambda_msmse, lambda_panmse | 1,1,8                            |
| Loss参数| ms_patch_size, pan_patch_size | 2, 4 |

# MAYBE TODO

| 改进点               | 备注/影响                                    |
| :------------------- | :------------------------------------------- |
| GNN 邻居选取方法     | 每个 patch 的邻居数应当是不一样的，可以手动设限，来改进邻居选取方法。 |
| GNN 特征提取的新方法 | 保留这个想法。                               |
| GNN 特征卷处理方案   | 在特征卷小以后把特征和邻居拼起来是个糟糕的方案。 |
| UNet Conv 改成 RNN 形式 | 效果：单图↓。                                |
| UNet Sparse-Skip     | 效果：单图↓。                                |
| UNet 1x1Conv 改为 tiny-cuda-nn | -                                            |
| UNet canconv         | 不行                                         |
| UNet 多 Group 或者 Span Conv | 效果：单图↓。                                |
| UNet 使用 ConvTranspose | 效果：单图↓。                                |
| UNet 上采样感受野    | 理解很重要。                                 |
| UNet 结尾+Attn (NSAAttn) | 可以试试预训练效果怎么样。                   |
| WTConv 改进          | 单图↓，保留这个想法。                        |
| CanConv 核和偏置生成方式 | kernel 和 bias 生成方式很重要。              |
| 优化器 Muon          | 目前看来对小batch的训练并不友好。            |
|采用逐层递减的初始化方式||

# Paper

## TODO

- 实验部分要增加内容(实验细节，参数，实验过程，更多的解释和结果)

- 丰富文章内容

- Loss证明*(定性就可以了)

1. Task1

- Related Work

- Methods

- CV 

- Recall Track Program 

2. Task2

- Prior

- Network Init

- 比一篇最新的

## Experience

- 不是自己的东西尽量一笔带过
