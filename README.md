# [RepNeXt: A Fast Multi-Scale CNN using Structural Reparameterization](https://arxiv.org/abs/2406.16004)

[![license](https://img.shields.io/github/license/suous/RepNeXt)](https://github.com/suous/RepNeXt/blob/main/LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2406.16004-red)](https://arxiv.org/abs/2406.16004)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/suous/RepNeXt/blob/main/demo/feature_map_visualization.ipynb)

<p align="center">
  <img src="figures/latency.png" width=70%> <br>
  The top-1 accuracy is tested on ImageNet-1K and the latency is measured by an iPhone 12 with iOS 16 across 20 experimental sets.
</p>

[RepNeXt: A Fast Multi-Scale CNN using Structural Reparameterization](https://arxiv.org/abs/2406.16004).\
Mingshu Zhao, Yi Luo, and Yong Ouyang
[[`arXiv`](https://arxiv.org/abs/2406.16004)]

![architecture](./figures/architecture.png)

## Abstract

We introduce RepNeXt, a novel model series integrates multi-scale feature representations and incorporates both serial and parallel structural reparameterization (SRP) to enhance network depth and width without compromising inference speed.
Extensive experiments demonstrate RepNeXt's superiority over current leading lightweight CNNs and ViTs, providing advantageous latency across various vision benchmarks.
RepNeXt-M4 matches RepViT-M1.5's 82.3% accuracy on ImageNet within 1.5ms on an iPhone 12, outperforms its AP$^{box}$ by 1.3 on MS-COCO, and reduces parameters by 0.7M.

![transforms](./figures/transforms.png)

<details>
  <summary>
  <font size="+1">Conclusion</font>
  </summary>
    In this paper, we introduced a multi-scale depthwise convolution integrated with both serial and parallel SRP mechanisms, enhancing feature diversity and expanding the network’s expressive capacity without compromising inference speed. 
    Specifically, we designed a reparameterized medium-kernel convolution to imitate the human foveal vision system. 
    Additionally, we proposed our light-weight, general-purpose RepNeXts that employed the distribute-transform-aggregate design philosophy across inner-stage blocks as well as downsampling layers, achieving comparable or superior accuracy-efficiency trade-off across various vision benchmarks, especially on downstream tasks. 
    Moreover, our flexible multi-branch design functions as a grouped-depthwise convolution with additional inductive bias and efficiency trade-offs. 
    It can also be reparameterized into a single-branch large-kernel depthwise convolution, enabling potential optimization towards different accelerators.

For example, the large-kernel depthwise convolution can be accelerated by the implicit GEMM algorithm: `DepthWiseConv2dImplicitGEMM` of [RepLKNet](https://github.com/DingXiaoH/RepLKNet-pytorch).

Many token mixers can be generalized as a **distribute-transform-aggregate** process:

| Token Mixer | Distribution |   Transforms   | Aggregation |
|:-----------:|:------------:|:--------------:|:-----------:|
|  ChunkConv  |    Split     | Conv, Identity |     Cat     |
|  CopyConv   |    Clone     |      Conv      |     Cat     |
|   MixConv   |    Split     |      Conv      |     Cat     |
|    MHSA     |    Split     |      Attn      |     Cat     |
|  RepBlock   |    Clone     |      Conv      |     Add     |

`ChunkConv` and `CopyConv` can be viewed as grouped depthwise convolutions.

- **Chunk Conv**

```python
class ChunkConv(nn.Module):
    def __init__(self, in_channels, bias=True):
        super().__init__()
        self.bias = bias
        in_channels = in_channels // 4
        kwargs = {"in_channels": in_channels, "out_channels": in_channels, "groups": in_channels, "bias": bias}
        self.conv_i = nn.Identity()
        self.conv_s = nn.Conv2d(kernel_size=3, padding=1, **kwargs)
        self.conv_m = nn.Conv2d(kernel_size=7, padding=3, **kwargs)
        self.conv_l = nn.Conv2d(kernel_size=11, padding=5, **kwargs)

    def forward(self, x):
        i, s, m, l = torch.chunk(x, chunks=4, dim=1)
        return torch.cat((self.conv_i(i), self.conv_s(s), self.conv_m(m), self.conv_l(l)), dim=1)

    @torch.no_grad()
    def fuse(self):
        conv_s_w, conv_s_b = self.conv_s.weight, self.conv_s.bias
        conv_m_w, conv_m_b = self.conv_m.weight, self.conv_m.bias
        conv_l_w, conv_l_b = self.conv_l.weight, self.conv_l.bias

        conv_i_w = torch.nn.functional.pad(torch.ones(conv_l_w.shape[0], conv_l_w.shape[1], 1, 1), [5, 5, 5, 5])
        conv_s_w = nn.functional.pad(conv_s_w, [4, 4, 4, 4])
        conv_m_w = nn.functional.pad(conv_m_w, [2, 2, 2, 2])

        in_channels = self.conv_l.in_channels*4
        conv = nn.Conv2d(in_channels, in_channels, kernel_size=11, padding=5, bias=self.bias, groups=in_channels)
        conv.weight.data.copy_(torch.cat((conv_i_w, conv_s_w, conv_m_w, conv_l_w), dim=0))

        if self.bias:
            conv_i_b = torch.zeros_like(conv_s_b)
            conv.bias.data.copy_(torch.cat((conv_i_b, conv_s_b, conv_m_b, conv_l_b), dim=0))
        return conv
```

- **Copy Conv**

```python
class CopyConv(nn.Module):
    def __init__(self, in_channels, bias=True):
        super().__init__()
        self.bias = bias
        kwargs = {"in_channels": in_channels, "out_channels": in_channels, "groups": in_channels, "bias": bias, "stride": 2}
        self.conv_s = nn.Conv2d(kernel_size=3, padding=1, **kwargs)
        self.conv_l = nn.Conv2d(kernel_size=7, padding=3, **kwargs)
         
    def forward(self, x):
        B, C, H, W = x.shape
        s, l = self.conv_s(x), self.conv_l(x)
        return torch.stack((s, l), dim=2).reshape(B, C*2, H//2, W//2)

    @torch.no_grad()
    def fuse(self):
        conv_s_w, conv_s_b = self.conv_s.weight, self.conv_s.bias
        conv_l_w, conv_l_b = self.conv_l.weight, self.conv_l.bias

        conv_s_w = nn.functional.pad(conv_s_w, [2, 2, 2, 2])

        in_channels = self.conv_l.in_channels
        conv = nn.Conv2d(in_channels, in_channels*2, kernel_size=7, padding=3, bias=self.bias, stride=self.conv_l.stride, groups=in_channels)
        conv.weight.data.copy_(torch.stack((conv_s_w, conv_l_w), dim=1).reshape(conv.weight.shape))

        if self.bias:
            conv.bias.data.copy_(torch.stack((conv_s_b, conv_l_b), dim=1).reshape(conv.bias.shape))
        return conv
```

In summary, by focusing solely on the simplicity of the model’s overall architecture and disregarding its efficiency and parameter count, we can ultimately consolidate it into the single-branch structure shown in the figure below:

![equivalent](./figures/equivalent.png)
</details>

<br/>

**UPDATES** 🔥
- **2024/10/13**: Added M0-M2 single branch equivalent form ImageNet-1K results using StarNet's training recipe.
- **2024/09/19**: Added M0-M2 ImageNet-1K results using StarNet's training recipe (distilled). Hit 80.6% top-1 accuracy within 1ms on an iPhone 12.
- **2024/09/08**: Added RepNext-M0 ImageNet-1K result using StarNet's training recipe. Achieving 73.8% top-1 accuracy without distillation.
- **2024/08/26**: RepNext-M0 (distilled) has been released, achieving 74.2% top-1 accuracy within 0.6ms on an iPhone 12.
- **2024/08/23**: Finished compact model (M0) ImageNet-1K experiments.
- **2024/07/23**: Updated readme about further simplified model structure.
- **2024/06/25**: Uploaded checkpoints and training logs of RepNext-M1 - M5.

## Classification on ImageNet-1K

### Models under the RepVit training strategy

We report the top-1 accuracy on ImageNet-1K with and without distillation using the same training strategy as [RepViT](https://github.com/THU-MIG/RepViT).

| Model | Top-1(distill) / Top-1 | #params | MACs | Latency |                                                                                                 Ckpt                                                                                                 |                                               Core ML                                               |                                                   Log                                                   |
|:------|:----------------------:|:-------:|:----:|:-------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------:|
| M0    |      74.2 \| 72.6      |  2.3M   | 0.4G | 0.59ms  | [fused 300e](https://github.com/suous/RepNeXt/releases/download/v1.0/repnext_m0_distill_300e_fused.pt) / [300e](https://github.com/suous/RepNeXt/releases/download/v1.0/repnext_m0_distill_300e.pth) | [300e](https://github.com/suous/RepNeXt/releases/download/v1.0/repnext_m0_distill_300e_224.mlmodel) | [distill 300e](./logs/repnext_m0_distill_300e.txt) / [300e](./logs/repnext_m0_without_distill_300e.txt) |
| M1    |      78.8 \| 77.5      |  4.8M   | 0.8G | 0.86ms  | [fused 300e](https://github.com/suous/RepNeXt/releases/download/v1.0/repnext_m1_distill_300e_fused.pt) / [300e](https://github.com/suous/RepNeXt/releases/download/v1.0/repnext_m1_distill_300e.pth) | [300e](https://github.com/suous/RepNeXt/releases/download/v1.0/repnext_m1_distill_300e_224.mlmodel) | [distill 300e](./logs/repnext_m1_distill_300e.txt) / [300e](./logs/repnext_m1_without_distill_300e.txt) |
| M2    |      80.1 \| 78.9      |  6.5M   | 1.1G | 1.00ms  | [fused 300e](https://github.com/suous/RepNeXt/releases/download/v1.0/repnext_m2_distill_300e_fused.pt) / [300e](https://github.com/suous/RepNeXt/releases/download/v1.0/repnext_m2_distill_300e.pth) | [300e](https://github.com/suous/RepNeXt/releases/download/v1.0/repnext_m2_distill_300e_224.mlmodel) | [distill 300e](./logs/repnext_m2_distill_300e.txt) / [300e](./logs/repnext_m2_without_distill_300e.txt) |
| M3    |      80.7 \| 79.4      |  7.8M   | 1.3G | 1.11ms  | [fused 300e](https://github.com/suous/RepNeXt/releases/download/v1.0/repnext_m3_distill_300e_fused.pt) / [300e](https://github.com/suous/RepNeXt/releases/download/v1.0/repnext_m3_distill_300e.pth) | [300e](https://github.com/suous/RepNeXt/releases/download/v1.0/repnext_m3_distill_300e_224.mlmodel) | [distill 300e](./logs/repnext_m3_distill_300e.txt) / [300e](./logs/repnext_m3_without_distill_300e.txt) |
| M4    |      82.3 \| 81.2      |  13.3M  | 2.3G | 1.48ms  | [fused 300e](https://github.com/suous/RepNeXt/releases/download/v1.0/repnext_m4_distill_300e_fused.pt) / [300e](https://github.com/suous/RepNeXt/releases/download/v1.0/repnext_m4_distill_300e.pth) | [300e](https://github.com/suous/RepNeXt/releases/download/v1.0/repnext_m4_distill_300e_224.mlmodel) | [distill 300e](./logs/repnext_m4_distill_300e.txt) / [300e](./logs/repnext_m4_without_distill_300e.txt) |
| M5    |      83.3 \| 82.4      |  21.7M  | 4.5G | 2.20ms  | [fused 300e](https://github.com/suous/RepNeXt/releases/download/v1.0/repnext_m5_distill_300e_fused.pt) / [300e](https://github.com/suous/RepNeXt/releases/download/v1.0/repnext_m5_distill_300e.pth) | [300e](https://github.com/suous/RepNeXt/releases/download/v1.0/repnext_m5_distill_300e_224.mlmodel) | [distill 300e](./logs/repnext_m5_distill_300e.txt) / [300e](./logs/repnext_m5_without_distill_300e.txt) |

### Models under the StarNet training strategy

We report the top-1 and top-5 accuracy on ImageNet-1K with and without distillation using the same training strategy as [StarNet](https://github.com/ma-xu/Rewrite-the-Stars).

> Models decorated with “E” denote the single-branch equivalent form of RepNeXts through channel-wise structural reparameterization.

| Model | Params (M) | MACs | Latency | Top-1 / top-5 |                                                                                                                     Download                                                                                                                      | Top-1 / top-5 (distill) |                                                                                                                                 Download                                                                                                                                  |
|:------|:----------:|:----:|:-------:|:-------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| M0    |    2.3     | 0.4G | 0.59ms  | 73.8 \| 91.6  | [args](./logs/strategy/repnext_m0_sz224_4xbs512_ep300_args.yaml) \| [log](./logs/strategy/repnext_m0_sz224_4xbs512_ep300_summary.csv) \| [model]( https://github.com/suous/RepNeXt/releases/download/v1.0/repnext_m0_sz224_4xbs512_ep300.pth.tar) |      75.4 \| 92.1       | [args](./logs/strategy/repnext_m0_sz224_4xbs256_ep300_distill_args.yaml) \| [log](./logs/strategy/repnext_m0_sz224_4xbs256_ep300_distill_summary.csv) \| [model]( https://github.com/suous/RepNeXt/releases/download/v1.0/repnext_m0_sz224_4xbs256_ep300_distill.pth.tar) |
| M0E   |    2.5     | 0.5G | 0.62ms  | 73.9 \| 91.9  |                                                      [args](./logs/strategy/repnext_m0_sz224_4xbs512_ep300_args.yaml) \| [log](./logs/strategy/repnext_m0e_sz224_4xbs512_ep300_summary.csv)                                                       |      75.6 \| 92.3       |                                                          [args](./logs/strategy/repnext_m0_sz224_4xbs256_ep300_distill_args.yaml) \| [log](./logs/strategy/repnext_m0e_sz224_4xbs256_ep300_distill_summary.csv)                                                           |
| M1    |    4.8     | 0.8G | 0.86ms  | 77.9 \| 94.0  | [args](./logs/strategy/repnext_m1_sz224_4xbs512_ep300_args.yaml) \| [log](./logs/strategy/repnext_m1_sz224_4xbs512_ep300_summary.csv) \| [model]( https://github.com/suous/RepNeXt/releases/download/v1.0/repnext_m1_sz224_4xbs512_ep300.pth.tar) |      79.7 \| 94.5       | [args](./logs/strategy/repnext_m1_sz224_4xbs256_ep300_distill_args.yaml) \| [log](./logs/strategy/repnext_m1_sz224_4xbs256_ep300_distill_summary.csv) \| [model]( https://github.com/suous/RepNeXt/releases/download/v1.0/repnext_m1_sz224_4xbs256_ep300_distill.pth.tar) |
| M1E   |    5.3     | 1.0G | 0.89ms  | 77.7 \| 93.9  |                                                      [args](./logs/strategy/repnext_m1_sz224_4xbs512_ep300_args.yaml) \| [log](./logs/strategy/repnext_m1e_sz224_4xbs512_ep300_summary.csv)                                                       |      79.5 \| 94.5       |                                                          [args](./logs/strategy/repnext_m1_sz224_4xbs256_ep300_distill_args.yaml) \| [log](./logs/strategy/repnext_m1e_sz224_4xbs256_ep300_distill_summary.csv)                                                           |
| M2    |    6.5     | 1.1G | 1.00ms  | 78.8 \| 94.5  | [args](./logs/strategy/repnext_m2_sz224_4xbs512_ep300_args.yaml) \| [log](./logs/strategy/repnext_m2_sz224_4xbs512_ep300_summary.csv) \| [model]( https://github.com/suous/RepNeXt/releases/download/v1.0/repnext_m2_sz224_4xbs512_ep300.pth.tar) |      80.6 \| 95.1       | [args](./logs/strategy/repnext_m2_sz224_4xbs256_ep300_distill_args.yaml) \| [log](./logs/strategy/repnext_m2_sz224_4xbs256_ep300_distill_summary.csv) \| [model]( https://github.com/suous/RepNeXt/releases/download/v1.0/repnext_m2_sz224_4xbs256_ep300_distill.pth.tar) |
| M2E   |    7.0     | 1.3G | 1.04ms  | 78.8 \| 94.3  |                                                      [args](./logs/strategy/repnext_m2_sz224_4xbs512_ep300_args.yaml) \| [log](./logs/strategy/repnext_m2e_sz224_4xbs512_ep300_summary.csv)                                                       |      80.6 \| 95.2       |                                                          [args](./logs/strategy/repnext_m2_sz224_4xbs256_ep300_distill_args.yaml) \| [log](./logs/strategy/repnext_m2e_sz224_4xbs256_ep300_distill_summary.csv)                                                           |

Model evaluation:

```bash
python moganet_valid.py \
--model repnext_m0 \
--img_size 224 \
--crop_pct 0.9 \
--data_dir data/imagenet \                  
--checkpoint repnext_m0_sz224_4xbs512_ep300.pth.tar
```

Tips: Convert a training-time RepNeXt into the inference-time structure
```
from timm.models import create_model
import utils

model = create_model('repnext_m1')
utils.replace_batchnorm(model)
```

## Latency Measurement 

The latency reported in RepNeXt for iPhone 12 (iOS 16) uses the benchmark tool from [XCode 14](https://developer.apple.com/videos/play/wwdc2022/10027/).

<details>
<summary>
RepNeXt-M1
</summary>
<img src="./figures/latency/repnext_m1_latency.png" width=70%>
</details>

<details>
<summary>
RepNeXt-M2
</summary>
<img src="./figures/latency/repnext_m2_latency.png" width=70%>
</details>

<details>
<summary>
RepNeXt-M3
</summary>
<img src="./figures/latency/repnext_m3_latency.png" width=70%>
</details>

<details>
<summary>
RepNeXt-M4
</summary>
<img src="./figures/latency/repnext_m4_latency.png" width=70%>
</details>

<details>
<summary>
RepNeXt-M5
</summary>
<img src="./figures/latency/repnext_m5_latency.png" width=70%>
</details>

Tips: export the model to Core ML model
```
python export_coreml.py --model repnext_m1 --ckpt pretrain/repnext_m1_distill_300e.pth
```
Tips: measure the throughput on GPU
```
python speed_gpu.py --model repnext_m1
```

## ImageNet  

### Prerequisites
`conda` virtual environment is recommended. 
```
conda create -n repnext python=3.8
pip install -r requirements.txt
```

### Data preparation

Download and extract ImageNet train and val images from http://image-net.org/. The training and validation data are expected to be in the `train` folder and `val` folder respectively:

```bash
# script to extract ImageNet dataset: https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh
# ILSVRC2012_img_train.tar (about 138 GB)
# ILSVRC2012_img_val.tar (about 6.3 GB)
```

```
# organize the ImageNet dataset as follows:
imagenet
├── train
│   ├── n01440764
│   │   ├── n01440764_10026.JPEG
│   │   ├── n01440764_10027.JPEG
│   │   ├── ......
│   ├── ......
├── val
│   ├── n01440764
│   │   ├── ILSVRC2012_val_00000293.JPEG
│   │   ├── ILSVRC2012_val_00002138.JPEG
│   │   ├── ......
│   ├── ......
```

### Training
To train RepNeXt-M1 on an 8-GPU machine:

```
python -m torch.distributed.launch --nproc_per_node=8 --master_port 12346 --use_env main.py --model repnext_m1 --data-path ~/imagenet --dist-eval
```
Tips: specify your data path and model name! 

Training with the helper script:

```bash
sh dist_train_cifar.sh
```

### Testing 
For example, to test RepNeXt-M1:
```
python main.py --eval --model repnext_m1 --resume pretrain/repnext_m1_distill_300e.pth --data-path ~/imagenet
```

### Fused model evaluation
For example, to evaluate RepNeXt-M1 with the fused model: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/suous/RepNeXt/blob/main/demo/fused_model_evaluation.ipynb)
```
python fuse_eval.py --model repnext_m1 --resume pretrain/repnext_m1_distill_300e_fused.pt --data-path ~/imagenet
```

## Downstream Tasks
[Object Detection and Instance Segmentation](detection/README.md)<br>

| Model      | $AP^b$ | $AP_{50}^b$ | $AP_{75}^b$ | $AP^m$ | $AP_{50}^m$ | $AP_{75}^m$ | Latency |                                       Ckpt                                        |                     Log                     |
|:-----------|:------:|:---:|:--:|:------:|:--:|:--:|:-------:|:---------------------------------------------------------------------------------:|:-------------------------------------------:|
| RepNeXt-M3 |  40.8  | 62.4   | 44.7  | 37.8   | 59.5  | 40.6 |  5.1ms  | [M3](https://github.com/suous/RepNeXt/releases/download/v1.0/repnext_m3_coco.pth) | [M3](./detection/logs/repnext_m3_coco.json) |
| RepNeXt-M4 |  42.9  | 64.4   | 47.2  |  39.1  | 61.7  | 41.7 |  6.6ms  | [M4](https://github.com/suous/RepNeXt/releases/download/v1.0/repnext_m4_coco.pth) | [M4](./detection/logs/repnext_m4_coco.json) |
| RepNeXt-M5 |  44.7  | 66.0   | 49.2  |  40.7  | 63.5  | 43.6 | 10.4ms  | [M5](https://github.com/suous/RepNeXt/releases/download/v1.0/repnext_m5_coco.pth) | [M5](./detection/logs/repnext_m5_coco.json) |

[Semantic Segmentation](segmentation/README.md)

| Model      | mIoU | Latency |                                        Ckpt                                         |                       Log                        |
|:-----------|:----:|:-------:|:-----------------------------------------------------------------------------------:|:------------------------------------------------:|
| RepNeXt-M3 |   40.6   |  5.1ms  | [M3](https://github.com/suous/RepNeXt/releases/download/v1.0/repnext_m3_ade20k.pth) | [M3](./segmentation/logs/repnext_m3_ade20k.json) |
| RepNeXt-M4 |   43.3   |  6.6ms  | [M4](https://github.com/suous/RepNeXt/releases/download/v1.0/repnext_m4_ade20k.pth) | [M4](./segmentation/logs/repnext_m4_ade20k.json) |
| RepNeXt-M5 |   45.0   | 10.4ms  | [M5](https://github.com/suous/RepNeXt/releases/download/v1.0/repnext_m5_ade20k.pth) | [M5](./segmentation/logs/repnext_m5_ade20k.json) |

## Feature Map Visualization
Run feature map visualization demo: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/suous/RepNeXt/blob/main/demo/feature_map_visualization.ipynb)

<table border=0 align=center>
	<tbody>
    <tr>
			<td align="center"> Original Image </td>
			<td align="center"> Identity </td>
			<td align="center"> RepDWConvS </td>
			<td align="center"> RepDWConvM </td>
			<td align="center"> DWConvL </td>
		</tr>
		<tr>
			<td width="20%"> <img src="https://raw.githubusercontent.com/shicai/MobileNet-Caffe/master/cat.jpg"> </td>
			<td width="20%"> <img src="./demo/figures/cat-stage0-block1-conv_i.png"> </td>
            <td width="20%"> <img src="./demo/figures/cat-stage0-block1-conv_s.png"> </td>
            <td width="20%"> <img src="./demo/figures/cat-stage0-block1-conv_m.png"> </td>
            <td width="20%"> <img src="./demo/figures/cat-stage0-block1-conv_l.png"> </td>
		</tr>
		<tr>
			<td width="20%"> <img src="https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/dog2.jpg"> </td>
			<td width="20%"> <img src="./demo/figures/dog2-stage0-block1-conv_i.png"> </td>
            <td width="20%"> <img src="./demo/figures/dog2-stage0-block1-conv_s.png"> </td>
            <td width="20%"> <img src="./demo/figures/dog2-stage0-block1-conv_m.png"> </td>
            <td width="20%"> <img src="./demo/figures/dog2-stage0-block1-conv_l.png"> </td>
		</tr>
	</tbody>
</table>

## Ablation Study

### Downsampling Layer Design

The downsampling layer between each stage is a modified version of the MetaNeXt block, where the shortcut connection bypasses the channel mixer.

<details>
  <summary>
  When replace downsampling layers of ConvNeXt-femto with our designs, the top-1 accuracy is improved by 1.8%.
  </summary>

|                          Model                           | Top-1(%) | Params(M) | GMACs | Throughput(im/s) |                          Log                          |                                                 Ckpt                                                 |
|:--------------------------------------------------------:|:--------:|:---------:|:-----:|:----------------:|:-----------------------------------------------------:|:----------------------------------------------------------------------------------------------------:|
| [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) |  72.37   |   5.22    | 0.78  |       3636       | [femto](./logs/ablation/convnext_femto_120e_7237.txt) | [baseline](https://github.com/suous/RepNeXt/releases/download/v1.0/convnext_femto_120e_baseline_7237.pth) |
|                         ModifiedA                        |  74.28   |   5.25    | 0.79  |       3544       | [femto](./logs/ablation/convnext_femto_120e_7428.txt) | [replaced](https://github.com/suous/RepNeXt/releases/download/v1.0/convnext_femto_120e_replaced_7428.pth) |
|                         ModifiedB                        |  74.19   |   5.25    | 0.79  |       3544       | [femto](./logs/ablation/convnext_femto_120e_7419.txt) | [replaced](https://github.com/suous/RepNeXt/releases/download/v1.0/convnext_femto_120e_replaced_7419.pth) |

![top1_acc](./figures/ablation_convnext.png)

```python
# ModifiedA
class Downsample(nn.Module):
    def __init__(self, dim, mlp_ratio):
        super().__init__()
        out_dim = dim * 2
        self.dwconv = nn.Conv2d(dim, out_dim, kernel_size=7, padding=3, groups=dim, stride=2)
        self.norm = LayerNorm(out_dim, eps=1e-6)
        self.pwconv1 = nn.Linear(out_dim, mlp_ratio * out_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(mlp_ratio * out_dim, out_dim)

    def forward(self, x):
        x = self.dwconv(x)        # token mixer: (N, C, H, W) -> (N, 2C, H/2, W/2)
        input = x                 # bypass the channel mixer and the normalization layer
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        return input + x
```

> The code below is our downsample layer design, where the shortcut connection only bypasses the channel mixer. 
> This design enables the batch normalization layer to be fused into the previous convolution layer.

```python
# ModifiedB
class Downsample(nn.Module):
    def __init__(self, dim, mlp_ratio=2):
        super().__init__()
        out_dim = dim * 2
        self.dwconv = nn.Conv2d(dim, out_dim, kernel_size=7, padding=3, groups=dim, stride=2)
        self.norm = LayerNorm(out_dim, eps=1e-6, data_format="channels_first")
        self.pwconv1 = nn.Linear(out_dim, mlp_ratio * out_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(mlp_ratio * out_dim, out_dim)

    def forward(self, x):
        x = self.dwconv(x)        # token mixer: (N, C, H, W) -> (N, 2C, H/2, W/2)
        x = self.norm(x)          # normalization layer
        input = x                 # bypass the channel mixer
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        return input + x
```

</details>

### Compact Model Experiments

#### Cifar-100

|    Model    | Latency (ms) | Params (M) | GMACs | Top-1 (100e) | Top-1 (200e) | Top-1 (300e) | Top-1 (400e) |                                                                                                                    Log                                                                                                                    |  
|:-----------:|:------------:|:----------:|:-----:|:------------:|:------------:|:------------:|:------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| RepViT-M0.6 |     0.6      |    2.2     | 0.39  |    71.73     |    78.45     |    80.16     |    80.60     |   [100e](./logs/compact/cifar/repvit_m0_6_cifar_100e.txt) / [200e](./logs/compact/cifar/repvit_m0_6_cifar_200e.txt) / [300e](./logs/compact/cifar/repvit_m0_6_cifar_300e.txt) / [400e](./logs/compact/cifar/repvit_m0_6_cifar_400e.txt)   |
| RepNeXt-M0E |     0.7      |    2.3     | 0.46  |    73.56     |    79.79     |  **81.89**   |  **82.07**   | [100e](./logs/compact/cifar/repnext_m0_e_cifar_100e.txt) / [200e](./logs/compact/cifar/repnext_m0_e_cifar_200e.txt) / [300e](./logs/compact/cifar/repnext_m0_e_cifar_300e.txt) / [400e](./logs/compact/cifar/repnext_m0_e_cifar_400e.txt) |
| RepNeXt-M0  |     0.6      |    2.0     | 0.39  |  **74.53**   |  **80.14**   |    81.73     |    81.97     |     [100e](./logs/compact/cifar/repnext_m0_cifar_100e.txt) / [200e](./logs/compact/cifar/repnext_m0_cifar_200e.txt) / [300e](./logs/compact/cifar/repnext_m0_cifar_300e.txt) / [400e](./logs/compact/cifar/repnext_m0_cifar_400e.txt)     |

![top1_acc_cifar_100](./figures/compact_models_cifar_100.png)

`RepNeXt-M0E` is the equivalent form of `RepNeXt-M0` where the multi-branch design is replaced by the single-branch large-kernel depthwise convolution.
Our multi-branch reparameter design helps the model converge faster, with a smaller size and lower latency.

#### ImageNet-1K

|    Model    | Latency (ms) | Params (M) | GMACs | Top-1 (100e) | Top-1 (200e) | Top-1 (300e) | Top-1 (400e) |                                                                                                                        Log                                                                                                                        |  
|:-----------:|:------------:|:----------:|:-----:|:------------:|:------------:|:------------:|:------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| RepViT-M0.6 |     0.6      |    2.5     | 0.39  |    69.29     |    71.63     |    72.34     |  **72.87**   |   [100e](./logs/compact/imgnet/repvit_m0_6_imgnet_100e.txt) / [200e](./logs/compact/imgnet/repvit_m0_6_imgnet_200e.txt) / [300e](./logs/compact/imgnet/repvit_m0_6_imgnet_300e.txt) / [400e](./logs/compact/imgnet/repvit_m0_6_imgnet_400e.txt)   |
| RepNeXt-M0E |     0.7      |    2.5     | 0.46  |    69.69     |    71.57     |    72.18     |    72.53     | [100e](./logs/compact/imgnet/repnext_m0_e_imgnet_100e.txt) / [200e](./logs/compact/imgnet/repnext_m0_e_imgnet_200e.txt) / [300e](./logs/compact/imgnet/repnext_m0_e_imgnet_300e.txt) / [400e](./logs/compact/imgnet/repnext_m0_e_imgnet_400e.txt) |
| RepNeXt-M0  |     0.6      |    2.3     | 0.39  |  **70.14**   |  **71.93**   |  **72.56**   |    72.78     |     [100e](./logs/compact/imgnet/repnext_m0_imgnet_100e.txt) / [200e](./logs/compact/imgnet/repnext_m0_imgnet_200e.txt) / [300e](./logs/compact/imgnet/repnext_m0_imgnet_300e.txt) / [400e](./logs/compact/imgnet/repnext_m0_imgnet_400e.txt)     |

![top1_acc_cifar_100](./figures/compact_models_imagenet_1k.png)

`RepNeXt-M0E` is the equivalent form of `RepNeXt-M0` where the multi-branch design is replaced by the single-branch large-kernel depthwise convolution.
Our multi-branch reparameter design helps the model converge faster, with a smaller size and lower latency.

## Acknowledgement

Classification (ImageNet) code base is partly built with [LeViT](https://github.com/facebookresearch/LeViT), [PoolFormer](https://github.com/sail-sg/poolformer), [EfficientFormer](https://github.com/snap-research/EfficientFormer),  [RepViT](https://github.com/THU-MIG/RepViT), and [MogaNet](https://github.com/Westlake-AI/MogaNet).

The detection and segmentation pipeline is from [MMCV](https://github.com/open-mmlab/mmcv) ([MMDetection](https://github.com/open-mmlab/mmdetection) and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)). 

Thanks for the great implementations! 

## Citation

If our code or models help your work, please cite our paper:
```BibTeX
@misc{zhao2024repnext,
      title={RepNeXt: A Fast Multi-Scale CNN using Structural Reparameterization},
      author={Mingshu Zhao and Yi Luo and Yong Ouyang},
      year={2024},
      eprint={2406.16004},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
