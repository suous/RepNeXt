# Semantic Segmentation 

Segmentation on ADE20K is implemented based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).

## Models
| Model      | mIoU | Latency |                                        Ckpt                                         |                 Log                 |
|:-----------|:----:|:-------:|:-----------------------------------------------------------------------------------:|:-----------------------------------:|
| RepNeXt-M3 |   40.6   |  5.1ms  | [M3](https://github.com/suous/RepNeXt/releases/download/v1.0/repnext_m3_ade20k.pth) | [M3](./logs/repnext_m3_ade20k.json) |
| RepNeXt-M4 |   43.3   |  6.6ms  | [M4](https://github.com/suous/RepNeXt/releases/download/v1.0/repnext_m4_ade20k.pth) | [M4](./logs/repnext_m4_ade20k.json) |
| RepNeXt-M5 |   45.0   | 10.4ms  | [M5](https://github.com/suous/RepNeXt/releases/download/v1.0/repnext_m5_ade20k.pth) | [M5](./logs/repnext_m5_ade20k.json) |

The backbone latency is measured with image crops of 512x512 on iPhone 12 by Core ML Tools.

## Requirements
Install [mmcv-full](https://github.com/open-mmlab/mmcv) and [MMSegmentation v0.30.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.30.0). 
Later versions should work as well. 
The easiest way is to install via [MIM](https://github.com/open-mmlab/mim)
```
pip install -U openmim
mim install mmcv-full==1.7.1
mim install mmseg==0.30.0
```

## Data preparation

We benchmark RepNeXt on the challenging ADE20K dataset, which can be downloaded and prepared following [insructions in MMSeg](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets). 
The data should appear as: 
```
├── segmentation
│   ├── data
│   │   ├── ade
│   │   │   ├── ADEChallengeData2016
│   │   │   │   ├── annotations
│   │   │   │   │   ├── training
│   │   │   │   │   ├── validation
│   │   │   │   ├── images
│   │   │   │   │   ├── training
│   │   │   │   │   ├── validation

```



## Testing

We provide a multi-GPU testing script, specify config file, checkpoint, and number of GPUs to use: 
```
./tools/dist_test.sh config_file path/to/checkpoint #GPUs --eval mIoU
```

For example, to test RepNeXt-M3 on ADE20K on an 8-GPU machine, 

```
./tools/dist_test.sh configs/sem_fpn/fpn_repnext_m3_ade20k_40k.py path/to/repnext_m3_ade20k.pth 8 --eval mIoU
```

## Training 
Download ImageNet-1K pretrained weights into `./pretrain` 

We provide PyTorch distributed data parallel (DDP) training script `dist_train.sh`, for example, to train RepNeXt-M3 on an 8-GPU machine: 
```
./tools/dist_train.sh configs/sem_fpn/fpn_repnext_m3_ade20k_40k.py 8
```
Tips: specify configs and #GPUs!
