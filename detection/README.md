# Object Detection and Instance Segmentation

Detection and instance segmentation on MS COCO 2017 is implemented based on [MMDetection](https://github.com/open-mmlab/mmdetection).

## Models
| Model      | $AP^b$ | $AP_{50}^b$ | $AP_{75}^b$ | $AP^m$ | $AP_{50}^m$ | $AP_{75}^m$ | Latency |                                       Ckpt                                        |                Log                |
|:-----------|:------:|:---:|:--:|:--:|:--:|:--:|:-------:|:---------------------------------------------------------------------------------:|:---------------------------------:|
| RepNeXt-M3 |  40.8  | 62.4   | 44.7  | 37.8   | 59.5  | 40.6 |  5.1ms  | [M3](https://github.com/suous/RepNeXt/releases/download/v1.0/repnext_m3_coco.pth) | [M3](./logs/repnext_m3_coco.json) |
| RepNeXt-M4 |  42.9  | 64.4   | 47.2  |  39.1  | 61.7  | 41.7 |  6.6ms  | [M4](https://github.com/suous/RepNeXt/releases/download/v1.0/repnext_m4_coco.pth) | [M4](./logs/repnext_m4_coco.json) |
| RepNeXt-M5 |  44.7  | 66.0   | 49.2  | 40.7   | 63.5  | 43.6 | 10.4ms  | [M5](https://github.com/suous/RepNeXt/releases/download/v1.0/repnext_m5_coco.pth) | [M5](./logs/repnext_m5_coco.json) |

## Installation

Install [mmcv-full](https://github.com/open-mmlab/mmcv) and [MMDetection v2.28.2](https://github.com/open-mmlab/mmdetection/tree/v2.28.2),
Later versions should work as well. 
The easiest way is to install via [MIM](https://github.com/open-mmlab/mim)
```
pip install -U openmim
mim install mmcv-full==1.7.1
mim install mmdet==2.28.2
```

## Data preparation

Prepare COCO 2017 dataset according to the [instructions in MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md#test-existing-models-on-standard-datasets).
The dataset should be organized as 
```
detection
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
```

## Testing

We provide a multi-GPU testing script, specify config file, checkpoint, and number of GPUs to use: 
```
./dist_test.sh config_file path/to/checkpoint #GPUs --eval bbox segm
```

For example, to test RepNeXt-M3 on COCO 2017 on an 8-GPU machine, 

```
./dist_test.sh configs/mask_rcnn_repnext_m3_fpn_1x_coco.py path/to/repnext_m3_coco.pth 8 --eval bbox segm
```

## Training
Download ImageNet-1K pretrained weights into `./pretrain` 

We provide PyTorch distributed data parallel (DDP) training script `dist_train.sh`, for example, to train RepNeXt-M3 on an 8-GPU machine: 
```
./dist_train.sh configs/mask_rcnn_renext_m3_fpn_1x_coco.py 8
```
Tips: specify configs and #GPUs!

## Hacking issues incompatible with torch>=2.0

1. `AttributeError: 'MMDistributedDataParallel' object has no attribute '_use_replicated_tensor_module'`

[Solution](https://github.com/microsoft/Cream/issues/179#issuecomment-1892997366): edit `/home/someone/micromamba/envs/detection/lib/python3.8/site-packages/mmcv/parallel/distributed.py` line **160** in `_run_ddp_forward` function.

```python
# comment below two lines
# module_to_run = self._replicated_tensor_module if \
#     self._use_replicated_tensor_module else self.module
# replace with below line
module_to_run = self.module
```

2. `AttributeError: 'int' object has no attribute 'type'`

[Solution](https://github.com/open-mmlab/mmdetection/issues/10720#issuecomment-1727317155): edit `/home/someone/micromamba/envs/detection/lib/python3.8/site-packages/mmcv/parallel/_functions.py` line **75** in `forward` function.

```python
# comment below line
# streams = [_get_stream(device) for device in target_gpus]
# replace with below line
streams = [_get_stream(torch.device("cuda", device)) for device in target_gpus]
```
