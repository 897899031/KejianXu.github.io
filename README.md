# YOLOAX: YOLOX With Attention
<div align=center><img src="https://github.com/KejianXu/yoloax/assets/134375672/3061a843-4493-488d-8695-f59dba513886"></div>

# Performance
MS COCO
| Model | Test Size |   AP<sup>test</sup> | AP<sup>val</sup> | AP<sub>50</sub><sup>test</sup> | AP<sub>75</sub><sup>test</sup> | batch 1 fps |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| YOLOAX-S  | 640 | **42.1%** | **42.0%**	| **62.8%** |	**47.7%** | 119 fps |
| YOLOAX-M  | 640 | **51.3%** | **51.0%**	| **69.5%**	| **53.3%** | 95 fps  |
| YOLOAX-L  | 640 | **53.8%** | **53.5%** |	**71.2%** |	**57.2%** | 84 fps  |
| YOLOAX-X  | 640 | **54.2%** | **54.2%** |	**72.3%** |	**58.4%** | 72 fps  |

# Training
Data preparation
+ Download MS COCO 2017 dataset images (train, val, test) and labels
+ Download PASCAL VOC 2012 dataset images (train, val, test) and labels

Single GPU training

Multiple GPU training
```
# DP
CUDA_VISIBLE_DEVICES=0,1 python train.py

# DDP
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
```
# Acknowledgements
+ <a name = "ref1" href="https://github.com/AlexeyAB/darknet">https://github.com/AlexeyAB/darknet</a>
+ <a name = "ref2" href="https://github.com/Megvii-BaseDetection/YOLOX">https://github.com/Megvii-BaseDetection/YOLOX</a>
+ <a name = "ref3" href="https://github.com/WongKinYiu/yolov7">https://github.com/WongKinYiu/yolov7</a>
+ <a name = "ref4" href="https://github.com/WongKinYiu/yolor">https://github.com/WongKinYiu/yolor</a>
+ <a name = "ref5" href="https://github.com/WongKinYiu/PyTorch_YOLOv4">https://github.com/WongKinYiu/PyTorch_YOLOv4</a>
+ <a name = "ref6" href="https://github.com/ultralytics/yolov3">https://github.com/ultralytics/yolov3</a>
+ <a name = "ref7" href="https://github.com/ultralytics/yolov5">https://github.com/ultralytics/yolov5</a>
+ <a name = "ref8" href="https://github.com/DingXiaoH/RepVGG">https://github.com/DingXiaoH/RepVGG</a>
+ <a name = "ref9" href="https://github.com/bubbliiiing/yolox-pytorch">https://github.com/bubbliiiing/yolox-pytorch</a>
+ <a name = "ref10" href="https://github.com/bubbliiiing/yolov7-pytorch">https://github.com/bubbliiiing/yolov7-pytorch</a>





