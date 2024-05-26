# NNDL_Midterm

## Introduction

This project contains two tasks: the image classification and the objective detection.

---

## TASK ONE:  Image Classification with Convolutional Neural Networks 

### Introduction

This is the first task of the midterm project, we implement Resnet-18 and VGG16 for the image classification on [CUB_200_2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) dataset.

### Structure

```
.
├── README.md
├── __init__.py
├── conf
│   ├── __init__.py
│   ├── __pycache__
│   └── settings.py
├── criterion
│   ├── LabelSmoothing.py
│   ├── __init__.py
│   └── __pycache__
├── dataset
│   ├── __init__.py
│   ├── __pycache__
│   └── dataset.py
├── lr_scheduler
│   ├── WarmUpLR.py
│   ├── __init__.py
│   └── __pycache__
├── models
│   ├── __init__.py
│   ├── resnet.py
│   └── vgg.py
├── train.py
├── transforms
│   ├── __init__.py
│   ├── __pycache__
│   └── transforms.py
└── utils.py
```

## Usage

- Resnet18

```
python train.py -data "your data path for CUB_200_2011" -net resnet18 -b 16 -e 2 -pretrained 1 -lr 0.04 -warm 5 -device 'cuda:0'
```

- VGG16

```
python train.py -data "your data path for CUB_200_2011" -net vgg16 -b 16 -e 2 -pretrained 1 -lr 0.04 -warm 5 -device 'cuda:0'
```

---

#### parameters

```
parser.add_argument('-net', type=str, required=True, help='net type')
parser.add_argument('-data', type=str, required=True, help='the data path')
parser.add_argument('-pretrained', type=int, default=1, help='whether to use the pretrained weights')
parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
parser.add_argument('-b', type=int, default=256, help='batch size for dataloader')
parser.add_argument('-lr', type=float, default=0.04, help='initial learning rate')
parser.add_argument('-e', type=int, default=450, help='training epoches')
parser.add_argument('-warm', type=int, default=5, help='warm up phase')
parser.add_argument('-parallel', type=bool, default=False, help='whether to use the parallel training')
parser.add_argument('-device', type=str, default='cuda:0', help='the device if we do not use the parallel training')
parser.add_argument('-gpus', nargs='+', type=int, default=0, help='gpu device')
```

### result


| Setting | Model | Pretrained | LR | Batch Size | Warm-Up Step | Loss | Cutout & RE | Testing Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |:----------------:|
| 1 | Resnet18 | TRUE | 0.04 | 64 | 5 | LSR | FALSE |   **74.78 \%**   |
| 2 | VGG16 | TRUE | 0.04 | 64 | 5 | LSR | FALSE |     62.20 \%     |
| 3 | Resnet18 | FALSE | 0.04 | 64 | 5 | LSR | FALSE |     61.84 \%     |
| 4 | Resnet18 | FALSE | 0.04 | 128 | 5 | LSR | FALSE |     59.56 \%     |
| 5 | Resnet18 | FALSE | 0.04 | 256 | 5 | LSR | FALSE |     56.49 \%     |
| 6 | Resnet18 | FALSE | 0.02 | 64 | 5 | LSR | FALSE |     61.34 \%     |
| 7 | Resnet18 | FALSE | 0.06 | 64 | 5 | LSR | FALSE |     62.39 \%     |
| 8 | Resnet18 | FALSE | 0.08 | 64 | 5 | LSR | FALSE |     62.98 \%     |
| 9 | Resnet18 | FALSE | 0.04 | 64 | 10 | LSR | FALSE |     62.77 \%     |
| 10 | Resnet18 | FALSE | 0.04 | 64 | 20 | LSR | FALSE |     61.67 \%     |
| 11 | Resnet18 | FALSE | 0.04 | 64 | 5 | CE | FALSE |     56.63 \%     |
| 12 | Resnet18 | FALSE | 0.04 | 64 | 5 | LSR | TRUE |     54.23 \%     |

---

**Some trained models could be find in [https://pan.baidu.com/s/1T6nm4gLfxOdZifMD3IeUcw?pwd=6666](https://pan.baidu.com/s/1T6nm4gLfxOdZifMD3IeUcw?pwd=6666)** with the password 6666.


## TASK TWO: Objective Detection Using Faster R-CNN and YOLOv3 under the `mmdetection` Framework
this repository is designed for training Faster RCNN and YOLOv3 models on PASCAL-VOC dataset using mmdetection framework.

### Structure


```
.
├── README.md
├── data
│   └── VOC0712
│       ├── VOC2007 -> your data path
│       ├── VOC2012 -> your data path
│       └── annotations -> your data path
├── faster-rcnn_on_voc0712_coco.py
├── plot.py
├── train.py
├── voc2coco.py
└── yolov3_on_voc0712_coco.py
```


### Prepare Environment
please refer to [https://github.com/open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)

### Prepare Dataset

1. download the PASCAL-VOC datasets (2007 and 2013) and decompress it.
2. switch the VOC data format to COCO data format

```
python voc2coco.py "the path for the VOC dataset" -o "save dir/cocoVOCdevkit" --out-format coco
```
3. create a data soft connection

```
mkdir data
cd data
mkdir VOC0712
cd VOC0712

ln -s save dir/cocoVOCdevkit ./annotations
ln -s the path for the VOC dataset/VOC2007/ ./VOC2007 
ln -s the path for the VOC dataset/VOC2012/ ./VOC2012 

```

### Train

**Note that you need to modify the `data_root` variable in the config file in order to train !!!**

#### Simple Training

- Train the Faster R-CNN
```
CUDA_VISIBLE_DEVICES=1 python train.py ./faster-rcnn_on_voc0712_coco.py
```

- Train the YOLOv3
```
CUDA_VISIBLE_DEVICES=1 python train.py ./yolov3_on_voc0712_coco.py
```

---

you can also use tensorboard to check the loss curves during the training process
```
# create logfile
python create_tensorboard.py # remenber to modify base_dir, train/eval/test, step
tensorboard --logdir=faster_rcnn/log_file/train
```
test the performance of the model
```
# need to download the pth file using mmdet tool
python tools/test.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py faster_rcnn/latest.pth --work-dir faster_rcnn --eval bbox`
```

Besides, you can also use the pretrained model and fine-tuning on the VOC dataset
```
CUDA_VISIBLE_DEVICES=1 python train.py ./faster-rcnn_on_voc0712_coco.py --resume "the pretrained model download from the website"
CUDA_VISIBLE_DEVICES=1 python train.py ./yolov3_on_voc0712_coco.py --resume "the pretrained model download from the website"
```
---

### Inference

you can modify the `plot.py` to inference other images. Moreover, you can use this file to see the region proposals in the first stage of the Faster-RCNN

### My Model
you can download my pretrained model from Baidu NetDisk [https://pan.baidu.com/s/1T-qT3BrHFfYr8DtCzt4wvQ](https://pan.baidu.com/s/1T-qT3BrHFfYr8DtCzt4wvQ), access code is 6666.

### Results

| Settings | Model | Pre-trained | LR | gamma | Batch Size | MAP@0.5 : 0.95 |    MAP@0.5    |
| :---: | :---: | :---: | :---: |:-----:| :---: | :---: |:-------------:|
| 1 | Faster R-CNN | False | 0.001 |  0.2  | 12 | 0.452 |     0.799     |
| 2 | Faster R-CNN | True | 0.001 |  0.2  | 12 | 0.623 |     0.883     |
| 3 | Faster R-CNN | True | 0.0001 |  0.1  | 4 | **0.630** |   **0.891**   |
| 4 | Faster R-CNN | True | 0.005 |  0.2  | 12 | 0.503 |     0.822     |
| 5 | Faster R-CNN | True | 0.01 |  0.1  | 12 | 0.526 |     0.837     |
| 6 | YOLO v3 | False | 0.0001 |  0.1  | 8 | 0.384 |     0.725     |
| 7 | YOLO v3 | True | 0.0001 |  0.1  | 8 | **0.578** |   **0.883**   |
| 8 | YOLO v3 | False | 0.0001 |  0.2  | 8 | 0.375 |     0.711     |
| 9 | YOLO v3 | False | 0.00005 |  0.2  | 8 | 0.268 |     0.555     |

