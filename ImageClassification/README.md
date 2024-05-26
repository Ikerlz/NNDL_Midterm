# Image Classification with Convolutional Neural Networks 

## Introduction

This is the first task of the midterm project, we implement Resnet-18 and VGG16 for the image classification on [CUB_200_2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) dataset.

## Structure

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

### parameters

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

## result


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




