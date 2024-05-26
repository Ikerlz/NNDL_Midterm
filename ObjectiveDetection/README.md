# Objective Detection Using Faster R-CNN and YOLOv3 under the `mmdetection` Framework
this repository is designed for training Faster RCNN and YOLOv3 models on PASCAL-VOC dataset using mmdetection framework.

## Structure


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


## Prepare Environment
please refer to [https://github.com/open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)

## Prepare Dataset

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

## Train

**Note that you need to modify the `data_root` variable in the config file in order to train !!!**

### Simple Training

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

## Inference

In order to implement the visualization, we need to modify the `two_stage.py` file in the installed “mmdet” package, which is located in the `models/detectors` path of the `mmdet` package. We simply need to open this file and add the following two lines just before the last return statement in

```
for nn in range(len(batch_data_samples)):
    batch_data_samples[nn].rpn_res = rpn_results_list[nn]
```

Then, you can modify the `plot.py` to inference other images. Moreover, you can use this file to see the region proposals in the first stage of the Faster-RCNN

## My Model
you can download my pretrained model from Baidu NetDisk [https://pan.baidu.com/s/1T-qT3BrHFfYr8DtCzt4wvQ](https://pan.baidu.com/s/1T-qT3BrHFfYr8DtCzt4wvQ), access code is 6666.

## Results

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
