#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2024/5/10 16:31 
# @Author : Iker Zhe 
# @Version：V 0.1
# @File : plot.py
# @desc :
import os

import numpy as np
from mmdet.apis import init_detector, inference_detector
import mmcv
import matplotlib.pyplot as plt
from mmdet.registry import VISUALIZERS
from mmengine.visualization import Visualizer



compare_yolo_rcnn = 1
if compare_yolo_rcnn:
    root_path = "fill your data path"
    checkpoint_file1 = root_path + "faster-rcnn_epoch_20.pth"
    config1 = root_path + "faster-rcnn_on_voc0712_coco.py"
    model1 = init_detector(config1, checkpoint_file1, device='cpu')

    checkpoint_file2 = root_path + "yolov3_epoch_20.pth"
    config2 = root_path + "yolov3_on_voc0712_coco.py"
    model2 = init_detector(config2, checkpoint_file2, device='cpu')

    img1 = mmcv.imread(root_path + "demo/cat_bus.png", channel_order='rgb')
    img2 = mmcv.imread(root_path + "demo/person_bike.png", channel_order='rgb')
    img3 = mmcv.imread(root_path + "demo/person_dog.png", channel_order='rgb')
    imgs = [img1, img2, img3]
    result1 = inference_detector(model1, imgs)
    visualizer1 = VISUALIZERS.build(model1.cfg.visualizer)
    visualizer1.dataset_meta = model1.dataset_meta

    result2 = inference_detector(model2, imgs)
    visualizer2 = VISUALIZERS.build(model2.cfg.visualizer)
    visualizer2.dataset_meta = model2.dataset_meta

    imgs_name = ["cat_bus", "person_bike", "person_dog"]

    # Show the results
    for i in range(3):
        visualizer1.add_datasample(
            'result',
            imgs[i],
            data_sample=result1[i],
            draw_gt=False,
            show=False, out_file=root_path + imgs_name[i] + "_faster-rcnn.jpg")

        visualizer2.add_datasample(
            'result',
            imgs[i],
            data_sample=result2[i],
            draw_gt=False,
            show=False, out_file=root_path + imgs_name[i] + "_yolov3.jpg")


plot_rpn = 1
if plot_rpn:

    root_path = "fill your data path"
    save_path = "fill the save path"


    def plot_proposals(img_path, scale_factor=1, bboxes=None, plot_num=20,
                       save_path="./proposals_plot"):

        image = mmcv.imread(img_path, channel_order='rgb')
        visualizer = Visualizer(image=image, vis_backends=[dict(type='LocalVisBackend')], save_dir=save_path)
        # 绘制多个检测框
        visualizer.draw_bboxes(bboxes=bboxes[1:plot_num]/scale_factor, line_widths=1, edge_colors="red")

        visualizer.add_image(img_path.split("/")[-1] + '_proposals', visualizer.get_image())



    checkpoint_file1 = save_path + "faster-rcnn_epoch_20.pth"
    config1 = save_path + "faster-rcnn_on_voc0712_coco.py"
    model1 = init_detector(config1, checkpoint_file1, device='cpu')

    imgs = ["004948.jpg", "004949.jpg", "004950.jpg", "004951.jpg"]

    result1 = inference_detector(model1, [root_path + x for x in imgs])
    visualizer1 = VISUALIZERS.build(model1.cfg.visualizer)
    visualizer1.dataset_meta = model1.dataset_meta


    bboxes = [result1[x].rpn_res.bboxes.cpu() for x in range(len(imgs))]
    scales = [result1[x].scale_factor[0] for x in range(len(imgs))]

    for i in range(4):


        plot_proposals(img_path=root_path + imgs[i],
                       scale_factor=scales[i], bboxes=bboxes[i], plot_num=20,
                       save_path=save_path)

        visualizer1.add_datasample(
            'result',
            mmcv.imread(root_path + imgs[i], channel_order='rgb'),
            data_sample=result1[i],
            draw_gt=False,
            show=False, out_file=save_path + imgs[i].split(".")[0] + "_faster-rcnn.jpg")





