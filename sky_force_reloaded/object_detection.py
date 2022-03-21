import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from models.common import DetectMultiBackend
from sky_force_reloaded import config
from utils.augmentations import letterbox
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


class ObjectDetection:
    def __init__(self):
        # Load model
        self.device = select_device()
        self.model = DetectMultiBackend(config.WEIGHTS, device=self.device, dnn=False, data=None, fp16=False)
        self.stride, self.names, pt = self.model.stride, self.model.names, self.model.pt

        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

        imgsz = config.IMG_SIZE
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size

        self.auto = True

        bs = 1  # batch_size

        # Run inference
        self.model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup

        pass

    def detect(self, im0, draw_box=False):
        # 检测结果列表
        result_list = []

        # Padded resize
        img = letterbox(im0, self.imgsz, stride=self.stride, auto=self.auto)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        # self.imgsz = check_img_size(imgsz, s=stride)  # check image size

        im = torch.from_numpy(img).to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        pred = self.model(im, augment=False, visualize=False)

        conf_thres = 0.25  # confidence threshold
        iou_thres = 0.45  # NMS IOU threshold
        classes = None  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms = False  # class-agnostic NMS
        max_det = 1000  # maximum detections per image

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        for i, det in enumerate(pred):  # per image

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):

                    # 坐标
                    x1 = int(xyxy[0])
                    y1 = int(xyxy[1])
                    x2 = int(xyxy[2])
                    y2 = int(xyxy[3])
                    # 类别
                    cls_name = self.names[int(cls)]
                    # 置信度
                    conf = round(float(conf), 4)
                    # 保存到list
                    result_list.append((cls_name, conf, x1, y1, x2, y2))

                    # 画框
                    if draw_box:
                        # if save_img or view_img:  # Add bbox to image
                        label = '%s %s' % (cls_name, conf)
                        plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=3)
                        pass
                    pass
                pass
            pass
        pass

        return result_list, im0
