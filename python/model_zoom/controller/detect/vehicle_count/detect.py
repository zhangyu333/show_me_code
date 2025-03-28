# coding=utf-8
# Created : 2023/1/13 12:31
# Author  : Zy
import cv2
import numpy as np
import onnxruntime as ort
from common.utils import Util


class VehicleDetection:
    def __init__(self):
        super(VehicleDetection, self).__init__()
        self.__ssd_model = ort.InferenceSession(Util.app_path() + "/models/vision/vehicle_count.onnx",
                                                provider_options=["CPUExecutionProvider"])
        self.classes = ["BACKGROUND", "car", "bus", "motorcycle", "bicycle", "truck"]
        self.prob_threshold = 0.5
        self.nms_threshold = 0.5

    def __preProc(self, image):
        height, width, _ = image.shape
        new_h, new_w = 240, 320
        new_img = cv2.resize(image, (new_w, new_h)).astype(np.float32).transpose((2, 0, 1))
        new_img -= 127
        new_img /= 128
        new_img = new_img[None]
        new_img = np.ascontiguousarray(new_img)
        return height, width, new_img

    def __forward(self, inp):
        outputs = self.__ssd_model.run(None, {"inputs": inp})
        return outputs

    def __postProc(self, outputs, height, width):
        scores = outputs[0][0]
        boxes = outputs[1][0]
        box_probs = []
        for class_index in range(1, scores.shape[1]):
            probs = scores[:, class_index]
            mask = probs > self.prob_threshold
            probs = probs[mask]
            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]
            indices = cv2.dnn.NMSBoxes(subset_boxes, probs, self.prob_threshold, self.nms_threshold)
            bboxes = subset_boxes[indices]
            bboxes[:, 0] *= width
            bboxes[:, 1] *= height
            bboxes[:, 2] *= width
            bboxes[:, 3] *= height
            bboxes = bboxes.astype(int)
            my_socres = probs[indices]
            for i in range(len(bboxes)):
                x1, y1 = bboxes[i][0], bboxes[i][1]
                x2, y2 = bboxes[i][2], bboxes[i][3]
                box_probs.append(
                    (x1, y1, x2, y2, self.classes[class_index], round(my_socres[i], 2))
                )
        return box_probs

    def detect(self, frame):
        image_p = frame.copy()
        height, width, inp = self.__preProc(image_p)
        outputs = self.__forward(inp)
        box_probs = self.__postProc(outputs, height, width)
        return box_probs
