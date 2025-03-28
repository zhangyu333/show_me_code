# coding=utf-8
# Created : 2023/1/16 09:44
# Author  : Zy
import cv2
import numpy as np
import onnxruntime as ort
from common.utils import Util


class VegetableRec():
    def __init__(self):
        super(VegetableRec, self).__init__()
        self.classees = ['豆科植物', '苦瓜', '葫芦瓜', '茄子', '西兰花', '甘蓝', '辣椒', '胡萝卜',
                         '花椰菜', '黄瓜', '番木瓜', '马铃薯', '南瓜', '萝卜', '番茄']
        self.model = ort.InferenceSession(Util.app_path() + '/models/cls/vegetable.onnx')

    def infer(self, filename):
        img = cv2.imread(filename)
        img = cv2.resize(img, (112, 112))
        img = img.astype('float32')
        img = img.transpose((2, 0, 1))
        img = img / 255.
        img = img[None]
        output = self.model.run(None, {"input.1": img})[0][0]
        idxs = np.argsort(output)[::-1][:3]
        scores = [output[idx] for idx in idxs]
        labels = [self.classees[idx] for idx in idxs]
        scores = [float(s) for s in scores]
        return {
            "labels": labels,
            "scores": scores,
        }
