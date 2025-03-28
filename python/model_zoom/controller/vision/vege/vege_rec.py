# -*- coding: utf-8 -*-
# @Time    : 2022/12/5 16:44
# @Author  : ZhangY
# @File    : vage_rec.py
# @Project : model_zoo
import cv2
import numpy as np
import onnxruntime as ort
from common.utils import Util


class VegeClassify:
    def __init__(self):
        super(VegeClassify, self).__init__()
        self.types = ['扁豆', '节瓜', '西兰花', '包菜', '彩椒', '胡萝卜', '菜花', '黄瓜', '木瓜', '土豆', '南瓜', '白萝卜', '番茄']
        self.shapes = ['不规则脑状', '球状', '椭圆状', '不规则多面体状', '叶状', '不规则扇形状', '条形椭圆状', '条形扁平状', '长条圆形状']
        self.pannels_types = ['表面光滑凹凸', '表面凹凸光滑', '表面颗粒凹凸', '表面光滑']
        self.model = ort.InferenceSession(Util.app_path() + "/models/cls/vege_classify.onnx")

    def __call__(self, img):
        img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None]
        x = np.ascontiguousarray(img).astype("float32")
        outputs = self.model.run(None, {"input": x})
        type_idx = np.argmax(outputs[0], 1)
        shapes_idx = np.argmax(outputs[1], 1)
        pannels_type_idx = np.argmax(outputs[2], 1)
        type_score = outputs[0][0][type_idx]
        shape_score = outputs[1][0][shapes_idx]
        pannels_type_score = outputs[2][0][pannels_type_idx]
        result = {
            "类型": self.types[int(type_idx)],
            "类型得分": float(type_score),
            "形状": self.shapes[int(shapes_idx)],
            "形状得分": float(shape_score),
            "表面": self.pannels_types[int(pannels_type_idx)],
            "表面得分": float(pannels_type_score)
        }
        return result


if __name__ == '__main__':
    pass
