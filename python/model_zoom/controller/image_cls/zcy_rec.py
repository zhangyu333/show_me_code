# -*- coding:utf-8 -*-
# @FileName  :zcy_rec.py
# @Time      :2023/4/18 09:16
# @Author    :ZY
import cv2
import numpy as np
import onnxruntime as ort
from common.utils import Util


class RecZCY:
    def __init__(self):
        super(RecZCY, self).__init__()
        self.__model = ort.InferenceSession(Util.app_path() + "/models/cls/zcy.onnx")
        self.labels = ['阿胶', '板蓝根', '陈皮', '党参', '枸杞']

    def preProc(self, image):
        image = cv2.resize(image, (112, 112)).astype("float32")
        image = np.expand_dims(image, 0)
        return image

    def postProc(self, output):
        classid = int(np.argmax(output))
        label = self.labels[classid]
        score = float(output[0][classid])
        return label, score

    def __infer(self, input_im):
        return self.__model.run(None, {"input": input_im})[0]

    def main(self, img_path):
        image = cv2.imread(img_path)
        input_im = self.preProc(image)
        output = self.__infer(input_im)
        label, score = self.postProc(output)
        return {
            "label": label,
            "score": score,
        }


if __name__ == '__main__':
    zcy = RecZCY()
    img_path = "/Users/zhangyu/Desktop/ml/_project_pytorch/中草药识别/ajiao_0063.jpg"
    res = zcy.main(img_path)
    print(res)
