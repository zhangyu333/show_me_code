# -*- coding: utf-8 -*-
# @Time    : 2023/2/13 9:02
# @Author  : ZhangY
import cv2
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import onnxruntime
from common.utils import Util


class HandwriteRec():
    def __init__(self):
        super(HandwriteRec, self).__init__()

        self.charset = np.load(Util.app_path() + "/static/ocr/charset.npy")
        self.ort_session = onnxruntime.InferenceSession(Util.app_path() + '/models/ocr/common_old.onnx')

    def predict(self, remote_file):
        bio = BytesIO(requests.get(remote_file).content)
        image = Image.open(bio)
        image = image.resize((int(image.size[0] * (64 / image.size[1])), 64), Image.ANTIALIAS).convert('L')
        image = np.array(image).astype(np.float32)
        image = np.expand_dims(image, axis=0) / 255.
        image = (image - 0.5) / 0.5
        ort_inputs = {'input1': np.array([image])}
        ort_outs = self.ort_session.run(None, ort_inputs)
        result = []
        last_item = 0
        for item in ort_outs[0][0]:
            if item == last_item:
                continue
            else:
                last_item = item
            if item != 0:
                result.append(self.charset[item])
        bio.close()
        return ''.join(result)
