# coding=utf-8
# Created : 2023/3/31 11:10
# Author  : Zy
import cv2
import torch
import onnxruntime
import numpy as np
from typing import Tuple, Any
from common.utils import Util
from common.file_utils import clearCache
from common.oss import OSS

image_oss = OSS("hz-images")


class DefineStyleMove():
    def __init__(self):
        super(DefineStyleMove, self).__init__()
        self.__encode = onnxruntime.InferenceSession(Util.app_path() + "/models/vision/style/encode.onnx")
        self.__decode = onnxruntime.InferenceSession(Util.app_path() + "/models/vision/style/decode.onnx")
        self.__style_swap = torch.jit.load(Util.app_path() + "/models/vision/style/stylemodel.bin")
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def getFeaturePreProc(self, image: np.ndarray) -> Tuple[Any, int, int]:
        image_h, image_w = image.shape[:2]
        max_r = min(300 / image_h, 300 / image_w)
        new_h, new_w = int(max_r * image_h), int(max_r * image_w)
        image = cv2.resize(image, (new_w, new_h), cv2.INTER_AREA)
        image = image / 255.0
        image -= self.mean
        image /= self.std
        image = image.astype("float32")
        image = image.transpose((2, 0, 1))
        image = image[None]
        image = np.ascontiguousarray(image)
        return image, image_h, image_w

    def postProc(self, decode_out: np.ndarray, image_h: int, image_w: int) -> np.ndarray:
        decode_out *= self.std
        decode_out += self.mean
        decode_out[decode_out < 0] = 0
        decode_out[decode_out > 1] = 1
        decode_out *= 255.0
        decode_out = decode_out.astype(np.uint8)
        decode_out = cv2.resize(decode_out, (image_w, image_h), cv2.INTER_AREA)
        return decode_out

    def __infer(self, content_input, style_input):
        content_features = self.__encode.run(None, {"images": content_input})[0]
        style_features = self.__encode.run(None, {"images": style_input})[0]
        style_swap_res = self.__style_swap(
            torch.Tensor(content_features), torch.Tensor(style_features)).numpy().astype(
            "float32")
        decode_out = self.__decode.run(None, {"inputs": style_swap_res})[0][0].transpose((1, 2, 0))
        return decode_out

    def styleMove(self, content_path, style_path):
        content_image = cv2.imread(content_path)
        style_image = cv2.imread(style_path)
        content_input, image_h, image_w = self.getFeaturePreProc(content_image)
        style_input, _, _ = self.getFeaturePreProc(style_image)
        decode_out = self.__infer(content_input, style_input)
        decode_out = self.postProc(decode_out, image_h, image_w)
        local_path = Util.generate_temp_file_path(suffix=Util.extract_file_suffix(content_path))
        cv2.imwrite(local_path, decode_out)
        remote_path = image_oss.upload(local_path)
        clearCache(local_path)
        return remote_path


if __name__ == '__main__':
    pass
