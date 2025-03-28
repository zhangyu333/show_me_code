# coding=utf-8
# Created : 2023/2/27 17:15
# Author  : Zy
import cv2
import torch
import threading
import numpy as np
from common.oss import OSS
from common.utils import Util
from common.file_utils import clearCache

image_oss = OSS("hz-images")


class StyleMove:
    def __init__(self):
        super(StyleMove, self).__init__()
        self.root_path = Util.app_path()
        self.__candy_style_model = torch.jit.load(self.root_path + "/models/vision/style/candy.script", map_location="cpu")
        self.__mosaic_style_model = torch.jit.load(self.root_path + "/models/vision/style/mosaic.script", map_location="cpu")
        self.__starry_night_style_model = torch.jit.load(self.root_path + "/models/vision/style/starry-night.script",
                                                         map_location="cpu")
        self.__udnie_style_model = torch.jit.load(self.root_path + "/models/vision/style/udnie.script", map_location="cpu")
        self.__candy_style_model.eval()
        self.__mosaic_style_model.eval()
        self.__starry_night_style_model.eval()
        self.__udnie_style_model.eval()

    def __preProc(self, img_path):
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        img = cv2.resize(img, (512, 512))
        img = img.transpose((2, 0, 1))
        img = img.astype("float32")
        img = img[None]
        img = torch.Tensor(img)
        return h, w, img

    def __postProc(self, output, w, h, savepath):
        output = output.clamp(0, 255).detach().numpy()
        output = output.squeeze(0)
        output = output.astype(np.uint8)
        output = output.transpose(1, 2, 0)
        output = cv2.resize(output, (w, h))
        cv2.imwrite(savepath, output)

    def __predict(self, model, input, w, h, savepath):
        output = model(input)
        self.__postProc(output, w, h, savepath)

    def predict(self, local_path):
        h, w, input_im = self.__preProc(local_path)
        suffix = Util.extract_file_suffix(local_path)
        candy_style_path = Util.generate_temp_file_path(suffix=suffix)
        mosaic_style_path = Util.generate_temp_file_path(suffix=suffix)
        starry_night_style_path = Util.generate_temp_file_path(suffix=suffix)
        udnie_style_path = Util.generate_temp_file_path(suffix=suffix)
        candy_style_process = threading.Thread(
            target=self.__predict,
            args=(self.__candy_style_model, input_im, w, h, candy_style_path))
        mosaic_style_process = threading.Thread(
            target=self.__predict,
            args=(self.__mosaic_style_model, input_im, w, h, mosaic_style_path))
        starry_night_style_process = threading.Thread(
            target=self.__predict,
            args=(self.__starry_night_style_model, input_im, w, h, starry_night_style_path))
        udnie_style_process = threading.Thread(
            target=self.__predict,
            args=(self.__udnie_style_model, input_im, w, h, udnie_style_path))

        candy_style_process.start()
        mosaic_style_process.start()
        starry_night_style_process.start()
        udnie_style_process.start()
        print("candy_style线程启动！！\nmosaic_style线程启动！！\nstarry_night_style线程启动！！\nudnie_style线程启动！！")
        candy_style_process.join()
        mosaic_style_process.join()
        starry_night_style_process.join()
        udnie_style_process.join()
        print("candy_style线程结束！！\nmosaic_style线程结束！！\nstarry_night_style线程结束！！\nudnie_style线程结束！！")

        candy_style_remote_path = image_oss.upload(candy_style_path)
        mosaic_style_remote_path = image_oss.upload(mosaic_style_path)
        starry_night_style_remote_path = image_oss.upload(starry_night_style_path)
        udnie_style_remote_path = image_oss.upload(udnie_style_path)
        clearCache(candy_style_path)
        clearCache(mosaic_style_path)
        clearCache(starry_night_style_path)
        clearCache(udnie_style_path)
        return {
            "candy_style": candy_style_remote_path,
            "mosaic_style": mosaic_style_remote_path,
            "starry_night_style": starry_night_style_remote_path,
            "udnie_style": udnie_style_remote_path,
        }


if __name__ == '__main__':
    pass
