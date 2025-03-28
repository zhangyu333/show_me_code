# -*- coding:utf-8 -*-
# @FileName  :vege.py
# @Time      :2023/5/4 13:47
# @Author    :ZY
import cv2
from common.oss import OSS
from common.utils import Util
from common.file_utils import clearCache
from controller.vision.vege.vege_detect import VegeDetect
from controller.vision.vege.vege_rec import VegeClassify

image_oss = OSS("hz-images")


class VT:
    def __init__(self):
        self.vd = VegeDetect()
        self.vc = VegeClassify()

    def __call__(self, filename):
        img, blocks = self.vd(filename)
        results = []
        for idx, block in enumerate(blocks):
            result = self.vc(block)
            result['id'] = idx
            results.append(result)
        localpath = Util.generate_temp_file_path(suffix="png")
        cv2.imwrite(localpath, img)
        remote_url = image_oss.upload(localpath)
        clearCache(localpath)
        resp = {
            "results": results,
            "remote_url": remote_url
        }
        return resp

