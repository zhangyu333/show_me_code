# -*- coding:utf-8 -*-
# @FileName  :control.py
# @Time      :2023/4/7 10:37
# @Author    :ZY
import cv2
import jieba.analyse
from common.oss import OSS
from common.utils import Util
from common.file_utils import clearCache
from controller.vision.fast_style_move.neural_style_script import StyleMove
from controller.vision.style_move_custom import DefineStyleMove
from controller.vision.clipcap_ch_beautiful import ClipCapChinese
from controller.vision.vege_main import VT
from controller.vision.zzsfp_ocr import zzsfp
from controller.vision.face import FaceAttrs
faceattrs = FaceAttrs()

# from modelscope.pipelines import pipeline
# facefusion = pipeline('image-face-fusion', 'damo/cv_unet-image-face-fusion_damo')

ccc = ClipCapChinese()
stylemove = StyleMove()
vt = VT()
defineStyleMove = DefineStyleMove()
image_oss = OSS("hz-images")


def styleMove(local_path):
    result = stylemove.predict(local_path)
    return result


def jugerStyleContent(local_path, alpha):
    noise_img = cv2.imread(Util.app_path() + "/static/vision/noise_img.png")
    starry_night_style_img = cv2.imread(local_path)
    h, w = starry_night_style_img.shape[:2]
    noise_img = cv2.resize(noise_img, (w, h))
    beta = 1 - alpha
    dst = cv2.addWeighted(noise_img, alpha, starry_night_style_img, beta, 1)
    dst_path = Util.generate_temp_file_path(suffix=Util.extract_file_suffix(local_path))
    cv2.imwrite(dst_path, dst)
    remote_dst_path = image_oss.upload(dst_path)
    clearCache(dst_path)
    return remote_dst_path


def customStyleMove(content_path, style_path):
    remote_path = defineStyleMove.styleMove(content_path, style_path)
    return remote_path


def vegetableRec(local_path):
    return vt(local_path)


def image2Caption(local_path):
    caption = ccc.predict(local_path)
    return caption


def image2CaptionTextAnalysis(text):
    result = jieba.analyse.extract_tags(text, allowPOS=('n'))
    return result


def zzsfpRec(filename):
    result = zzsfp.predict(filename)
    return result


def faceFusion(template_remote_path, user_remote_path):
    output_img = facefusion(
        {'template': template_remote_path,
         'user': user_remote_path})['output_img']
    output_path = Util.generate_temp_file_path(suffix=Util.extract_file_suffix(template_remote_path))
    cv2.imwrite(output_path, output_img)
    remote_dst_path = image_oss.upload(output_path)
    clearCache(output_path)
    return remote_dst_path


def faceAttrGet(local_path):
    response = faceattrs(local_path)
    return response


if __name__ == "__main__":
    pass
