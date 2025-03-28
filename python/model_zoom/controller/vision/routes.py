# -*- coding:utf-8 -*-
# @FileName  :routes.py
# @Time      :2023/4/7 10:37
# @Author    :ZY
from flask import request
from flask import Blueprint
from controller.vision.control import *
from common.file_utils import downloadImageFile

vision = Blueprint('vision', __name__)


@vision.route("/style-move", methods=["POST"])
def styleMove_():
    formdata = request.form
    remote_path = formdata.get("remote_path")
    local_path = downloadImageFile(remote_path)
    result = styleMove(local_path)
    clearCache(local_path)
    return {"data": result, "code": 200, "message": ""}


@vision.route("/style-move-juger", methods=["POST"])
def jugerStyleContent_():
    formdata = request.form
    remote_path = formdata.get("remote_path")
    alpha = formdata.get("alpha")
    local_path = downloadImageFile(remote_path)
    remote_dst_path = jugerStyleContent(local_path, float(alpha))
    clearCache(local_path)
    return {"data": {"remote_dst_url": remote_dst_path}, "code": 200, "message": ""}


@vision.route("/style-move-custom", methods=["POST"])
def customStyleMove_():
    formdata = request.form
    content_path = formdata.get("content_path")
    style_path = formdata.get("style_path")
    local_path1 = downloadImageFile(content_path)
    local_path2 = downloadImageFile(style_path)
    remote_path = customStyleMove(local_path1, local_path2)
    clearCache(local_path1)
    clearCache(local_path2)
    return {"data": {"remote_path": remote_path}, "code": 200, "message": ""}


@vision.route("/vegetable-rec", methods=["POST"])
def vegetableRec_():
    formdata = request.form
    remote_path = formdata.get("remote_path")
    local_path = downloadImageFile(remote_path)
    result = vegetableRec(local_path)
    clearCache(local_path)
    return {"data": result, "code": 200, "message": ""}


@vision.route("/image-to-caption", methods=["POST"])
def image2Caption_():
    formdata = request.form
    remote_path = formdata.get("remote_path")
    local_path = downloadImageFile(remote_path)
    caption = image2Caption(local_path)
    clearCache(local_path)
    return {"data": {'caption': caption}, "code": 200, "message": ""}


@vision.route("/image-to-caption-text-analysis", methods=["POST"])
def image2CaptionTextAnalysis_():
    formdata = request.form
    text = formdata.get("text")
    result = image2CaptionTextAnalysis(text)
    return {"data": {"result": result}, "code": 200, "message": ""}


@vision.route("/zzsfp-recognize", methods=["POST"])
def zzsfpRec_():
    formdata = request.form
    remote_path = formdata.get("remote_path")
    local_path = downloadImageFile(remote_path)
    result = zzsfpRec(local_path)
    clearCache(local_path)
    return {"data": {"result": result}, "code": 200, "message": ""}


@vision.route("/face-fusion", methods=["POST"])
def faceFusion_():
    formdata = request.form
    template_remote_path = formdata.get("template_remote_path")
    user_remote_path = formdata.get("user_remote_path")
    remote_dst_path = faceFusion(template_remote_path, user_remote_path)
    return remote_dst_path


@vision.route("/face-get-attrs", methods=["POST"])
def faceGetAttrs():
    formdata = request.form
    remote_path = formdata.get("template_remote_path")
    local_path = downloadImageFile(remote_path)
    response = faceAttrGet(local_path)
    clearCache(local_path)
    return response


if __name__ == "__main__":
    pass
