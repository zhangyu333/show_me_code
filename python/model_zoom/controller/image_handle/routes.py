# coding=utf-8
# Created : 2023/1/9 17:51
# Author  : Zy
from flask import request
from flask import Blueprint
from common.file_utils import downloadImageFile
from controller.image_handle.control import *

image_handle = Blueprint('image-handle', __name__)


@image_handle.route("/image2gray", methods=["POST"])
def imgGrayScale_():
    formdata = request.form
    remote_path = formdata.get("remote_path")
    local_path = downloadImageFile(remote_path)
    remote_url = imgGrayScale(local_path)
    clearCache(local_path)
    return {"data": {"remote_url": remote_url}, "code": 200, "message": ""}


@image_handle.route("/image_rgb_split", methods=["POST"])
def imgRGBsplit_():
    formdata = request.form
    remote_path = formdata.get("remote_path")
    local_path = downloadImageFile(remote_path)
    split_result = imgRGBSplit(local_path)
    clearCache(local_path)
    return {
        "message": "",
        "data": split_result,
        "code": 200
    }


@image_handle.route("/image_binary_sep", methods=["POST"])
def imgBinarySep_():
    formdata = request.form
    remote_path = formdata.get("remote_path")
    local_path = downloadImageFile(remote_path)
    binary_remote_url = imgBinarySep(local_path)
    clearCache(local_path)
    return {
        "message": "",
        "data": {"binary_remote_url": binary_remote_url},
        "code": 200
    }


@image_handle.route("/image_binary_detect", methods=["POST"])
def imgBinaryDetect_():
    formdata = request.form
    remote_path = formdata.get("remote_path")
    local_path = downloadImageFile(remote_path)
    result = imgBinaryDetect(local_path)
    clearCache(local_path)
    return {
        "message": "",
        "data": {"result": result},
        "code": 200
    }


@image_handle.route("/image_correct", methods=["POST"])
def imgCorrect_():
    formdata = request.form
    remote_path = formdata.get("remote_path")
    corners = formdata.get("corners")
    corners = eval(corners)
    local_path = downloadImageFile(remote_path)
    remote_url = imgCorrect(local_path, corners)
    clearCache(local_path)
    return {
        "message": "",
        "data": {"correct_img_remote_url": remote_url},
        "code": 200
    }


@image_handle.route("/data-augment", methods=["POST"])
def dataAugment_():
    formdata = request.form
    remote_path = formdata.get("remote_path")
    local_path = downloadImageFile(remote_path)
    augment_rsults = dataAugment(local_path)
    clearCache(local_path)
    return {
        "message": "",
        "data": {"augment_rsults": augment_rsults},
        "code": 200
    }
