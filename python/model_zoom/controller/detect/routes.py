# coding=utf-8
# Created : 2022/12/30 14:14
# Author  : Zy
from flask import request
from flask import Blueprint
from controller.detect.control import *
from common.oss import OSS
from common.file_utils import *
from common.utils import MyResp

image_oss = OSS("hz-images")
obj_detect = Blueprint('detect', __name__)


@obj_detect.route("/mask_detect", methods=["POST"])
def maskDetect_():
    formdata = request.form
    remote_path = formdata.get("remote_path")
    local_path = downloadImageFile(remote_path)
    detect_res = maskDetect(local_path)
    detect_result_path = detect_res.get("local_file")
    detect_result_path1 = detect_res.get("local_file1")
    label_info = detect_res.get("label_info")
    remote_detect_result_path = image_oss.upload(detect_result_path)
    remote_detect_result_path1 = image_oss.upload(detect_result_path1)
    clearCache(local_path)
    clearCache(detect_result_path)
    clearCache(detect_result_path1)
    maskdetect_result = MyResp()()
    maskdetect_result['data'] = {
        "label_info": label_info,
        "remote_detect_result_path": remote_detect_result_path,
        "remote_detect_result_path1": remote_detect_result_path1,
    }
    return maskdetect_result


@obj_detect.route("/face_entry", methods=["POST"])
def faceEntry_():
    formdata = request.form
    remote_path = formdata.get("remote_path")
    local_path = downloadImageFile(remote_path)
    status = faceEntry(local_path)
    result = MyResp()()
    result['code'] = status
    clearCache(local_path)
    return result


@obj_detect.route("/face_match", methods=["POST"])
def faceMatch_():
    formdata = request.form
    remote_path = formdata.get("remote_path")
    max_score = formdata.get("max_score", 80)
    max_score = float(max_score)
    local_path = downloadImageFile(remote_path)
    match_result = faceMatch(local_path, max_score)
    clearCache(local_path)
    return match_result


@obj_detect.route("/handwrite_rec", methods=["POST"])
def handwriteRec_():
    formdata = request.form
    remote_path = formdata.get("remote_path")
    text = handwriteRec(remote_path)
    result = MyResp()()
    result['data']['text'] = text
    return result


@obj_detect.route("/only_face_match", methods=["POST"])
def onlyFaceMatch_():
    formdata = request.form
    remote_path1 = formdata.get("remote_path1")
    remote_path2 = formdata.get("remote_path2")
    local_path1 = downloadImageFile(remote_path1)
    local_path2 = downloadImageFile(remote_path2)
    match_result = onlyFaceMatch(local_path1, local_path2)
    result = MyResp()()
    result['data'] = match_result
    clearCache(local_path1)
    clearCache(local_path2)
    return result


@obj_detect.route("/face_detect", methods=["POST"])
def faceDetect_():
    formdata = request.form
    remote_path = formdata.get("remote_path")
    local_path = downloadImageFile(remote_path)
    detect_fign = faceDetect(local_path)
    result = MyResp()()
    info = "未检测到人脸,请重新上传!!!" if not detect_fign else "检测到人脸!!"
    code = 200 if detect_fign else 201
    result['code'] = code
    result['message'] = info
    clearCache(local_path)
    return result


@obj_detect.route("/trafficLight_detect", methods=["POST"])
def trafficLightDetect_():
    formdata = request.form
    remote_path = formdata.get("remote_path")
    local_path = downloadImageFile(remote_path)
    detect_result = trafficLightDetect(local_path)
    clearCache(local_path)
    result = MyResp()()
    result['data'] = detect_result
    return result


@obj_detect.route("/vehicle-count", methods=["POST"])
def vehicleCount_():
    formdata = request.form
    remote_path = formdata.get("remote_path")
    corners1 = formdata.get("corners1")
    corners2 = formdata.get("corners2")
    corners1 = eval(corners1) if corners1 else None
    corners2 = eval(corners2) if corners2 else None
    local_video_path = downloadFile(remote_path)
    count_result = vehicleCount(local_video_path, corners1, corners2)
    clearCache(local_video_path)
    result = MyResp()()
    result['data'] = count_result
    return result


@obj_detect.route("/vehicle-plate-rec", methods=["POST"])
def vehiclePlateRec():
    formdata = request.form
    remote_path = formdata.get("remote_path")
    local_path = downloadFile(remote_path)
    rec_result = licensePlateRec(local_path)
    clearCache(local_path)
    result = MyResp()()
    result['data'] = rec_result
    return result
