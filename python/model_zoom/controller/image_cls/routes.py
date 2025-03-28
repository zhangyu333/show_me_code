# coding=utf-8
# Created : 2023/1/9 17:51
# Author  : Zy
from flask import request, jsonify
from flask import Blueprint
from controller.image_cls.control import *
from common.file_utils import downloadImageFile
from common.file_utils import clearCache
from common.utils import MyResp

image_cls = Blueprint('image-cls', __name__)


@image_cls.route("/money-rec", methods=["POST"])
def moneyRec_():
    formdata = request.form
    remote_path = formdata.get("remote_path")
    local_path = downloadImageFile(remote_path)
    result = moneyRec(local_path)
    myresp = MyResp()()
    myresp['data'] = result
    clearCache(local_path)
    return jsonify(myresp)


@image_cls.route("/vegetable-rec", methods=["POST"])
def vegetableRec_():
    formdata = request.form
    remote_path = formdata.get("remote_path")
    local_path = downloadImageFile(remote_path)
    result = vegetableRec(local_path)
    myresp = MyResp()()
    myresp['data'] = result
    clearCache(local_path)
    return jsonify(myresp)



@image_cls.route("/zcy-rec", methods=["POST"])
def recZCY_():
    formdata = request.form
    remote_path = formdata.get("remote_path")
    local_path = downloadImageFile(remote_path)
    result = recZCY(local_path)
    myresp = MyResp()()
    myresp['data'] = result
    clearCache(local_path)
    return jsonify(myresp)