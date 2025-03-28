# coding=utf-8
# Created : 2023/1/9 17:51
# Author  : Zy
from flask import request
from flask import Blueprint
from controller.voice.control import *
from common.oss import OSS
from common.file_utils import *

voiceoss = OSS("hz-voices")
voice = Blueprint('voice', __name__)


@voice.route("/word_to_voice", methods=["POST"])
def wordToVoice():
    formdata = request.form
    text = formdata.get("text")
    speed = formdata.get("speed")
    volume = formdata.get("volume")
    speed = float(speed)
    volume = int(volume)
    local_path = Util.generate_temp_file_path(suffix="wav")
    status = myTTS(text, local_path)
    temp_path = Util.generate_temp_file_path(suffix="wav")
    changeVoiceSpeed(local_path, speed, temp_path)
    temp_path1 = Util.generate_temp_file_path(suffix="wav")
    changeVoiceVolume(temp_path, volume, temp_path1)
    remote_path = voiceoss.upload(temp_path1)
    clearCache(local_path)
    clearCache(temp_path)
    clearCache(temp_path1)
    return {
        "message": "",
        "data": {"remote_path":remote_path},
        "code": status
    }


@voice.route("/voice_feature_image", methods=["POST"])
def getVoiceFeatureImg():
    formdata = request.form
    remote_file = formdata.get("remote_file")
    local_path = voiceoss.download(remote_file)
    wav_path = mp3ToWav(local_path)
    result = saveVoiceFeatureImages(wav_path)
    clearCache(local_path)
    clearCache(wav_path)
    return {
        "message": "",
        "data": result,
        "code": 200
    }



@voice.route("/voice_text2pinyin", methods=["POST"])
def voiceModelMatch_():
    formdata = request.form
    text = formdata.get("text")
    result = voiceModelMatch(text)
    return {
        "message": "",
        "data": result,
        "code": 200
    }


@voice.route("/voice_rec", methods=["POST"])
def voiceRec_():
    formdata = request.form
    remote_url = formdata.get("remote_url")
    local_path = voiceoss.download(remote_url)
    wav_path = mp3ToWav(local_path)
    text = voiceRec(wav_path)
    clearCache(local_path)
    clearCache(wav_path)
    return {
        "message": "",
        "data": {"text": text},
        "code": 200
    }


@voice.route("/voiceprint_register", methods=["POST"])
def voiceprintRegister_():
    formdata = request.form
    remote_url = formdata.get("remote_url")
    wav_id = formdata.get("ID")
    experiment_id = formdata.get("experimentID")
    if not experiment_id or not wav_id or not remote_url:
        return {"message": "params error!! please check your params","data": {},"code": 203}
    local_path = voiceoss.download(remote_url)
    wav_path = mp3ToWav(local_path)
    result = voiceprintRegister(wav_path, wav_id, experiment_id)
    clearCache(local_path)
    clearCache(wav_path)
    return {
        "message": result["message"],
        "data": {},
        "code": result["code"]
    }


@voice.route("/voiceprint_match", methods=["POST"])
def voiceprintMatch_():
    formdata = request.form
    remote_url = formdata.get("remote_url")
    experiment_id = formdata.get("experimentID")
    if not experiment_id or not remote_url:
        return {"message": "params error!! please check your params","data": {},"code": 203}
    local_path = voiceoss.download(remote_url)
    wav_path = mp3ToWav(local_path)
    result = voiceprintMatch(wav_path,experiment_id)
    clearCache(local_path)
    clearCache(wav_path)
    return {
        "message": result["message"],
        "data": {"result": result["result"]},
        "code": result["code"]
    }


@voice.route("/voiceprint_onlymatch", methods=["POST"])
def voiceprintOnlyMatch_():
    formdata = request.form
    remote_url1 = formdata.get("remote_url1")
    remote_url2 = formdata.get("remote_url2")
    local_path1 = voiceoss.download(remote_url1)
    local_path2 = voiceoss.download(remote_url2)
    wav_path1 = mp3ToWav(local_path1)
    wav_path2 = mp3ToWav(local_path2)
    coef = voiceprintOnlyMatch(wav_path1, wav_path2)
    clearCache(local_path1)
    clearCache(local_path2)
    clearCache(wav_path2)
    clearCache(wav_path1)
    return {
        "message": "",
        "data": {"coef": float(coef)},
        "code": 200
    }

