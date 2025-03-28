# coding=utf-8
# Created : 2023/1/5 16:10
# Author  : Zy
from controller.detect.face import FaceRec
from controller.detect.mask_detect import MaskDetect
from controller.detect.hwtRec import HandwriteRec
from controller.detect.faceMatch import FaceMatch
from controller.detect.trafficLightDetect import TrafficLightDetect
from controller.detect.vehicle_count.count import VehicleCount
from controller.detect.car_plate_rec import LicensePlateRec

license_plate_rec = LicensePlateRec()
trafficlightdetect = TrafficLightDetect()
handwriterec = HandwriteRec()
maskdetect = MaskDetect()
facerec = FaceRec()
facematcher = FaceMatch()


def maskDetect(filename: str):
    detect_res = maskdetect.detect(filename)
    return detect_res


def faceEntry(filename: str):
    status = facerec.faceRegiste(filename)
    return status


def faceMatch(filename: str, max_score: float):
    match_result = facerec.matchFace(filename, max_score)
    return match_result


def handwriteRec(filename: str):
    text = handwriterec.predict(filename)
    return text


def onlyFaceMatch(filename1: str, filename2: str):
    result = facematcher.faceMatch(filename1, filename2)
    return result


def faceDetect(filename: str):
    sign = facematcher.faceDetect(filename)
    return sign


def trafficLightDetect(filename: str):
    detect_result = trafficlightdetect.detect(filename)
    return detect_result


def vehicleCount(video_path, corners1, corners2):
    result = VehicleCount(video_path, corners1, corners2)()
    return result

def licensePlateRec(filename: str):
    result = license_plate_rec(filename)
    return result
