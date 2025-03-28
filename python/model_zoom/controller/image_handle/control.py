# coding=utf-8
# Created : 2023/1/9 17:51
# Author  : Zy
import cv2
import numpy as np
from common.oss import OSS
from common.utils import Util
from common.file_utils import clearCache
from controller.image_handle.data_augment import dataAugmentMain

image_oss = OSS("hz-images")


def imgGrayScale(filename: str):
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    suffix = Util.extract_file_suffix(filename)
    save_path = Util.generate_temp_file_path(suffix=suffix)
    cv2.imwrite(save_path, gray)
    remote_url = image_oss.upload(save_path)
    clearCache(save_path)
    return remote_url


def imgRGBSplit(filename: str):
    image = cv2.imread(filename)
    image_b, image_g, image_r = cv2.split(image)
    suffix = Util.extract_file_suffix(filename)
    b_save_path = Util.generate_temp_file_path(suffix=suffix)
    g_save_path = Util.generate_temp_file_path(suffix=suffix)
    r_save_path = Util.generate_temp_file_path(suffix=suffix)
    cv2.imwrite(b_save_path, image_b)
    cv2.imwrite(g_save_path, image_g)
    cv2.imwrite(r_save_path, image_r)
    b_remote_url = image_oss.upload(b_save_path)
    g_remote_url = image_oss.upload(g_save_path)
    r_remote_url = image_oss.upload(r_save_path)
    clearCache(b_save_path)
    clearCache(g_save_path)
    clearCache(r_save_path)
    return {
        "b_remote_url": b_remote_url,
        "g_remote_url": g_remote_url,
        "r_remote_url": r_remote_url,
    }


def imgBinarySep(filename: str):
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    suffix = Util.extract_file_suffix(filename)
    binary_save_path = Util.generate_temp_file_path(suffix=suffix)
    cv2.imwrite(binary_save_path, binary)
    binary_remote_url = image_oss.upload(binary_save_path)
    clearCache(binary_save_path)
    return binary_remote_url


def imgBinaryDetect(filename: str):
    image = cv2.imread(filename)
    h, w, c = image.shape
    max_r = max(100 / h, 100 / w)
    new_h, new_w = int(max_r * h), int(max_r * w)
    image = cv2.resize(image, (new_w, new_h))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.sum(binary) > new_h * new_w * 255 / 2:
        _, binary = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY_INV)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.dilate(binary, element, 1)
    contours, _ = cv2.findContours(binary, 0, 2)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append([x, y, x + w, y + h])
    for box in boxes:
        x, y, x1, y1 = box
        cv2.rectangle(image, (x, y), (x1, y1), [0, 175, 0], 2)

    boxes.sort(key=lambda box: box[0])
    binarys = []
    for box in boxes:
        x1, y1, x2, y2 = box
        block = binary[y1:y2, x1:x2]
        block[block > 1] = 1
        binarys.append(block.tolist())

    suffix = Util.extract_file_suffix(filename)
    save_path = Util.generate_temp_file_path(suffix=suffix)
    cv2.imwrite(save_path, image)
    remote_url = image_oss.upload(save_path)
    clearCache(save_path)
    return {"remote_url": remote_url, "binarys": binarys}


def imgCorrect(filename: str, corners: list):
    img = cv2.imread(filename)
    blank = np.zeros(img.shape[:2], np.uint8)
    cv2.line(blank, corners[0], corners[1], (255, 255, 255))
    cv2.line(blank, corners[1], corners[2], (255, 255, 255))
    cv2.line(blank, corners[2], corners[3], (255, 255, 255))
    cv2.line(blank, corners[3], corners[0], (255, 255, 255))
    contours, _ = cv2.findContours(blank, 0, 1)
    bx, by, bw, bh = cv2.boundingRect(contours[0])
    rect = cv2.minAreaRect(contours[0])
    points = cv2.boxPoints(rect)
    point0 = [points[0][0], points[0][1]]
    point1 = [points[3][0], points[3][1]]
    point2 = [points[2][0], points[2][1]]
    point3 = [points[1][0], points[1][1]]
    src = np.float32([point0, point1, point2, point3])
    dst = np.float32([[0, 0], [0, bh], [bw, bh], [bw, 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, M, (bw, bh))
    suffix = Util.extract_file_suffix(filename)
    save_path = Util.generate_temp_file_path(suffix=suffix)
    cv2.imwrite(save_path, out)
    remote_url = image_oss.upload(save_path)
    clearCache(save_path)
    return remote_url

def dataAugment(filename: str):
    return dataAugmentMain(filename)


if __name__ == '__main__':
    pass