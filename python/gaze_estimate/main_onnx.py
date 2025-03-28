# -*- coding:utf-8 -*-
# @FileName  :main_onnx.py
# @Time      :2024/5/23 11:59
# @Author    :ZY

#  这个可以做的东西有很多， 然后 c++那边还有一个头部姿态估计， 可以估计头部姿态方向
#  然后根据这个gaze模型 去估计出眼部的的视线姿态， 可以做出智能考试监控系统，
#  或者 三维计算，视线空间计算算法，更加真实的MR计算等等的东西

import cv2
import numpy as np
from face_detect import FastFaceDetect
from gaze_estimate import GazeEstimate

ge_model = GazeEstimate()
fd_model = FastFaceDetect()


def drawGaze(a, b, c, d, image, pitch_yaw, thickness=2, color=(255, 255, 0)):
    length = c
    pos = (int(a + c / 2.0), int(b + d / 2.0))
    dx = -length * np.sin(pitch_yaw[0]) * np.cos(pitch_yaw[1])
    dy = -length * np.sin(pitch_yaw[1])
    cv2.arrowedLine(image, tuple(np.round(pos).astype(np.int32)),
                    tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                    thickness, cv2.LINE_AA, tipLength=0.18)
    return image


def render(frame, bboxes, pitchlist, yawlist):
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

    for i in range(len(pitchlist)):
        bbox = bboxes[i]
        pitch = pitchlist[i]
        yaw = yawlist[i]

        x_min = int(bbox[0])
        y_min = int(bbox[1])
        x_max = int(bbox[2])
        y_max = int(bbox[3])

        # Compute sizes
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        drawGaze(x_min, y_min, bbox_width, bbox_height, frame, (pitch, yaw), color=(0, 0, 255))

    return frame


if __name__ == "__main__":
    image = cv2.imread("test.jpg")
    fd_results = fd_model(image)
    boxes = fd_results.get("boxes")
    faces = []
    for (x1, y1, x2, y2) in boxes:
        face = image[y1:y2, x1:x2]
        faces.append(face)
    pitch, yaw = ge_model(faces)
    # pitch 注视前方默认为 0,0  pitch 左正右负, yaw 上正下负
    image = render(image, boxes, pitch, yaw)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
