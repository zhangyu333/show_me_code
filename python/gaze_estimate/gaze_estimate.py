# -*- coding:utf-8 -*-
# @FileName  :gaze_estimate_onnx.py
# @Time      :2024/5/23 11:27
# @Author    :ZY
import cv2
import numpy as np
import onnxruntime
from scipy.special import softmax


def preProc(input_image):
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(input_image, (448, 448)).astype("float32")
    input_image /= 255.
    input_image -= np.array([0.485, 0.456, 0.406])
    input_image /= np.array([0.229, 0.224, 0.225])
    input_image = input_image.transpose((2, 0, 1))
    return input_image


idx_tensor = np.array([idx for idx in range(90)], dtype=np.float32)


class GazeEstimate:
    def __init__(self):
        super(GazeEstimate, self).__init__()
        self.model = onnxruntime.InferenceSession("models/gaze_estimate.onnx")

    def __call__(self, faces):
        inputs = [preProc(face) for face in faces]
        inputs = np.stack(inputs)
        outputs = self.model.run(None, {"inputs": inputs})
        gaze_pitches, gaze_yaws = outputs
        pitch_predicted = softmax(gaze_pitches, axis=1)
        yaw_predicted = softmax(gaze_yaws, axis=1)

        pitch_predicted = [np.sum(pitch * idx_tensor) * 4 - 180 for pitch in pitch_predicted]
        yaw_predicted = [np.sum(yaw * idx_tensor) * 4 - 180 for yaw in yaw_predicted]
        pitch_predicted = [pitch * np.pi / 180.0 for pitch in pitch_predicted]
        yaw_predicted = [yaw * np.pi / 180.0 for yaw in yaw_predicted]
        return pitch_predicted, yaw_predicted
