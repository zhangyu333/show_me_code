# -*- coding:utf-8 -*-
# @FileName  :main.py
# @Time      :2024/5/23 10:46
# @Author    :ZY
import cv2
import torch
import utils
import numpy as np
import torch.nn as nn
from typing import Union
from face_detect import FastFaceDetect

model = utils.getArch("ResNet50", 90)
model.load_state_dict(torch.load("models/L2CSNet_gaze360.pkl", map_location="cpu"))
model.eval()

fd = FastFaceDetect()

softmax = nn.Softmax(dim=1)

idx_tensor = [idx for idx in range(90)]
idx_tensor = torch.FloatTensor(idx_tensor)


def predictGaze(frame: Union[np.ndarray, torch.Tensor]):
    # Prepare input
    if isinstance(frame, np.ndarray):
        img = utils.prep_input_numpy(frame, "cpu")
    elif isinstance(frame, torch.Tensor):
        img = frame
    else:
        raise RuntimeError("Invalid dtype for input")

    # Predict
    # torch.onnx.export(
    #     model,
    #     img,
    #     "models/gaze_estimate.onnx",
    #     input_names=["inputs"],
    #     output_names=["gaze_pitchs", "gaze_yaws"],
    #     dynamic_axes={
    #         "inputs":[0],
    #         "gaze_pitchs":[0],
    #         "gaze_yaws":[0],
    #     }
    # )
    gaze_pitch, gaze_yaw = model(img)
    pitch_predicted = softmax(gaze_pitch)
    yaw_predicted = softmax(gaze_yaw)

    # Get continuous predictions in degrees.
    pitch_predicted = torch.sum(pitch_predicted.data * idx_tensor, dim=1) * 4 - 180
    yaw_predicted = torch.sum(yaw_predicted.data * idx_tensor, dim=1) * 4 - 180

    pitch_predicted = pitch_predicted.cpu().detach().numpy() * np.pi / 180.0
    yaw_predicted = yaw_predicted.cpu().detach().numpy() * np.pi / 180.0

    return pitch_predicted, yaw_predicted


def draw_gaze(a, b, c, d, image_in, pitchyaw, thickness=2, color=(255, 255, 0), sclae=2.0):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    length = c
    pos = (int(a + c / 2.0), int(b + d / 2.0))
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[0]) * np.cos(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[1])
    cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                    tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                    thickness, cv2.LINE_AA, tipLength=0.18)
    return image_out


def draw_bbox(frame: np.ndarray, bbox: np.ndarray):
    x_min = int(bbox[0])
    if x_min < 0:
        x_min = 0
    y_min = int(bbox[1])
    if y_min < 0:
        y_min = 0
    x_max = int(bbox[2])
    y_max = int(bbox[3])

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

    return frame


def render(frame, bboxes, pitchlist, yawlist):
    # Draw bounding boxes
    for bbox in bboxes:
        frame = draw_bbox(frame, bbox)

    # Draw Gaze
    for i in range(len(pitchlist)):

        bbox = bboxes[i]
        pitch = pitchlist[i]
        yaw = yawlist[i]

        # Extract safe min and max of x,y
        x_min = int(bbox[0])
        if x_min < 0:
            x_min = 0
        y_min = int(bbox[1])
        if y_min < 0:
            y_min = 0
        x_max = int(bbox[2])
        y_max = int(bbox[3])

        # Compute sizes
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        draw_gaze(x_min, y_min, bbox_width, bbox_height, frame, (pitch, yaw), color=(0, 0, 255))

    return frame


if __name__ == "__main__":
    image = cv2.imread("test.jpg")
    results = fd(image)
    boxes = results.get("boxes")
    faces = []
    for (x1, y1, x2, y2) in boxes:
        face = image[y1:y2, x1:x2]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        faces.append(face)

    pitch, yaw = predictGaze(np.stack(faces))
    # [-1.7801286448003404, -1.0303932710343067, 1.3177827074319186, 0.476214215586955]
    # [-0.6774866097113286, 0.07750784141733912, -0.2537004486081095, -0.2643914424881236]

    # [-1.7500647 -1.0411583  1.2898939  0.4604688]
    # [-0.6766139   0.0705791  -0.27184165 -0.21597224]
    pitch = pitch.tolist()
    yaw = yaw.tolist()

    image = render(image, boxes, pitch, yaw)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
