# coding=utf-8
# Created : 2023/1/9 14:59
# Author  : Zy
import cv2
import numpy as np
import onnxruntime as ort
from common.utils import Util
from PIL import Image, ImageFont, ImageDraw


class YOLOOnnxDetect:
    def __init__(self, onnx_filepath, labels, colors, nms_threshold=0.9, conf_thres=0.25):
        super(YOLOOnnxDetect, self).__init__()
        self.nms_threshold = nms_threshold
        self.conf_thres = conf_thres
        self.colors = colors
        self.labels = labels
        self.model = ort.InferenceSession(onnx_filepath, providers=['CPUExecutionProvider'])

    def padIm(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        blank_im = np.ones((640, 640, 3), dtype=np.uint8) * 114
        r = max(h, w)
        nh, nw = (640, int(w * 640 / r)) if r == h else (int(h * 640 / r), 640)
        new_im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        blank_im[int((640 - nh) / 2):int((640 - nh) / 2) + nh,
        int((640 - nw) / 2):int((640 - nw) / 2) + nw] = new_im
        return blank_im

    def preProc(self, padded_im: np.ndarray) -> np.ndarray:
        input_im = padded_im.transpose((2, 0, 1))[::-1]
        input_im = np.ascontiguousarray(input_im)
        input_im = np.array(input_im, dtype=np.float32)
        input_im = input_im / 255
        input_im = input_im[None] if len(input_im.shape) == 3 else input_im
        return input_im

    def xywh2xyxy(self, x: np.ndarray) -> np.ndarray:
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def postProc(self, prediction: np.ndarray, conf_thres: float = 0.25):
        bs = prediction.shape[0]
        output = [np.zeros((0, 6))] * bs
        xc = prediction[..., 4] > conf_thres
        for xi, x in enumerate(prediction):
            x = x[xc[xi]]
            if not x.shape[0]:
                continue
            boxes = self.xywh2xyxy(x[:, :4])
            x[:, 5:] *= x[:, 4:5]
            scores = np.max(x[:, 5:], axis=1)
            classid = np.argmax(x[:, 5:], axis=1)
            indices = cv2.dnn.NMSBoxes(boxes, scores, nms_threshold=self.nms_threshold, score_threshold=conf_thres)
            boxes = boxes[indices, :]
            scores = scores[indices][None].T
            classid = classid[indices][None].T
            results = np.concatenate((boxes, scores, classid), axis=1)
            output[xi] = results
        return output

    def scaleCoords(self, img1_shape, coords, img0_shape):
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        self.clipCoords(coords, img0_shape)
        return coords

    def clipCoords(self, boxes: np.ndarray, shape):
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

    def drowIm(self, img: np.ndarray, box: list, conf: float, cls: int):
        text = f"{conf:.2f}|{self.labels[cls]}"
        color = self.colors[cls]
        x = int(box[0])
        y = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])
        cv2.rectangle(img, (x, y), (x1, y1), color, 2)
        text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
        text_weight = text_size[0]
        text_height = text_size[1]
        cv2.rectangle(img, (x - 1, y - text_height - 1), (x + text_weight, y), color, -1)
        img = self.cv2ImgAddText(img, text, x - 1, y - text_height, (255, 255, 255), text_height)
        return img

    def cv2ImgAddText(self, img, text, left, top, textColor=(255, 255, 255, 0), textSize=20):
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        fontText = ImageFont.truetype(Util.app_path()+"/static/simsun.ttc", textSize, encoding="utf-8")
        draw.text((left, top), text, textColor, font=fontText)
        return cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)
