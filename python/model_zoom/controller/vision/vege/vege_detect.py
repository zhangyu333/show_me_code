# coding=utf-8
# Created : 2022/11/15 13:41
# Author  : Zy
import cv2
import random
import numpy as np
import onnxruntime as ort
from common.utils import Util

class VegeDetect():
    def __init__(self):
        super(VegeDetect, self).__init__()
        self.nms_threshold = 0.5
        self.conf_thres = 0.8
        self.model = ort.InferenceSession(Util.app_path() + "/models/detect/vege_detect.onnx")

    def padIm(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        blank_im = np.ones((640, 640, 3), dtype=np.uint8) * 114
        r = max(h, w)
        nh, nw = (640, int(w * 640 / r)) if r == h else (int(h * 640 / r), 640)
        new_im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        blank_im[int((640 - nh) / 2):int((640 - nh) / 2) + nh, int((640 - nw) / 2):int((640 - nw) / 2) + nw] = new_im
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

    def clipCoords(self, boxes, shape):
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

    def setColor(self):
        return [random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)]

    def __call__(self, filename: str):
        img = cv2.imread(filename)
        padded_im = self.padIm(img)
        input_im = self.preProc(padded_im)
        outputs = self.model.run([self.model.get_outputs()[0].name], {self.model.get_inputs()[0].name: input_im})[0]
        post_results = self.postProc(outputs, self.conf_thres)[0]
        post_results[:, :4] = self.scaleCoords(input_im.shape[2:], post_results[:, :4], img.shape).round()
        blocks = []
        idx = 0
        for *box, conf, cls in reversed(post_results):
            x = int(box[0])
            y = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])
            block = img[y:y1, x:x1]
            blocks.append(block)
            color = self.setColor()
            cv2.rectangle(img, (x, y), (x1, y1), color)
            text_width, text_height = cv2.getTextSize(str(idx), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(img, (x - text_width, y), (x, y + text_height), color, -1)
            cv2.putText(img, str(idx), (x - text_width, y + text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        [255, 255, 255], 1)
            idx += 1
        return img, blocks
