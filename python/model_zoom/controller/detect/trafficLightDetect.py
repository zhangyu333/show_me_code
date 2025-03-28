# coding=utf-8
# Created : 2023/2/13 17:44
# Author  : Zy
import cv2
import random
import copy
from common.oss import OSS
from common.utils import Util
from common.vision.yolo_detect import YOLOOnnxDetect
from common.file_utils import clearCache


class TrafficLightDetect(YOLOOnnxDetect):
    def __init__(self):
        super(TrafficLightDetect, self).__init__(
            onnx_filepath=Util.app_path() + '/models/detect/trafficlightdetect.onnx',
            labels=['RL', 'RR', 'RS', 'RF', 'YF', 'YL', 'YR', 'YS', 'GL', 'GR', 'GS', 'GF'],
            colors=[[random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)] for i in
                    range(19)]
        )
        self.label_map = {
            'RL': '红色左转',
            'RR': '红色右转',
            'RS': '禁止通行',
            'RF': '红色直行',
            'YF': '黄色直行',
            'YL': '黄色左转',
            'YR': '黄色右转',
            'YS': '通行警告',
            'GL': '绿色左转',
            'GR': '绿色右转',
            'GS': '允许通行',
            'GF': '绿色直行'
        }
        self.imageoss = OSS("hz-images")

    def drowIm(self, img, box: list, conf: float, cls: int):
        text = self.labels[cls]
        x = int(box[0])
        y = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])
        cv2.rectangle(img, (x, y), (x1, y1), [0, 0, 255], 5)
        cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 255], 3)
        return img

    def detect(self, filename: str):
        suffix = Util.extract_file_suffix(filename)
        img = cv2.imread(filename)
        img1 = copy.deepcopy(img)
        padded_im = self.padIm(img)
        padded_result_savepath = Util.generate_temp_file_path(suffix=suffix)
        cv2.imwrite(padded_result_savepath, padded_im)
        padded_result_remote_savepath = self.imageoss.upload(padded_result_savepath)
        clearCache(padded_result_savepath)
        input_im = self.preProc(padded_im)
        outputs = self.model.run([self.model.get_outputs()[0].name], {self.model.get_inputs()[0].name: input_im})[0]
        post_results = self.postProc(outputs, self.conf_thres)[0]
        post_results[:, :4] = self.scaleCoords(input_im.shape[2:], post_results[:, :4], img.shape).round()
        results = []
        idx = 1
        for *xyxy, conf, cls in reversed(post_results):
            if not self.labels[int(cls)]:
                continue
            img = self.drowIm(img, xyxy, conf, int(cls))
            img1 = cv2.rectangle(img1, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), [0, 0, 255], 5)
            label = self.labels[int(cls)]
            results.append([idx, self.label_map[label], conf])
            idx += 1
        detect_result_savepath = Util.generate_temp_file_path(suffix=suffix)
        cv2.imwrite(detect_result_savepath, img)
        detect_result_remote_savepath = self.imageoss.upload(detect_result_savepath)
        clearCache(detect_result_savepath)
        only_detect_result_savepath = Util.generate_temp_file_path(suffix=suffix)
        cv2.imwrite(only_detect_result_savepath, img1)
        only_detect_result_remote_savepath = self.imageoss.upload(only_detect_result_savepath)
        clearCache(only_detect_result_savepath)
        detect_results = {
            "result": results,
            "detect_result_remote_savepath": detect_result_remote_savepath,
            "padded_result_remote_savepath": padded_result_remote_savepath,
            "only_detect_result_savepath": only_detect_result_remote_savepath,
            "total": len(results)
        }
        return detect_results


if __name__ == '__main__':
    tld = TrafficLightDetect()
    detect_results = tld.detect("/Users/zhangyu/Desktop/ml/_project_pytorch/红绿灯检测/test2.jpg")
    print(detect_results)
