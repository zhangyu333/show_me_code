# coding=utf-8
# Created : 2022/11/15 13:41
# Author  : Zy
import cv2
import copy
from common.utils import Util
from common.vision.yolo_detect import YOLOOnnxDetect


class MaskDetect(YOLOOnnxDetect):
    def __init__(self):
        super(MaskDetect, self).__init__(
            onnx_filepath=Util.app_path() + '/models/detect/mask_detect.onnx',
            labels=['戴口罩', '没带口罩', '未正确佩戴'],
            colors=[[0, 175, 0], [0, 0, 175], [0, 175, 175]],
        )

    def detect(self, filename: str):
        img = cv2.imread(filename)
        img1 = copy.deepcopy(img)
        padded_im = self.padIm(img)
        input_im = self.preProc(padded_im)
        outputs = self.model.run(
            [self.model.get_outputs()[0].name],
            {self.model.get_inputs()[0].name: input_im})[0]
        post_results = self.postProc(outputs, self.conf_thres)[0]
        post_results[:, :4] = self.scaleCoords(
            input_im.shape[2:], post_results[:, :4],
            img.shape).round()
        label_info = {k: 0 for k in self.labels}
        label_info["total"] = 0
        for *xyxy, conf, cls in reversed(post_results):
            if conf > self.conf_thres:
                img = self.drowIm(img, xyxy, conf, int(cls))
                img1 = cv2.rectangle(img1, (int(xyxy[0]),int(xyxy[1])), (int(xyxy[2]),int(xyxy[3])),[0,175,175],2)
                label = self.labels[int(cls)]
                label_info[label] += 1
                label_info["total"] += 1
        local_file = Util.generate_temp_file_path(
            suffix=Util.extract_file_suffix(filename)
        )
        local_file1 = Util.generate_temp_file_path(
            suffix=Util.extract_file_suffix(filename)
        )
        cv2.imwrite(local_file, img)
        cv2.imwrite(local_file1, img1)
        return {
            "local_file": local_file,
            "local_file1": local_file1,
            "label_info": label_info
        }
