# coding=utf-8
# Created : 2023/1/31 10:20
# Author  : Zy
import cv2
import copy
import onnxruntime
import numpy as np
from common.oss import OSS
from common.utils import Util
from common.file_utils import clearCache
from PIL import Image, ImageDraw, ImageFont

image_oss = OSS("hz-images")


class LicensePlateRec:
    def __init__(self):
        super(LicensePlateRec, self).__init__()
        self.plateName = r"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕" \
                         r"甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"
        self.mean_value = 0.588
        self.std_value = 0.193
        self.img_size = (640, 640)
        self.__detect_model = onnxruntime.InferenceSession(Util.app_path() + "/models/ocr/plate_detect.onnx")
        self.__rec_model = onnxruntime.InferenceSession(Util.app_path() + "/models/ocr/plate_rec.onnx")
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
        self.result = {}
        self.plate_image_remote_urls = []

    def decodePlate(self, preds):
        pre = 0
        newPreds = []
        for i in range(len(preds)):
            if preds[i] != 0 and preds[i] != pre:
                newPreds.append(preds[i])
            pre = preds[i]
        plate = ""
        for i in newPreds:
            plate += self.plateName[int(i)]
        return plate

    def recPreProc(self, img):
        img = cv2.resize(img, (168, 48))
        plate_image_local_path = Util.generate_temp_file_path(suffix="jpg")
        cv2.imwrite(plate_image_local_path, img)
        plate_image_remote_path = image_oss.upload(plate_image_local_path)
        clearCache(plate_image_local_path)
        self.plate_image_remote_urls.append(plate_image_remote_path)
        img = img.astype(np.float32)
        img = (img / 255 - self.mean_value) / self.std_value
        img = img.transpose(2, 0, 1)
        img = img.reshape(1, *img.shape)
        return img

    def detectPreProc(self, img, img_size):
        img, r, left, top = self.letterBox(img, img_size)
        detect_pre_proc_image_local_path = Util.generate_temp_file_path(suffix="jpg")
        cv2.imwrite(detect_pre_proc_image_local_path, img)
        detect_pre_proc_image_remote_path = image_oss.upload(detect_pre_proc_image_local_path)
        clearCache(detect_pre_proc_image_local_path)
        self.result["detect_pre_proc_image_remote_url"] = detect_pre_proc_image_remote_path
        img = img[:, :, ::-1].transpose(2, 0, 1).copy().astype(np.float32)
        img = img / 255
        img = img.reshape(1, *img.shape)
        return img, r, left, top

    def getPlateResult(self, img):
        img = self.recPreProc(img)
        y_onnx = self.__rec_infer(img)
        index = np.argmax(y_onnx[0], axis=1)
        plate_no = self.decodePlate(index)
        return plate_no

    def getSpliMerge(self, img):
        h, w, c = img.shape
        img_upper = img[0:int(5 / 12 * h), :]
        img_lower = img[int(1 / 3 * h):, :]
        img_upper = cv2.resize(img_upper, (img_lower.shape[1], img_lower.shape[0]))
        new_img = np.hstack((img_upper, img_lower))
        return new_img

    def orderPoints(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def fourPointTransform(self, image, pts):
        rect = self.orderPoints(pts)
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    def letterBox(self, img, size=(640, 640)):  #
        h, w, c = img.shape
        r = min(size[0] / h, size[1] / w)
        new_h, new_w = int(h * r), int(w * r)
        top = int((size[0] - new_h) / 2)
        left = int((size[1] - new_w) / 2)

        bottom = size[0] - new_h - top
        right = size[1] - new_w - left
        img_resize = cv2.resize(img, (new_w, new_h))
        img = cv2.copyMakeBorder(img_resize, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT,
                                 value=(114, 114, 114))
        return img, r, left, top

    def xywh2xyxy(self, boxes):
        xywh = copy.deepcopy(boxes)
        xywh[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xywh[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xywh[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xywh[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return xywh

    def nms(self, boxes, iou_thresh):
        index = np.argsort(boxes[:, 4])[::-1]
        keep = []
        while index.size > 0:
            i = index[0]
            keep.append(i)
            x1 = np.maximum(boxes[i, 0], boxes[index[1:], 0])
            y1 = np.maximum(boxes[i, 1], boxes[index[1:], 1])
            x2 = np.minimum(boxes[i, 2], boxes[index[1:], 2])
            y2 = np.minimum(boxes[i, 3], boxes[index[1:], 3])

            w = np.maximum(0, x2 - x1)
            h = np.maximum(0, y2 - y1)

            inter_area = w * h
            union_area = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]) + (
                    boxes[index[1:], 2] - boxes[index[1:], 0]) * (boxes[index[1:], 3] - boxes[index[1:], 1])
            iou = inter_area / (union_area - inter_area)
            idx = np.where(iou <= iou_thresh)[0]
            index = index[idx + 1]
        return keep

    def restoreBox(sefl, boxes, r, left, top):  # 返回原图上面的坐标
        boxes[:, [0, 2, 5, 7, 9, 11]] -= left
        boxes[:, [1, 3, 6, 8, 10, 12]] -= top

        boxes[:, [0, 2, 5, 7, 9, 11]] /= r
        boxes[:, [1, 3, 6, 8, 10, 12]] /= r
        return boxes

    def detectPostProc(self, dets, r, left, top, conf_thresh=0.3, iou_thresh=0.5):  # 检测后处理
        choice = dets[:, :, 4] > conf_thresh
        dets = dets[choice]
        dets[:, 13:15] *= dets[:, 4:5]
        box = dets[:, :4]
        boxes = self.xywh2xyxy(box)
        score = np.max(dets[:, 13:15], axis=-1, keepdims=True)
        index = np.argmax(dets[:, 13:15], axis=-1).reshape(-1, 1)
        output = np.concatenate((boxes, score, dets[:, 5:13], index), axis=1)
        reserve_ = self.nms(output, iou_thresh)
        output = output[reserve_]
        output = self.restoreBox(output, r, left, top)
        return output

    def recPlate(self, outputs, img0):
        dict_list = []
        cut_block_remote_paths = []
        for output in outputs:
            result_dict = {}
            rect = [int(i) for i in output[:4].tolist()]
            land_marks = output[5:13].reshape(4, 2)
            roi_img = self.fourPointTransform(img0, land_marks)
            label = int(output[-1])
            score = output[4]
            if label == 1:
                roi_img = self.getSpliMerge(roi_img)
            cut_block_local_path = Util.generate_temp_file_path(suffix="jpg")
            cv2.imwrite(cut_block_local_path, roi_img)
            cut_block_remote_path = image_oss.upload(cut_block_local_path)
            clearCache(cut_block_local_path)
            cut_block_remote_paths.append(cut_block_remote_path)
            plate_no = self.getPlateResult(roi_img)
            result_dict['rect'] = rect
            result_dict['landmarks'] = land_marks.astype(int).tolist()
            result_dict['plate_no'] = plate_no
            result_dict['roi_height'] = roi_img.shape[0]
            result_dict['score'] = round(score, 2)
            dict_list.append(result_dict)

        self.result["cut_plate_remote_urls"] = " ".join(cut_block_remote_paths)
        self.result["cut_plate_proc_remote_urls"] = " ".join(self.plate_image_remote_urls)

        return dict_list

    def cv2ImgAddText(self, img, text, left, top, textColor=(0, 255, 0), textSize=20):  # 将识别结果画在图上
        if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        fontText = ImageFont.truetype(
            Util.app_path() + "/static/platech.ttf", textSize, encoding="utf-8")
        draw.text((left, top), text, textColor, font=fontText)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    def drawResult(self, orgimg, dict_list):
        only_detect_img = copy.deepcopy(orgimg)
        result_str = ""
        for result in dict_list:
            rect_area = result['rect']

            x, y, w, h = rect_area[0], rect_area[1], rect_area[2] - rect_area[0], rect_area[3] - rect_area[1]
            padding_w = 0.05 * w
            padding_h = 0.11 * h
            rect_area[0] = max(0, int(x - padding_w))
            rect_area[1] = min(orgimg.shape[1], int(y - padding_h))
            rect_area[2] = max(0, int(rect_area[2] + padding_w))
            rect_area[3] = min(orgimg.shape[0], int(rect_area[3] + padding_h))

            height_area = result['roi_height']
            landmarks = result['landmarks']
            result = result['plate_no']
            result_str += result + " "
            for i in range(4):
                cv2.circle(only_detect_img, (int(landmarks[i][0]), int(landmarks[i][1])), 5, self.colors[i], -1)
                cv2.circle(orgimg, (int(landmarks[i][0]), int(landmarks[i][1])), 5, self.colors[i], -1)
            cv2.rectangle(orgimg, (rect_area[0], rect_area[1]), (rect_area[2], rect_area[3]), (0, 0, 255), 2)  # 画框
            cv2.rectangle(only_detect_img, (rect_area[0], rect_area[1]), (rect_area[2], rect_area[3]), (0, 0, 255),
                          2)  # 画框

            detect_result_local_path = Util.generate_temp_file_path(suffix="jpg")
            cv2.imwrite(detect_result_local_path, only_detect_img)
            detect_result_remote_path = image_oss.upload(detect_result_local_path)
            clearCache(detect_result_local_path)
            self.result["detect_result_remote_url"] = detect_result_remote_path

            if len(result) >= 1:
                orgimg = self.cv2ImgAddText(orgimg, result, rect_area[0] - height_area, rect_area[1] - height_area - 10,
                                            (255, 0, 0), height_area)
        return orgimg

    def __detect_infer(self, img):
        return self.__detect_model.run(
            [self.__detect_model.get_outputs()[0].name],
            {self.__detect_model.get_inputs()[0].name: img})[0]

    def __rec_infer(self, img):
        return self.__rec_model.run(
            [self.__rec_model.get_outputs()[0].name],
            {self.__rec_model.get_inputs()[0].name: img})[0]

    def __call__(self, img_path):
        self.result = {}
        self.plate_image_remote_urls = []
        img = cv2.imread(img_path)
        img0 = copy.deepcopy(img)
        img, r, left, top = self.detectPreProc(img, self.img_size)
        detect_result = self.__detect_infer(img)
        detect_outputs = self.detectPostProc(detect_result, r, left, top)
        results = self.recPlate(detect_outputs, img0)
        self.result["ocr_results"] = results
        draw = self.drawResult(img0, results)
        result_local_path = Util.generate_temp_file_path(suffix="jpg")
        cv2.imwrite(result_local_path, draw)
        result_remote_path = image_oss.upload(result_local_path)
        clearCache(result_local_path)
        self.result["result_remote_url"] = result_remote_path
        return self.result


if __name__ == '__main__':
    lpr = LicensePlateRec()
    print(lpr("/Users/zhangyu/Desktop/ml/_project_pytorch/车牌检测识别/test.jpeg"))
