# coding=utf-8
# Created : 2023/2/13 12:31
# Author  : Zy
import cv2
import dlib
import copy
import numpy as np
from common.oss import OSS
from common.utils import Util
from common.file_utils import clearCache

class FaceMatch():
    def __init__(self):
        super(FaceMatch, self).__init__()
        # 加载模型
        self.face_detect_model_path = Util.app_path() + '/models/face/mmod_human_face_detector.dat'
        self.face_shape_predictor_path = Util.app_path() + '/models/face/shape_predictor_68_face_landmarks.dat'
        self.face_rec_model_path = Util.app_path() + '/models/face/dlib_face_recognition_resnet_model_v1.dat'
        self.face_detector = dlib.cnn_face_detection_model_v1(self.face_detect_model_path)
        self.face_shape_predictor = dlib.shape_predictor(self.face_shape_predictor_path)
        self.face_recognition_model = dlib.face_recognition_model_v1(self.face_rec_model_path)
        # 参数
        self.face_padding = 0.15
        self.face_resample = 100
        self.imageoss = OSS("hz-images")

    def cacuEuclideanDistance(self, source_representation, test_representation):
        euclidean_distance = source_representation - test_representation
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)
        return euclidean_distance

    def drawFaceFeatures(self, img, features):
        for feature_point in features:
            cv2.circle(img, (feature_point.x, feature_point.y), 2, [0, 255, 0], 2)
        return img

    def drawFaceDetect(self, img, detection):
        left, top, right, bottom = detection.left(), detection.top(), detection.right(), detection.bottom()
        cv2.rectangle(img, (left, top), (right, bottom), [255, 0, 0], 2)
        return img

    def imgProcess(self, image_path):
        img = cv2.imread(image_path)
        h, w, c = img.shape
        r = min(300 / h, 300 / w)
        new_h, new_w = int(h * r), int(w * r)
        img = cv2.resize(img, (new_w, new_h))
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, image

    def faceMatch(self, image_path1, image_path2):
        suffix = Util.extract_file_suffix(image_path1)
        img1, image1 = self.imgProcess(image_path1)
        img2, image2 = self.imgProcess(image_path2)
        results1 = self.featureExtract(img1, image1)
        results2 = self.featureExtract(img2, image2)
        face_feature1 = results1['face_feature']
        face_feature2 = results2['face_feature']
        distance = self.cacuEuclideanDistance(face_feature1, face_feature2)
        score = (1 - distance) * 100
        only_68_marks_draw1 = results1['68_marks_draw']
        only_68_marks_draw2 = results2['68_marks_draw']
        face_detect1 = results1['face_detect_draw']
        face_detect2 = results2['face_detect_draw']
        total_draw1 = results1['total_draw']
        total_draw2 = results2['total_draw']

        image_h, image_w = face_detect1.shape[:2]
        face_detect_image_r = cv2.resize(face_detect2, (image_w, image_h))
        detect_result = np.hstack((face_detect1, face_detect_image_r))

        face_68_feature_image = cv2.resize(only_68_marks_draw2, (image_w, image_h))
        face_68_feature_result = np.hstack((only_68_marks_draw1, face_68_feature_image))

        face_all_result_image = cv2.resize(total_draw2, (image_w, image_h))
        face_all_feature_result = np.hstack((total_draw1, face_all_result_image))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(face_all_feature_result, f"score:{score:.2f}", (0, 20), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        detect_result_savepath = Util.generate_temp_file_path(suffix=suffix)
        face_68_feature_result_savepath = Util.generate_temp_file_path(suffix=suffix)
        face_all_feature_result_savepath = Util.generate_temp_file_path(suffix=suffix)

        cv2.imwrite(detect_result_savepath, detect_result)
        cv2.imwrite(face_68_feature_result_savepath, face_68_feature_result)
        cv2.imwrite(face_all_feature_result_savepath, face_all_feature_result)

        detect_result_remote_savepath = self.imageoss.upload(detect_result_savepath)
        face_68_feature_result_remote_savepath = self.imageoss.upload(face_68_feature_result_savepath)
        face_all_feature_result_remote_savepath = self.imageoss.upload(face_all_feature_result_savepath)

        clearCache(detect_result_savepath)
        clearCache(face_68_feature_result_savepath)
        clearCache(face_all_feature_result_savepath)

        result = {
            "score": score,
            "detect_url": detect_result_remote_savepath,
            "68_feature_url": face_68_feature_result_remote_savepath,
            "all_feature_result_url": face_all_feature_result_remote_savepath
        }
        return result

    def featureExtract(self, img, image):
        img1 = copy.deepcopy(img)
        face_detections = self.face_detector(image, 1)
        face_shape = self.face_shape_predictor(image, face_detections[0].rect)
        face_68_features = face_shape.parts()
        face_detect_result = face_detections[0].rect
        only_68_marks_draw = self.drawFaceFeatures(img, face_68_features)
        only_face_detect_draw = self.drawFaceDetect(img1, face_detect_result)
        only_68_marks_draw_c = only_68_marks_draw.copy()
        total_draw = self.drawFaceDetect(only_68_marks_draw_c, face_detect_result)
        face_descriptor = self.face_recognition_model.compute_face_descriptor(image, face_shape, self.face_resample,
                                                                              self.face_padding)
        face_feature = np.ascontiguousarray(face_descriptor)
        result = {
            "face_feature": face_feature,
            "68_marks_draw": only_68_marks_draw,
            "face_detect_draw": only_face_detect_draw,
            "total_draw": total_draw
        }
        return result

    def faceDetect(self, image_path):
        img, image = self.imgProcess(image_path)
        face_detections = self.face_detector(image, 1)
        if len(face_detections) < 1:
            return False
        else:
            return True

if __name__ == '__main__':
    pass
