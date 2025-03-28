# coding=utf-8
# Created : 2022/10/21 10:17
# Author  : Zy
import os
import cv2
import dlib
import numpy as np
from loguru import logger
from common.oss import OSS
from common.utils import Util
from common.file_utils import clearCache


class FaceRec():
    def __init__(self):
        super(FaceRec, self).__init__()
        self.face_detect_model_path = Util.app_path() + "/models/face/mmod_human_face_detector.dat"
        self.face_shape_predictor_path = Util.app_path() + "/models/face/shape_predictor_68_face_landmarks.dat"
        self.face_rec_model_path = Util.app_path() + "/models/face/dlib_face_recognition_resnet_model_v1.dat"
        self.face_detector = dlib.cnn_face_detection_model_v1(self.face_detect_model_path)
        self.face_shape_predictor = dlib.shape_predictor(self.face_shape_predictor_path)
        self.face_recognition_model = dlib.face_recognition_model_v1(self.face_rec_model_path)
        # params
        self.face_padding = 0.15
        self.face_resample = 100
        # db path
        self.face_features_path = Util.app_path() + "/cache/face_features.npy"
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

    def matchFace(self, image_path, max_score):
        logger.info("人脸查找中 ...")
        img = cv2.imread(image_path)
        img_cp = img.copy()
        suffix = Util.extract_file_suffix(image_path)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_detections = self.face_detector(image, 1)
        if len(face_detections) == 0:
            return {
                "code": 202,
                "message": "未检测到人脸",
                "data": {}
            }
        face_shape = self.face_shape_predictor(image, face_detections[0].rect)
        face_68_features = face_shape.parts()
        face_detect_result = face_detections[0].rect
        face_descriptor = self.face_recognition_model.compute_face_descriptor(image, face_shape, self.face_resample,
                                                                              self.face_padding)
        face_feature = np.ascontiguousarray(face_descriptor)
        for idx, db_face_features in enumerate(np.load(self.face_features_path, allow_pickle=True)):
            db_face_detect_image, db_face_68_feature_image, db_face_all_result_image, db_face_feature = db_face_features
            distance = self.cacuEuclideanDistance(face_feature, db_face_feature)
            score = (1 - distance) * 100
            score = float(score)
            if score > max_score:
                logger.info(f"人脸查找完毕: 人脸匹配得分:{score:.2f}")
                db_image_h, db_image_w = db_face_detect_image.shape[:2]
                face_detect_image = self.drawFaceDetect(img, face_detect_result)
                single_detect_result_savepath = Util.generate_temp_file_path(suffix=suffix)
                cv2.imwrite(single_detect_result_savepath, face_detect_image)
                single_detect_result_remote_savepath = self.imageoss.upload(single_detect_result_savepath)

                face_detect_image_r = cv2.resize(face_detect_image, (db_image_w, db_image_h))
                detect_result = np.hstack((db_face_detect_image, face_detect_image_r))

                face_68_feature_image = self.drawFaceFeatures(img_cp, face_68_features)
                face_68_feature_image = cv2.resize(face_68_feature_image, (db_image_w, db_image_h))
                face_68_feature_result = np.hstack((db_face_68_feature_image, face_68_feature_image))

                face_all_result_image = self.drawFaceFeatures(face_detect_image, face_68_features)
                face_all_result_image = cv2.resize(face_all_result_image, (db_image_w, db_image_h))
                face_all_feature_result = np.hstack((db_face_all_result_image, face_all_result_image))
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(face_all_feature_result, f"score:{score:.2f}", (0, 20), font, 0.7, (0, 0, 255), 2,
                            cv2.LINE_AA)

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
                clearCache(single_detect_result_savepath)
                clearCache(face_68_feature_result_savepath)
                clearCache(face_all_feature_result_savepath)

                result = {
                    "score": score,
                    "detect_url": detect_result_remote_savepath,
                    "68_feature_url": face_68_feature_result_remote_savepath,
                    "all_feature_result_url": face_all_feature_result_remote_savepath,
                    "single_detect_url": single_detect_result_remote_savepath,
                }
                return {
                    "code": 200,
                    "data": result,
                    "message": ""
                }

        logger.warning("人脸未注册!!!")
        return {
            "code": 201,
            "message": "人脸未注册!!!",
            "data": {}
        }

    def faceRegiste(self, image_path):
        logger.info("人脸注册中...")
        img = cv2.imread(image_path)
        img_cp = img.copy()
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_detections = self.face_detector(image, 1)
        if len(face_detections) == 0:
            # 300 未检测到人脸
            return 300
        face_shape = self.face_shape_predictor(image, face_detections[0].rect)
        face_68_features = face_shape.parts()
        face_detect_result = face_detections[0].rect
        face_detect_image = self.drawFaceDetect(img, face_detect_result)
        face_68_feature_image = self.drawFaceFeatures(img_cp, face_68_features)
        face_detect_image_c = face_detect_image.copy()
        face_all_result_image = self.drawFaceFeatures(face_detect_image_c, face_68_features)
        face_descriptor = self.face_recognition_model.compute_face_descriptor(image, face_shape, self.face_resample,
                                                                              self.face_padding)
        face_feature = np.ascontiguousarray([face_descriptor])
        face_feature = np.array([[face_detect_image, face_68_feature_image, face_all_result_image, face_feature]],
                                dtype=object)
        if not os.path.isfile(self.face_features_path):
            np.save(self.face_features_path, face_feature)
            logger.info("人脸注册成功!!!")
            return 200
        else:
            face_features = np.load(self.face_features_path, allow_pickle=True)
            face_features = np.vstack((face_features, face_feature))
            np.save(self.face_features_path, face_features)
            logger.info("人脸注册成功!!!")
            return 200
