import cv2
import numpy as np
import face_recognition
import onnxruntime as Ort
from common.utils import Util
from common.file_utils import clearCache
from common.oss import OSS


class FaceAttrs:

    def __init__(self):
        self.gender = ["男性", "女性"]
        self.face_emotions = ["愤怒", "厌恶", "恐惧", "快乐", "中性", "悲伤", "惊讶"]
        self.__face_age_model = Ort.InferenceSession(Util.app_path() + "/models/face/face_age_sex.onnx")
        self.__fe_model = Ort.InferenceSession(Util.app_path() + "/models/face/face_emotion_rec.bin")

    def __infer_age(self, face):
        inputs = np.transpose(cv2.resize(face, (64, 64)), (2, 0, 1))
        inputs = np.expand_dims(inputs, 0).astype(np.float32) / 255.
        predictions = self.__face_age_model.run(None, {"input": inputs})[0]
        gender_idx = int(np.argmax(predictions[0, :2]))
        age = int(predictions[0, 2])
        return {"gender": self.gender[gender_idx], "age": age}

    def __infer_emotion(self, face):
        inp = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype("float32")
        inp = cv2.resize(inp, (48, 48))
        inp /= 255.
        inp -= 0.5
        inp /= 0.5
        inp = inp[None]
        inp = inp[None]
        output = self.__fe_model.run(None, {"input.1": inp})[0][0]
        return output.tolist()

    def __call__(self, filanme):
        if isinstance(filanme, str):
            img = cv2.imread(filanme)
        else:
            img = filanme
        face_marks = face_recognition.face_landmarks(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None, "large")
        face_location = face_recognition.face_locations(img)[0]
        y1, x2, y2, x1 = face_location
        face = img[y1:y2, x1:x2]
        age_info = self.__infer_age(face)
        emotion_info = self.__infer_emotion(face)
        return {
            "face_loaction": [y2, x2, y1, x1],
            "face_landmarks": face_marks,
            "face_age_info": age_info,
            "face_emotion_info": emotion_info
        }


if __name__ == "__main__":
    pass
