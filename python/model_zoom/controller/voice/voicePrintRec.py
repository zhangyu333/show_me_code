import os
import torch
import librosa
import numpy as np
from common.utils import Util


class VoiceprintRec():
    def __init__(self):
        super(VoiceprintRec, self).__init__()
        self.model = torch.jit.load(Util.app_path() + "/models/voices/voiceprint.pth", map_location="cpu")
        self.model.eval()

    def load_audio(self, audio_path, win_length=400, sr=16000, hop_length=160, n_fft=512, spec_len=257):
        wav, sr_ret = librosa.load(audio_path, sr=sr)
        extended_wav = np.append(wav, wav[::-1])
        linear = librosa.stft(extended_wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
        mag, _ = librosa.magphase(linear)
        spec_mag = mag[:, :spec_len]
        mean = np.mean(spec_mag, 0, keepdims=True)
        std = np.std(spec_mag, 0, keepdims=True)
        spec_mag = (spec_mag - mean) / (std + 1e-5)
        spec_mag = spec_mag[np.newaxis, :]
        return spec_mag

    def predict(self, data):
        data = data[np.newaxis, :]
        data = torch.tensor(data, dtype=torch.float32)
        feature = self.model(data)
        return feature.data.cpu().numpy()

    def calculateSimCoef(self, feature1, feature2):
        feature1 = feature1[0]
        feature2 = feature2[0]
        dist = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
        return dist

    def voiceprintRegister(self, wav_path, wav_id, experiment_id):
        features_path = Util.app_path() + f"/cache/{experiment_id}_voiceprintFeatures.npy"
        info = {"code": 200, "message": "声纹注册成功"}
        audio = self.load_audio(wav_path)
        feature = self.predict(audio)
        features_one = np.array([[wav_id, feature]], dtype=object)
        if not os.path.exists(features_path):
            np.save(features_path, features_one, allow_pickle=True)
            print(f"ID:{wav_id} experimentID:{experiment_id} <<==>> 声纹注册成功!!!!!!")
            return info
        else:
            new_features = []
            features = np.load(features_path, allow_pickle=True)
            replace_sign = False
            for feature_dbs in  features:
                feature_id, feature_db = feature_dbs
                if feature_id == wav_id:
                    replace_sign = True
                    new_features.append([wav_id, feature])
                    print(f"ID:{wav_id} experimentID:{experiment_id} <<==>> 声纹替换成功!!!!!!")
                else:
                    new_features.append(feature_dbs)
            new_features = np.array(new_features, dtype=object)
            if not replace_sign:
                new_features = np.vstack((new_features,features_one))
            np.save(features_path,new_features , allow_pickle=True)
            print(f"ID:{wav_id} experimentID:{experiment_id} <<==>> 声纹注册成功!!!!!!")
            return info

    def voiceprintMatch(self, wav_path, experiment_id):
        features_path =  Util.app_path() + f"/cache/{experiment_id}_voiceprintFeatures.npy"
        audio = self.load_audio(wav_path)
        feature = self.predict(audio)
        if not os.path.exists(features_path):
            return {"message": f"experimentID:{experiment_id}此声纹库不存在 请先注册声纹!!", "code": 401 ,"result":""}
        features = np.load(features_path, allow_pickle=True)
        max_coef = 0
        rec_result = ""
        for feature_db_ in features:
            wav_id_db, feature_db = feature_db_
            sim_coef = self.calculateSimCoef(feature, feature_db)
            print(f"sim_coef:{sim_coef}")
            if sim_coef > max_coef:
                max_coef = sim_coef
                rec_result = wav_id_db
        if max_coef < 0.71:
            return {"code": 200, "result":"", "message": "声纹库没有此声纹"}
        # os.remove(features_path)
        # print(f"experimentID:{experiment_id} 此实验结束，声纹库已经销毁")
        return {"code": 200, "result": rec_result, "message": "声纹库匹配成功"}

    def voiceprintOnlyMatch(self, wav_path1, wav_path2):
        audio1 = self.load_audio(wav_path1)
        audio2 = self.load_audio(wav_path2)
        feature1 = self.predict(audio1)
        feature2 = self.predict(audio2)
        coef = self.calculateSimCoef(feature1, feature2)
        return coef


if __name__ == '__main__':
    pass
