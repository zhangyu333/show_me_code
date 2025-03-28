# coding=utf-8
# Created : 2023/2/11 16:33
# Author  : Zy
import torch
import librosa
# from transformers import Wav2Vec2ForCTC, AutoTokenizer
# from transformers import Wav2Vec2FeatureExtractor


class ASR():
    def __init__(self):
        super(ASR, self).__init__()
        # MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
        # self.featureExtractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
        # self.model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
        # self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    def predict(self, voice_path):
        waveform, _ = librosa.load(voice_path, sr=16_000)
        inputs = self.featureExtractor(waveform, sampling_rate=16_000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = self.model(inputs.input_values).logits
        predicted_id = torch.argmax(logits, dim=-1)[0]
        predicted_sentence = self.tokenizer.decode(predicted_id)
        return predicted_sentence