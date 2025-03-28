# coding=utf-8
# Created : 2023/3/3 10:09
# Author  : Zy
import re
import numpy as np
import onnxruntime as ort
from scipy.io import wavfile
from controller.voice.fastspeech2 import text_to_sequence
from pypinyin import pinyin, Style
from common.utils import Util


class FastSpeech2Onnx:
    def __init__(self):
        super(FastSpeech2Onnx, self).__init__()
        self.__model_path = Util.app_path() + "/models/fastspeech/model.onnx"
        self.__vocoder_path = Util.app_path() + "/models/fastspeech/vocoder.onnx"
        self.__lex_path = Util.app_path() + "/models/fastspeech/pinyin-lexicon-r.txt"
        self.__model = ort.InferenceSession(self.__model_path)
        self.__vocoder = ort.InferenceSession(self.__vocoder_path)

    def readLexicon(self):
        lexicon = {}
        with open(self.__lex_path, encoding="utf-8") as f:
            for line in f:
                temp = re.split(r"\s+", line.strip("\n"))
                word = temp[0]
                phones = temp[1:]
                if word.lower() not in lexicon:
                    lexicon[word.lower()] = phones
        return lexicon

    def preprocessMandarin(self, text):
        lexicon = self.readLexicon()

        phones = []
        pinyins = [
            p[0]
            for p in pinyin(
                text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
            )
        ]
        for p in pinyins:
            if p in lexicon:
                phones += lexicon[p]
            else:
                phones.append("sp")

        phones = "{" + " ".join(phones) + "}"
        sequence = np.array(
            text_to_sequence(
                phones, []
            )
        )

        return np.array([sequence], dtype=np.int64), np.array([len(sequence)], dtype=np.int64)

    def expand(self, values, durations):
        out = list()
        for value, d in zip(values, durations):
            out += [value] * max(0, int(d))
        return np.array(out)

    def __vocoderInfer(self, mel_predictions, length):
        wav = self.__vocoder.run(None, {"mels": mel_predictions})[0].squeeze(1)
        wav = (wav * 32768.0).astype("int16")[0]
        wav = wav[: length]
        return wav

    def dataPreProc(self, text, speaker_id):
        texts, src_lens = self.preprocessMandarin(text)
        speakers = np.array([speaker_id], dtype=np.int64)
        texts_seq_len = texts.shape[1]
        texts = np.hstack((texts, np.zeros((1, 300 - texts_seq_len), dtype=np.int64)))
        return texts, src_lens, speakers, texts_seq_len

    def fastspeech2Infer(self, text, save_path):
        texts, src_lens, speakers, texts_seq_len = self.dataPreProc(text, 10)
        predictions = self.__model.run(None, {"speakers": speakers, "texts": texts, "src_lens.1": src_lens})
        mel_prediction = predictions[1].transpose((0, 2, 1))
        length = predictions[9][0] * (texts_seq_len / 300) * 256
        wav = self.__vocoderInfer(mel_prediction, int(length))
        wavfile.write(save_path, 22050, wav)


if __name__ == '__main__':
    text = "他喜欢说他是一个狗人"
    fs = FastSpeech2Onnx()
    save_path = "output.wav"
    fs.fastspeech2Infer(text, save_path)
