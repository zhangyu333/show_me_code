# coding=utf-8
# Created : 2023/1/9 17:51
# Author  : Zy
from controller.voice.tts_onnx import ttsMain
from controller.voice.fastspeech2_onnx import FastSpeech2Onnx
from controller.voice.psp_tts.data_process import ttsDataprocess
from controller.voice.voice_feature_map import extractVoiceFeatureMap
from controller.voice.voicePrintRec import VoiceprintRec
from controller.voice.asr_onnx import ASR
from common.file_utils import *
from common.utils import Util
from xpinyin import Pinyin

asr = ASR()
p = Pinyin()
voiceprintrec = VoiceprintRec()


def myTTS(text: str, save_path: str):

    x, x_lengths = ttsDataprocess(text)
    status = ttsMain(x, x_lengths, save_path)
    clearCache(Util.app_path() + '/cache/temp1.txt')
    clearCache(Util.app_path() + '/cache/temp.txt')
    return status


def changeVoiceSpeed(input_file, speed, out_file):
    cmd = "ffmpeg -y -i %s -filter_complex \"atempo=tempo=%s\" %s" % (input_file, speed, out_file)
    osCall(cmd)


def changeVoiceVolume(input_file, volume, out_file):
    cmd = f'ffmpeg  -i {input_file} -filter:a "volume={volume}dB" {out_file}'
    osCall(cmd)


def mp3ToWav(mp3_path):
    wav_path = Util.generate_temp_file_path(suffix="wav")
    cmd = f'ffmpeg  -i {mp3_path} -f wav {wav_path}'
    osCall(cmd)
    return wav_path

def saveVoiceFeatureImages(local_path: str):
    remote_mel_path, remote_spectrum_path, remote_amplitude_path = extractVoiceFeatureMap(local_path)
    return {
        "remote_mel_path": remote_mel_path,
        "remote_spectrum_path": remote_spectrum_path,
        "remote_amplitude_path": remote_amplitude_path,
    }


def voiceModelMatch(text: str):
    transform_result = p.get_pinyin(text, tone_marks='numbers')
    results = transform_result.split("-")
    pinyins = []
    tones = []
    idxs = []
    for idx, result in enumerate(results):
        pinyins.append(result[:-1])
        tones.append(result[-1])
        idxs.append(idx + 1)

    return {
        "pinyins": pinyins,
        "tones": tones,
        "idxs": idxs,
    }
    


def voiceRec(voice_path: str):
    text = asr.predict(voice_path=voice_path)
    return text
    
    
def voiceprintRegister(wav_path: str, wav_id: str,experiment_id):
    result = voiceprintrec.voiceprintRegister(wav_path, wav_id,experiment_id)
    return result

def voiceprintMatch(wav_path: str,experiment_id):
    result = voiceprintrec.voiceprintMatch(wav_path,experiment_id)
    return result


def voiceprintOnlyMatch(wav_path1, wav_path2):
    coef = voiceprintrec.voiceprintOnlyMatch(wav_path1, wav_path2)
    return coef





