import re
import numpy as np
from controller.voice.psp_tts import prosody_txt
from scipy.io import wavfile
from static.tts.pinyin_dict import pinyin_dict
from controller.voice.psp_tts.text import cleaned_text_to_sequence

from pypinyin import Style
from pypinyin.contrib.neutral_tone import NeutralToneWith5Mixin
from pypinyin.converter import DefaultConverter
from pypinyin.core import Pinyin, load_phrases_dict
from common.utils import Util


class MyConverter(NeutralToneWith5Mixin, DefaultConverter):
    pass


def load_pinyin_dict():
    my_dict = {}
    with open(Util.app_path() + "/static/tts/pypinyin-local.dict", "r", encoding='utf-8') as f:
        content = f.readlines()
        for line in content:
            cuts = line.strip().split()
            hanzi = cuts[0]
            pinyin = cuts[1:]
            tmp = []
            for one in pinyin:
                onelist = [one]
                tmp.append(onelist)
            my_dict[hanzi] = tmp
    load_phrases_dict(my_dict)


def get_phoneme4pinyin(pinyins):
    result = []
    for pinyin in pinyins:
        if pinyin[:-1] in pinyin_dict:
            tone = pinyin[-1]
            a = pinyin[:-1]
            a1, a2 = pinyin_dict[a]
            result += [a1, a2 + tone, "#0"]
    result.append("sil")
    return result


def chinese_to_phonemes(pinyin_parser, text, single_zw):
    all = 'sil'
    zw_index = 0
    py_list_all = pinyin_parser.pinyin(text, style=Style.TONE3, errors="ignore")
    py_list = [single[0] for single in py_list_all]
    for single in single_zw:
        if single == '#':
            all = all[:-2]
            all += single
        elif single.isdigit():
            all += single
        else:
            pyname = pinyin_dict.get(py_list[zw_index][:-1])
            all += ' ' + pyname[0] + ' ' + pyname[1] + py_list[zw_index][-1] + ' ' + '#0'
            zw_index += 1
    all = all + ' ' + 'sil' + ' ' + 'eos'
    return all


def save_wav(wav, path, rate):
    wav *= 32767 / max(0.01, np.max(np.abs(wav))) * 0.6
    wavfile.write(path, rate, wav.astype(np.int16))


def get_text(phones):
    text_norm = cleaned_text_to_sequence(phones)
    return text_norm


load_pinyin_dict()
pinyin_parser = Pinyin(MyConverter())
yl_model = prosody_txt.init_model()


def ttsDataprocess(message: str):
    message = re.sub(r'[^\u4e00-\u9fa5]', "，", message)
    message = message[:99] + "。"
    single_zw = ''
    prosody_txt.run_auto_labels(yl_model, message)
    with open(Util.app_path() + '/cache/temp.txt', 'r') as r:
        for line in r.readlines():
            line = line.strip()
            single_zw += line + '#3'
    single_zw = single_zw[:-1] + '4'
    phonemes = chinese_to_phonemes(pinyin_parser, message, single_zw)
    input_ids = get_text(phonemes)
    input_ids_lengths = np.array([len(input_ids)], dtype=np.int64)
    input_ids = np.array([input_ids], dtype=np.int64)
    return input_ids, input_ids_lengths
