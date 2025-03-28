# coding=utf-8
# Created : 2023/1/12 09:20
# Author  : Zy
import numpy as np
import onnxruntime as ort
from scipy.io import wavfile
from common.utils import Util

TextEncoder = ort.InferenceSession(Util.app_path() + "/models/tts/TextEncoder.onnx", providers=['CPUExecutionProvider'])
SDP = ort.InferenceSession(Util.app_path() + "/models/tts/sdp.onnx", providers=['CPUExecutionProvider'])
RCB = ort.InferenceSession(Util.app_path() + "/models/tts/flow.onnx", providers=['CPUExecutionProvider'])
Generator = ort.InferenceSession(Util.app_path() + "/models/tts/dec.onnx", providers=['CPUExecutionProvider'])


def sequenceMask(length, max_length=None):
    if max_length is None:
        max_length = np.max(length)
    x = np.arange(max_length, dtype=length.dtype)
    x = np.expand_dims(x, 0)
    length = np.expand_dims(length, 1)
    return x < length


def generatePath(duration, mask):
    b, _, t_y, t_x = mask.shape
    cum_duration = np.cumsum(duration, -1)
    cum_duration_flat = cum_duration[0][0]
    path = sequenceMask(cum_duration_flat, t_y).astype(mask.dtype)
    path = path[None]
    path = path - np.pad(path, [[0, 0], [1, 0], [0, 0]])[:, :-1]
    path = np.expand_dims(path, 1).transpose(0, 1, 3, 2) * mask
    return path


def save_wav(wav, path, rate):
    wav *= 32767 / max(0.01, np.max(np.abs(wav))) * 0.6
    wavfile.write(path, rate, wav.astype(np.int16))


def ttsMain(x, x_lengths, save_path, sample_rate=16000):
    outputs = TextEncoder.run(None, {"x": x, "x_lengths": x_lengths})
    x, m_p, logs_p, x_mask = outputs
    logw = SDP.run(None, {"x": x, "x_mask": x_mask})[0]
    w = np.exp(logw) * x_mask
    w_ceil = np.ceil(w)
    y_lengths = np.array([np.sum(w_ceil)])
    y_lengths[y_lengths < 1] = 1
    y_mask = sequenceMask(y_lengths).astype(x_mask.dtype)[None]
    attn_mask = np.expand_dims(x_mask, 2) * np.expand_dims(y_mask, -1)
    attn = generatePath(w_ceil, attn_mask)
    m_p = np.matmul(attn.squeeze(1), m_p.transpose(0, 2, 1)).transpose(0, 2, 1)
    logs_p = np.matmul(attn.squeeze(1), logs_p.transpose(0, 2, 1)).transpose(0, 2, 1)
    z_p = m_p + np.random.randn(m_p.shape[0], m_p.shape[1], m_p.shape[2]) * np.exp(logs_p) * .0
    z_p = z_p.astype(np.float32)
    y_mask = y_mask.astype(np.float32)
    z = RCB.run(None, {"z_p": z_p, "y_mask": y_mask})[0]
    b, d, l = z.shape
    pad_l = 1881 - l
    radio = l / 1881
    pad = np.zeros((b, d, pad_l))
    z = np.concatenate((z, pad), axis=2).astype("float32")
    z - np.ascontiguousarray(z)
    o = Generator.run(None, {"z": z})[0][0, 0]
    o = o[:int(radio * o.shape[0])]
    save_wav(o, save_path, sample_rate)
    print("TTS SUCCESS SAVE PATH:")
    print(f"\t\t{save_path}")
    return 200


if __name__ == '__main__':
    pass
