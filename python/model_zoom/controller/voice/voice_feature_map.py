# coding=utf-8
# Created : 2023/1/13 14:54
# Author  : Zy
import matplotlib
matplotlib.use('Agg')

import pylab
import numpy as np
import librosa.display
from common.oss import OSS
from common.utils import Util
from common.file_utils import clearCache

imageoss = OSS("hz-images")


def extractVoiceFeatureMap(local_path):
    y, sr = librosa.load(local_path)
    local_mel_path = Util.generate_temp_file_path(suffix="png")
    local_spectrum_path = Util.generate_temp_file_path(suffix="png")
    local_amplitude_path = Util.generate_temp_file_path(suffix="png")
    pylab.axis('off')
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    pylab.savefig(local_mel_path, bbox_inches=None, pad_inches=0)
    pylab.close()

    cent = librosa.feature.spectral_centroid(y=y, sr=sr)  # 频谱
    pylab.plot(cent[0])
    #pylab.axis('off')
    #pylab.xticks([])
    pylab.xlabel("|amplitude|")
    pylab.ylabel("|frequency|")
    pylab.savefig(local_spectrum_path) # , bbox_inches='tight', pad_inches=-0.1
    pylab.close()

    librosa.display.waveplot(y, sr=sr)  # 波形
    #pylab.axis('off')
    #pylab.xticks([])
    pylab.savefig(local_amplitude_path)  # 注意两个参数 , bbox_inches='tight', pad_inches=-0.1
    pylab.close()

    remote_mel_path = imageoss.upload(local_mel_path)
    remote_spectrum_path = imageoss.upload(local_spectrum_path)
    remote_amplitude_path = imageoss.upload(local_amplitude_path)

    clearCache(local_mel_path)
    clearCache(local_spectrum_path)
    clearCache(local_amplitude_path)

    return remote_mel_path, remote_spectrum_path, remote_amplitude_path
