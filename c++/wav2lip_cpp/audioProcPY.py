import warnings

warnings.filterwarnings("ignore")
import librosa
import numpy as np
import librosa.filters
from scipy import signal


class HParams:
    def __init__(self, **kwargs):
        self.data = {}

        for key, value in kwargs.items():
            self.data[key] = value

    def __getattr__(self, key):
        if key not in self.data:
            raise AttributeError("'HParams' object has no attribute %s" % key)
        return self.data[key]

    def set_hparam(self, key, value):
        self.data[key] = value

hp = HParams(
    num_mels=80,
    rescale=True,
    rescaling_max=0.9,
    n_fft=800,
    hop_size=200,
    win_size=800,
    sample_rate=16000,
    frame_shift_ms=None,
    signal_normalization=True,
    allow_clipping_in_normalization=True,
    symmetric_mels=True,
    max_abs_value=4.,
    preemphasize=True,
    preemphasis=0.97,
    min_level_db=-100,
    ref_level_db=20,
    fmin=55,
    fmax=7600,
)

_mel_basis = None


class Audio:

    @staticmethod
    def load_wav(path, sr=16000):
        return librosa.core.load(path, sr=sr)[0]

    @staticmethod
    def melspectrogram(wav):
        wav_preemphasis = Audio.preemphasis(wav, hp.preemphasis, hp.preemphasize)
        D = Audio._stft(wav_preemphasis)
        S = Audio._amp_to_db(Audio._linear_to_mel(np.abs(D))) - hp.ref_level_db

        if hp.signal_normalization:
            return Audio._normalize(S)
        return S

    @staticmethod
    def _normalize(S):
        if hp.allow_clipping_in_normalization:
            if hp.symmetric_mels:
                return np.clip((2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value,
                               -hp.max_abs_value, hp.max_abs_value)
            else:
                return np.clip(hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db)), 0, hp.max_abs_value)

        assert S.max() <= 0 and S.min() - hp.min_level_db >= 0
        if hp.symmetric_mels:
            return (2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value
        else:
            return hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db))

    @staticmethod
    def _build_mel_basis():
        assert hp.fmax <= hp.sample_rate // 2
        return librosa.filters.mel(n_fft=hp.n_fft, sr=hp.sample_rate, n_mels=hp.num_mels,
                                   fmin=hp.fmin, fmax=hp.fmax)

    @staticmethod
    def _linear_to_mel(spectogram):
        global _mel_basis
        if _mel_basis is None:
            _mel_basis = Audio._build_mel_basis()
        return np.dot(_mel_basis, spectogram)

    @staticmethod
    def _amp_to_db(x):
        min_level = np.exp(hp.min_level_db / 20 * np.log(10))
        return 20 * np.log10(np.maximum(min_level, x))

    @staticmethod
    def preemphasis(wav, k, preemphasize=True):
        if preemphasize:
            return signal.lfilter([1, -k], [1], wav)
        return wav

    @staticmethod
    def get_hop_size():
        hop_size = hp.hop_size
        if hop_size is None:
            assert hp.frame_shift_ms is not None
            hop_size = int(hp.frame_shift_ms / 1000 * hp.sample_rate)
        return hop_size

    @staticmethod
    def _stft(y):
        return librosa.stft(y=y, n_fft=hp.n_fft, hop_length=Audio.get_hop_size(), win_length=hp.win_size)


mel_step_size = 16
fps = 25


def audioProc(wav):
    mel = Audio.melspectrogram(wav)
    mel_chunks = []
    mel_idx_multiplier = 80. / fps
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
        i += 1
    mel_chunks = [item.tolist() for item in mel_chunks]
    return mel_chunks