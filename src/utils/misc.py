import time
import random
import os
import yaml
import numpy as np
import soundfile
import librosa
import torch
import matplotlib.pyplot as plt

def get_dirname(base_dir):
    dirname = time.strftime(
        f"%y-%m-%d-%H%M%S-{random.randint(0, 1e6):06}", time.gmtime(time.time())
    )
    log_path = os.path.join(base_dir, dirname)

    return log_path


def one_hot_to_wave(one_hot):
    one_hot = one_hot.squeeze()
    quantized = one_hot.T.argmax(-1)
    return MuLaw.decode(quantized.squeeze().numpy())


def load_txts(dir):
    """Create a dictionary with all the text of the audio transcriptions."""
    utterences = dict()
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in fnames:
                if fname.endswith(".txt"):
                    with open(os.path.join(root, fname), "r") as f:
                        fname_no_ext = os.path.basename(
                            fname).rsplit(".", 1)[0]
                        utterences[fname_no_ext] = f.readline()
    return utterences


def parse_audio(y , noiseInjector = False, noise_prob=0.4, sample_rate=16000, window_size=0.02, window_stride=0.01, window='hamming', normalize=True):

    #class noiseInjector isnt implemented. Left this just in case
    ############################################################
    if noiseInjector:
        add_noise = np.random.binomial(1, noise_prob)
        if add_noise:
            y = noiseInjector.inject_noise(y)
    #############################################################

    n_fft = int(sample_rate * window_size)
    win_length = n_fft
    hop_length = int(sample_rate * window_stride)
    # STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=window)
    spect, _ = librosa.magphase(D)
    # S = log(S+1)
    spect = np.log1p(spect)
    spect = torch.FloatTensor(spect)
    if normalize:
        mean = spect.mean()
        std = spect.std()
        spect.add_(-mean)
        spect.div_(std)

    return spect


def plot_pcolormesh(data, fig, x=None, y=None, axis=None):
    axis = plt.gca() if axis is None else axis  # default axis if None
    x = np.arange(data.shape[1]) if x is None else x  # default x shape if None
    y = np.arange(data.shape[0]) if y is None else y  # default y shape if None
    c = axis.pcolormesh(x, y, data)
    fig.colorbar(c, ax=axis)


def compute_unified_time_scale(shape, winstep=0.01, downsampling_factor=1):
    return np.arange(shape) * winstep * downsampling_factor

class ConfigParser:
    @staticmethod
    def parse_yaml_config(path):
        with open(path, "r") as f:
            config = yaml.safe_load(f)

        return config


class MuLaw(object):
    @staticmethod
    def encode(x, mu=256):
        x = x.astype(np.float32)
        y = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
        y = np.digitize(y, 2 * np.arange(mu) / mu - 1) - 1
        return y.astype(np.long)

    @staticmethod
    def decode(y, mu=256):
        y = y.astype(np.float32)
        y = 2 * y / mu - 1
        x = np.sign(y) / mu * ((mu) ** np.abs(y) - 1)
        return x.astype(np.float32)
