import time
import random
import os
import yaml
import numpy as np
import soundfile


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
