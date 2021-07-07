from pathlib import Path
import torch
import numpy as np
import json
from .misc import get_dirname


class Logger:
    _instance = None

    @staticmethod
    def get_logger(root_dir=".", verbose=True):
        if Logger._instance == None:
            Logger(root_dir, verbose)
        return Logger._instance

    def __init__(self, root_dir, quiet):
        if Logger._instance != None:
            raise Exception("This class is a singleton!")
        else:
            Logger._instance = self._initialize(root_dir, quiet)

    def __del__(self):
        self._info_file.close()

    @property
    def root_dir(self):
        return self._root_dir

    def _initialize(self, root_dir, verbose=False):
        self._verbose = verbose

        self._root_dir = Path(get_dirname(root_dir))
        self._root_dir.mkdir(exist_ok=True)

        self._checkpoints_dir = self._root_dir.joinpath("checkpts")
        self._checkpoints_dir.mkdir(exist_ok=True)

        self._info_file = open(self._root_dir.joinpath("info.txt"), "a")
        return self

    def info(self, message):
        self._info_file.write(f"{message}\n")

        if not self._verbose:
            print(message)

    def debug(self, message):
        print(message)

    def save_model(self, model, filename):
        path = self._checkpoints_dir.joinpath(filename).with_suffix(".pth")
        torch.save(model.state_dict(), path)

    def to_csv(self, data, filename):
        savepath = self._root_dir.joinpath(filename).with_suffix(".csv")
        np.savetxt(savepath, data, delimiter=",")

    def log_config(self, config):
        with open(self._root_dir.joinpath("config.json"), "w") as fp:
            json.dump(config, fp, indent=4)

    def log_training(self, info):
        for k, v in info.items():
            self.to_csv(np.array(v), k)
