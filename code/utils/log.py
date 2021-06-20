from pathlib import Path
import torch
import numpy as np
import json


class Logger:
    _instance = None

    @staticmethod
    def get_logger(opts):
        if Logger._instance == None:
            Logger(opts)
        return Logger._instance

    def __init__(self, opts):
        if Logger._instance != None:
            raise Exception("This class is a singleton!")
        else:
            Logger._instance = self._initialize(opts)

    def __del__(self):
        self._info_file.close()

    def _initialize(self, opts):
        self._quiet = opts.quiet

        self._root_dir = Path(opts.base_dir)
        self._root_dir.mkdir(exist_ok=True)

        self._checkpoints_dir = self._root_dir.joinpath("checkpts")
        self._checkpoints_dir.mkdir(exist_ok=True)

        self._info_file = open(self.root_dir.joinpath("info.txt"), "a")
        return self

    def info(self, message):
        self._info_file.write(f"{message}\n")

        if not self._quiet:
            print(message)

    def save_model(self, model, filename):
        path = self._checkpoints_dir.joinpath(filename).with_suffix(".pth")
        torch.save(model.state_dict(), path)

    def to_csv(self, data, filename):
        savepath = self._root_dir.joinpath(filename).with_suffix(".csv")
        np.savetxt(savepath, data, delimiter=",")

    def log_config(self, config):
        with open(self._root_dir.joinpath("config.json"), "w") as fp:
            json.dump(vars(config), fp)

    def log_training(self, info):
        for k, v in info.items():
            self.to_csv(np.array(v), k)
