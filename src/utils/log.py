from pathlib import Path
import torch
import numpy as np
import json
from .misc import get_dirname
import matplotlib.pyplot as plt


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

        self._plots_dir = self._root_dir.joinpath("plots")
        self._plots_dir.mkdir(exist_ok=True)

        self._stats_dir = self._root_dir.joinpath("stats")
        self._stats_dir.mkdir(exist_ok=True)

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

    def save_plot(self, fig, filename):
        path = self._plots_dir.joinpath(filename)
        fig.savefig(path, bbox_inches="tight", pad_inches=0)
        plt.close()

    def to_npy(self, data, filename):
        np.save(self._stats_dir.joinpath(filename), data)

    def to_csv(self, data, filename):
        savepath = self._root_dir.joinpath(filename).with_suffix(".csv")
        np.savetxt(savepath, data, delimiter=",")

    def log_config(self, config):
        with open(self._root_dir.joinpath("config.json"), "w") as fp:
            json.dump(config, fp, indent=4)

    def log_training(self, info):
        for k, v in info.items():
            self.to_npy(np.array(v), k)
        self._plot_training(info)

    def _plot_training(self, info):
        fig, axs = plt.subplots(3, 2, figsize=(18, 6))
        flat_axs = axs.flatten()
        for i, (k, v) in enumerate(info.items()):
            self._plot(v, k, flat_axs[i])

        self.save_plot(fig, "stats.png")

    def _plot(self, data, title, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.plot(np.arange(len(data)), data)
        ax.set_ylabel(title)
        ax.grid(True)
