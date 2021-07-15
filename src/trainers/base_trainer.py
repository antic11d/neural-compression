from datetime import datetime
import numpy as np

from tqdm.std import tqdm
from src.utils.log import Logger

import torch


class BaseTrainer(object):
    def __init__(self, device, configuration):
        self._device = device
        self._configuration = configuration

    def train(self, datastream):
        logger = Logger.get_logger()
        logger.info(
            "Training started: {} ".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        )

        for epoch in range(self._configuration["num_epochs"]):
            with tqdm(datastream.training_loader) as train_bar:
                for batch in train_bar:
                    losses, perplexity_value = self.iterate(
                        batch, epoch, datastream.speaker_dic
                    )
                    if losses is None or perplexity_value is None:
                        continue

                if epoch % self._configuration["log_period"] == 0:
                    log_msg = "Epoch {}: loss {:.4f} perplexity {:.3f}".format(
                        epoch + 1, losses["loss"], perplexity_value
                    )
                    logger.info(log_msg)
                    train_bar.set_description(log_msg)
                if epoch % self._configuration["checkpoint_period"] == 0:
                    logger.save_model(self._model, "m_{}".format(epoch + 1))
        # TODO: return training info dict
        return {}

    def iterate(self, data, epoch):
        raise NotImplementedError
