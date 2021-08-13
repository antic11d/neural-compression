from datetime import datetime
from collections import defaultdict
import numpy as np

from tqdm.std import tqdm
from src.utils.log import Logger


class BaseTrainer(object):
    def __init__(self, device, configuration):
        self._device = device
        self._configuration = configuration

    def train(self, datastream):
        logger = Logger.get_logger()
        logger.info(
            "Training started: {} ".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        )
        training_stats = defaultdict(list)
        for epoch in range(self._configuration["num_epochs"]):
            epoch_losses = defaultdict(list)
            with tqdm(datastream.training_loader) as train_bar:
                for batch in train_bar:
                    losses, perplexity_value = self.iterate(
                        batch, epoch, datastream.speaker_dic
                    )
                    if losses is None or perplexity_value is None:
                        continue
                    for k, v in losses.items():
                        epoch_losses[k].append(v)

                if epoch % self._configuration["log_period"] == 0:
                    log_msg = "Epoch {}: loss {:.4f} perplexity {:.3f}".format(
                        epoch + 1, losses["loss"], perplexity_value
                    )
                    logger.info(log_msg)
                    train_bar.set_description(log_msg)
                if epoch % self._configuration["checkpoint_period"] == 0:
                    logger.save_model(self._model, "m_{}".format(epoch + 1))
            for k, v in epoch_losses.items():
                training_stats[k].append(np.mean(v))

        logger.log_training(training_stats)
        return training_stats

    def iterate(self, data, epoch):
        raise NotImplementedError
