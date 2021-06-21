from datetime import datetime
from utils.log import Logger


class BaseTrainer(object):
    def __init__(self, device, configuration):
        self._device = device
        self._configuration = configuration

    def train(self, dataloader):
        logger = Logger.get_logger()
        logger.info(
            "Training started: {} ".format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        )

        for epoch in range(self._configuration["num_epochs"]):
            for batch in dataloader:
                losses, perplexity_value = self.iterate(batch, epoch)
                if losses is None or perplexity_value is None:
                    continue

                if epoch % self._configuration["log_period"] == 0:
                    logger.info(
                        "Epoch {}: loss {:.4f} perplexity {:.3f}".format(
                            epoch + 1, losses["loss"], perplexity_value
                        )
                    )
                if epoch % self._configuration["checkpoint_period"] == 0:
                    # TODO: Implement checkpoints
                    logger.debug("Should checkpoint models")
        # TODO: return training info dict
        return {}

    def iterate(self, data, epoch):
        raise NotImplementedError
