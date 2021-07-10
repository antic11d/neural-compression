from torch import nn
import torch.optim as optim

from .base_trainer import BaseTrainer


class WavenetTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        device,
        configuration,
        optimizer_name="adam",
        criterion=None,
        **kwargs
    ):
        super().__init__(device, configuration)

        self._model = model
        self._criterion = criterion if criterion is not None else nn.MSELoss()
        if optimizer_name == "adam":
            self._optimizer = optim.Adam(
                self._model.parameters(),
                lr=configuration["learning_rate"],
                amsgrad=True,
            )
        else:
            raise ValueError(
                "Optimizer name {} not supported. (use 'adam')".format(optimizer_name)
            )

    def iterate(self, data, epoch, speaker_dic):
        source = data["input_features"].permute(0, 2, 1).float().to(self._device)
        speaker_id = data["speaker_id"].to(self._device)
        target = (
            data["preprocessed_audio"].to(self._device).contiguous().float().squeeze()
        )

        self._optimizer.zero_grad()

        (
            reconstructed_x,
            x_dec,
            vq_loss,
            losses,
            perplexity,
            encoding_indices,
            concatenated_quantized,
        ) = self._model(source, data["one_hot"], speaker_id)

        reconstruction_loss = self._criterion(reconstructed_x, target)

        loss = vq_loss + reconstruction_loss
        losses["reconstruction_loss"] = reconstruction_loss.item()
        losses["loss"] = loss.item()

        loss.backward()

        self._optimizer.step()

        perplexity_value = perplexity.item()

        return losses, perplexity_value
