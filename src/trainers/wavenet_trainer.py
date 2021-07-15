from torch import nn
import torch.optim as optim

from .base_trainer import BaseTrainer
import torch


class WavenetTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        device,
        configuration,
        optimizer_name="adam",
        criterion=None,
        **kwargs,
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
        # with torch.autograd.detect_anomaly():
        debug = False
        source = data["input_features"].permute(0, 2, 1).float().to(self._device)
        speaker_id = data["speaker_id"].to(self._device)
        target = data["one_hot"].to(self._device).contiguous()

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
        print(torch.isfinite(reconstructed_x).all())
        reconstruction_loss = self._criterion(reconstructed_x, target.squeeze(1))
        # for name, param in self._model._encoder.named_parameters():
        #     # print(name, p)
        #     print(self._model._encoder._conv_1.weight)
        #     print(name, torch.isfinite(param.grad).all())
        loss = vq_loss + reconstruction_loss
        losses["reconstruction_loss"] = reconstruction_loss.item()
        losses["loss"] = loss.item()
        print(loss.item())

        # if debug and epoch > 1:
        #     print("Before backward")
        #     for name, param in self._model.named_parameters():
        #         print(
        #             f"{name}\t{torch.isfinite(param.grad).all()}\t{torch.max(abs(param.grad))}"
        #         )
        loss.backward()
        # vq_loss.backward(retain_graph=True)
        # reconstruction_loss.backward()

        if debug:
            print("After backward")
            for name, param in self._model.named_parameters():
                print(name)
                print(
                    f"{name}\t{torch.isfinite(param.grad).all()}\t{torch.max(abs(param.grad))}"
                )

        self._optimizer.step()
        if debug:
            print("After step")
            for name, param in self._model.named_parameters():
                print(
                    f"{name}\t{torch.isfinite(param.grad).all()}\t{torch.max(abs(param.grad))}"
                )

            print(f"Epoch: {epoch}")
            print()
            print()

        perplexity_value = perplexity.item()

        return losses, perplexity_value
