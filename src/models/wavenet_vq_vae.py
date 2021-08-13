from .conv_encoder import ConvolutionalEncoder
from .wavenet_decoder import WaveNetDecoder
from .vq import VectorQuantizer

import torch
import torch.nn as nn


class WaveNetVQVAE(nn.Module):
    def __init__(self, configuration, speaker_dic, device):
        super(WaveNetVQVAE, self).__init__()

        self._encoder = ConvolutionalEncoder(
            in_channels=configuration["input_features_dim"],
            num_hiddens=configuration["num_hiddens"],
            num_residual_layers=configuration["num_residual_layers"],
            num_residual_hiddens=configuration["residual_channels"],
            use_kaiming_normal=configuration["use_kaiming_normal"],
            input_features_type=configuration["input_features_type"],
            features_filters=configuration["input_features_filters"] * 3
            if configuration["augment_input_features"]
            else configuration["input_features_filters"],
            sampling_rate=configuration["sampling_rate"],
            device=device,
        )

        self._pre_vq_conv = nn.Conv1d(
            in_channels=configuration["num_hiddens"],
            out_channels=configuration["embedding_dim"],
            kernel_size=1,
            stride=1,
            padding=1,
        )

        self._vq = VectorQuantizer(
            num_embeddings=configuration["num_embeddings"],
            embedding_dim=configuration["embedding_dim"],
            commitment_cost=configuration["commitment_cost"],
            device=device,
        )

        self._decoder = WaveNetDecoder(configuration, speaker_dic, device)

        self._device = device
        self._record_codebook_stats = configuration["record_codebook_stats"]

    @property
    def vq(self):
        return self._vq

    @property
    def pre_vq_conv(self):
        return self._pre_vq_conv

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder

    def forward(self, x_enc, x_dec, global_condition):
        z = self._encoder(x_enc)

        z = self._pre_vq_conv(z)

        (
            vq_loss,
            quantized,
            perplexity,
            _,
            _,
            encoding_indices,
            losses,
            _,
            _,
            _,
            concatenated_quantized,
        ) = self._vq(z, record_codebook_stats=self._record_codebook_stats)

        local_condition = quantized
        local_condition = local_condition.squeeze(-1)
        x_dec = x_dec.squeeze(-1)

        reconstructed_x = self._decoder(x_dec, local_condition, global_condition)
        reconstructed_x = reconstructed_x.unsqueeze(-1)
        x_dec = x_dec.unsqueeze(-1)

        return (
            reconstructed_x,
            x_dec,
            vq_loss,
            losses,
            perplexity,
            encoding_indices,
            concatenated_quantized,
        )

    def save(self, path):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(path, configuration, speaker_dic, device):
        model = WaveNetVQVAE(configuration, speaker_dic, device)
        model.load_state_dict(torch.load(path, map_location=device))
        return model
