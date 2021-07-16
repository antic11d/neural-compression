from .conv_encoder import ConvolutionalEncoder
from .deconv_decoder import DeconvolutionalDecoder
from .vq import VectorQuantizer
from src.utils.log import Logger

import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalVQVAE(nn.Module):
    def __init__(self, configuration, device):
        super(ConvolutionalVQVAE, self).__init__()

        self._output_features_filters = (
            configuration["output_features_filters"] * 3
            if configuration["augment_output_features"]
            else configuration["output_features_filters"]
        )
        self._output_features_dim = configuration["output_features_dim"]
        self._verbose = configuration["verbose"]
        if self._verbose:
            self._logger = Logger.get_logger()

        self._encoder = ConvolutionalEncoder(
            in_channels=configuration["input_features_dim"],
            num_hiddens=configuration["num_hiddens"],
            num_residual_layers=configuration["num_residual_layers"],
            num_residual_hiddens=configuration["num_hiddens"],
            use_kaiming_normal=configuration["use_kaiming_normal"],
            input_features_type=configuration["input_features_type"],
            features_filters=configuration["input_features_filters"] * 3
            if configuration["augment_input_features"]
            else configuration["input_features_filters"],
            sampling_rate=configuration["sampling_rate"],
            device=device,
            verbose=self._verbose,
        )

        self._pre_vq_conv = nn.Conv1d(
            in_channels=configuration["num_hiddens"],
            out_channels=configuration["embedding_dim"],
            kernel_size=3,
            padding=1,
        )

        self._vq = VectorQuantizer(
            num_embeddings=configuration["num_embeddings"],
            embedding_dim=configuration["embedding_dim"],
            commitment_cost=configuration["commitment_cost"],
            device=device,
        )

        self._decoder = DeconvolutionalDecoder(
            in_channels=configuration["embedding_dim"],
            out_channels=self._output_features_filters,
            num_hiddens=configuration["num_hiddens"],
            num_residual_layers=configuration["num_residual_layers"],
            num_residual_hiddens=configuration["residual_channels"],
            use_kaiming_normal=configuration["use_kaiming_normal"],
            use_jitter=configuration["use_jitter"],
            jitter_probability=configuration["jitter_probability"],
            use_speaker_conditioning=configuration["use_speaker_conditioning"],
            device=device,
            verbose=self._verbose,
        )

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

    def forward(self, x, speaker_dic, speaker_id):
        x = x.permute(0, 2, 1).contiguous().float()

        z = self._encoder(x)
        if self._verbose:
            self._logger.debug("[ConvVQVAE] _encoder output size: {}".format(z.size()))

        z = self._pre_vq_conv(z)
        if self._verbose:
            self._logger.debug(
                "[ConvVQVAE] _pre_vq_conv output size: {}".format(z.size())
            )

        (
            vq_loss,
            quantized,
            perplexity,
            encodings,
            distances,
            encoding_indices,
            losses,
            encoding_distances,
            embedding_distances,
            frames_vs_embedding_distances,
            concatenated_quantized,
        ) = self._vq(z, record_codebook_stats=self._record_codebook_stats)

        reconstructed_x = self._decoder(quantized, speaker_dic, speaker_id)

        input_features_size = x.size(2)
        output_features_size = reconstructed_x.size(2)

        reconstructed_x = reconstructed_x.view(
            -1, self._output_features_filters, output_features_size
        )
        reconstructed_x = reconstructed_x[
            :, :, : -(output_features_size - input_features_size)
        ]
        return (
            reconstructed_x,
            vq_loss,
            losses,
            perplexity,
            quantized,
            encodings,
            distances,
            encoding_indices,
            encoding_distances,
            embedding_distances,
            frames_vs_embedding_distances,
            concatenated_quantized,
        )
