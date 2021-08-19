from .modules import (
    ResidualStack,
    Conv1DBuilder,
    ConvTranspose1DBuilder,
    GlobalConditioning,
    Jitter,
)
from src.utils.log import Logger
import torch.nn as nn
import torch.nn.functional as F
import torch


class DeconvolutionalDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_hiddens,
        num_residual_layers,
        num_residual_hiddens,
        use_kaiming_normal,
        use_jitter,
        jitter_probability,
        use_speaker_conditioning,
        device,
        verbose=False,
    ):

        super(DeconvolutionalDecoder, self).__init__()

        self._use_jitter = use_jitter
        self._use_speaker_conditioning = use_speaker_conditioning
        self._device = device
        self._verbose = verbose
        if self._verbose:
            self._logger = Logger.get_logger()

        if self._use_jitter:
            self._jitter = Jitter(jitter_probability)

        in_channels = (
            in_channels + 40 if self._use_speaker_conditioning else in_channels
        )

        self._conv_1 = Conv1DBuilder.build(
            in_channels=in_channels,
            out_channels=num_hiddens,
            kernel_size=3,
            padding=1,
            use_kaiming_normal=use_kaiming_normal,
        )

        self._upsample = nn.Upsample(scale_factor=2)

        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            use_kaiming_normal=use_kaiming_normal,
        )

        self._conv_trans_1 = ConvTranspose1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            padding=1,
            use_kaiming_normal=use_kaiming_normal,
        )

        self._conv_trans_2 = ConvTranspose1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            padding=0,
            use_kaiming_normal=use_kaiming_normal,
        )

        self._conv_trans_3 = ConvTranspose1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=out_channels,
            kernel_size=2,
            padding=0,
            use_kaiming_normal=use_kaiming_normal,
        )

    def forward(self, inputs, speaker_dic, speaker_id):
        x = inputs
        if self._verbose:
            self._logger.debug("[FEATURES_DEC] input size: {}".format(x.size()))

        if self._use_jitter and self.training:
            x = self._jitter(x)

        if self._use_speaker_conditioning:
            speaker_embedding = GlobalConditioning.compute(
                speaker_dic,
                speaker_id,
                x,
                device=self._device,
                gin_channels=40,
                expand=True,
            )
            x = torch.cat([x, speaker_embedding], dim=1).to(self._device)

        x = self._conv_1(x)
        if self._verbose:
            self._logger.debug(
                "[FEATURES_DEC] _conv_1 output size: {}".format(x.size())
            )

        x = self._upsample(x)
        if self._verbose:
            self._logger.debug(
                "[FEATURES_DEC] _upsample output size: {}".format(x.size())
            )

        x = self._residual_stack(x)
        if self._verbose:
            self._logger.debug(
                "[FEATURES_DEC] _residual_stack output size: {}".format(x.size())
            )

        x = F.relu(self._conv_trans_1(x))
        if self._verbose:
            self._logger.debug(
                "[FEATURES_DEC] _conv_trans_1 output size: {}".format(x.size())
            )

        x = F.relu(self._conv_trans_2(x))
        if self._verbose:
            self._logger.debug(
                "[FEATURES_DEC] _conv_trans_2 output size: {}".format(x.size())
            )

        x = self._conv_trans_3(x)
        if self._verbose:
            self._logger.debug(
                "[FEATURES_DEC] _conv_trans_3 output size: {}".format(x.size())
            )

        return x
