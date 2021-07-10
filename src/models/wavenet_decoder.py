from .modules import Conv1DBuilder, Jitter
from .wavenet_vocoder.wavenet import WaveNet

import torch.nn as nn


class WaveNetDecoder(nn.Module):
    def __init__(self, configuration, speaker_dic, device):
        super(WaveNetDecoder, self).__init__()

        self._use_jitter = configuration["use_jitter"]

        # Apply the randomized time-jitter regularization
        if self._use_jitter:
            self._jitter = Jitter(configuration["jitter_probability"])

        """
        The jittered latent sequence is passed through a single
        convolutional layer with filter length 3 and 128 hidden
        units to mix information across neighboring timesteps.
        """
        self._conv_1 = Conv1DBuilder.build(
            in_channels=64,
            out_channels=768,
            kernel_size=2,
            use_kaiming_normal=configuration["use_kaiming_normal"],
        )

        # self._wavenet = WaveNetFactory.build(wavenet_type)
        self._wavenet = WaveNet(
            configuration["quantize"],
            configuration["n_layers"],
            configuration["n_loop"],
            configuration["residual_channels"],
            configuration["gate_channels"],
            configuration["skip_out_channels"],
            configuration["filter_size"],
            cin_channels=configuration["local_condition_dim"],
            gin_channels=configuration["global_condition_dim"],
            n_speakers=len(speaker_dic),
            upsample_conditional_features=True,
            upsample_scales=[2, 2, 2, 2, 2, 12]  # 768
            # upsample_scales=[2, 2, 2, 2, 12]
        )

        self._device = device

    def forward(self, y, local_condition, global_condition):
        if self._use_jitter and self.training:
            local_condition = self._jitter(local_condition)

        local_condition = self._conv_1(local_condition)

        # x = self._wavenet(y, local_condition, global_condition)
        x = self._wavenet(y)

        return x
