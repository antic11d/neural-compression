import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Conv1DBuilder(object):
    @staticmethod
    def build(
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        use_kaiming_normal=False,
    ):
        conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        if use_kaiming_normal:
            conv = nn.utils.weight_norm(conv)
            nn.init.kaiming_normal_(conv.weight)
        return conv


class Residual(nn.Module):
    def __init__(
        self, in_channels, num_hiddens, num_residual_hiddens, use_kaiming_normal
    ):
        super(Residual, self).__init__()

        relu_1 = nn.ReLU(True)
        conv_1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=num_residual_hiddens,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        if use_kaiming_normal:
            conv_1 = nn.utils.weight_norm(conv_1)
            nn.init.kaiming_normal_(conv_1.weight)

        relu_2 = nn.ReLU(True)
        conv_2 = nn.Conv1d(
            in_channels=num_residual_hiddens,
            out_channels=num_hiddens,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        if use_kaiming_normal:
            conv_2 = nn.utils.weight_norm(conv_2)
            nn.init.kaiming_normal_(conv_2.weight)

        # All parameters same as specified in the paper
        self._block = nn.Sequential(relu_1, conv_1, relu_2, conv_2)

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(
        self,
        in_channels,
        num_hiddens,
        num_residual_layers,
        num_residual_hiddens,
        use_kaiming_normal,
    ):
        super(ResidualStack, self).__init__()

        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList(
            [
                Residual(
                    in_channels, num_hiddens, num_residual_hiddens, use_kaiming_normal
                )
            ]
            * self._num_residual_layers
        )

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class ConvTranspose1DBuilder(object):
    @staticmethod
    def build(
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        use_kaiming_normal=False,
    ):
        conv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        if use_kaiming_normal:
            conv = nn.utils.weight_norm(conv)
            nn.init.kaiming_normal_(conv.weight)
        return conv


class Jitter(nn.Module):
    """
    Jitter implementation from [Chorowski et al., 2019].
    During training, each latent vector can replace either one or both of
    its neighbors. As in dropout, this prevents the model from
    relying on consistency across groups of tokens. Additionally,
    this regularization also promotes latent representation stability
    over time: a latent vector extracted at time step t must strive
    to also be useful at time steps t − 1 or t + 1.
    """

    def __init__(self, probability=0.12):
        super(Jitter, self).__init__()

        self._probability = probability

    def forward(self, quantized):
        original_quantized = quantized.detach().clone()
        length = original_quantized.size(2)
        for i in range(length):
            """
            Each latent vector is replace with either of its neighbors with a certain probability
            (0.12 from the paper).
            """
            replace = [True, False][
                np.random.choice([1, 0], p=[self._probability, 1 - self._probability])
            ]
            if replace:
                if i == 0:
                    neighbor_index = i + 1
                elif i == length - 1:
                    neighbor_index = i - 1
                else:
                    """
                    "We independently sample whether it is to
                    be replaced with the token right after
                    or before it."
                    """
                    neighbor_index = i + np.random.choice([-1, 1], p=[0.5, 0.5])
                quantized[:, :, i] = original_quantized[:, :, neighbor_index]

        return quantized


class GlobalConditioning(object):
    @staticmethod
    def compute(
        speaker_dic, speaker_ids, x_one_hot, device, gin_channels=128, expand=True
    ):
        speakers_embedding = GlobalConditioning._Embedding(
            len(speaker_dic), gin_channels, padding_idx=None, std=0.1
        ).to(device)

        # Extract the batch size and the signal length
        B, _, T = x_one_hot.size()

        # (B x 1) -> (B x 1 x gin_channels)
        global_conditioning = speakers_embedding(speaker_ids.view(B, -1).long())

        # (B x gin_channels x 1)
        global_conditioning = global_conditioning.transpose(1, 2)

        # Check if the result have the right dimension
        assert global_conditioning.dim() == 3

        """
        Return the global conditioning if the expand
        option is set to False
        """
        if not expand:
            return global_conditioning

        # Expand global conditioning features to all time steps
        expanded_global_conditioning = GlobalConditioning._expand_global_features(
            B, T, global_conditioning, bct=True
        )

        return expanded_global_conditioning

    @staticmethod
    def _Embedding(num_embeddings, embedding_dim, padding_idx, std=0.01):
        m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        m.weight.data.normal_(0, std)
        return m

    @staticmethod
    def _expand_global_features(B, T, g, bct=True):
        """
        Expand global conditioning features to all time steps

        Args:
            B (int): Batch size.
            T (int): Time length.
            g (Tensor): Global features, (B x C) or (B x C x 1).
            bct (bool) : returns (B x C x T) if True, otherwise (B x T x C)

        Returns:
            Tensor: B x C x T or B x T x C or None
        """
        if g is None:
            return None
        g = g.unsqueeze(-1) if g.dim() == 2 else g
        if bct:
            g_bct = g.expand(B, -1, T)
            return g_bct.contiguous()
        else:
            g_btc = g.expand(B, -1, T).transpose(1, 2)
            return g_btc.contiguous()
