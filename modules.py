from collections import OrderedDict
import torch
import torch.nn as nn

from torch import bucketize as f0_bucketize
from torch import bucketize as energy_bucketize

import utils


class VarianceAdaptor(nn.Module):
    """ Variance Adaptor """

    def __init__(self, cfg, f0_min, f0_max, energy_min, energy_max):
        super().__init__()

        self.duration_offset = cfg.log_offset

        self.duration_predictor = VariancePredictor(cfg)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(cfg)
        self.energy_predictor = VariancePredictor(cfg)
        self.pitch_bins = nn.Parameter(
            torch.linspace(f0_min, f0_max, cfg.n_bins - 1),
            requires_grad=False,
        )
        self.energy_bins = nn.Parameter(
            torch.linspace(energy_min, energy_max, cfg.n_bins - 1),
            requires_grad=False,
        )

        self.pitch_embedding = nn.Embedding(cfg.n_bins, cfg.encoder_hidden)
        self.energy_embedding = nn.Embedding(cfg.n_bins, cfg.encoder_hidden)

    def regulate_length(self, x, x_features, log_duration_prediction, duration_target, mel_mask, max_len):
        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(
                torch.round(torch.exp(log_duration_prediction) - self.duration_offset),
                min=0,
            ).long()
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = utils.get_mask_from_lengths(mel_len)

        x_features, _ = self.length_regulator(x_features, duration_rounded, max_len)

        return x, x_features, duration_rounded, mel_len, mel_mask

    def forward(
        self,
        x,
        x_features,
        src_mask,
        mel_mask=None,
        duration_target=None,
        pitch_target=None,
        energy_target=None,
        max_len=None,
    ):

        log_duration_prediction = self.duration_predictor(x_features, src_mask)
        x, x_features, duration_rounded, mel_len, mel_mask = self.regulate_length(x, x_features, log_duration_prediction, duration_target, mel_mask, max_len)
        src_mask = mel_mask

        pitch_prediction = self.pitch_predictor(x_features, src_mask)
        if pitch_target is not None:
            pitch_embedding = self.pitch_embedding(
                f0_bucketize(pitch_target, self.pitch_bins)
            )
        else:
            pitch_embedding = self.pitch_embedding(
                f0_bucketize(pitch_prediction, self.pitch_bins)
            )

        energy_prediction = self.energy_predictor(x_features, src_mask)
        if energy_target is not None:
            energy_embedding = self.energy_embedding(
                energy_bucketize(energy_target, self.energy_bins)
            )
        else:
            energy_embedding = self.energy_embedding(
                energy_bucketize(energy_prediction, self.energy_bins)
            )

        x = x + pitch_embedding + energy_embedding

        return (
            x,
            log_duration_prediction,
            duration_rounded,
            pitch_prediction,
            energy_prediction,
            mel_len,
            mel_mask,
        )


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self):
        super().__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = utils.pad(output, max_len)
        else:
            output = utils.pad(output)

        return output, torch.LongTensor(mel_len).cuda()

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(int(expand_size), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class VariancePredictor(nn.Module):
    """ Duration, Pitch and Energy Predictor """

    def __init__(self, cfg, out_c=1):
        super().__init__()

        self.input_size = cfg.encoder_hidden
        self.filter_size = cfg.variance_predictor_filter_size
        self.kernel = cfg.variance_predictor_kernel_size
        self.conv_output_size = cfg.variance_predictor_filter_size
        self.dropout = cfg.variance_predictor_dropout

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, out_c)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)

        if mask is not None:
            out = out.masked_fill(mask.unsqueeze(-1), 0.0)
        out = out.squeeze(-1)
        return out


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x
