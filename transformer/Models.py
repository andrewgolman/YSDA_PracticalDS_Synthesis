import torch
import torch.nn as nn
import numpy as np

import transformer.Constants as Constants
from transformer.Layers import FFTBlock
from text.symbols import symbols


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)


class Encoder(nn.Module):
    """ Encoder """

    def __init__(
        self,
        cfg,
        n_src_vocab=len(symbols) + 1,
    ):
        super(Encoder, self).__init__()
        self.block_type = cfg.block_type if hasattr(cfg, "block_type") else 'fft'
        self.max_seq_len = cfg.max_seq_len
        self.d_model = cfg.encoder_hidden
        d_word_vec = cfg.encoder_hidden
        n_layers = cfg.encoder_layer
        n_head = cfg.encoder_head
        d_k = cfg.encoder_hidden // cfg.encoder_head
        d_v = cfg.encoder_hidden // cfg.encoder_head
        d_inner = cfg.fft_conv1d_filter_size
        dropout = cfg.encoder_dropout
        n_position = self.max_seq_len + 1

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=Constants.PAD)
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )
        self.layer_stack = nn.ModuleList([
                FFTBlock(self.d_model, d_inner, n_head, d_k, d_v, cfg.fft_conv1d_kernel_size, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, src_seq, mask, return_attns=False):
        enc_slf_attn_list = []
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]

        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        enc_output = self.src_word_emb(src_seq)
        if not self.training and src_seq.shape[1] > self.max_seq_len:
            enc_output = enc_output + get_sinusoid_encoding_table(
                src_seq.shape[1], self.d_model
            )[: src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                src_seq.device
            )
        else:
            enc_output = enc_output + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output


class Decoder(nn.Module):
    """ Decoder """

    def __init__(
        self,
        cfg,
    ):
        super(Decoder, self).__init__()

        self.block_type = cfg.block_type if hasattr(cfg, "block_type") else 'fft'
        self.max_seq_len = cfg.max_seq_len
        self.d_model = cfg.decoder_hidden
        d_word_vec = cfg.encoder_hidden
        n_layers = cfg.decoder_layer
        n_head = cfg.decoder_head
        d_k = cfg.decoder_hidden // cfg.decoder_head
        d_v = cfg.decoder_hidden // cfg.decoder_head
        d_inner = cfg.fft_conv1d_filter_size
        dropout = cfg.decoder_dropout
        n_position = self.max_seq_len + 1

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList([
            FFTBlock(self.d_model, d_inner, n_head, d_k, d_v, cfg.fft_conv1d_kernel_size, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, enc_seq, mask, return_attns=False):

        dec_slf_attn_list = []
        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]

        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # -- Forward
        if not self.training and enc_seq.shape[1] > self.max_seq_len:
            dec_output = enc_seq + get_sinusoid_encoding_table(
                enc_seq.shape[1], self.d_model
            )[: enc_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                enc_seq.device
            )
        else:
            dec_output = enc_seq + self.position_enc[:, :max_len, :].expand(
                batch_size, -1, -1
            )

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output
