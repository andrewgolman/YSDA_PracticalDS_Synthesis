import torch
from scipy.signal import lfilter

import audio.stft as stft

_stft = None


def create_stft(cfg):
    global _stft
    if _stft is None:
        _stft = stft.TacotronSTFT(
            cfg.filter_length,
            cfg.hop_length,
            cfg.win_length,
            cfg.n_mel_channels,
            cfg.sampling_rate,
            cfg.mel_fmin,
            cfg.mel_fmax,
        )


def get_mel_from_wav(audio, cfg):
    create_stft(cfg)

    sampling_rate = cfg.sampling_rate
    if sampling_rate != _stft.sampling_rate:
        raise ValueError(
            "{} {} SR doesn't match target {} SR".format(
                sampling_rate, _stft.sampling_rate
            )
        )

    audio = lfilter([1, -cfg.preemph], [1], audio)
    audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
    audio = torch.autograd.Variable(audio, requires_grad=False)
    melspec, energy = _stft.mel_spectrogram(audio)
    melspec = torch.squeeze(melspec, 0)
    energy = torch.squeeze(energy, 0)

    return melspec, energy
