import torch
import numpy as np
from scipy.io import wavfile
from utils import get_list_from_lengths, pad_1D, process_text


def get_vocoder():
    vocoder = torch.hub.load(
        "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
    )
    vocoder.mel2wav.eval()
    vocoder.mel2wav.cuda()
    return vocoder


def vocode(mels, lengths=None, max_wav_value=32768.0):
    with torch.no_grad():
        wavs = vocoder.inverse(mels / np.log(10)).cpu().numpy() * max_wav_value
    wavs = wavs.astype("int16")
    if lengths is not None:
        wavs = [x[:length] for x, length in zip(wavs, lengths)]
    return wavs


def vocoder_infer(mels, paths, sampling_rate, lengths=None):
    wavs = vocode(mels, lengths=lengths)
    for i in range(len(mels)):
        wav = wavs[i]
        path = paths[i]
        if lengths is not None:
            length = lengths[i]
            wavfile.write(path, sampling_rate, wav[:length])
        else:
            wavfile.write(path, sampling_rate, wav)


vocoder = get_vocoder()


def synthesize(
    model,
    data_cfg,
    speaker_ids,
    texts,
    mel_len,
    Ds,
    f0s,
    energies,
    speaker_embedding=None
):
    src_len = torch.from_numpy(np.array([len(t) for t in texts])).cuda()
    texts = [
        process_text(text) for text in texts
    ]
    texts = torch.from_numpy(pad_1D(texts)).cuda()
    Ds = torch.from_numpy(pad_1D(Ds)).cuda() if Ds is not None else None
    f0s = torch.from_numpy(pad_1D(f0s)).cuda() if f0s is not None else None
    energies = torch.from_numpy(pad_1D(energies)).cuda() if energies is not None else None
    mel_len = torch.from_numpy(np.array(mel_len)).long().cuda() if mel_len is not None else None
    speakers = torch.from_numpy(np.array(speaker_ids)).cuda() if speaker_ids is not None else None

    with torch.no_grad():
        (
            mel,
            mel_postnet,
            log_duration_output,
            duration_output,
            f0_output,
            energy_output,
            _,
            _,
            mel_len,
        ) = model(
            texts,
            src_len,
            mel_len=mel_len,
            d_target=Ds,
            p_target=f0s,
            e_target=energies,
            max_src_len=torch.max(src_len).item(),
            speaker=speakers,
            speaker_embedding=speaker_embedding
        )

        wavs = vocode(mel_postnet.transpose(1, 2), lengths=mel_len * data_cfg.hop_length)
        duration_output = duration_output if Ds is None else Ds
        energy_output = energy_output if energies is None else energies

    mels = get_list_from_lengths(mel_postnet, mel_len)
    f0s = get_list_from_lengths(f0_output, mel_len)
    energies = get_list_from_lengths(energy_output, mel_len)
    Ds = get_list_from_lengths(duration_output, src_len)

    return wavs, mels, Ds, f0s, energies
