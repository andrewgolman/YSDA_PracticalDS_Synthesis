import numpy as np
import pyworld as pw
import audio
import os
from pathlib import Path
from scipy.interpolate import interp1d


def extract_f0(wav, max_duration, data_cfg):
    # Compute fundamental frequency
    f0, t = pw.dio(
        wav.astype(np.float64),
        data_cfg.sampling_rate,
        frame_period=data_cfg.hop_length / data_cfg.sampling_rate * 1000,
    )
    f0 = pw.stonemask(wav.astype(np.float64), f0, t, data_cfg.sampling_rate).astype(np.float32)
    f0 = f0[:max_duration]

    nonzero_ids = np.where(f0 != 0)[0]
    if len(nonzero_ids) > 2:
        interp_fn = interp1d(
            nonzero_ids,
            f0[nonzero_ids],
            fill_value=(f0[nonzero_ids[0]], f0[nonzero_ids[-1]]),
            bounds_error=False,
        )
        f0 = interp_fn(np.linspace(0, len(f0) - 1, max_duration))
        f0 = np.log(f0)

    if np.sum(f0 != 0) <= 1:
        return None
    return f0


def extract_mel_energy(wav, max_duration, data_cfg):
    # Compute mel-scale spectrogram and energy
    mel_spectrogram, energy = audio.tools.get_mel_from_wav(wav, data_cfg)
    mel_spectrogram = mel_spectrogram.numpy().astype(np.float32)[:, :max_duration]
    energy = np.log(energy + data_cfg.energy_log_offset)
    energy = energy[:max_duration]
    return mel_spectrogram.T, energy


def build_feature_stat(base_path, train_filenames):
    f0_min = 1e9
    f0_max = -1e9
    energy_min = 1e9
    energy_max = -1e9

    for filename in train_filenames:
        f0 = np.load(base_path / "f0" / f"f0-{filename}.npy")
        f0_min = min(f0_min, f0.min())
        f0_max = max(f0_max, f0.max())
        energy = np.load(base_path / "energy" / f"energy-{filename}.npy")
        energy_min = min(energy_min, energy.min())
        energy_max = max(energy_max, energy.max())

    return {
        "f0_min": float(f0_min),
        "f0_max": float(f0_max),
        "energy_min": float(energy_min),
        "energy_max": float(energy_max),
    }


def train_test_split(base_path):
    np.random.seed(42)
    train_out = []
    dev_out = []
    speakers = set()

    with open(base_path / "train.txt", "w", encoding="utf-8") as ftrain:
        with open(base_path / "dev.txt", "w", encoding="utf-8") as fdev:
            for filename in os.listdir(base_path / "f0"):
                basename = Path(filename).stem.split("-")[1]
                speaker = filename.split("-")[1].split("_")[0]
                speakers.add(speaker)
                if np.random.random() < 0.9:
                    print(basename, speaker, file=ftrain)
                    train_out.append(basename)
                else:
                    print(basename, speaker, file=fdev)
                    dev_out.append(basename)

    return train_out, dev_out, speakers
