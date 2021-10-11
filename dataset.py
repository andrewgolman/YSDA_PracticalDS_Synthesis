import math
import os
import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

import utils
from utils import pad_1D, pad_2D
from text import text_to_sequence


class TrainDataset(Dataset):
    def __init__(self, datafolder, metafile="train.txt", sort=True, speaker_map_path=None):
        self.datafolder = Path(datafolder)
        self.sort = sort

        if speaker_map_path is None:
            speaker_map_path = self.datafolder / "speakers.json"
        with open(speaker_map_path) as f:
            self.speaker_map = json.load(f)
        self.basename, self.speaker = self.process_meta(self.datafolder / metafile)

    def __len__(self):
        return len(self.basename)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        sample = {
            "id": basename,
            "speaker": speaker_id,
        }

        text_path = self.datafolder / "phone" / "phone-{}.npy".format(basename)
        sample['text'] = utils.process_text(np.load(text_path))

        mel_path = self.datafolder / "mel" / "mel-{}.npy".format(basename)
        sample["mel_target"] = np.load(mel_path)

        D_path = self.datafolder / "alignment" / "alignment-{}.npy".format(basename)
        if not os.path.exists(D_path):
            D_path = self.datafolder / "alignment" / "ali-{}.npy".format(basename)
        sample["D"] = np.load(D_path)

        f0_path = self.datafolder / "f0" / "f0-{}.npy".format(basename)
        sample["f0"] = np.load(f0_path)

        energy_path = self.datafolder / "energy" / "energy-{}.npy".format(basename)
        sample["energy"] = np.load(energy_path)

        assert len(sample["f0"]) == len(sample["energy"]) and len(sample["f0"]) == sample["mel_target"].shape[0], \
            f"{basename}, {len(sample['f0'])}, {len(sample['energy'])}, {sample['mel_target'].shape[0]}"

        return sample

    @staticmethod
    def process_meta(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            speaker = []
            name = []
            for line in f.readlines():
                n, s = line.strip().split()
                name.append(n)
                speaker.append(s)
            return name, speaker

    def reprocess(self, batch, cut_list):
        ids = [batch[ind]["id"] for ind in cut_list]
        speakers = [batch[ind]["speaker"] for ind in cut_list]
        energies = [batch[ind]["energy"] for ind in cut_list]
        texts = [batch[ind]["text"] for ind in cut_list]
        mel_targets = [batch[ind]["mel_target"] for ind in cut_list]
        Ds = [batch[ind]["D"] for ind in cut_list]
        f0s = [batch[ind]["f0"] for ind in cut_list]

        for text, D, id_ in zip(texts, Ds, ids):
            if len(text) != len(D):
                assert False, str((text, text.shape, D, D.shape, id_))
        length_text = np.array(list())
        for text in texts:
            length_text = np.append(length_text, text.shape[0])

        length_mel = np.array(list())
        for mel in mel_targets:
            length_mel = np.append(length_mel, mel.shape[0])

        speakers = np.array(speakers)
        texts = pad_1D(texts)
        mel_targets = pad_2D(mel_targets)
        Ds = pad_1D(Ds)
        f0s = pad_1D(f0s)
        energies = pad_1D(energies)

        out = {
            "id": ids,
            "speaker": speakers,
            "text": texts,
            "mel_target": mel_targets,
            "D": Ds,
            "f0": f0s,
            "energy": energies,
            "src_len": length_text,
            "mel_len": length_mel,
        }

        return out

    def collate_fn(self, batch):
        len_arr = np.array([d["text"].shape[0] for d in batch])
        index_arr = np.argsort(-len_arr)
        batchsize = len(batch)
        real_batchsize = int(math.sqrt(batchsize))

        cut_list = list()
        for i in range(real_batchsize):
            if self.sort:
                cut_list.append(
                    index_arr[i * real_batchsize : (i + 1) * real_batchsize]
                )
            else:
                cut_list.append(np.arange(i * real_batchsize, (i + 1) * real_batchsize))

        output = list()
        for i in range(real_batchsize):
            output.append(self.reprocess(batch, cut_list[i]))

        return output
