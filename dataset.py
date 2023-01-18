import os
import math
import json
import kaldiio

import numpy as np

from os.path import join
from text import text_to_sequence, phone_to_index
from torch.utils.data import Dataset
from utils.tools import pad_1D, pad_2D


class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, batch_size, sort=False, drop_last=False, use_teacher_forcing=False,
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.symbol_path = preprocess_config["path"]["symbol_path"]
        self.batch_size = batch_size

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filename,
        )

        self.use_teacher_forcing = use_teacher_forcing

        with open(self.symbol_path, 'r') as symbol_file:
            self.symbols = json.load(symbol_file)

        try:
            self.xvectors = self.load_xvectors(join(self.preprocessed_path, "spk_xvector.ark"))
        except:
            self.xvectors = None

        with open(join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)

        self.sort = sort
        self.drop_last = drop_last

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(phone_to_index(self.text[idx], self.symbols))

        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            f"{speaker}-mel-{basename}.npy",
        )
        mel = np.load(mel_path)

        pitch_per_phoneme_path = os.path.join(
            self.preprocessed_path,
            "pitch_per_phoneme",
            f"{speaker}-pitch_per_phoneme-{basename}.npy"
        )
        pitch_per_phoneme = np.load(pitch_per_phoneme_path)

        cwt_path = os.path.join(
            self.preprocessed_path,
            "pitch_cwt",
            f"{speaker}-pitch_cwt-{basename}.npy"
        )
        pitch_cwt = np.load(cwt_path, allow_pickle=True).item()

        pitch_per_frame_path = os.path.join(
            self.preprocessed_path,
            "pitch_per_frame",
            f"{speaker}-pitch_per_frame-{basename}.npy"
        )
        pitch_per_frame = np.load(pitch_per_frame_path)

        pitch_max_per_phoneme_path = os.path.join(
            self.preprocessed_path,
            "pitch_max_per_phoneme",
            f"{speaker}-pitch_max_per_phoneme-{basename}.npy"
        )
        pitch_max_per_phoneme = np.load(pitch_max_per_phoneme_path)

        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            f"{speaker}-energy-{basename}.npy"
        )
        energy = np.load(energy_path)

        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            f"{speaker}-duration-{basename}.npy"
        )
        duration = np.load(duration_path)

        if self.xvectors is not None:
            xvector = self.xvectors[speaker]

        cwt, scales, mean, std = pitch_cwt

        cwt = pitch_cwt[cwt]
        scales = pitch_cwt[scales]
        mean = pitch_cwt[mean]
        std = pitch_cwt[std]

        pitch_per_phoneme = np.pad(pitch_per_phoneme, (0, 1), 'constant', constant_values=0)
        pitch_per_frame = np.pad(pitch_per_frame, (0, 1), 'constant', constant_values=0)
        pitch_max_per_phoneme = np.pad(pitch_max_per_phoneme, (0, 1), 'constant', constant_values=0)
        energy = np.pad(energy, (0, 1), 'constant', constant_values=0)
        duration = np.pad(duration, (0, 1), 'constant', constant_values=1)
        cwt = np.pad(cwt, ((0, 1), (0, 0)), 'constant', constant_values=0)

        if self.xvectors is not None:
            sample = {
                "id": basename,
                "speaker": speaker_id,
                "text": phone,
                "raw_text": raw_text,
                "mel": mel,
                "pitch_per_phoneme": pitch_per_phoneme,
                "pitch_per_frame": pitch_per_frame,
                "cwt": cwt,
                "cwt_scale": scales,
                "cwt_mean": mean,
                "cwt_std": std,
                "energy": energy,
                "duration": duration,
                "xvector": xvector,
            }
        else:
            sample = {
                "id": basename,
                "speaker": speaker_id,
                "text": phone,
                "raw_text": raw_text,
                "mel": mel,
                "pitch_per_phoneme": pitch_per_phoneme,
                "pitch_per_frame": pitch_per_frame,
                "pitch_max_per_phoneme": pitch_max_per_phoneme,
                "cwt": cwt,
                "cwt_scale": scales,
                "cwt_mean": mean,
                "cwt_std": std,
                "energy": energy,
                "duration": duration,
            }

        return sample

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)

            return name, speaker, text, raw_text

    def load_xvectors(self, xvector_path):
        xvectors = {k: np.array(v) for k, v in kaldiio.load_ark(xvector_path)}
        return xvectors

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches_per_phoneme = [data[idx]["pitch_per_phoneme"] for idx in idxs]
        pitches_per_frame = [data[idx]["pitch_per_frame"] for idx in idxs]
        pitches_max_per_phoneme = [data[idx]["pitch_max_per_phoneme"] for idx in idxs]

        cwts = [data[idx]["cwt"] for idx in idxs]
        cwt_scales = [data[idx]["cwt_scale"] for idx in idxs]
        cwt_means = [data[idx]["cwt_mean"] for idx in idxs]
        cwt_stds = [data[idx]["cwt_std"] for idx in idxs]

        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]

        if self.xvectors is not None:
            xvector = [data[idx]["xvector"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        pitch_per_phoneme_lens = np.array([pitch.shape[0] for pitch in pitches_per_phoneme])
        pitch_per_frame_lens = np.array([pitch.shape[0] for pitch in pitches_per_frame])
        pitch_max_per_phoneme_lens = np.array([pitch.shape[0] for pitch in pitches_max_per_phoneme])
        cwt_lens = np.array([cwt.shape[0] for cwt in cwts])
        energy_lens = np.array([energy.shape[0] for energy in energies])
        duration_lens = np.array([duration.shape[0] for duration in durations])

        speakers = np.array(speakers)
        cwt_scales = np.array(cwt_scales)
        cwt_means = np.array(cwt_means)
        cwt_stds = np.array(cwt_stds)

        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches_per_phoneme = pad_1D(pitches_per_phoneme)
        pitches_per_frame = pad_1D(pitches_per_frame)
        pitches_max_per_phoneme = pad_1D(pitches_max_per_phoneme)
        cwts = pad_2D(cwts)
        energies = pad_1D(energies)
        durations = pad_1D(durations)

        if self.xvectors is not None:
            xvector = np.array(xvector)
            if self.use_teacher_forcing:
                return {
                    "base_name": ids,
                    # raw_texts,
                    # speakers,
                    "text": texts,
                    "text_lengths": text_lens,
                    # max(text_lens),
                    "feats": mels,
                    "feats_lengths": mel_lens,
                    # max(mel_lens),
                    "pitch_per_phoneme": pitches_per_phoneme,
                    "pitch_per_phoneme_lengths": pitch_per_phoneme_lens,
                    "pitch_per_frame": pitches_per_frame,
                    "pitch_per_frame_lengths": pitch_per_frame_lens,
                    "pitch_max_per_phoneme": pitches_max_per_phoneme,
                    "pitch_max_per_phoneme_lengths": pitch_max_per_phoneme_lens,
                    "cwt": cwts,
                    "cwt_lengths": cwt_lens,
                    "cwt_scales": cwt_scales,
                    "cwt_means": cwt_means,
                    "cwt_stds": cwt_stds,
                    "energy": energies,
                    "energy_lengths": energy_lens,
                    "durations": durations,
                    "durations_lengths": duration_lens,
                    "spemb": xvector,
                }
            else:
                return {
                    # raw_texts,
                    # speakers,
                    "text": texts,
                    "text_lengths": text_lens,
                    # max(text_lens),
                    "feats": mels,
                    "feats_lengths": mel_lens,
                    # max(mel_lens),
                    "pitch_per_phoneme": pitches_per_phoneme,
                    "pitch_per_phoneme_lengths": pitch_per_phoneme_lens,
                    "pitch_per_frame": pitches_per_frame,
                    "pitch_per_frame_lengths": pitch_per_frame_lens,
                    "pitch_max_per_phoneme": pitches_max_per_phoneme,
                    "pitch_max_per_phoneme_lengths": pitch_max_per_phoneme_lens,
                    "cwt": cwts,
                    "cwt_lengths": cwt_lens,
                    "cwt_scales": cwt_scales,
                    "cwt_means": cwt_means,
                    "cwt_stds": cwt_stds,
                    "energy": energies,
                    "energy_lengths": energy_lens,
                    "durations": durations,
                    "durations_lengths": duration_lens,
                    "spemb": xvector,
                }
        else:
            if self.use_teacher_forcing:
                return {
                    "base_name": ids,
                    # raw_texts,
                    "sids": speakers,
                    "text": texts,
                    "text_lengths": text_lens,
                    # max(text_lens),
                    "feats": mels,
                    "feats_lengths": mel_lens,
                    # max(mel_lens),
                    "pitch_per_phoneme": pitches_per_phoneme,
                    "pitch_per_phoneme_lengths": pitch_per_phoneme_lens,
                    "pitch_per_frame": pitches_per_frame,
                    "pitch_per_frame_lengths": pitch_per_frame_lens,
                    "pitch_max_per_phoneme": pitches_max_per_phoneme,
                    "pitch_max_per_phoneme_lengths": pitch_max_per_phoneme_lens,
                    "cwt": cwts,
                    "cwt_lengths": cwt_lens,
                    "cwt_scales": cwt_scales,
                    "cwt_means": cwt_means,
                    "cwt_stds": cwt_stds,
                    "energy": energies,
                    "energy_lengths": energy_lens,
                    "durations": durations,
                    "durations_lengths": duration_lens,
                }
            else:
                return {
                    # raw_texts,
                    "sids": speakers,
                    "text": texts,
                    "text_lengths": text_lens,
                    # max(text_lens),
                    "feats": mels,
                    "feats_lengths": mel_lens,
                    # max(mel_lens),
                    "pitch_per_phoneme": pitches_per_phoneme,
                    "pitch_per_phoneme_lengths": pitch_per_phoneme_lens,
                    "pitch_per_frame": pitches_per_frame,
                    "pitch_per_frame_lengths": pitch_per_frame_lens,
                    "pitch_max_per_phoneme": pitches_max_per_phoneme,
                    "pitch_max_per_phoneme_lengths": pitch_max_per_phoneme_lens,
                    "cwt": cwts,
                    "cwt_lengths": cwt_lens,
                    "cwt_scales": cwt_scales,
                    "cwt_means": cwt_means,
                    "cwt_stds": cwt_stds,
                    "energy": energies,
                    "energy_lengths": energy_lens,
                    "durations": durations,
                    "durations_lengths": duration_lens,
                }

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        # idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        # output = list()
        # for idx in idx_arr:
            # output.append(self.reprocess(data, idx))
        output = self.reprocess(data, idx_arr)

        return output


class TextDataset(Dataset):
    def __init__(self, filepath, preprocess_config):
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filepath
        )
        with open(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"], "speakers.json"
            )
        ) as f:
            self.speaker_map = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))

        return (basename, speaker_id, phone, raw_text)

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def collate_fn(self, data):
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]
        text_lens = np.array([text.shape[0] for text in texts])

        texts = pad_1D(texts)

        return ids, raw_texts, speakers, texts, text_lens, max(text_lens)

