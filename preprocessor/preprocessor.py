import os
import tgt
import json
import torch
import random
import numpy as np
import pyworld as pw
import audio as Audio
import soundfile as sf

from tqdm import tqdm
from os import makedirs, listdir
from os.path import join, exists
from scipy.interpolate import interp1d
from feats_extract.dio_pitch import Dio
from feats_extract.energy import Energy
from feats_extract.cwt_pitch import cwt_extract
from sklearn.preprocessing import StandardScaler
from feats_extract.log_mel_fbank import LogMelFbank


import audio as Audio


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.align_dir = config["path"]["alignment_path"]
        self.val_size = config["preprocessing"]["val_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]

        assert config["preprocessing"]["pitch"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        assert config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        self.pitch_phoneme_averaging = (
            config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )
        self.energy_phoneme_averaging = (
            config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )

        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]
        self.mel_normalization = config["preprocessing"]["mel"]["normalization"]

        self.LogMelExtractor = LogMelFbank(
            fs=config["preprocessing"]["audio"]["sampling_rate"],
            n_fft=config["preprocessing"]["stft"]["filter_length"],
            win_length=config["preprocessing"]["stft"]["win_length"],
            hop_length=config["preprocessing"]["stft"]["hop_length"],
            fmin=config["preprocessing"]["mel"]["mel_fmin"],
            fmax=config["preprocessing"]["mel"]["mel_fmax"],
        )

        self.EnergyExtractor = Energy(
            fs=config["preprocessing"]["audio"]["sampling_rate"],
            n_fft=config["preprocessing"]["stft"]["filter_length"],
            win_length=config["preprocessing"]["stft"]["win_length"],
            hop_length=config["preprocessing"]["stft"]["hop_length"],
        )

        self.PitchExtractor = Dio(
            fs=config["preprocessing"]["audio"]["sampling_rate"],
            n_fft=config["preprocessing"]["stft"]["filter_length"],
            hop_length=config["preprocessing"]["stft"]["hop_length"],

        )

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    def build_from_path(self):
        makedirs(join(self.out_dir, "mel"), exist_ok=True)
        makedirs(join(self.out_dir, "pitch_per_phoneme"), exist_ok=True)
        # makedirs(join(self.out_dir, "pitch"), exist_ok=True)
        makedirs(join(self.out_dir, "pitch_per_frame"), exist_ok=True)
        makedirs(join(self.out_dir, "pitch_cwt"), exist_ok=True)
        makedirs(join(self.out_dir, "pitch_max_per_phoneme"), exist_ok=True)
        makedirs(join(self.out_dir, "energy"), exist_ok=True)
        makedirs(join(self.out_dir, "duration"), exist_ok=True)
        makedirs(join(self.out_dir, "vocoder_wav"), exist_ok=True)

        print("Processing Data ...")
        out = list()
        n_frames = 0
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()
        mel_scaler = StandardScaler()

        speakers = dict()
        for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))):
            speakers[speaker] = i
            for wav_name in os.listdir(join(self.in_dir, speaker)):
                if ".wav" not in wav_name:
                    continue

                basename = wav_name.split(".")[0]
                tg_path = join(
                    self.align_dir,
                    speaker,
                    f"{basename}.TextGrid"
                )
                if exists(tg_path):
                    print(basename)
                    # self.process_utterance(speaker, basename)
                    ret = self.process_utterance(speaker, basename)
                    if ret is None:
                        continue
                    else:
                        info, pitch, energy, mel = ret
                        # info, pitch = ret
                    out.append(info)

                if len(pitch) > 0:
                    pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                if len(energy) > 0:
                    energy_scaler.partial_fit(energy.reshape((-1, 1)))
                if len(mel) > 0:
                    mel_scaler.partial_fit(mel.transpose(1, 0))

                n_frames += len(mel[0])

        print("Computing statistic quantities ...")
        if self.pitch_normalization:
            pitch_mean = pitch_scaler.mean_[0]
            pitch_std = pitch_scaler.scale_[0]
        else:
            pitch_mean = 0
            pitch_std = 1

        if self.energy_normalization:
            energy_mean = energy_scaler.mean_[0]
            energy_std = energy_scaler.scale_[0]
        else:
            energy_mean = 0
            energy_std = 1

        if self.mel_normalization:
            mel_mean = mel_scaler.mean_
            mel_std = mel_scaler.scale_
        else:
            mel_mean = np.zeros(80)
            mel_std = np.ones(80)

        # # pitch_norm & cwt
        # print("Pitch normalization...")
        # pitch_min, pitch_max = self.normalize(
        #     join(self.out_dir, "pitch"), pitch_mean, pitch_std
        # )
        # print("Extracting CWT from normalized pitch...")
        # cwt_extract(join(self.out_dir, "pitch"), join(self.out_dir, "pitch_cwt_v2"))

        print("Energy normalization...")
        energy_min, energy_max = self.normalize(
            join(self.out_dir, "energy"), energy_mean, energy_std
        )

        print("Mel spectrogram normalization...")
        _, _ = self.normalize(
            join(self.out_dir, "mel"), mel_mean, mel_std, False
        )

        with open(join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        with open(join(self.out_dir, "stats_v2.json"), "w") as f:
            stats = {
                "pitch": [
                    # float(pitch_min),
                    # float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
                ],
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
                ],
                "mel": [
                    list(mel_mean),
                    list(mel_std),
                ]
            }
            f.write(json.dumps(stats))

        print(
            f"Total time: {n_frames * self.hop_length / self.sampling_rate / 3600} hours"
        )

        random.shuffle(out)
        out = [r for r in out if r is not None]

        with open(join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size:]:
                f.write(m + "\n")
        with open(join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[:self.val_size]:
                f.write(m + "\n")

        return out

    def process_utterance(self, speaker, basename):
        wav_path = join(self.in_dir, speaker, f"{basename}.wav")
        text_path = join(self.in_dir, speaker, f"{basename}.lab")
        tg_path = join(self.align_dir, speaker, f"{basename}.TextGrid")

        textgrid = tgt.io.read_textgrid(tg_path, include_empty_intervals=True)
        phone, duration, start, end = self.get_alignment(
            textgrid.get_tier_by_name("phones")
        )
        text = "{" + " ".join(phone) + "}"
        if start >= end:
            return None

        wav, _ = sf.read(wav_path)

        wav = wav[
            int(self.sampling_rate * start):int(self.sampling_rate * end)
        ].astype(np.float32)

        wav_name = wav_path.split('/')[-1]

        sf.write(join(self.out_dir, "vocoder_wav", wav_name), wav, samplerate=self.sampling_rate)

        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")

        pitch_per_frame, pitch_per_phoneme, pitch_length, pitch_cwt, pitch_max_per_phoneme = self.PitchExtractor(
            torch.tensor(wav, dtype=torch.float64).unsqueeze(0),
            durations=torch.tensor(duration).unsqueeze(0),
        )
        energy, energy_length = self.EnergyExtractor(
            torch.tensor(wav, dtype=torch.float64).unsqueeze(0),
            durations=torch.tensor(duration).unsqueeze(0),
        )
        mel_spectrogram, mel_length = self.LogMelExtractor(
            torch.tensor(wav, dtype=torch.float64).unsqueeze(0),
        )

        pitch_per_phoneme = pitch_per_phoneme.squeeze().cpu().numpy()
        pitch_per_frame = pitch_per_frame.squeeze().cpu().numpy()
        pitch_max_per_phoneme = pitch_max_per_phoneme.squeeze().cpu().numpy()
        energy = energy.squeeze().cpu().numpy()
        mel_spectrogram = mel_spectrogram.squeeze().transpose(1, 0).cpu().numpy()

        if np.sum(pitch_per_phoneme != 0) <= 1:
            return None

        dur_filename = f"{speaker}-duration-{basename}.npy"
        np.save(join(self.out_dir, "duration", dur_filename), duration)

        pitch_per_phoneme_filename = f"{speaker}-pitch_per_phoneme-{basename}.npy"
        np.save(join(self.out_dir, "pitch_per_phoneme", pitch_per_phoneme_filename), pitch_per_phoneme)

        # pitch_per_phoneme_filename = f"{speaker}-pitch-{basename}.npy"
        # np.save(join(self.out_dir, "pitch", pitch_per_phoneme_filename), pitch_per_phoneme)

        pitch_per_frame_filename = f"{speaker}-pitch_per_frame-{basename}.npy"
        np.save(join(self.out_dir, "pitch_per_frame", pitch_per_frame_filename), pitch_per_frame)

        pitch_max_per_frame_filename = f"{speaker}-pitch_max_per_phoneme-{basename}.npy"
        np.save(join(self.out_dir, "pitch_max_per_phoneme", pitch_max_per_frame_filename), pitch_max_per_phoneme)

        pitch_cwt_filename = f"{speaker}-pitch_cwt-{basename}.npy"
        np.save(join(self.out_dir, "pitch_cwt", pitch_cwt_filename), pitch_cwt)

        energy_filename = f"{speaker}-energy-{basename}.npy"
        np.save(join(self.out_dir, "energy", energy_filename), energy)

        mel_filename = f"{speaker}-mel-{basename}"
        np.save(join(self.out_dir, "mel", mel_filename), mel_spectrogram.T)

        return (
            "|".join([basename, speaker, text, raw_text]),
            pitch_per_phoneme,
            energy,
            mel_spectrogram,
        )

    def get_alignment(self, tier):

        sil_phones = ["sil", "sp", "spn", ""]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0

        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75-p25)
        upper = p75 + 1.5 * (p75-p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, in_dir, mean, std, flag=True):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max

        for filename in listdir(in_dir):
            filename = join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            if flag:
                max_value = max(max_value, max(values))
                min_value = min(min_value, min(values))

        return min_value, max_value