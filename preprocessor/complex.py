import os
import numpy as np
import soundfile as sf
import shutil

from os import makedirs
from os.path import join, exists
from scipy.io import wavfile
from tqdm import tqdm
from text import _clean_text
from glob import glob


def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]

    for speaker in tqdm(os.listdir(in_dir)):
        if "english_info" in speaker:
            continue
        for file_path in glob(join(in_dir, speaker, '**', '*.wav'), recursive=True):
            base_name = file_path.split('/')[-1].split('.')[0]
            wav_path = file_path
            text_path = file_path.replace('wav', 'txt')
            wav_output_path = join(out_dir, speaker, f"{base_name}.wav")
            text_output_path = join(out_dir, speaker, f"{base_name}.lab")

            if exists(wav_output_path):
                continue
            if exists(text_output_path):
                continue

            try:
                with open(text_path, encoding="utf-8") as f:
                    text = f.readline().strip("\n")
            except:
                continue
            text = _clean_text(text, cleaners)

            makedirs(join(out_dir, speaker), exist_ok=True)

            shutil.copy(wav_path, wav_output_path)

            with open(
                text_output_path,
                "w",
            ) as f:
                f.write(text)
