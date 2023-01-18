import os
import shutil

from pydub import AudioSegment
from os import makedirs
from os.path import join, exists
from text import _clean_text
from glob import glob


def prepare_align(config):
    print("Start align")
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]

    wav_dir = join(in_dir, 'wav')
    wav_list = glob(join(wav_dir, '**', '*.wav'), recursive=True)

    for wav_dir in wav_list:
        emotion = wav_dir.split('/')[-2]
        os.makedirs(join(out_dir, emotion), exist_ok=True)

        base_name = wav_dir.split('/')[-1].split('.')[0]
        text_dir = wav_dir.replace('wav', 'txt')
        wav_output_path = join(out_dir, emotion, f"{base_name}.wav")
        text_output_path = join(out_dir, emotion, f"{base_name}.lab")

        if exists(wav_output_path):
            continue
        if exists(text_output_path):
            continue

        try:
            with open(text_dir, encoding="utf-8") as f:
                text = f.readline().strip("\n")
        except:
            continue
        text = _clean_text(text, cleaners)

        makedirs(join(out_dir, emotion), exist_ok=True)

        signal = AudioSegment.from_wav(wav_dir)
        signal = signal.set_channels(1)
        signal.export(wav_output_path, format="wav")
        # shutil.copy(wav_dir, wav_output_path)

        with open(
            text_output_path,
            "w",
        ) as f:
            f.write(text)
