import os
import shutil

from os import makedirs
from os.path import join, exists
from tqdm import tqdm
from text import _clean_text
from glob import glob


def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]

    for langauge in os.listdir(in_dir):
        for speaker in os.listdir(join(in_dir, langauge)):
            for file_path in glob(join(in_dir, langauge, speaker, '**', '*.wav'), recursive=True):
                base_name = file_path.split('/')[-1].split('.')[0]
                wav_path = file_path
                text_path = file_path.replace('wav', 'txt')
                wav_output_path = join(out_dir, f'{langauge}_{speaker}', f'{langauge}_{base_name}.wav')
                text_output_path = join(out_dir, f'{langauge}_{speaker}', f'{langauge}_{base_name}.lab')

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

                makedirs(join(out_dir, f'{langauge}_{speaker}'), exist_ok=True)

                shutil.copy(wav_path, wav_output_path)

                with open(
                    text_output_path,
                    "w",
                ) as f:
                    f.write(text)
