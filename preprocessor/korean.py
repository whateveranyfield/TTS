import os
import re
import shutil

from os import makedirs
from os.path import join, exists
from glob import glob


def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]

    for speaker in os.listdir(in_dir):
        for file_path in glob(join(in_dir, speaker, '**', '*.wav'), recursive=True):
            base_name = file_path.split('/')[-1].split('.')[0]
            wav_path = file_path
            text_path = file_path.replace('wav', 'txt')
            wav_output_path = join(out_dir, f'{speaker}', f'{base_name}.wav')
            text_output_path = join(out_dir, f'{speaker}', f'{base_name}.lab')

            # if exists(wav_output_path):
            #     continue
            # if exists(text_output_path):
            #     continue

            try:
                with open(text_path, encoding="utf-8") as f:
                    text = f.readline().strip("\n")
                    text = re.sub(r"([~/*\",:―·ㆍ`!?°“”’‘≪≫〈〉<>（）「」《》{}|=\';.\(\)\[\]\-\s+])", " ", text)
                    text = text.replace('π', '파이').replace('花', '화').replace('華', '').replace('萬古風霜','만고풍상').replace('處士','처사').replace('非', '비').replace('%', '퍼센트').replace('ㅘ', '와').replace('㎐', '헤르츠').replace('℃', '도씨').replace('㎢', '제곱킬로미터')
            except:
                continue

            makedirs(join(out_dir, f'{speaker}'), exist_ok=True)

            # shutil.copy(wav_path, wav_output_path)

            with open(
                text_output_path,
                "w",
            ) as f:
                f.write(text)
