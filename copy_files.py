import os
import shutil

from os.path import join
from glob import glob
from joblib import Parallel, delayed


def copy_audio(wav_dir, output_path):
    file_name = wav_dir.split('/')[-1]
    speaker_name = wav_dir.split('/')[-3]

    folder_path = join(output_path, speaker_name)
    os.makedirs(folder_path, exist_ok=True)

    shutil.copy(wav_dir, join(folder_path, file_name))


def copy_text(text_dir, output_path):
    file_name = text_dir.split('/')[-1][7:]
    speaker_name = text_dir.split('/')[-2][7:]

    folder_path = join(output_path, speaker_name)
    os.makedirs(folder_path, exist_ok=True)

    shutil.copy(text_dir, join(folder_path, file_name))


def copy_files(data_list, output_path, cmd_function):

    Parallel(n_jobs=-1, verbose=1)(
        delayed(cmd_function)(file_dir, output_path) for file_dir in data_list
    )


if __name__ == '__main__':
    wav_data_path = '/mnt/NAS/AI_dev_team/DB/TTS/Voucher/original/korean/'
    txt_data_path = '/mnt/NAS/AI_dev_team/DB/TTS/Voucher/edited_txt/'

    output_path = '/mnt/NAS/AI_dev_team/DB/TTS/Korean/original/'

    txt_list = glob(join(txt_data_path, '**', '*.txt'), recursive=True)

    copy_files(glob(join(wav_data_path, '**', '*.wav'), recursive=True), output_path, copy_audio)
    copy_files(glob(join(txt_data_path, '**', '*.txt'), recursive=True), output_path, copy_text)
