import os
import shutil

from glob import glob
from os.path import join
from joblib import Parallel, delayed
from audio.db_normalization import cmd_function

if __name__ == '__main__':
    data_path = '/mnt/NAS/AI_dev_team/users/matt/TTS/DB/matt/'
    wav_list = glob(join(data_path, '**', '*.wav'), recursive=True)
    output_path = '/mnt/NAS/AI_dev_team/users/matt/TTS/DB/matt/data/'

    os.makedirs(output_path, exist_ok=True)

    # for wav_dir in wav_list:
    #     wav_name = wav_dir.split('/')[-1]
    #     txt_dir = wav_dir.replace('wav', 'txt')
    #     txt_name = wav_name.replace('wav', 'txt')
    #
    #     shutil.copy(wav_dir, join(output_path, wav_name))
    #     shutil.copy(txt_dir, join(output_path, txt_name))

    wav_list = glob(join(output_path, '**', '*.wav'), recursive=True)

    Parallel(n_jobs=-1, verbose=1)(
        delayed(cmd_function)(wav_file) for wav_file in wav_list
    )