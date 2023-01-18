import os
import numpy as np
import soundfile as sf

from glob import glob
from os.path import join, exists
from joblib import Parallel, delayed


def wav_padding(wav_file):
    signal, sr = sf.read(wav_file)
    if len(signal) < (sr * 3.1):
        temp = np.zeros(int(sr * 3.1))
        start_point = int((int(sr * 3.1) - len(signal)) / 2)
        temp[start_point:start_point + len(signal)] = signal
        sf.write(wav_file, temp, sr)


def cmd_function(wav_file):
    wav_padding(wav_file)
    cmd_order = f'ffmpeg-normalize -f {wav_file} -o {wav_file} -ar 24000'
    print(wav_file)
    os.system(cmd_order)


def amplitude_normalize(config):
    print("Start amplitude normalize")
    wav_dir = config["path"]["raw_path"]
    wav_list = glob(join(wav_dir, '**', '*.wav'), recursive=True)

    Parallel(n_jobs=-1, verbose=1)(
        delayed(cmd_function)(wav_file) for wav_file in wav_list
    )

    # for wav_file in wav_list:
    #     cmd_function(wav_file)
