#!/usr/bin/env python3

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Trim silence at the beginning and the end of audio."""

import librosa
import numpy

import resampy
import soundfile as sf

from glob import glob
from os.path import join


def _time_to_str(time_idx):
    time_idx = time_idx * 10**4
    return "%06d" % time_idx


def trim_silence(config):
    """Run silence trimming and generate segments."""

    out_dir = config["path"]["raw_path"]
    fs = config["preprocessing"]["audio"]["sampling_rate"]
    win_length = 2048
    shift_length = 512
    threshold = 35
    # normalize = 16
    min_silence = 0.01

    wav_list = glob(join(out_dir, "**", "*.wav"))

    for wav_path in wav_list:
        array, rate = sf.read(wav_path)
        array = array.astype(numpy.float32)
        # if normalize is not None and normalize != 1:
        #     array = array / (1 << (normalize - 1))
        if rate != fs:
            array = resampy.resample(array, rate, fs, axis=0)
        array_trim, idx = librosa.effects.trim(
            y=array,
            top_db=threshold,
            frame_length=win_length,
            hop_length=shift_length,
        )

        start, end = idx / fs

        # added minimum silence part
        start = max(0.0, start - min_silence)
        end = min(len(array) / fs, end + min_silence)

        sf.write(wav_path, array[int(start * fs):int(end * fs)], fs)