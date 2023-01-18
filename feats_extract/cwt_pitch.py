import os
import numpy as np
import pycwt as wavelet

from os.path import join


def cwt_extract(pitch_dir, output_dir):

    for file_dir in os.listdir(pitch_dir):
        pitch = np.load(join(pitch_dir, file_dir))

        pitch_cwt_dict = dict()
        mean = np.mean(pitch)
        std = np.std(pitch)

        lf0_norm = np.divide((pitch - mean), std, out=np.zeros_like((pitch - mean)), where=std != 0)
        mother = wavelet.MexicanHat()

        dt = 0.005
        dj = 1
        s0 = dt * 2
        J = 9

        wavelet_lf0, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(np.squeeze(lf0_norm), dt, dj, s0, J, mother)

        cwt = np.real(wavelet_lf0).T

        pitch_cwt_dict["pitch_cwt"] = cwt
        pitch_cwt_dict["scales"] = scales
        pitch_cwt_dict["mean"] = mean
        pitch_cwt_dict["std"] = std

        speaker, _, file_name = file_dir.split('-')

        pitch_cwt_filename = f"{speaker}-pitch_cwt-{file_name}"
        np.save(join(output_dir, pitch_cwt_filename), pitch_cwt_dict)