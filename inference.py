import argparse
import os
import kaldiio
import yaml
from glob import glob
import torch
import json
import numpy as np
import soundfile as sf
import librosa


from os.path import join
from scipy.io.wavfile import write, read
from bin.tts_inference import Text2Speech
from audio.trim_silence import trim_silence
from parallel_wavegan.utils import load_model
from feats_extract.log_mel_fbank import LogMelFbank


def trim_silence(wav):
    """Run silence trimming and generate segments."""

    fs = 24000
    win_length = 1200
    shift_length = 300
    threshold = 35
    # normalize = 16
    min_silence = 0.01

    array = wav.astype(np.float32)
    # if normalize is not None and normalize != 1:
    #     array = array / (1 << (normalize - 1))
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

    return array[int(start * fs):int(end * fs)]


def main(configs):
    fs = 24000
    vocoder_step = 2500000
    # acoustic_step = 62
    acoustic_step = 326
    # acoustic_step = 112

    # acoustic_dir = f"/home/dmlab/Matt/output/new_toolkit/{acoustic_step}epoch.pth"
    # acoustic_dir = f"/mnt/NAS/AI_dev_team/users/matt/TTS/multi_CWT/{acoustic_step}epoch.pth"
    # acoustic_dir = f"/mnt/NAS/AI_dev_team/users/matt/TTS/model_output/CWT_extracted_by_unnorm_pitch_p_out/{acoustic_step}epoch.pth"

    model_name = 'korean'
    version_name = 'style_equalizer_with_attention'
    acoustic_dir = f"/mnt/NAS/AI_dev_team/users/matt/TTS/model_output/{model_name}/{version_name}/{acoustic_step}epoch.pth"
    # acoustic_dir = f"/mnt/NAS/AI_dev_team/users/matt/TTS/model_output/Heather/{acoustic_step}epoch.pth"
    emotion = 'Disgust'
    # speaker = 'spk001_F'
    # wav_reference_dir = f"/home/matt/Downloads/ref_mel/disgust_141-168_0161.wav"
    # wav_reference_dir = "/mnt/NAS/AI_dev_team/users/matt/TTS/DB/matt/data/matt_2021_06_01_0002.wav"
    mel_reference_dir = f"/media/matt/새 볼륨/TTS/Heather/preprocessed/mel/{emotion}-mel-{emotion}_000610.npy"
    # mel_reference_dir = f"/mnt/NAS/AI_dev_team/DB/TTS/Korean/preprocessed/mel/{speaker}-mel-{speaker}_0001.npy"
    # pitch_reference_dir = f"/media/matt/새 볼륨/TTS/Heather/preprocessed/pitch_per_frame/{emotion}-pitch_per_frame-{emotion}_000610.npy"
    # voco_dir = f"D:/TTS/Vocoder/Complex/checkpoint-{vocoder_step}steps.pkl"
    # voco_dir = f"/media/matt/새 볼륨/vocoder/Heather/checkpoint-{vocoder_step}steps.pkl"
    voco_dir = f"/mnt/NAS/AI_dev_team/users/matt/Vocoder/models/Korean/HifiGAN/checkpoint-{vocoder_step}steps.pkl"
    output_dir = f"/home/matt/model_results/{model_name}/{version_name}"
    # wav_output_dir = join(output_dir, 'wav', f'{model_name}_{version_name}_{acoustic_step}_{vocoder_step}')
    # feature_output_dir = join(output_dir, 'features', f'{model_name}_{version_name}_{acoustic_step}_{vocoder_step}')
    wav_output_dir = join(output_dir, 'wav', f'{model_name}_{version_name}_{emotion}_{acoustic_step}')
    feature_output_dir = join(output_dir, 'features', f'{model_name}_{version_name}_{emotion}_{acoustic_step}')
    # wav_output_dir = join(output_dir, 'wav', f'{model_name}_{version_name}_{speaker}_{acoustic_step}')
    # feature_output_dir = join(output_dir, 'features', f'{model_name}_{version_name}_{speaker}_{acoustic_step}')

    stats_dir = '/mnt/NAS/AI_dev_team/DB/TTS/Korean/preprocessed/stats.json'

    text2speech = Text2Speech(
        train_configs=configs,
        model_dir=acoustic_dir,
        device='cpu',
        threshold=0.5,
        minlenratio=0.0,
        maxlenratio=10.0,
        backward_window=1,
        forward_window=3,
    )

    preprocess_config, _ = configs

    LogMelExtractor = LogMelFbank(
        fs=preprocess_config["preprocessing"]["audio"]["sampling_rate"],
        n_fft=preprocess_config["preprocessing"]["stft"]["filter_length"],
        win_length=preprocess_config["preprocessing"]["stft"]["win_length"],
        hop_length=preprocess_config["preprocessing"]["stft"]["hop_length"],
        fmin=preprocess_config["preprocessing"]["mel"]["mel_fmin"],
        fmax=preprocess_config["preprocessing"]["mel"]["mel_fmax"],
    )

    vocoder = load_model(voco_dir).to('cpu').eval()

    # with open(stats_dir, 'r') as stats_file:
    #     stats = json.load(stats_file)
    #
    # mean, std = stats['mel']
    #
    # wav, _ = sf.read(wav_reference_dir)
    # wav = trim_silence(wav)
    #
    # mel, _ = LogMelExtractor(
    #     torch.tensor(wav, dtype=torch.float64).unsqueeze(0)
    # )
    #
    # mel = mel.squeeze().numpy()
    # mel = (mel - mean) / std
    # mel = torch.from_numpy(mel).float()

    mel = torch.from_numpy(np.load(mel_reference_dir)).float()
    # pitch = torch.from_numpy(np.load(pitch_reference_dir)).float()

    input_sentences = [
        "재치 부리다 보면 조금은 거짓말을 하게 된다, 나의 가로등지기 이야기도 그렇게 정직했던 것은 아니다.",
        "지구를 잘 알지 못하는 사람들에게 자칫, 지구에 대한 잘못된 생각을 불러일으킬 수 있으니까."
    ]

    with open('/mnt/NAS/AI_dev_team/DB/TTS/Korean/preprocessed/speakers.json', 'r') as speaker_file:
        speakers = json.load(speaker_file)

    for spk_id in range(75):

        for key in speakers.keys():
            if spk_id == speakers[key]:
                spk = key
                break
            else:
                pass

        # os.makedirs(f"{wav_output_dir}/{spk}", exist_ok=True)
        os.makedirs(f"{wav_output_dir}/{spk}", exist_ok=True)
        os.makedirs(f"{feature_output_dir}/{spk}", exist_ok=True)
        # os.makedirs(f"{wav_output_dir}/{mel_reference_dir.split('/')[-1].split('-')[-1][:-4]}", exist_ok=True)
        # os.makedirs(f"{feature_output_dir}/{mel_reference_dir.split('/')[-1].split('-')[-1][:-4]}", exist_ok=True)

        utt_id = 0
        for input_sentence in input_sentences:
            # synthesis
            with torch.no_grad():
                # Soft
                # output_dict = text2speech(input_sentence, pitch=pitch)
                # Complex_embedding
                # output_dict = text2speech(input_sentence, pitch=pitch, speech=mel)
                # spk_embedding & emo_embedding
                # output_dict = text2speech(input_sentence, sids=torch.from_numpy(np.array(spk_id)).long().unsqueeze(0))
                # style_equalizer
                output_dict = text2speech(input_sentence, speech=mel.float().squeeze())
                # GST
                # output_dict = text2speech(input_sentence, speech=mel.float().squeeze())
                # GST_embedding
                # output_dict = text2speech(input_sentence, speech=mel.float().squeeze(), sids=torch.from_numpy(np.array(spk_id)).long().unsqueeze(0))
                # GST manual
                # output_dict = text2speech(input_sentence, speech=speech, gst_mask=mask, spembs=spembs)
                # Meta
                # output_dict = text2speech(input_sentence, speech=speech, pitch=pitch)

                # output_dict = text2speech(input_sentence, spembs=spembs, mean=mean, std=std)
                # output_dict = text2speech(input_sentence, spembs=spembs)
                # output_dict = text2speech(input_sentence, speech=mel)
                wav = vocoder.inference(output_dict["feat_gen"])

                write(f"{wav_output_dir}/{spk}/out{int(utt_id)}.wav",
                      fs, wav.cpu().numpy())
                print("wrote utterence %i" % (int(utt_id)))
                # np.save(
                #     f"{feature_output_dir}/{spk_id}/{int(utt_id)}_pitch.npy",
                #     output_dict['pitch'].squeeze().detach().numpy())
                # np.save(
                #     f"{feature_output_dir}/{spk_id}/{int(utt_id)}_cwt.npy",
                #     output_dict['cwt'].squeeze().detach().numpy())
                # np.save(
                #     f"{feature_output_dir}/{spk_id}/{int(utt_id)}_style.npy",
                #     output_dict['style'].squeeze().detach().numpy())

                # write(f"{wav_output_dir}/{mel_reference_dir.split('/')[-1].split('-')[-1][:-4]}/out{int(utt_id)}.wav", fs, wav.cpu().numpy())
                # print("wrote utterence %i" % (int(utt_id)))
                # np.save(f"{feature_output_dir}/{mel_reference_dir.split('/')[-1].split('-')[-1][:-4]}/{int(utt_id)}_pitch.npy",
                #         output_dict['pitch'].squeeze().detach().numpy())
                # np.save(f"{feature_output_dir}/{mel_reference_dir.split('/')[-1].split('-')[-1][:-4]}/{int(utt_id)}_cwt.npy",
                #         output_dict['cwt'].squeeze().detach().numpy())
                # np.save(f"{feature_output_dir}/{mel_reference_dir.split('/')[-1].split('-')[-1][:-4]}/{int(utt_id)}_style.npy",
                #         output_dict['style'].squeeze().detach().numpy())
            utt_id += 1

        break
        # spk_id+=1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--preprocess_config", type=str, default="./config/korean/preprocess.yaml", help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, default="./config/korean/model.yaml", help="path to model.yaml"
    )
    args = parser.parse_args()

    preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)

    configs = (preprocess_config, model_config)

    main(configs)