from glob import glob
from os.path import join

if __name__ == '__main__':
    data_path = '/mnt/NAS/AI_dev_team/DB/TTS/Korean/original/'
    txt_list = glob(join(data_path, '**', '*.txt'), recursive=True)

    for txt_dir in txt_list:
        with open(txt_dir, 'r', encoding='utf-8') as txt_file:
            print(f'{txt_dir.split("/")[-1]}: {txt_file.readline()}'.strip("\n"))