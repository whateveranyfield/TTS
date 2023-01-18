import re

from g2pk import G2p
from tqdm import tqdm
from glob import glob
from os.path import join

if __name__ == '__main__':
    data_path = '/mnt/NAS/AI_dev_team/DB/TTS/Korean/data/'
    lab_list = glob(join(data_path, '**', '*.lab'), recursive=True)

    g2p = G2p()

    for lab_dir in tqdm(lab_list):
        with open(lab_dir, 'r') as lab_file:
            text = lab_file.readline()
            # text = re.sub(r"([~/*\",:·`“”’‘≪〈〉<>（）「」《》{}|=\';.\(\)\[\]\-\?\!\s+])", " ", text)
            # text = text.replace('π', '파이').replace('花', '화').replace('華', '').replace('萬古風霜', '만고풍상').replace(
            #     '處士', '처사').replace('非', '비').replace('%', '퍼센트').replace('ㅘ', '와')
            text = g2p(text)

        with open(lab_dir, 'w') as lab_output_file:
            lab_output_file.write(text)