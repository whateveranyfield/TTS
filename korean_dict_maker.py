import os
import re
import yaml
import argparse

from g2pk import G2p
from os.path import join
from tqdm import tqdm
from jamo import h2j, hangul_to_jamo, j2hcj
from glob import glob


def main(config):
    g2p = G2p()
    lexicon_path = config["path"]["lexicon_path"]
    data_path = config["path"]["raw_path"]
    text_list = glob(join(data_path, '**', '*.lab'))

    lexicon_dict = dict()

    for text_file in tqdm(text_list):
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.readline().strip("\n")

        text = re.sub(r"([~/*\",:·`“”’‘≪〈〉<>（）「」《》{}|=\';.\(\)\[\]\-\?\!\s+])", " ", text)
        text = text.replace('π', '파이').replace('花', '화').replace('華', '').replace('萬古風霜', '만고풍상').replace('處士', '처사').replace('非', '비').replace('%', '퍼센트').replace('ㅘ', '와')

        print(text)
        words = text.split(' ')

        for _ in range(words.count('')):
            words.remove('')

        for word in words:
            filtered_word = word.strip('\"')
            filtered_word = filtered_word.strip('\'')
            if filtered_word == '':
                continue
            pho_line = ''
            pho_list = g2p(filtered_word)
            pho_jamo_list = h2j(pho_list)
            for i, pho in enumerate(pho_jamo_list):
                if i == (len(pho_jamo_list) - 1):
                    pho_line += pho
                else:
                    pho_line += pho + ' '

            lexicon_dict[filtered_word] = pho_line

    words = list(lexicon_dict.keys())
    words.sort()
    print(words)

    with open(lexicon_path, 'w', encoding='utf-8') as output_f:
        for word in tqdm(words):
            output_f.write(f"{word}\t{lexicon_dict[word]}\n")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/korean/preprocess.yaml", type=str,
                        help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    main(config)
