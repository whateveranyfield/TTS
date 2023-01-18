import tgt
import json
from glob import glob
from os.path import join

if __name__ == '__main__':
    dict_path = '/mnt/NAS/AI_dev_team/DB/TTS/Korean/preprocessed/TextGrid/'
    textgrid_list = glob(join(dict_path, '**', '*.TextGrid'), recursive=True)
    symbol_output_dir = '/mnt/NAS/AI_dev_team/DB/TTS/Korean/preprocessed/symbols.json'

    symbol_dict = dict()
    index_num = 0

    for textgrid_dir in textgrid_list:
        text_grid = tgt.io.read_textgrid(textgrid_dir, include_empty_intervals=True)
        text_grid = text_grid.get_tier_by_name("phones")

        print(textgrid_dir)

        for info in text_grid._objects:
            phone = info.text

            if not phone in symbol_dict.keys():
                symbol_dict[phone] = index_num
                index_num += 1
            else:
                pass

    with open(symbol_output_dir, 'w') as symbol_file:
        json.dump(symbol_dict, symbol_file, ensure_ascii=False, indent=4)
