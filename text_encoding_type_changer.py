import os.path
from glob import glob
from os.path import join

if __name__ == '__main__':
    data_path = '/mnt/NAS/AI_dev_team/DB/TTS/Voucher/original/korean/'
    text_list = glob(join(data_path, '**', '*.txt'), recursive=True)
    output_dir = '/mnt/NAS/AI_dev_team/DB/TTS/Voucher/edited_txt/'

    for text_dir in text_list:
        file_name = text_dir.split('/')[-1]
        speaker = text_dir.split('/')[-3]
        language = text_dir.split('/')[-4]

        os.makedirs(join(output_dir, f'{language}_{speaker}'), exist_ok=True)

        try:
            with open(text_dir, 'r', encoding='euc-kr') as text_file:
                text = text_file.readline()
            with open(join(output_dir, f'{language}_{speaker}', f'{language}_{file_name}'), 'w', encoding='utf-8') as output_file:
                output_file.write(text)
        except:
            pass
        try:
            if not os.path.exists(join(output_dir, f'{language}_{speaker}', f'{language}_{file_name}')):
                with open(text_dir, 'r', encoding='cp949') as text_file:
                    text = text_file.readline()
                with open(join(output_dir, f'{language}_{speaker}', f'{language}_{file_name}'), 'w', encoding='utf-8') as output_file:
                    output_file.write(text)
        except:
            pass
        try:
            if not os.path.exists(join(output_dir, f'{language}_{speaker}', f'{language}_{file_name}')):
                with open(text_dir, 'r', encoding='utf=8') as text_file:
                    text = text_file.readline()
                with open(join(output_dir, f'{language}_{speaker}', f'{language}_{file_name}'), 'w', encoding='utf-8') as output_file:
                    output_file.write(text)
        except:
            pass