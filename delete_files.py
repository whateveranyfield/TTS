import os

from os.path import join
from glob import glob

if __name__ == '__main__':
    data_path = '/mnt/NAS/AI_dev_team/DB/TTS/Korean/original/spk031_F/'
    files = glob(join(data_path, '*.*'), recursive=True)

    for file_path in files:
        if len(file_path.split('/')[-1]) == 17:
            os.remove(file_path)
