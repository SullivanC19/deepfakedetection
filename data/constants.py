import os
from typing import List

DIR_DATA_INFO = './140k-real-and-fake-faces/'
DIR_DATA = './140k-real-and-fake-faces/real_vs_fake/real-vs-fake/'
URL_DATA = 'https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces'

LABEL_FAKE = 0
LABEL_REAL = 1

IMAGE_SIZE = 128

DATA_TYPES = ['train', 'test', 'val']

def get_data_info_file(data_type: str, data_info_dir: List[str]=DIR_DATA_INFO) -> str:
    assert data_type in DATA_TYPES
    return os.path.join(data_info_dir, f'{data_type}.csv')

