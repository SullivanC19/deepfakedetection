import os
from typing import List

DIR_DATA_INFO = './140k-real-and-fake-faces/'

DIR_DATA = './140k-real-and-fake-faces/real_vs_fake/real-vs-fake/'
DIR_DATA_TPDNE = './tpdne-60k-128x128/'
DIR_DATA_CELEBA = './celeba-dataset/img_align_celeba/img_align_celeba/'

URL_DATA = 'https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces'
URL_DATA_TPDNE = 'https://www.kaggle.com/datasets/potatohd404/tpdne-60k-128x128'
URL_DATA_CELEBA = 'https://www.kaggle.com/jessicali9530/celeba-dataset'

LABEL_FAKE = 0
LABEL_REAL = 1

IMAGE_SIZE = 64
TEST_SIZE = 1000

DATA_TYPES = ['train', 'test', 'valid']

def get_data_info_file(data_type: str, data_info_dir: List[str]=DIR_DATA_INFO) -> str:
    assert data_type in DATA_TYPES
    return os.path.join(data_info_dir, f'{data_type}.csv')

