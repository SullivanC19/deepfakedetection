import pandas as pd
import os
import opendatasets as od

from .constants import (
  URL_DATA,
  URL_DATA_TPDNE,
  URL_DATA_CELEBA,
  get_data_info_file,
  DIR_DATA,
  DIR_DATA_TPDNE,
  DIR_DATA_CELEBA,
  LABEL_FAKE,
  LABEL_REAL,
  TEST_SIZE,
)

def load_data_info(data_type, source='local'):
  if source == 'local':
    data_info = pd.read_csv(get_data_info_file(data_type))
    data_info['image'] = data_info['path'].apply(lambda path: os.path.join(DIR_DATA, path))
    data_info['label'] = data_info['label'].astype(int)
    if data_type == 'test':
      data_info = data_info.sample(TEST_SIZE)
    return data_info[['image', 'label']]
  
  if source == 'tpdne':
    data_info = pd.DataFrame(columns=['image', 'label'])
    data_info['image'] = pd.Series([f"{DIR_DATA_TPDNE}/{f_img}" for f_img in os.listdir(DIR_DATA_TPDNE)[:TEST_SIZE]])
    data_info['label'] = LABEL_FAKE
    return data_info
  
  if source == 'celeba':
    data_info = pd.DataFrame(columns=['image', 'label'])
    data_info['image'] = pd.Series([f"{DIR_DATA_CELEBA}/{f_img}" for f_img in os.listdir(DIR_DATA_CELEBA)[:TEST_SIZE]])
    data_info['label'] = LABEL_REAL
    return data_info

def download_data():
  od.download(URL_DATA, force=True)

def download_wild_data():
  od.download(URL_DATA_TPDNE, force=True)
  od.download(URL_DATA_CELEBA, force=True)
