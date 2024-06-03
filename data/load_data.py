import pandas as pd
import os
import opendatasets as od

from .constants import URL_DATA, get_data_info_file, DIR_DATA

def load_data_info(data_type):
  data_info = pd.read_csv(get_data_info_file(data_type))
  data_info['image'] = data_info['path'].apply(lambda path: os.path.join(DIR_DATA, path))
  data_info['label'] = data_info['label'].astype(int)
  return data_info[['image', 'label']]

def download_data():
  od.download(URL_DATA, force=True)

if __name__ == '__main__':
  download_data()