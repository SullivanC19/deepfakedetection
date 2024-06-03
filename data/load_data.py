import pandas as pd
import os
import opendatasets as od

from .constants import DIR_DATA_TRAIN, FAKE_DIR, REAL_DIR


def load_train_data():
  real_images = os.listdir(os.path.join(DIR_DATA_TRAIN, REAL_DIR))
  fake_images = os.listdir(os.path.join(DIR_DATA_TRAIN, FAKE_DIR))

  return pd.DataFrame({
    'image': real_images + fake_images,
    'label': ['real'] * len(real_images) + ['fake'] * len(fake_images)
  })

def download_data():
  dataset = "https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces"
  od.download(dataset)

if __name__ == '__main__':
  download_data()