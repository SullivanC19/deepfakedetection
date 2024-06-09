import os
from data.constants import DIR_DATA
from data.load_data import download_data, download_wild_data

def main():
  print("Downloading data...")
  if os.path.exists(DIR_DATA):
    print("Data already exists")
    return
  download_data()
  download_wild_data()
  print("Data downloaded")

if __name__ == '__main__':
  main()