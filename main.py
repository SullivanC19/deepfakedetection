import os

import torch.nn as nn
import torch

from data.load_data import download_data
from data.dataset import FaceImageDataset, compute_mean_and_std
from data.constants import DIR_DATA
from train.trainer import train_model

from models.cnn import cnn_model
from models.lin import lin_model
from models.lcn import lcn_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Device:", device)

print("Loading data...")
if not os.path.exists(DIR_DATA):
  download_data()

print("Getting mean and std...")

# mean, std = compute_mean_and_std(FaceImageDataset('train'))
# mean_fft, std_fft = compute_mean_and_std(FaceImageDataset('train', do_fft=True))
# mean_aug, std_aug = compute_mean_and_std(FaceImageDataset('train', do_augment=True))
mean = torch.tensor([0.5212, 0.4263, 0.3810])
std = torch.tensor([0.2444, 0.2190, 0.2164])
mean_fft = torch.tensor([7.7721, 7.2894, 7.1348])
std_fft = torch.tensor([73.8876, 61.3502, 56.0426])
mean_aug = torch.tensor([0.4852, 0.3929, 0.3501])
std_aug = torch.tensor([0.2643, 0.2313, 0.2203])

print("Mean and std computed")
print(f"\tMean: {mean}")
print(f"\tStd: {std}")
print(f"\tMean FFT: {mean_fft}")
print(f"\tStd FFT: {std_fft}")
print(f"\tMean Aug: {mean_aug}")
print(f"\tStd Aug: {std_aug}")

MODELS = [
  ('lin', lin_model, False, False, mean, std),
  ('cnn', cnn_model, False, False, mean, std),
  ('lcn', lcn_model, False, False, mean, std),
  ('lin-fft', lin_model, True, False, mean_fft, std_fft),
  ('cnn-fft', cnn_model, True, False, mean_fft, std_fft),
  ('lin-aug', lin_model, False, True, mean_aug, std_aug),
  ('cnn-aug', cnn_model, False, True, mean_aug, std_aug),
  ('lcn-aug', lcn_model, False, True, mean_aug, std_aug),
]

def main():
  print("Starting training...")
  for spec in MODELS:
    model_name, f_model, do_fft, do_aug, mean, std = spec
    model = f_model()
    model = nn.DataParallel(model).to(device)
    train_data = FaceImageDataset('train', mean=mean, std=std, do_fft=do_fft, do_augment=do_aug)
    val_data = FaceImageDataset('valid', mean=mean, std=std, do_fft=do_fft, do_augment=do_aug)
    
    print(f"Training {model_name}...")
    train_model(model_name, model, train_data, val_data) 

if __name__ == '__main__':
  main()
