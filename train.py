from argparse import ArgumentParser
import torch.nn as nn
import torch

from data.dataset import FaceImageDataset
from train.trainer import train_model
from constants import SPECS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Device:", device)

print("Getting mean and std...")

def main():
  parser = ArgumentParser()
  parser.add_argument('--job_index', '-i', required=False, type=int)
  args = parser.parse_args()

  job_i = args.job_index

  specs = SPECS
  if job_i is not None:
    specs = [SPECS[job_i]]
  
  for spec in specs:
    run_w_spec(spec)

def run_w_spec(spec):
  model_name, f_model, do_fft, do_aug, mean, std = spec

  print(f"Starting training of model {model_name}...")
  model = f_model()
  model = nn.DataParallel(model).to(device)
  train_data = FaceImageDataset('train', mean=mean, std=std, do_fft=do_fft, do_augment=do_aug)
  val_data = FaceImageDataset('valid', mean=mean, std=std, do_fft=do_fft, do_augment=do_aug)
  
  print(f"Training {model_name}...")
  train_model(model_name, model, train_data, val_data)

if __name__ == '__main__':
  main()
