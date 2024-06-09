import torch
from models.cnn import cnn_model
from models.lin import lin_model
from models.lcn import lcn_model

import os
SAVED_MODEL_DIR = 'saved_models'

def get_most_recent_model(model):
  return f'{SAVED_MODEL_DIR}/{model}/{sorted(os.listdir(f"{SAVED_MODEL_DIR}/{model}"))[-1]}'

LIN_MODELS = [
  ('lin', get_most_recent_model('lin')),
  ('lin-aug', get_most_recent_model('lin-aug')),
  ('lin-fft', get_most_recent_model('lin-fft')),
]

CNN_MODELS = [
  ('cnn', get_most_recent_model('cnn')),
  ('cnn-aug', get_most_recent_model('cnn-aug')),
  ('cnn-fft', get_most_recent_model('cnn-fft')),
]

DATASET_SPECS = [
  ('normal', 'train', 0, 1, False, False),
  ('aug', 'train', 0, 1, True, False),
  ('fft', 'train', 0, 1, False, True),
]

# print("Getting mean and std...")
# mean, std = compute_mean_and_std(FaceImageDataset('train'))
# mean_fft, std_fft = compute_mean_and_std(FaceImageDataset('train', do_fft=True))
# mean_aug, std_aug = compute_mean_and_std(FaceImageDataset('train', do_augment=True))

# Results from the above commented out code
mean = torch.tensor([0.5209, 0.4259, 0.3807])
std = torch.tensor([0.2399, 0.2145, 0.2120])
mean_fft = torch.tensor([5.4393, 4.9986, 4.8708])
std_fft = torch.tensor([36.5962, 30.3230, 27.6568])
mean_aug = torch.tensor([0.4853, 0.3930, 0.3504])
std_aug = torch.tensor([0.2594, 0.2264, 0.2155])

print("Mean and std computed")
print(f"\tMean: {mean}")
print(f"\tStd: {std}")
print(f"\tMean FFT: {mean_fft}")
print(f"\tStd FFT: {std_fft}")
print(f"\tMean Aug: {mean_aug}")
print(f"\tStd Aug: {std_aug}")

SPECS = [
  ('lin', lin_model, False, False, mean, std),
  ('cnn', cnn_model, False, False, mean, std),
  ('lcn', lcn_model, False, False, mean, std),
  ('lin-fft', lin_model, True, False, mean_fft, std_fft),
  ('cnn-fft', cnn_model, True, False, mean_fft, std_fft),
  ('lin-aug', lin_model, False, True, mean_aug, std_aug),
  ('cnn-aug', cnn_model, False, True, mean_aug, std_aug),
  ('lcn-aug', lcn_model, False, True, mean_aug, std_aug),
]
