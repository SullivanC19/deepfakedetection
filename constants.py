import torch
from models.cnn import cnn_model
from models.lin import lin_model
from models.lcn import lcn_model

# TODO get most recent models

LIN_MODELS = [
  ('lin', 'saved_models/lin/2024-06-04_23-12-26.pt'),
  ('lin-aug', 'saved_models/lin-aug/2024-06-04_23-12-37.pt'),
  ('lin-fft', 'saved_models/lin-fft/2024-06-04_23-11-52.pt'),
]

CNN_MODELS = [
  ('cnn', 'saved_models/cnn/2024-06-04_23-12-26.pt'),
  ('cnn-aug', 'saved_models/cnn-aug/2024-06-04_23-12-37.pt'),
  ('cnn-fft', 'saved_models/cnn-fft/2024-06-04_23-12-37.pt'),
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
