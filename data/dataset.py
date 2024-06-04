import torch
import torchvision
from PIL import Image
import torchvision.datasets.utils
import torchvision.transforms.functional
from typing import Tuple

from load_data import load_data_info
from constants import IMAGE_SIZE

def compute_mean_and_std(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    mean = 0.0
    std = 0.0
    nb_samples = 0.0

    for data in loader:
        data = data[0]
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    return mean, std

class FaceImageDataset(torch.utils.data.Dataset):

    def __init__(self, data_type: str, mean: torch.Tensor, std: torch.Tensor, do_augment: bool=False, do_fft: bool=False, mean=None, std=None):
        self.data_type = data_type
        self.mean = mean
        self.std = std
        self.do_augment = do_augment
        self.do_fft = do_fft

        image_info = load_data_info(data_type)
        self.image_files = image_info['image'].values
        self.image_labels = image_info['label'].values

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx])

        # TODO deal with complex numbers
        # TODO deal with colors

        preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize((128, 128)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=self.mean, std=self.std)
        ])

        augment = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(180),
            torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            torchvision.transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        ])

        if self.do_augment:
            image = augment(image)
        
        image = preprocess(image)

        if self.do_fft:
            image = torch.fft.fft2(image)
            
        return (image, self.image_labels[idx])