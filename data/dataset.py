import os
import csv
import torch
import torchvision
from PIL import Image
import torchvision.transforms.functional

from load_data import load_data_info

class FaceImageDataset(torch.utils.data.Dataset):

    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize((128, 128)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=0, std=1)
    ])

    augment = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(180),
        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        torchvision.transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    ])

    def __init__(self, data_type: str, do_augment: bool=False, do_fft: bool=False):
        self.data_type = data_type
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

        if self.do_augment:
            image = self.augment(image)
        
        image = self.preprocess(image)

        if self.do_fft:
            image = torch.fft.fft2(image)
            
        return (image, self.image_labels[idx])