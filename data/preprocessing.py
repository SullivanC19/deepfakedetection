import os
import csv
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


class CustomDataset(Dataset):
    def __init__(self, data_folder, data_set, transform=None):
        self.data_folder = data_folder
        self.transform = transform

        # read csv
        with open(os.path.join(data_folder, data_set), mode='r') as infile:
            reader = csv.reader(infile)
            # key = path to image, value = real (1) or fake (0)
            mydict = {rows[5]:rows[3] for rows in reader}

        # Get lists of image file names and labels
        items = mydict.items()
        self.image_files = [k for (k, v) in items]
        self.image_labels = [v for (k, v) in items]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_folder, 'real_vs_fake/real-vs-fake/' + self.image_files[idx])
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)
        return (image, self.image_labels[idx])

# Define the path to your data csv 
# (assuming running this file using 'python data/preprocessing.py' from deepfakesonly)
data_folder = './140k-real-and-fake-faces'
data_set = 'train.csv'

# Define the transformations to apply to the images
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the image to 256x256
    transforms.ToTensor(),  # Convert PIL image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize image pixels
])

# Create an instance of your custom dataset
custom_dataset = CustomDataset(data_folder, data_set, transform=transform)



# Create a DataLoader to load the data in batches
batch_size = 128
data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

# Iterate over the data loader to access batches of preprocessed images
for batch_idx, batch in enumerate(data_loader):
    # Process each batch as needed
    # batch is a tensor of shape (batch_size, channels, height, width)
    # For example, you can perform some operations on the batch like printing its shape
    print(f"Batch {batch_idx + 1} shape:", batch[0].shape)
    print(batch[1])

    # Convert the batch tensor to a NumPy array
    batch_np = batch[0].numpy()

    # Iterate over images in the batch
    for i in range(len(batch_np)):
        image_np = np.transpose(batch_np[i], (1, 2, 0))  # Change from (C, H, W) to (H, W, C)
        # Display the image using matplotlib
        plt.imshow(image_np)
        plt.title(f"Batch {batch_idx + 1}, Image {i + 1}")
        plt.axis('off')  # Turn off axis labels
        plt.show()
        print('label: ', batch[1][i])

    # Optionally, break the loop after processing a few batches
    if batch_idx == 1:
        break
