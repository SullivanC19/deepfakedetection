import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


class CustomDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform

        # Get a list of image file names in the data folder
        self.image_files = [f for f in os.listdir(data_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_folder, self.image_files[idx])
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image

# Define the path to your data folder
data_folder = "/Users/harshalagrawal/deepfakesonly/trial"

# Define the transformations to apply to the images
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the image to 256x256
    transforms.ToTensor(),  # Convert PIL image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize image pixels
])

# Create an instance of your custom dataset
custom_dataset = CustomDataset(data_folder, transform=transform)



# Create a DataLoader to load the data in batches
batch_size = 32
data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

# Iterate over the data loader to access batches of preprocessed images
for batch_idx, batch in enumerate(data_loader):
    # Process each batch as needed
    # batch is a tensor of shape (batch_size, channels, height, width)
    # For example, you can perform some operations on the batch like printing its shape
    print(f"Batch {batch_idx + 1} shape:", batch.shape)

    # Convert the batch tensor to a NumPy array
    batch_np = batch.numpy()

    # Iterate over images in the batch
    for i in range(len(batch_np)):
        image_np = np.transpose(batch_np[i], (1, 2, 0))  # Change from (C, H, W) to (H, W, C)
        # Display the image using matplotlib
        plt.imshow(image_np)
        plt.title(f"Batch {batch_idx + 1}, Image {i + 1}")
        plt.axis('off')  # Turn off axis labels
        plt.show()

    # Optionally, break the loop after processing a few batches
    if batch_idx == 4:
        break
