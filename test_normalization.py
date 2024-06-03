import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# Custom dataset for loading images from disk
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def compute_mean_and_std(dataset):
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4)
    mean = 0.0
    std = 0.0
    nb_samples = 0.0

    for data in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    return mean, std

def main():
    # Directory containing images
    image_dir = 'trial_directory'
    
    # List all image files in the directory
    image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.lower().endswith(('png', 'jpg', 'jpeg'))]

    if not image_paths:
        print("No images found in the directory.")
        return

    # Create a dataset instance without transformations for mean and std computation
    dataset = CustomImageDataset(image_paths, transform=transforms.ToTensor())

    # Compute mean and std
    mean, std = compute_mean_and_std(dataset)

    # Use the computed mean and std for normalization
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize the image to 256x256
        transforms.ToTensor(),  # Convert PIL image to tensor
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())  # Normalize image pixels
    ])

    # Create dataset with the new transform
    normalized_dataset = CustomImageDataset(image_paths, transform=transform)

    # Create DataLoaders to load the data in batches
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    normalized_loader = DataLoader(normalized_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Iterate over the DataLoader and display the first 3 normalized images alongside the original images
    for idx, (data, normalized_data) in enumerate(zip(loader, normalized_loader)):
        if idx >= 3:
            break
        
        image = data.numpy()[0]
        image = np.transpose(image, (1, 2, 0))  # Change from (C, H, W) to (H, W, C)

        normalized_image = normalized_data.numpy()[0]
        normalized_image = np.transpose(normalized_image, (1, 2, 0))  # Change from (C, H, W) to (H, W, C)

        # Plot original and normalized images side by side
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
        axs[0].imshow(image)
        axs[0].set_title(f"Original Image {idx + 1}")
        axs[0].axis('off')
        
        axs[1].imshow(normalized_image)
        axs[1].set_title(f"Normalized Image {idx + 1}")
        axs[1].axis('off')
        
        plt.show()

if __name__ == '__main__':
    main()
