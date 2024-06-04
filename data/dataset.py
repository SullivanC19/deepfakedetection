import torch
import torch.utils.data as dt
import torchvision
from PIL import Image
import torchvision.datasets.utils
import torchvision.transforms.functional

from tqdm import tqdm

from .load_data import load_data_info
from .constants import IMAGE_SIZE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_mean_and_std(dataset):
    loader = dt.DataLoader(dataset, batch_size=64, shuffle=False)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    num_samples = 0.0

    for (image, _) in tqdm(loader):
        batch_samples = image.size(0)
        color_channels = image.size(1)
        image = image.view(batch_samples, color_channels, -1)
        mean += image.mean(2).sum(0)
        std += image.std(2).sum(0)
        num_samples += batch_samples

    mean /= num_samples
    std /= num_samples

    mean = mean.to(device)
    std = std.to(device)

    return mean, std

class FaceImageDataset(torch.utils.data.Dataset):

    def __init__(self, data_type: str, mean: torch.Tensor=0, std: torch.Tensor=1, do_augment: bool=False, do_fft: bool=False):
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

        augment = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(180),
            torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            torchvision.transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        ])
        fft = lambda x: torch.fft.fft2(x).abs()
        tensorize = torchvision.transforms.Compose([
            torchvision.transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            torchvision.transforms.ToTensor(),
        ])
        normalize = torchvision.transforms.Normalize(mean=self.mean, std=self.std)

        if self.do_augment:
            image = augment(image)

        image = tensorize(image)
                
        if self.do_fft:
            image = fft(image)

        image = normalize(image)

        image = image.to(device)
        label = torch.tensor(self.image_labels[idx], dtype=torch.float).expand(1).to(device)
            
        return (image, label)