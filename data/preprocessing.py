import os
import csv
from torch.utils.data import Dataset
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, data_folder, data_set, transform=None):
        self.data_folder = data_folder
        self.transform = transform

        # read csv
        with open(os.path.join(data_folder, data_set), mode='r') as infile:
            next(infile) # skip first row
            reader = csv.reader(infile)
            # key = path to image, value = real (1) or fake (0)
            mydict = {rows[5]:rows[3] for rows in reader}

        # Get lists of image file names and labels
        items = mydict.items()
        self.image_files = [k for (k, v) in items]
        self.image_labels = [int(v) for (k, v) in items]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_folder, 'real_vs_fake/real-vs-fake/' + self.image_files[idx])
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)
        return (image, self.image_labels[idx])