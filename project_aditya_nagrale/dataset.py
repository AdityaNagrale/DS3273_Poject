import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
from .config import resize_x, resize_y, batch_size

class GalaxyDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx, 0]
        # Handle cases where img_id might be a float or int
        img_path = os.path.join(self.img_dir, f"{int(img_id)}.jpg")
        image = Image.open(img_path).convert('RGB')
        label = self.df.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
        return image, label

def get_dataloader(csv_file, img_dir, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((resize_x, resize_y)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = GalaxyDataset(csv_file, img_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
