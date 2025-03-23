import torch 
import os
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.io import decode_image, read_image
from tqdm import tqdm

class CelebaDataset(Dataset):
    def __init__(self, data_path: Path, transform=None, merging=False, cropped=False):
        if cropped:
            self.data_path = os.path.join(data_path, "Cropped_CelebA-HQ-img")
        else:
            self.data_path = os.path.join(data_path, "CelebA-HQ-img")
        self.transform = transform
        self.images_names = os.listdir(self.data_path)
        self.merging = merging

    def __len__(self):
        return len(self.images_names)
    
    def __getitem__(self, idx):

        img_name = self.images_names[idx]
        img_path = os.path.join(self.data_path, img_name)
        image = read_image(img_path).float()/255.0
        if self.transform:
            image = self.transform(image)
        if self.merging:
            target_name = self.images_names[-idx]
            target_path = os.path.join(self.data_path, target_name)
            target = read_image(target_path).float()/255.0
            return image, target
        target = image
        return image, target
        
    def calculate_mean_variance(self):
        mean = 0
        std = 0
        for name in tqdm(self.images_names, desc="Calculating mean and standard deviation"):
            image = read_image(os.path.join(self.data_path, name)).float()/255.0
            mean += image.mean()
            std += image.std()
        l = len(self.images_names)
        mean /= l
        std /= l
        return mean, std