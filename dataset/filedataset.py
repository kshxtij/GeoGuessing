import os

import pandas as pd
from PIL import Image as image
from tqdm import tqdm

from .basedataset import BaseDataset


class FileDataset(BaseDataset):
    def __init__(self, dataset_file, dataset_folder, transform=None):
        self.dataset_folder = dataset_folder
        self.transform = transform
        # self.images, self.coordinates = self.load_dataset(dataset_file)

    def load_dataset(self, dataset_file):
        try:
            dataset_info = pd.read_csv(dataset_file)
        except Exception as e:
            raise IOError(f"Error reading {dataset_file}: {e}")

        images = []
        coordinates = []

        for _, row in tqdm(dataset_info.iterrows(), desc="Loading image paths and coordinates"):
            filename = os.path.join(self.dataset_folder, row[self.image_column_name])
            if os.path.exists(filename):
                images.append(filename)
                latitude = float(row['lat'])
                longitude = float(row['lon'])
                coordinates.append((latitude, longitude))

        self.images, self.coordinates = images, coordinates
    
    def __len__(self):
        return len(self.images)
    
    def set_columns(
        self,
        image_column_name: str,
        latitude_column_name: str,
        longitude_column_name: str,
    ):
        self.image_column_name = image_column_name
        self.latitude_column_name = latitude_column_name
        self.longitude_column_name = longitude_column_name

    def __getitem__(self, idx):
        img_path = self.images[idx]
        gps = self.coordinates[idx]

        img = image.open(os.path.join(self.dataset_folder, img_path)).convert('RGB')
        
        if self.transform:
            img = self.transform(img)

        return img, gps