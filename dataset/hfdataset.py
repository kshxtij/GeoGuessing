from collections.abc import Callable

from datasets import load_dataset

from .basedataset import BaseDataset


class HFImageDataset(BaseDataset):
    def __init__(
        self, huggingface_dataset_name: str, split: str, transform: Callable = None
    ):
        super().__init__()
        self.dataset = load_dataset(huggingface_dataset_name, split=split, trust_remote_code=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

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
        if (
            not hasattr(self, "image_column_name")
            or not hasattr(self, "latitude_column_name")
            or not hasattr(self, "longitude_column_name")
        ):
            raise ValueError(
                "Columns not set. Use set_columns() to set the column names."
            )

        data = self.dataset[idx]
        image = data[self.image_column_name]
        latitude = data[self.latitude_column_name]
        longitude = data[self.longitude_column_name]

        if self.transform:
            image = self.transform(image)

        return image, (latitude, longitude)
