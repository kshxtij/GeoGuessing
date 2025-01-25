from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):

    def __init__(self):
        super().__init__()

    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def set_columns(
        self,
        image_column_name: str,
        latitude_column_name: str,
        longitude_column_name: str,
    ):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError
