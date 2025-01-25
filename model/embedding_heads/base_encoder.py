from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseEncoder(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def load_weights(self, mlp_weights_path: str):
        raise NotImplementedError

    @abstractmethod
    def to(self, device):
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, image):
        raise NotImplementedError

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError
