import torch
import torch.nn as nn
from .base_encoder import BaseEncoder
from transformers import AutoModel, AutoProcessor

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.*")


class DINOImageEncoder(BaseEncoder):
    def __init__(self):
        super(DINOImageEncoder, self).__init__()
        self.DINO = AutoModel.from_pretrained('facebook/dinov2-base')
        self.image_processor = AutoProcessor.from_pretrained(
            "facebook/dinov2-base"
        )
        self.mlp = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 512))

        # Freeze CLIP
        for param in self.DINO.parameters():
            param.requires_grad = False

    def load_weights(self, mlp_weights_path: str):
        self.mlp.load_state_dict(torch.load(mlp_weights_path, weights_only=True))

    def to(self, device):
        self.DINO.to(device)
        self.mlp.to(device)

    def preprocess(self, image):
        x = self.image_processor(images=image, return_tensors="pt")["pixel_values"]
        return x

    def forward(self, x):
        x = self.DINO(pixel_values=x).last_hidden_state.mean(dim=1)
        x = self.mlp(x)
        return x
