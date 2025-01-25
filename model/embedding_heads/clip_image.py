import torch
import torch.nn as nn
from .base_encoder import BaseEncoder
from transformers import CLIPModel, AutoProcessor

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.*")


class CLIPImageEncoder(BaseEncoder):
    def __init__(self):
        super(CLIPImageEncoder, self).__init__()
        self.CLIP = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.image_processor = AutoProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.mlp = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 512))

        # Freeze CLIP
        for param in self.CLIP.parameters():
            param.requires_grad = False

    def load_weights(self, mlp_weights_path: str):
        self.mlp.load_state_dict(torch.load(mlp_weights_path, weights_only=True))

    def to(self, device):
        self.CLIP.to(device)
        self.mlp.to(device)

    def preprocess(self, image):
        x = self.image_processor(images=image, return_tensors="pt")["pixel_values"]
        return x

    def forward(self, x):
        x = self.CLIP.get_image_features(pixel_values=x)
        x = self.mlp(x)
        return x
