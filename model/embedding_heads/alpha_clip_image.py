import warnings

import alpha_clip
import torch
import torch.nn as nn
from transformers import AutoProcessor, CLIPModel

from .base_encoder import BaseEncoder

warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.*")


class CLIPImageEncoder(BaseEncoder):
    def __init__(self):
        super(CLIPImageEncoder, self).__init__()
        self.CLIP = alpha_clip.load("CLIP-L/14", alpha_vision_ckpt_pth='./weights/alpha_clip.pth')
        self.image_processor = AutoProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.mlp = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 512))

        # Freeze CLIP
        for param in self.CLIP.parameters():
            param.requires_grad = False

    def load_weights(self, mlp_weights_path: str):
        self.mlp.load_state_dict(torch.load(mlp_weights_path, weights_only=True))

    def to(self, device: str):
        self.CLIP.to(device)
        self.mlp.to(device)

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        x = self.image_processor(images=image, return_tensors="pt")["pixel_values"]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.CLIP.encode_image(pixel_values=x)
        x = self.mlp(x)
        return x
