import warnings

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizer

from .base_encoder import BaseEncoder

warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.*")


class CLIPTextEncoder(BaseEncoder):
    def __init__(self):
        super(CLIPTextEncoder, self).__init__()
        self.CLIP = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.text_tokenizer = CLIPTokenizer.from_pretrained(
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

    def preprocess(self, text: str) -> torch.Tensor:
        x = self.text_tokenizer(text, return_tensors="pt")["input_ids"]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.CLIP.get_text_features(input_ids=x)
        x = self.mlp(x)
        return x
