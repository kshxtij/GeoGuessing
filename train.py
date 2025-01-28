import os

import torch
import wandb
from torch.utils.data import DataLoader

from dataset.filedataset import FileDataset
from dataset.hfdataset import HFImageDataset
from dataset.transforms.geoclip import geoclip_train_transform
from evaluation.geoclip_eval import train
from model.embedding_heads.dino_image import DINOImageEncoder
from model.embedding_heads.rff_location import RFFLocationEncoder
from model.GeoClip import GeoCLIP
from utils.geoclip import save_weights

NAME = "geoclip_dino_rff_training"

dataset = FileDataset("/Users/kshitij/Documents/University/Year4/MLP/RealProject/data/10K/metadata.csv", "/Users/kshitij/Documents/University/Year4/MLP/RealProject/data/10K", transform=geoclip_train_transform())
dataset.set_columns("file_name", "lat", "lon")
dataset.load_dataset("/Users/kshitij/Documents/University/Year4/MLP/RealProject/data/10K/metadata.csv")
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

wandb.init(
    # set the wandb project where this run will be logged
    project="FAFO",
    name="Training GeoCLIP with DINO and RFF Location Encoder",
    # track hyperparameters and run metadata
    config={
        "learning_rate": 3e-5,
        "architecture": "GeoCLIP",
        "dataset": "Training on GeoGuessing/GeoTaggedImages",
        "epochs": 10,
    }
)

location_encoder = RFFLocationEncoder()
image_encoder = DINOImageEncoder()

model = GeoCLIP(image_encoder, location_encoder, '/Users/kshitij/Documents/University/Year4/MLP/RealProject/weights/pretrained_geoclip/coordinates_100K.csv', queue_size=4096)
model.to('mps')
optim = torch.optim.Adam(model.parameters(), lr=3e-5, weight_decay=1e-6)

for i in range(10):
    loss = train(dataloader, model, batch_size=256, device='mps', optimizer=optim, epoch=i)
    wandb.log({'loss': loss})
    if i % 2 == 0:
        os.makedirs(f'./checkpoints/{NAME}/{i}', exist_ok=True)
        save_weights(model, f'./checkpoints/{NAME}/{i}')
        wandb.save(f'./checkpoints/{NAME}/{i}/image_encoder_mlp_weights.pth')
        wandb.save(f'./checkpoints/{NAME}/{i}/location_encoder_weights.pth')
        wandb.save(f'./checkpoints/{NAME}/{i}/logit_scale_weights.pth')

wandb.finish()