import torch


def save_weights(model, path):
    torch.save(model.image_encoder.mlp.state_dict(), f"{path}/image_encoder_mlp_weights.pth")
    torch.save(model.location_encoder.state_dict(), f"{path}/location_encoder_weights.pth")
    torch.save(model.logit_scale, f"{path}/logit_scale_weights.pth")