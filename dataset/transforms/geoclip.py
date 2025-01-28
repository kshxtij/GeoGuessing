from torch import float
from torchvision import transforms


def geoclip_train_transform():
    train_transform_list = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                    )
                ],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(float),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return train_transform_list


def geoclip_val_transform():
    val_transform_list = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(float),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return val_transform_list
