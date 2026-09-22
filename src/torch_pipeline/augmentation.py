"""Field-condition data augmentation for leaf disease images (torchvision.transforms.v2)."""

import torch
from torchvision.transforms import v2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int = 224) -> v2.Compose:
    """Augmentations that simulate real field capture conditions.

    Order matters: geometric ops run on PIL/uint8 tensors (cheap, before
    float conversion), photometric ops run right before normalization, and
    cutout runs last since RandomErasing expects a normalized float tensor.
    """
    return v2.Compose([
        v2.ToImage(),
        # Simulates the leaf filling different fractions of the frame.
        v2.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.2),
        # Camera tilt / off-axis phone shots in the field.
        v2.RandomAffine(degrees=25, translate=(0.05, 0.05), shear=8),
        # Sun/shadow/white-balance variation between growers and time of day.
        v2.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35, hue=0.05),
        # Motion blur / focus miss, applied only some of the time.
        v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        # Coarse dropout / cutout: forces the model off any single occluded
        # region (e.g. a background patch it learned to key on).
        v2.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value="random"),
    ])


def get_val_transforms(img_size: int = 224) -> v2.Compose:
    """Deterministic preprocessing for validation, test, and inference."""
    return v2.Compose([
        v2.ToImage(),
        v2.Resize(int(img_size * 1.14)),
        v2.CenterCrop(img_size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
