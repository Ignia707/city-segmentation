import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# mapping from cityscapes raw label IDs to 19 train IDs
# 255 means ignore (not used in training)
LABEL_MAP = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255,
    5: 255, 6: 255, 7: 0,   8: 1,   9: 255,
    10: 255, 11: 2, 12: 3,  13: 4,  14: 255,
    15: 255, 16: 255, 17: 5, 18: 255, 19: 6,
    20: 7,  21: 8,  22: 9,  23: 10, 24: 11,
    25: 12, 26: 13, 27: 14, 28: 15, 29: 255,
    30: 255, 31: 16, 32: 17, 33: 18, -1: 255
}

# the 19 class names in order
CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation",
    "terrain", "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle"
]

def convert_labels(mask):
    """Convert raw cityscapes label IDs to 0-18 train IDs."""
    converted = np.full_like(mask, 255)
    for raw_id, train_id in LABEL_MAP.items():
        converted[mask == raw_id] = train_id
    return converted


class CityscapesDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        """
        root     : path to data/cityscapes/
        split    : "train", "val", or "test"
        transform: albumentations transform pipeline
        """
        self.root = root
        self.split = split
        self.transform = transform

        self.img_dir = os.path.join(root, "leftImg8bit", split)
        self.mask_dir = os.path.join(root, "gtFine", split)

        # collect all image paths
        self.images = []
        self.masks = []

        cities = os.listdir(self.img_dir)
        for city in sorted(cities):
            city_img_dir = os.path.join(self.img_dir, city)
            city_mask_dir = os.path.join(self.mask_dir, city)

            for fname in sorted(os.listdir(city_img_dir)):
                if not fname.endswith("_leftImg8bit.png"):
                    continue
                img_path = os.path.join(city_img_dir, fname)
                mask_fname = fname.replace(
                    "_leftImg8bit.png",
                    "_gtFine_labelIds.png"
                )
                mask_path = os.path.join(city_mask_dir, mask_fname)

                if os.path.exists(mask_path):
                    self.images.append(img_path)
                    self.masks.append(mask_path)

        print(f"Cityscapes {split}: {len(self.images)} images found")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # load image
        img = cv2.imread(self.images[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # load mask and convert labels
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        mask = convert_labels(mask)

        # apply transforms
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        return img, mask.long()


def get_transforms(split, crop_size=512):
    """Return albumentations transforms for train or val."""
    if split == "train":
        return A.Compose([
            A.RandomResizedCrop(
                size=(crop_size, crop_size),
                scale=(0.5, 1.0)
            ),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1,
                p=0.5
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(height=crop_size, width=crop_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])