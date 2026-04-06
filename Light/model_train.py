import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

os.environ['TORCH_HOME'] = r'd:\College\IVPLab\torch_cache'

# Cityscapes labelId -> trainId mapping
id_to_trainid = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
    19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
    25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18
}

class CityscapesDataset(Dataset):
    def __init__(self, root, split='train', img_size=(512, 1024)):
        self.root = root
        self.split = split
        self.img_size = img_size
        self.img_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.samples = self._load_pairs()
        print(f"[{split}] Found {len(self.samples)} image-label pairs")

    def _load_pairs(self):
        pairs = []
        img_dir = os.path.join(self.root, 'leftImg8bit', self.split)
        gt_dir  = os.path.join(self.root, 'gtFine',      self.split)

        for city in sorted(os.listdir(img_dir)):
            img_city = os.path.join(img_dir, city)
            gt_city  = os.path.join(gt_dir,  city)
            for fname in sorted(os.listdir(img_city)):
                if not fname.endswith('_leftImg8bit.png'):
                    continue
                img_path = os.path.join(img_city, fname)
                # match the ground truth file
                gt_fname = fname.replace('_leftImg8bit.png',
                                         '_gtFine_labelIds.png')
                gt_path  = os.path.join(gt_city, gt_fname)
                if os.path.exists(gt_path):
                    pairs.append((img_path, gt_path))
                else:
                    print(f"Warning: missing GT for {fname}")
        return pairs

    def _encode_labels(self, gt):
        """Convert labelIds to trainIds (0-18), rest -> 255"""
        gt_np = np.array(gt, dtype=np.int32)
        out   = np.ones_like(gt_np) * 255
        for label_id, train_id in id_to_trainid.items():
            out[gt_np == label_id] = train_id
        return torch.from_numpy(out).long()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, gt_path = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        gt  = Image.open(gt_path)

        # resize GT with nearest neighbor to preserve label values
        gt = gt.resize((self.img_size[1], self.img_size[0]),
                        Image.NEAREST)

        img = self.img_transform(img)
        gt  = self._encode_labels(gt)
        return img, gt

# ── Loss: Weighted Cross Entropy (handles class imbalance) ───
"""device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_weights = torch.ones(19).to(device)
class_weights[0] = 0.5   # road is dominant, downweight it
class_weights[11] = 2.0  # person — upweight rare classes
class_weights[12] = 2.0  # rider
criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)"""

# ── Data ─────────────────────────────────────────────────────
img_transform = transforms.Compose([
    transforms.Resize((512, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
target_transform = transforms.Compose([
    transforms.Resize((512, 1024), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.PILToTensor()
])

train_dataset = CityscapesDataset(
    r'd:\College\IVPLab\Project\Fast_Seg\cityscapes', split='train')
val_dataset   = CityscapesDataset(
    r'd:\College\IVPLab\Project\Fast_Seg\cityscapes', split='val')

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_dataset,   batch_size=4, shuffle=False, num_workers=4)


from architecture import FastSegNet

if __name__=='__main__':
    # ── Training Loop ─────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = FastSegNet(num_classes=19).to(device)

    class_weights = torch.ones(19).to(device)
    class_weights[0]  = 0.5
    class_weights[11] = 2.0
    class_weights[12] = 2.0
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    def compute_miou(pred, target, num_classes=19, ignore=255):
        pred   = pred.argmax(1).cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()
        mask   = target != ignore
        pred, target = pred[mask], target[mask]
        iou_per_class = []
        for c in range(num_classes):
            tp = ((pred == c) & (target == c)).sum()
            fp = ((pred == c) & (target != c)).sum()
            fn = ((pred != c) & (target == c)).sum()
            if tp + fp + fn == 0:
                continue
            iou_per_class.append(tp / (tp + fp + fn))
        return sum(iou_per_class) / len(iou_per_class)

    EPOCHS = 40
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch_idx, (imgs, masks) in enumerate(train_loader):
            imgs  = imgs.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            loss  = criterion(model(imgs), masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Print every 10 batches so you know it's running
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | "
                    f"Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f}")

        scheduler.step()
        print(f"Epoch {epoch+1} complete | Avg Loss: {total_loss/len(train_loader):.4f}")

        if (epoch + 1) % 5 == 0:
            model.eval()
            mious = []
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs = imgs.to(device)
                    out  = model(imgs)
                    mious.append(compute_miou(out, masks))
            print(f"Val mIoU: {sum(mious)/len(mious)*100:.2f}%")

        best_miou = 0.0

        if (epoch + 1) % 5 == 0:
            model.eval()
            mious = []
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs = imgs.to(device)
                    out  = model(imgs)
                    mious.append(compute_miou(out, masks))
            val_miou = sum(mious) / len(mious) * 100
            print(f"Val mIoU: {val_miou:.2f}%")

            # Only save if it's the best so far
            if val_miou > best_miou:
                best_miou = val_miou
                torch.save(model.state_dict(),
                        r'd:\College\IVPLab\Project\Fast_Seg\Light\best_model.pth')
                print(f"Saved best model with mIoU: {best_miou:.2f}%")