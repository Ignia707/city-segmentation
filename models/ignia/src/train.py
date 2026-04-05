import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import yaml
import os

from src.dataset import CityscapesDataset, get_transforms
from src.model import get_model, get_modified_model
from src.evaluate import MeanIoU
from torch.amp import autocast, GradScaler

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, accumulation_steps=1, scaler=None):
    model.train()
    total_loss = 0
    metric = MeanIoU(num_classes=19)

    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for i, (imgs, masks) in enumerate(pbar):
        imgs = imgs.to(device)
        masks = masks.to(device)

        # mixed precision forward pass
        with autocast(device_type='cuda', enabled=scaler is not None):
            outputs = model(imgs)
            loss = criterion(outputs, masks)

        # scale loss by accumulation steps
        loss = loss / accumulation_steps

        # backward with scaler if fp16
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        torch.cuda.empty_cache()

        if (i + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        preds = outputs.argmax(dim=1)
        metric.update(preds, masks)
        pbar.set_postfix(loss=f"{loss.item() * accumulation_steps:.4f}")

    avg_loss = total_loss / len(loader)
    miou, _ = metric.compute()

    return avg_loss, miou


def validate(model, loader, criterion, device, epoch):
    model.eval()
    total_loss = 0
    metric = MeanIoU(num_classes=19)

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]")
        for imgs, masks in pbar:
            imgs = imgs.to(device)
            masks = masks.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            metric.update(preds, masks)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    miou, per_class = metric.compute()
    return avg_loss, miou, per_class


def train(config):
    # setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # initialize wandb
    wandb.init(
        project="city-segmentation",
        name=config["run_name"],
        config=config
    )

    # datasets
    train_dataset = CityscapesDataset(
        root=config["data_root"],
        split="train",
        transform=get_transforms("train", config["crop_size"])
    )
    val_dataset = CityscapesDataset(
        root=config["data_root"],
        split="val",
        transform=get_transforms("val", config["crop_size"])
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True
    )

    # model
    use_modified =  config.get("use_modified", False)
    if use_modified:
        model = get_modified_model(num_classes=19).to(device)
        print("Using modified model with FEM attention and weighted fusion")
    else:
        model = get_model(num_classes=19).to(device)
        print("Using baseline model")

    # loss — ignore index 255
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    # warmup + cosine annealing scheduler
    warmup_epochs = config.get("warmup_epochs", 5)
    use_fp16 = config.get("fp16", False)
    scaler = GradScaler(device='cuda') if use_fp16 else None

    # warmup scheduler — linear ramp from 0 to lr
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_epochs
    )

    # cosine annealing after warmup
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"] - warmup_epochs,
        eta_min=1e-6
    )

    # combine both
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    # training loop
    best_miou = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, config["epochs"] + 1):
        # train
        train_loss, train_miou = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, accumulation_steps=config.get("accumulation_steps", 1), scaler=scaler
        )

        # validate
        val_loss, val_miou, per_class_iou = validate(
            model, val_loader, criterion, device, epoch
        )

        # update scheduler
        scheduler.step()

        # log to wandb
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/mIoU": train_miou,
            "val/loss": val_loss,
            "val/mIoU": val_miou,
            "lr": optimizer.param_groups[0]["lr"]
        })

        print(f"\nEpoch {epoch}/{config['epochs']}")
        print(f"Train — Loss: {train_loss:.4f}, mIoU: {train_miou:.4f}")
        print(f"Val   — Loss: {val_loss:.4f}, mIoU: {val_miou:.4f}")

        # save best model
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(
                model.state_dict(),
                f"checkpoints/best_model.pth"
            )
            print(f"New best model saved — mIoU: {best_miou:.4f}")
            wandb.run.summary["best_val_miou"] = best_miou

    wandb.finish()
    print(f"\nTraining complete. Best val mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    train(config)