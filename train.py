# coding=utf8
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from config import Config
from dataset import SegmentationDataset
from network import DMSFNet
from utils import calculate_metrics, set_seed


def run_epoch(model, loader, optimizer, criterion, device, mode="train"):
    if mode == "train":
        model.train()
    else:
        model.eval()

    total_loss = 0
    all_preds = []
    all_masks = []

    for images, masks in tqdm(loader, desc=mode.capitalize()):
        images, masks = images.to(device), masks.to(device)

        with torch.set_grad_enabled(mode == "train"):
            outputs = model(images)
            loss = criterion(outputs, masks)

            if mode == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item()

        # Binarize predictions for metric calculation
        preds = (outputs > 0.5).float()
        all_preds.append(preds.cpu().numpy())
        all_masks.append(masks.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    acc, f1, iou, prec, rec, dice = calculate_metrics(all_masks, all_preds)

    return total_loss / len(loader), acc, f1, iou, prec, rec, dice


def main():
    set_seed(Config.SEED)

    # Initialize Datasets
    train_dataset = SegmentationDataset(Config.TRAIN_IMG_DIR, Config.TRAIN_MASK_DIR)
    val_dataset = SegmentationDataset(Config.VAL_IMG_DIR, Config.VAL_MASK_DIR)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE // 2, shuffle=False)

    # Initialize Model
    model = DMSFNet(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.BCELoss()

    best_iou = 0.0

    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")

        train_metrics = run_epoch(model, train_loader, optimizer, criterion, Config.DEVICE, mode="train")
        val_metrics = run_epoch(model, val_loader, optimizer, criterion, Config.DEVICE, mode="val")

        print(f"Train Loss: {train_metrics[0]:.4f} | IoU: {train_metrics[3]:.4f}")
        print(f"Val Loss: {val_metrics[0]:.4f} | IoU: {val_metrics[3]:.4f}")

        # Save best model
        if val_metrics[3] > best_iou:
            best_iou = val_metrics[3]
            torch.save(model.state_dict(), Config.SAVE_PATH)
            print(f"Model saved with IoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()