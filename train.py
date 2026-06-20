# %%writefile train.py
"""Training Module for Histopathology Tumor Segmentation.

This module provides the training pipeline for both patch classification 
and semantic segmentation models. It features:
1. Custom Dice + BCE Combined Loss for robust segmentation boundary tracking.
2. PyTorch Automatic Mixed Precision (AMP) training utilizing torch.cuda.amp.
3. Gradient Scaling to prevent underflow of small gradients in float16 precision.
4. GPU VRAM optimization, invoking torch.cuda.empty_cache() between epochs.
5. Model checkpointing saved dynamically to Google Drive.
"""

import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Any, Optional

# Import configuration and pipeline modules
from config import config
from data_pipeline import HistopathologyDataset, HistopathologySegmentationDataset
from models import ResNet18Classifier, StandardUNet, AttentionUNet


class DiceLoss(nn.Module):
    """Dice Loss for Binary Semantic Segmentation.
    
    Dice Loss measures the overlap ratio between ground truth mask and prediction.
    Formula:
        Dice = 2 * |P intersect Y| / (|P| + |Y| + epsilon)
        DiceLoss = 1 - Dice
    """
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Unnormalized network outputs, shape [B, 1, H, W].
            targets: Binary ground truth masks, shape [B, 1, H, W].
            
        Returns:
            Scalar loss tensor.
        """
        probs = torch.sigmoid(logits)
        
        # Flatten predictions and targets to calculate overlap
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        denominator = probs_flat.sum() + targets_flat.sum()
        
        dice_coeff = (2. * intersection + self.smooth) / (denominator + self.smooth)
        return 1. - dice_coeff


class HybridBCEWithDiceLoss(nn.Module):
    """Hybrid loss combining Binary Cross Entropy (BCE) and Dice Loss.
    
    BCE evaluates pixel-level classification accuracy, while Dice Loss optimizes
    the global spatial intersection, resolving extreme class imbalances.
    """
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def calculate_metrics(pred_logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> Tuple[float, float]:
    """Calculates Dice and IoU metrics for predictions relative to target masks.
    
    Args:
        pred_logits: Logits output from model.
        targets: Target binary masks.
        threshold: Sigmoid probability threshold for positive label.
        smooth: Smoothing factor to avoid division by zero.
        
    Returns:
        Tuple of (dice, iou).
    """
    preds = (torch.sigmoid(pred_logits) > threshold).float()
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)
    
    intersection = (preds_flat * targets_flat).sum().item()
    union = preds_flat.sum().item() + targets_flat.sum().item() - intersection
    
    dice = (2.0 * intersection + smooth) / (preds_flat.sum().item() + targets_flat.sum().item() + smooth)
    iou = (intersection + smooth) / (union + smooth)
    return dice, iou


def train_epoch_segmentation(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    device: str
) -> float:
    """Trains a segmentation model for a single epoch using Mixed Precision.
    
    Underlying Mechanics of AMP (Automatic Mixed Precision):
        Histopathology segmentation models deal with large activations. Floating point 32-bit (FP32)
        arithmetic consumes significant VRAM. By utilizing `torch.cuda.amp.autocast`, PyTorch executes
        non-critical ops (such as convolutions and matrix multiplies) in half-precision 16-bit floats (FP16),
        effectively cutting the VRAM requirements in half. 
        To prevent numerical underflow (where small gradients underflow to zero in FP16), `GradScaler` multiplies 
        the loss by a scaling factor prior to backward pass and scales it down during optimizer step.
    """
    model.train()
    running_loss = 0.0
    
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad(set_to_none=True)  # set_to_none saves memory by deleting grad tensors instead of writing zeros
        
        # Runs forward pass in autocast context (mixed precision)
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, masks)
            
        # Scale loss and backpropagate gradients
        if device == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        running_loss += loss.item() * images.size(0)
        
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate_epoch_segmentation(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, float, float]:
    """Validates the model, calculating validation loss, Dice score, and IoU.
    
    Executes in torch.no_grad() context to prevent computation graph storage, 
    further preserving VRAM.
    """
    model.eval()
    running_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            running_loss += loss.item() * images.size(0)
            
            # Calculate metrics
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            # Dice & IoU metrics
            intersection = (preds * masks).sum(dim=(2, 3))
            union = preds.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
            
            # Dice metric
            dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
            # IoU metric
            iou = (intersection + 1e-6) / (union - intersection + 1e-6)
            
            total_dice += dice.sum().item()
            total_iou += iou.sum().item()
            total_samples += images.size(0)
            
    val_loss = running_loss / len(dataloader.dataset)
    avg_dice = total_dice / total_samples
    avg_iou = total_iou / total_samples
    
    return val_loss, avg_dice, avg_iou


def run_training_pipeline(model_type: str = "attention_unet") -> Dict[str, List[float]]:
    """Initializes data loaders, models, and runs the entire training execution loop.
    
    Args:
        model_type: One of 'resnet18_classifier', 'standard_unet', or 'attention_unet'.
        
    Returns:
        Dictionary tracking loss and metric histories.
    """
    # Ensure save paths exist
    config.setup_directories()
    
    # Identify patch images extracted
    image_paths = sorted(glob.glob(os.path.join(config.PATCH_DIR, "images", "*.png")))
    mask_paths = sorted(glob.glob(os.path.join(config.PATCH_DIR, "masks", "*.png")))
    
    if len(image_paths) == 0:
        print("Error: No patches detected in", config.PATCH_DIR)
        print("Aborting training. Please extract patches first using data_pipeline.py.")
        return {}
        
    print(f"Detected {len(image_paths)} patches for training.")
    
    # Split paths into Train & Validation subsets (80/20 split)
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )
    
    # Instantiate PyTorch datasets & dataloaders
    train_dataset = HistopathologySegmentationDataset(train_imgs, train_masks, augment=True)
    val_dataset = HistopathologySegmentationDataset(val_imgs, val_masks, augment=False)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )
    
    # Select architecture
    device = config.DEVICE
    print(f"Initializing {model_type} model on {device}...")
    
    if model_type == "resnet18_classifier":
        print("Note: resnet18_classifier requires patch classification pipeline. Standardizing to segmentation U-Nets.")
        model = StandardUNet()
    elif model_type == "standard_unet":
        model = StandardUNet()
    elif model_type == "attention_unet":
        model = AttentionUNet()
    else:
        raise ValueError(f"Unknown model architecture type: {model_type}")
        
    model = model.to(device)
    
    # Configure optimizer, loss criteria, and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    criterion = HybridBCEWithDiceLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    
    # Initialize Gradient Scaler for Mixed Precision
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and config.USE_AMP))
    
    # Performance tracking lists
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": []
    }
    
    best_dice = 0.0
    
    for epoch in range(1, config.EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{config.EPOCHS} ---")
        
        # Train
        train_loss = train_epoch_segmentation(model, train_loader, optimizer, criterion, scaler, device)
        # Validate
        val_loss, val_dice, val_iou = validate_epoch_segmentation(model, val_loader, criterion, device)
        
        # Step learning rate scheduler
        scheduler.step()
        
        # Track history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")
        
        # Save checkpoints (Google Drive directory)
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"{model_type}_latest.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_dice': val_dice,
            'val_iou': val_iou,
        }, checkpoint_path)
        
        # Keep track of the best checkpoint
        if val_dice > best_dice:
            best_dice = val_dice
            best_path = os.path.join(config.CHECKPOINT_DIR, f"{model_type}_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"New best model saved with Dice Score: {best_dice:.4f}")
            
        # --- GPU VRAM Memory Optimization ---
        # Between epochs, we call empty_cache to clean out residual activations
        # from memory. Essential for Colab notebooks containing multiple pipelines.
        if device == "cuda":
            torch.cuda.empty_cache()
            
    print("\nTraining completed successfully.")
    return history


if __name__ == "__main__":
    print("Train Module initialized successfully.")
