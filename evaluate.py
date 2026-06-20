"""Evaluation Module for Histopathology Tumor Segmentation.

This module provides features to evaluate trained segmentation models:
1. Reconstruct Whole Slide Image (WSI) probability heatmaps using sliding windows.
2. Run Ablation Studies evaluating the impact of Macenko stain normalization on color-shifted slide crops.
3. Plot Free-Response Receiver Operating Characteristic (FROC) curves to evaluate patch-level classification.
4. Visualize prediction overlays comparing original crop, normalized crop, ground truth, and predictions.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from typing import Tuple, List, Dict, Any, Optional

from config import config
from data_pipeline import (
    WSI_Simulator, 
    compute_otsu_tissue_mask, 
    MacenkoNormalizer, 
    HistopathologySegmentationDataset
)
from train import calculate_metrics, DiceLoss


def reconstruct_heatmap(
    slide_img: np.ndarray, 
    model: torch.nn.Module, 
    device: str, 
    patch_size: int = 128, 
    stride: int = 64
) -> np.ndarray:
    """Reconstructs the whole-slide tumor probability heatmap using a sliding window.
    
    Overlapping patches are stitched together and averaged using a weighting mask
    to avoid grid boundary artifacts.
    """
    h, w, _ = slide_img.shape
    heatmap = np.zeros((h, w), dtype=np.float32)
    weights = np.zeros((h, w), dtype=np.float32)
    
    model.eval()
    patch_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Pre-instantiate standard Macenko normalizer
    normalizer = MacenkoNormalizer()
    
    with torch.no_grad():
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                crop = slide_img[y:y+patch_size, x:x+patch_size]
                
                # Apply stain normalization (graceful fallback inside data_pipeline)
                crop_normalized = normalizer.normalize(crop)
                
                img_tensor = patch_transform(crop_normalized).unsqueeze(0).to(device)
                pred_logits = model(img_tensor)
                pred_probs = torch.sigmoid(pred_logits).squeeze().cpu().numpy()
                
                heatmap[y:y+patch_size, x:x+patch_size] += pred_probs
                weights[y:y+patch_size, x:x+patch_size] += 1.0
                
    weights = np.clip(weights, 1.0, None)
    return heatmap / weights


def apply_color_shift(img_np: np.ndarray) -> np.ndarray:
    """Applies a color shift simulation to histopathology patches.
    
    Simulates variations in slide staining procedures across different laboratories.
    Decreases red/blue channels and increases green channels to challenge the model.
    """
    img_shifted = img_np.astype(np.float32)
    img_shifted[..., 0] *= 0.85  # decrease red
    img_shifted[..., 1] *= 1.15  # increase green
    img_shifted[..., 2] *= 0.90  # decrease blue
    return np.clip(img_shifted, 0, 255).astype(np.uint8)


def run_ablation_study(
    test_slide: np.ndarray, 
    test_mask: np.ndarray, 
    model: torch.nn.Module, 
    device: str,
    output_path: Optional[str] = None
) -> Tuple[float, float]:
    """Runs stain normalization ablation study on color-shifted slide crops.
    
    Compares the U-Net model's mean Dice Overlap Coefficient on:
    1. Color-shifted patches WITHOUT stain normalization.
    2. Color-shifted patches WITH Macenko stain normalization.
    
    Returns:
        Tuple of (mean_dice_no_norm, mean_dice_with_norm).
    """
    # Create an un-transformed dataset of patches
    from data_pipeline import TissuePatchDataset
    ablation_dataset = TissuePatchDataset(
        [test_slide], [test_mask], patch_size=128, stride=64, transform=None
    )
    
    dice_no_norm = []
    dice_with_norm = []
    
    model.eval()
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    normalizer = MacenkoNormalizer()
    
    print("Running Ablation evaluation on color-shifted patches...")
    with torch.no_grad():
        for img_np, mask_np in zip(ablation_dataset.patches, ablation_dataset.masks):
            img_shifted = apply_color_shift(img_np)
            mask_tensor = torch.tensor(mask_np > 0, dtype=torch.float32).unsqueeze(0).to(device)
            
            # 1. Evaluate without stain normalization
            img_tensor_no = eval_transform(Image.fromarray(img_shifted)).unsqueeze(0).to(device)
            out_no = model(img_tensor_no)
            
            # Simple threshold calculation (Dice)
            probs_no = torch.sigmoid(out_no)
            preds_no = (probs_no > 0.5).float()
            intersection_no = (preds_no * mask_tensor).sum().item()
            dice_val_no = (2.0 * intersection_no + 1e-6) / (preds_no.sum().item() + mask_tensor.sum().item() + 1e-6)
            dice_no_norm.append(dice_val_no)
            
            # 2. Evaluate with Macenko stain normalization
            img_norm = normalizer.normalize(img_shifted)
            img_tensor_with = eval_transform(Image.fromarray(img_norm)).unsqueeze(0).to(device)
            out_with = model(img_tensor_with)
            
            probs_with = torch.sigmoid(out_with)
            preds_with = (probs_with > 0.5).float()
            intersection_with = (preds_with * mask_tensor).sum().item()
            dice_val_with = (2.0 * intersection_with + 1e-6) / (preds_with.sum().item() + mask_tensor.sum().item() + 1e-6)
            dice_with_norm.append(dice_val_with)
            
    mean_dice_no_norm = np.mean(dice_no_norm)
    mean_dice_with_norm = np.mean(dice_with_norm)
    
    print(f"Ablation Results:")
    print(f" -> Without Normalization: {mean_dice_no_norm:.4f}")
    print(f" -> With Macenko Normalization: {mean_dice_with_norm:.4f}")
    
    # Plot results
    categories = ['Color-Shifted\n(No Normalization)', 'Color-Shifted\n(Macenko Normalized)']
    scores = [mean_dice_no_norm, mean_dice_with_norm]
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(categories, scores, color=['#e74c3c', '#2ecc71'], width=0.4)
    plt.ylabel('Mean Dice Coefficient', fontsize=12)
    plt.title('Stain Normalization Ablation Study', fontsize=13)
    plt.ylim(0, 1.05)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Ablation plot saved to: {output_path}")
    plt.close()
    
    return mean_dice_no_norm, mean_dice_with_norm


def run_froc_curve(
    test_slide: np.ndarray,
    test_mask: np.ndarray,
    model: torch.nn.Module,
    device: str,
    output_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes and plots the Free-Response ROC (FROC) Curve for tumor detection.
    
    Measures the trade-off between tumor localization Sensitivity and the average
    number of False Positives per WSI patch.
    """
    from data_pipeline import TissuePatchDataset
    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset_patches = TissuePatchDataset(
        [test_slide], [test_mask], patch_size=128, stride=64, transform=val_test_transform, stain_norm=True
    )
    test_loader_patches = DataLoader(test_dataset_patches, batch_size=16, shuffle=False)
    
    all_probs = []
    all_masks = []
    
    model.eval()
    print("Computing probabilities for FROC analysis...")
    with torch.no_grad():
        for images, masks in test_loader_patches:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy().reshape(-1))
            all_masks.extend(masks.numpy().reshape(-1))
            
    all_probs = np.array(all_probs)
    all_masks = np.array(all_masks)
    
    thresholds = np.linspace(0.01, 0.99, 50)
    sensitivities = []
    false_positives = []
    
    for th in thresholds:
        preds = (all_probs > th).astype(np.float32)
        tp = np.sum((preds == 1) & (all_masks == 1))
        fn = np.sum((preds == 0) & (all_masks == 1))
        sens = tp / (tp + fn + 1e-8)
        
        fp = np.sum((preds == 1) & (all_masks == 0))
        # Avg FP per patch size (represented in units of 128x128 patches)
        avg_fp = fp / (128 * 128)
        
        sensitivities.append(sens)
        false_positives.append(avg_fp)
        
    # Plot FROC Curve
    plt.figure(figsize=(8, 5))
    plt.plot(false_positives, sensitivities, marker='o', color='purple', linewidth=2)
    plt.xlabel('Average False Positives per WSI (Patches)', fontsize=11)
    plt.ylabel('Sensitivity (True Positive Rate)', fontsize=11)
    plt.title('Free-Response ROC (FROC) Curve', fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, max(false_positives) * 1.05)
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"FROC curve saved to: {output_path}")
    plt.close()
    
    return np.array(false_positives), np.array(sensitivities)


def evaluate_single_crop(
    model: torch.nn.Module, 
    device: str, 
    slide_seed: int = 999, 
    output_path: Optional[str] = None
) -> float:
    """Generates a small test slide, crops a central patch, runs inference,
    plots results, and prints the Dice overlap metric.
    """
    print(f"Generating a mock slide (Seed {slide_seed}) for visual crop inference...")
    test_sim = WSI_Simulator(size=256, seed=slide_seed)
    test_slide, test_mask = test_sim.generate_slide()
    
    crop_img = test_slide[64:192, 64:192]
    crop_gt_mask = test_mask[64:192, 64:192]
    
    # Stain normalization
    normalizer = MacenkoNormalizer()
    normalized_img = normalizer.normalize(crop_img)
    
    # Run inference
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = eval_transform(Image.fromarray(normalized_img)).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.sigmoid(outputs).squeeze().cpu().numpy()
        binary_mask = (probabilities > 0.5).astype(np.uint8)
        
    # Plot panels
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(crop_img)
    axes[0].set_title("1. Original Crop", fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(normalized_img)
    axes[1].set_title("2. Macenko Normalized", fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(crop_gt_mask, cmap='gray')
    axes[2].set_title("3. Ground Truth Mask", fontsize=12)
    axes[2].axis('off')
    
    overlay = np.copy(normalized_img)
    overlay[binary_mask == 1] = [255, 0, 0]  # color tumor pixels red
    blended = (normalized_img * 0.5 + overlay * 0.5).astype(np.uint8)
    
    axes[3].imshow(blended)
    axes[3].set_title("4. Predicted Mask Overlay (Red)", fontsize=12)
    axes[3].axis('off')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Crop evaluation visualization saved to: {output_path}")
    plt.close()
    
    # Calculate dice coefficient
    intersection = np.sum((binary_mask == 1) & (crop_gt_mask == 1))
    union_sum = np.sum(binary_mask) + np.sum(crop_gt_mask)
    dice = (2.0 * intersection) / (union_sum + 1e-8)
    print(f"Dice Coefficient: {dice * 100:.2f}%")
    
    return dice


if __name__ == "__main__":
    print("Histopathology Evaluation Module initialized.")
