# End-to-End Histopathology Tumor Segmentation Pipeline (Notebook Code)
# Automatically extracted from notebookd62b3f86d4-2.ipynb

import os
import glob
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# Import custom modular helpers
from utils_segmentation import WSI_Simulator, TissuePatchDataset, otsu_tissue_mask, normalize_stain_macenko
from unet_model import ResNetUNet, AttentionUNet, DiceBCELoss

# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Model: {torch.cuda.get_device_name(0)}")

# 2. Configuration & Loaders (Kaggle Kumar Dataset paths)
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_glob = "/kaggle/input/datasets/ayush02102001/tnbc-seg/Kumar/kumar/train/Images/*.tif"
mask_glob = "/kaggle/input/datasets/ayush02102001/tnbc-seg/Kumar/kumar/train/Overlay/*.png"

image_paths_raw = glob.glob(image_glob)
mask_paths_raw = glob.glob(mask_glob)

# Pair matching images and masks
image_map = {os.path.splitext(os.path.basename(p))[0]: p for p in image_paths_raw}
mask_map = {os.path.splitext(os.path.basename(p))[0]: p for p in mask_paths_raw}

common_keys = sorted(list(set(image_map.keys()) & set(mask_map.keys())))
image_paths = [image_map[k] for k in common_keys]
mask_paths = [mask_map[k] for k in common_keys]

print(f"Total .tif images found: {len(image_paths_raw)}")
print(f"Total .png masks found:  {len(mask_paths_raw)}")
print(f"Successfully matched:    {len(image_paths)} image-mask pairs.")

# Try loading Kumar dataset if files exist
real_images = []
real_masks = []
if len(image_paths) > 0:
    print("\nLoading paired Kumar slides into memory...")
    for img_path, mask_path in zip(image_paths, mask_paths):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)
        real_images.append(img)
        real_masks.append(mask)
    print(f"Loaded {len(real_images)} slides into memory.")
    train_dataset = TissuePatchDataset(real_images, real_masks, patch_size=128, stride=64, transform=train_transform, stain_norm=True)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    print(f"✅ train_loader created successfully with {len(train_loader)} batches!")
else:
    print("Kumar dataset not found. Standard pipeline runs using simulated WSIs.")

# 3. Simulate WSI slides to build a fallback/robust dataset
print("\n--- Generating Simulated WSI Dataset ---")
train_slides, train_masks = [], []
for seed in [101, 102, 103]:
    sim = WSI_Simulator(size=1024, seed=seed)
    s, m = sim.generate_slide()
    train_slides.append(s)
    train_masks.append(m)

val_slides, val_masks = [], []
for seed in [201]:
    sim = WSI_Simulator(size=1024, seed=seed)
    s, m = sim.generate_slide()
    val_slides.append(s)
    val_masks.append(m)

val_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Extracting training patches...")
sim_train_dataset = TissuePatchDataset(train_slides, train_masks, patch_size=128, stride=64, transform=train_transform, stain_norm=True)
print("\nExtracting validation patches...")
sim_val_dataset = TissuePatchDataset(val_slides, val_masks, patch_size=128, stride=64, transform=val_test_transform, stain_norm=True)

sim_train_loader = DataLoader(sim_train_dataset, batch_size=16, shuffle=True)
sim_val_loader = DataLoader(sim_val_dataset, batch_size=16, shuffle=False)

print(f"\nSimulated Train Loader: {len(sim_train_loader)} batches.")
print(f"Simulated Validation Loader: {len(sim_val_loader)} batches.")

# 4. Model Training Settings
model_type = "unet"  # change to "attention" to train Attention U-Net
if model_type == "unet":
    model = ResNetUNet(n_class=1).to(device)
else:
    model = AttentionUNet(n_class=1).to(device)

criterion = DiceBCELoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)
print(f"Model '{model_type}' initialized and loaded to {device}.")

# 5. Training Loop Helper
def calculate_metrics(pred_logits, targets, threshold=0.5, smooth=1e-6):
    preds = (torch.sigmoid(pred_logits) > threshold).float()
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)
    intersection = (preds_flat * targets_flat).sum().item()
    union = preds_flat.sum().item() + targets_flat.sum().item() - intersection
    dice = (2.0 * intersection + smooth) / (preds_flat.sum().item() + targets_flat.sum().item() + smooth)
    iou = (intersection + smooth) / (union + smooth)
    return dice, iou

epochs = 5
print("\n--- Starting Model Training Loop ---")
active_loader = sim_train_loader
active_val_loader = sim_val_loader

for epoch in range(epochs):
    # Training Phase
    model.train()
    train_loss = 0.0
    train_dice, train_iou = [], []
    for images, masks in active_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        d, i = calculate_metrics(outputs, masks)
        train_dice.append(d)
        train_iou.append(i)
    epoch_loss = train_loss / len(active_loader.dataset)
    mean_tdice = np.mean(train_dice)
    mean_tiou = np.mean(train_iou)

    # Validation Phase
    model.eval()
    val_loss = 0.0
    val_dice, val_iou = [], []
    with torch.no_grad():
        for images, masks in active_val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)
            d, i = calculate_metrics(outputs, masks)
            val_dice.append(d)
            val_iou.append(i)
    epoch_val_loss = val_loss / len(active_val_loader.dataset)
    mean_vdice = np.mean(val_dice)
    mean_viou = np.mean(val_iou)
    
    print(f"Epoch {epoch+1}/{epochs} | "
          f"Train Loss: {epoch_loss:.4f} Dice: {mean_tdice:.4f} IoU: {mean_tiou:.4f} | "
          f"Val Loss: {epoch_val_loss:.4f} Dice: {mean_vdice:.4f} IoU: {mean_viou:.4f}")

# 6. Evaluation On Simulated Crop (Seed 456)
print("\n--- Running Single Crop Evaluation (Seed 456) ---")
test_sim = WSI_Simulator(size=256, seed=456)
test_slide, test_mask = test_sim.generate_slide()
crop_img = test_slide[64:192, 64:192]
crop_gt_mask = test_mask[64:192, 64:192]

try:
    normalized_img = normalize_stain_macenko(crop_img)
    processed_img = Image.fromarray(normalized_img)
except Exception as e:
    processed_img = Image.fromarray(crop_img)
    normalized_img = crop_img

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
img_tensor = eval_transform(processed_img).unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    outputs = model(img_tensor)
    probabilities = torch.sigmoid(outputs).squeeze().cpu().numpy()
    binary_mask = (probabilities > 0.5).astype(np.uint8)

intersection = np.sum((binary_mask == 1) & (crop_gt_mask == 1))
union_sum = np.sum(binary_mask) + np.sum(crop_gt_mask)
dice = (2.0 * intersection) / (union_sum + 1e-8)
print(f"Accuracy Metric for Seed 456 Crop -> Dice: {dice * 100:.2f}%")

# 7. Whole WSI Stitched Heatmap Reconstruction
def reconstruct_heatmap(slide_img, model, patch_size=128, stride=64):
    h, w, _ = slide_img.shape
    heatmap = np.zeros((h, w), dtype=np.float32)
    weights = np.zeros((h, w), dtype=np.float32)
    model.eval()
    patch_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    with torch.no_grad():
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                crop = slide_img[y:y+patch_size, x:x+patch_size]
                try:
                    crop = normalize_stain_macenko(crop)
                except:
                    pass
                img_tensor = patch_transform(crop).unsqueeze(0).to(device)
                pred_logits = model(img_tensor)
                pred_probs = torch.sigmoid(pred_logits).squeeze().cpu().numpy()
                heatmap[y:y+patch_size, x:x+patch_size] += pred_probs
                weights[y:y+patch_size, x:x+patch_size] += 1.0
    weights = np.clip(weights, 1.0, None)
    return heatmap / weights

print("Generating stitched WSI Tumor Probability Heatmap (Seed 301)...")
test_sim_wsi = WSI_Simulator(size=1024, seed=301)
test_slide_wsi, test_mask_wsi = test_sim_wsi.generate_slide()
prob_heatmap = reconstruct_heatmap(test_slide_wsi, model)
print("Stitching completed successfully.")

# 8. Stain Normalization Ablation Study
def apply_color_shift(img_np):
    img_shifted = img_np.astype(np.float32)
    img_shifted[..., 0] *= 0.85
    img_shifted[..., 1] *= 1.15
    img_shifted[..., 2] *= 0.90
    return np.clip(img_shifted, 0, 255).astype(np.uint8)

ablation_dataset = TissuePatchDataset([test_slide_wsi], [test_mask_wsi], patch_size=128, stride=64, transform=None)
dice_no_norm = []
dice_with_norm = []

print("Running Ablation evaluation on color-shifted patches...")
with torch.no_grad():
    for img_np, mask_np in zip(ablation_dataset.patches, ablation_dataset.masks):
        img_shifted = apply_color_shift(img_np)
        img_tensor_no = eval_transform(Image.fromarray(img_shifted)).unsqueeze(0).to(device)
        mask_tensor = torch.tensor(mask_np > 0, dtype=torch.float32).unsqueeze(0).to(device)
        out_no = model(img_tensor_no)
        d_no, _ = calculate_metrics(out_no, mask_tensor)
        dice_no_norm.append(d_no)
        
        try:
            img_norm = normalize_stain_macenko(img_shifted)
        except:
            img_norm = img_shifted
        img_tensor_with = eval_transform(Image.fromarray(img_norm)).unsqueeze(0).to(device)
        out_with = model(img_tensor_with)
        d_with, _ = calculate_metrics(out_with, mask_tensor)
        dice_with_norm.append(d_with)

print(f"Mean Dice - No Normalization: {np.mean(dice_no_norm):.4f}")
print(f"Mean Dice - Macenko Normalized: {np.mean(dice_with_norm):.4f}")

# 9. Free-Response ROC (FROC) Curve Calculations
thresholds = np.linspace(0.01, 0.99, 50)
sensitivities = []
false_positives = []

test_dataset_patches = TissuePatchDataset([test_slide_wsi], [test_mask_wsi], patch_size=128, stride=64, transform=val_test_transform, stain_norm=True)
test_loader_patches = DataLoader(test_dataset_patches, batch_size=16, shuffle=False)

all_probs = []
all_masks = []

with torch.no_grad():
    for images, masks in test_loader_patches:
        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits)
        all_probs.extend(probs.cpu().numpy().reshape(-1))
        all_masks.extend(masks.numpy().reshape(-1))

all_probs = np.array(all_probs)
all_masks = np.array(all_masks)

for th in thresholds:
    preds = (all_probs > th).astype(np.float32)
    tp = np.sum((preds == 1) & (all_masks == 1))
    fn = np.sum((preds == 0) & (all_masks == 1))
    sens = tp / (tp + fn + 1e-8)
    fp = np.sum((preds == 1) & (all_masks == 0))
    avg_fp = fp / (128 * 128)
    
    sensitivities.append(sens)
    false_positives.append(avg_fp)
print("FROC curve computation completed.")
print("Pipeline Execution successfully completed.")
