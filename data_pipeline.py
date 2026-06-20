# %%writefile data_pipeline.py
"""Data Pipeline Module for Histopathology Tumor Segmentation.

This module provides tools for working with gigapixel Whole Slide Images (WSIs).
It contains functions to read SV/TIF formats using OpenSlide, generate tissue masks
using Otsu thresholding, extract balanced patches of tumor and normal tissue,
and apply stain normalization using the Macenko method via torchstain.
"""

import os
import cv2
import numpy as np
from typing import Tuple, List, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset
from PIL import Image

# Try to import openslide. If it fails, define a mock or raise an informative error.
try:
    import openslide
except ImportError:
    # On non-Linux environments where openslide-tools isn't installed, 
    # we provide a mockup structure so importing does not fail immediately.
    openslide = None
    print("Warning: OpenSlide is not installed or missing binary dependencies. Please install openslide-tools.")

# Try to import torchstain for GPU-accelerated stain normalization.
try:
    import torchstain
    from torchstain.torch.normalizers import TorchMacenkoNormalizer
except ImportError:
    torchstain = None
    TorchMacenkoNormalizer = None
    print("Warning: torchstain is not installed. Using mock stain normalization.")


def get_best_level_for_magnification(slide, target_mag: float = 20.0) -> int:
    """Calculates the best slide level that corresponds to the target magnification.
    
    Many slides have objective magnification stored in properties. We compute the
    level relative to the slide's base magnification.
    
    Args:
        slide: OpenSlide object.
        target_mag: Desired magnification level (typically 20.0 or 40.0).
        
    Returns:
        The slide level index closest to the target magnification.
    """
    if openslide is None:
        return 0
        
    # Get base slide magnification (level 0)
    mag_str = slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
    if not mag_str:
        # Fallback assumption: level 0 is 40x
        base_mag = 40.0
    else:
        base_mag = float(mag_str)
        
    # Downsamples available
    downsamples = slide.level_downsamples
    
    # Target downsample ratio relative to level 0
    target_downsample = base_mag / target_mag
    
    # Find closest level to target downsample
    diffs = [abs(ds - target_downsample) for ds in downsamples]
    best_level = int(np.argmin(diffs))
    
    return best_level


def compute_otsu_tissue_mask(slide, level: int) -> np.ndarray:
    """Generates a binary tissue mask using Otsu's thresholding.
    
    Mechanics of Otsu's thresholding:
        Histopathology slides contain massive white background areas. This function reads a low-resolution
        representation of the slide (thumbnail level) and converts it to the HSV color space.
        We isolate the saturation (S) channel because background glass has low saturation (close to 0),
        while stained tissue contains rich pigments. Otsu's algorithm automatically determines the optimal
        threshold that minimizes intra-class variance (or maximizes inter-class variance) between the
        foreground (tissue) and background (glass).
        
    Args:
        slide: OpenSlide object.
        level: Slide pyramid level to read (low magnification thumbnail).
        
    Returns:
        Binary mask (numpy array of shape [height, width] containing 0 for background, 255 for tissue).
    """
    if openslide is None:
        # Return dummy mask for local testing
        return np.ones((512, 512), dtype=np.uint8) * 255

    # Read thumbnail at the designated low-res level
    size = slide.level_dimensions[level]
    img = slide.read_region((0, 0), level, size)
    img_rgb = np.array(img.convert("RGB"))
    
    # Convert RGB to HSV color space
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    saturation = img_hsv[:, :, 1]
    
    # Apply Otsu's thresholding to the Saturation channel
    _, tissue_mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Optional morphological cleanup to fill small holes inside tissue
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)
    
    return tissue_mask


class MacenkoNormalizer:
    """Macenko Stain Normalizer.
    
    Decomposes tissue images into Hematoxylin and Eosin optical density channels
    and standardizes them to a reference style to correct lab-specific staining variations.
    Uses GPU speedups if torchstain and CUDA are available.
    """
    def __init__(self, ref_image: Optional[np.ndarray] = None):
        self.torch_normalizer = None
        self.ref_image = ref_image
        
        if torchstain is not None and TorchMacenkoNormalizer is not None:
            # We initialize torchstain's normalizer class
            self.torch_normalizer = TorchMacenkoNormalizer()
            
            # Setup a standard reference image if none is passed
            if ref_image is None:
                # Synthesize a realistic histopathology-like patch (soft pink/purple hues)
                ref_image = np.ones((256, 256, 3), dtype=np.uint8) * 240
                ref_image[50:200, 50:200, 0] = 180  # H-like purple channel bias
                ref_image[50:200, 50:200, 1] = 100
                ref_image[50:200, 50:200, 2] = 190  # E-like pink channel bias
            
            ref_tensor = torch.from_numpy(ref_image).permute(2, 0, 1).contiguous()
            self.torch_normalizer.fit(ref_tensor)
            print("GPU-backed Macenko Normalizer successfully fitted to reference image.")

    def normalize(self, img_rgb: np.ndarray) -> np.ndarray:
        """Applies stain normalization to the input RGB image.
        
        Args:
            img_rgb: Input image of shape [H, W, 3] in RGB format, range [0, 255].
            
        Returns:
            Normalized image of shape [H, W, 3] in RGB format.
        """
        if self.torch_normalizer is not None:
            try:
                # Convert to torch tensor, shape [C, H, W]
                img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous()
                # Run torchstain normalization
                norm_tensor, _, _ = self.torch_normalizer.normalize(img_tensor)
                # Convert back to numpy array (TorchMacenkoNormalizer returns shape [H, W, C]) and cast to uint8
                return norm_tensor.numpy().astype(np.uint8)
            except Exception as e:
                # Fallback to local numpy-based Macenko implementation
                return normalize_stain_macenko(img_rgb)
        return normalize_stain_macenko(img_rgb)


def extract_balanced_patches(
    slide_path: str,
    mask_path: Optional[str],
    output_dir: str,
    patch_size: int = 256,
    target_mag: float = 20.0,
    patches_per_slide: int = 400,
    tumor_ratio: float = 0.5,
    tissue_threshold: float = 0.3,
    tumor_threshold: float = 0.5,
    normalizer: Optional[MacenkoNormalizer] = None
) -> Tuple[List[str], List[int]]:
    """Extracts balanced patch coordinate locations or saves patch files from slide.
    
    This function analyzes the slide, generates a tissue mask at low resolution,
    and maps the coordinate spaces between the high-resolution level (level 0) and
    the low-resolution level. It samples coordinates, checks if they contain tissue,
    determines if they overlap with the tumor mask, and saves the stain-normalized
    patches into tumor/normal folders.
    
    Args:
        slide_path: Path to the .tif/.svs whole slide image.
        mask_path: Path to the tumor mask slide or None if normal slide.
        output_dir: Parent folder to save extracted patch images.
        patch_size: Square patch size at the target magnification level.
        target_mag: Magnification level (typically 20x).
        patches_per_slide: Number of patches to extract.
        tumor_ratio: Proportion of tumor patches to sample (if mask is present).
        tissue_threshold: Minimum percentage of tissue pixels required.
        tumor_threshold: Minimum percentage of tumor pixels required to label as tumor.
        normalizer: Macenko stain normalizer.
        
    Returns:
        A list of paths to saved patch files and their corresponding binary labels (1=tumor, 0=normal).
    """
    if openslide is None:
        # Local fallback for non-OpenSlide environments (e.g. mock/simulation runs)
        print("OpenSlide is missing. Falling back to OpenCV image reader...")
        img_bgr = cv2.imread(slide_path)
        if img_bgr is None:
            print("Error: Could not read slide image:", slide_path)
            return [], [], []
        slide_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h_target, w_target, _ = slide_rgb.shape
        
        if mask_path and os.path.exists(mask_path):
            mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_bin = (mask_gray > 127).astype(np.uint8)
        else:
            mask_bin = np.zeros((h_target, w_target), dtype=np.uint8)
            
        tissue_mask = otsu_tissue_mask(slide_rgb)
        
        images_dir = os.path.join(output_dir, "images")
        masks_dir = os.path.join(output_dir, "masks")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)
        
        saved_paths = []
        saved_masks = []
        saved_labels = []
        
        tumor_candidates = []
        normal_candidates = []
        
        stride = patch_size
        for y_best in range(0, h_target - patch_size + 1, stride):
            for x_best in range(0, w_target - patch_size + 1, stride):
                mask_crop = tissue_mask[y_best:y_best + patch_size, x_best:x_best + patch_size]
                if mask_crop.size == 0:
                    continue
                tissue_ratio = np.mean(mask_crop)
                if tissue_ratio < tissue_threshold:
                    continue
                    
                crop_mask = mask_bin[y_best:y_best + patch_size, x_best:x_best + patch_size]
                tumor_ratio_val = np.mean(crop_mask)
                is_tumor = tumor_ratio_val >= tumor_threshold
                
                if is_tumor:
                    tumor_candidates.append((x_best, y_best, crop_mask))
                else:
                    normal_candidates.append((x_best, y_best, crop_mask))
                    
        total_tumor_needed = int(patches_per_slide * tumor_ratio) if mask_path else 0
        total_normal_needed = patches_per_slide - total_tumor_needed
        
        if len(tumor_candidates) > 0 and total_tumor_needed > 0:
            sampled_tumor_idx = np.random.choice(len(tumor_candidates), min(total_tumor_needed, len(tumor_candidates)), replace=False)
            sampled_tumor = [tumor_candidates[i] for i in sampled_tumor_idx]
        else:
            sampled_tumor = []
            total_normal_needed = patches_per_slide
            
        if len(normal_candidates) > 0:
            sampled_normal_idx = np.random.choice(len(normal_candidates), min(total_normal_needed, len(normal_candidates)), replace=False)
            sampled_normal = [normal_candidates[i] for i in sampled_normal_idx]
        else:
            sampled_normal = []
            
        slide_id = os.path.basename(slide_path).split('.')[0]
        all_samples = [(x_best, y_best, m_np, 1) for (x_best, y_best, m_np) in sampled_tumor] + \
                      [(x_best, y_best, m_np, 0) for (x_best, y_best, m_np) in sampled_normal]
                      
        for idx, (x_b, y_b, m_np, label) in enumerate(all_samples):
            patch_rgb = slide_rgb[y_b:y_b+patch_size, x_b:x_b+patch_size]
            if normalizer is not None:
                patch_rgb = normalizer.normalize(patch_rgb)
                
            img_name = f"{slide_id}_patch_{idx}_{x_b}_{y_b}.png"
            mask_name = f"{slide_id}_mask_{idx}_{x_b}_{y_b}.png"
            
            img_path = os.path.join(images_dir, img_name)
            mask_path_saved = os.path.join(masks_dir, mask_name)
            
            cv2.imwrite(img_path, cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(mask_path_saved, m_np * 255)
            
            saved_paths.append(img_path)
            saved_masks.append(mask_path_saved)
            saved_labels.append(label)
            
        print(f"Slide {slide_id}: Extracted {len(saved_paths)} patches ({len(sampled_tumor)} tumor, {len(sampled_normal)} normal).")
        return saved_paths, saved_masks, saved_labels

    # Open slide and determine scale factor mapping
    slide = openslide.OpenSlide(slide_path)
    best_level = get_best_level_for_magnification(slide, target_mag)
    
    # Get tissue mask at a low resolution level (level 4)
    low_res_level = min(4, len(slide.level_dimensions) - 1)
    tissue_mask = compute_otsu_tissue_mask(slide, low_res_level)
    
    # Scale conversion factors between level 0, best_level, and low_res_level
    best_to_zero_scale = slide.level_downsamples[best_level]
    low_res_to_zero_scale = slide.level_downsamples[low_res_level]
    
    # Scale from low_res to best_level (patch extraction coordinate level)
    best_to_low_res_scale = best_to_zero_scale / low_res_to_zero_scale
    
    # Open mask slide if provided
    mask_slide = openslide.OpenSlide(mask_path) if mask_path else None
    
    # Setup subdirectories
    images_dir = os.path.join(output_dir, "images")
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    saved_paths = []
    saved_masks = []
    saved_labels = []
    
    # Retrieve slide dimensions at target magnification level
    target_dims = slide.level_dimensions[best_level]
    w_target, h_target = target_dims[0], target_dims[1]
    
    # Compute patch dimensions at low resolution level for tissue threshold check
    low_res_patch_size = int(patch_size * best_to_low_res_scale)
    
    # Grid search coordinates or random sampling to satisfy target balance
    np.random.seed(42)
    
    # Separate lists to store candidates and their corresponding binary masks at best_level
    # We store (x_zero, y_zero, is_tumor, optional_mask_np)
    tumor_candidates = []
    normal_candidates = []
    
    # Grid stride
    stride = patch_size
    
    for y_best in range(0, h_target - patch_size, stride):
        for x_best in range(0, w_target - patch_size, stride):
            # Translate coordinates to low resolution tissue mask
            x_low = int(x_best * best_to_low_res_scale)
            y_low = int(y_best * best_to_low_res_scale)
            
            # Crop tissue mask region
            mask_crop = tissue_mask[y_low:y_low + low_res_patch_size, x_low:x_low + low_res_patch_size]
            if mask_crop.size == 0:
                continue
                
            # Check if there is enough tissue in this candidate patch
            tissue_ratio = np.sum(mask_crop == 255) / mask_crop.size
            if tissue_ratio < tissue_threshold:
                continue
                
            # Calculate coordinates at level 0 (needed for read_region)
            x_zero = int(x_best * best_to_zero_scale)
            y_zero = int(y_best * best_to_zero_scale)
            
            # Check tumor class mapping
            is_tumor = False
            mask_np = None
            if mask_slide is not None:
                # Read region from mask slide at target level
                mask_patch = mask_slide.read_region((x_zero, y_zero), best_level, (patch_size, patch_size))
                mask_np = np.array(mask_patch.convert("L"))  # Grayscale
                
                # Check tumor pixel ratio
                tumor_pixel_ratio = np.sum(mask_np > 0) / mask_np.size
                if tumor_pixel_ratio >= tumor_threshold:
                    is_tumor = True
            else:
                # No tumor mask slide means this is a healthy slide
                mask_np = np.zeros((patch_size, patch_size), dtype=np.uint8)
                    
            if is_tumor:
                tumor_candidates.append((x_zero, y_zero, mask_np))
            else:
                normal_candidates.append((x_zero, y_zero, mask_np))
                
    # Balance extraction using sampled candidates
    total_tumor_needed = int(patches_per_slide * tumor_ratio) if mask_slide else 0
    total_normal_needed = patches_per_slide - total_tumor_needed
    
    # Sample subsets
    if len(tumor_candidates) > 0 and total_tumor_needed > 0:
        sampled_tumor_idx = np.random.choice(len(tumor_candidates), min(total_tumor_needed, len(tumor_candidates)), replace=False)
        sampled_tumor = [tumor_candidates[i] for i in sampled_tumor_idx]
    else:
        sampled_tumor = []
        # If no tumor patches, fill with normal patches
        total_normal_needed = patches_per_slide
        
    if len(normal_candidates) > 0:
        sampled_normal_idx = np.random.choice(len(normal_candidates), min(total_normal_needed, len(normal_candidates)), replace=False)
        sampled_normal = [normal_candidates[i] for i in sampled_normal_idx]
    else:
        sampled_normal = []
        
    # Read, normalize and save patches
    slide_id = os.path.basename(slide_path).split('.')[0]
    all_samples = [(x_z, y_z, m_np, 1) for (x_z, y_z, m_np) in sampled_tumor] + \
                  [(x_z, y_z, m_np, 0) for (x_z, y_z, m_np) in sampled_normal]
    
    for idx, (x_z, y_z, m_np, label) in enumerate(all_samples):
        # Read the high resolution patch RGB region
        patch_rgba = slide.read_region((x_z, y_z), best_level, (patch_size, patch_size))
        patch_rgb = np.array(patch_rgba.convert("RGB"))
        
        # Stain normalization
        if normalizer is not None:
            patch_rgb = normalizer.normalize(patch_rgb)
            
        # Define filenames
        img_name = f"{slide_id}_patch_{idx}_{x_z}_{y_z}.png"
        mask_name = f"{slide_id}_mask_{idx}_{x_z}_{y_z}.png"
        
        img_path = os.path.join(images_dir, img_name)
        mask_path_saved = os.path.join(masks_dir, mask_name)
        
        # If mask is None (which shouldn't happen, but just in case), create zero mask
        if m_np is None:
            m_np = np.zeros((patch_size, patch_size), dtype=np.uint8)
            
        # OpenCV expects BGR to write image
        cv2.imwrite(img_path, cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(mask_path_saved, m_np)
        
        saved_paths.append(img_path)
        saved_masks.append(mask_path_saved)
        saved_labels.append(label)
        
    slide.close()
    if mask_slide:
        mask_slide.close()
        
    print(f"Slide {slide_id}: Extracted {len(saved_paths)} patches ({len(sampled_tumor)} tumor, {len(sampled_normal)} normal).")
    return saved_paths, saved_masks, saved_labels


class HistopathologyDataset(Dataset):
    """PyTorch Dataset for patch classification.
    
    Loads patch images and returns (image_tensor, class_label).
    """
    def __init__(self, patch_paths: List[str], labels: List[int], augment: bool = True):
        self.patch_paths = patch_paths
        self.labels = labels
        
        # Setup albumentations pipeline
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
                A.OneOf([
                    A.GridDistortion(p=0.3),
                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),
                ], p=0.3),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

    def __len__(self) -> int:
        return len(self.patch_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.patch_paths[idx]
        label = self.labels[idx]
        
        # Read image
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Apply transformation
        augmented = self.transform(image=img_rgb)
        img_tensor = augmented['image']
        
        return img_tensor, label


class HistopathologySegmentationDataset(Dataset):
    """PyTorch Dataset for pixel-level tumor segmentation.
    
    Loads matching (image, mask) pairs and returns (image_tensor, mask_tensor).
    """
    def __init__(self, patch_paths: List[str], mask_paths: List[str], augment: bool = True):
        self.patch_paths = patch_paths
        self.mask_paths = mask_paths
        self.augment = augment
        
        # For segmentation, image and mask must undergo the same spatial augmentations.
        # Albumentations handles this automatically when passing mask to the transform.
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
                A.OneOf([
                    A.GridDistortion(p=0.3),
                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),
                ], p=0.3),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

    def __len__(self) -> int:
        return len(self.patch_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.patch_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Read image
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Read mask
        mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Normalize mask to binary [0.0, 1.0] and add channel dimension
        mask_bin = (mask_gray > 0).astype(np.float32)
        
        # Apply transformation
        augmented = self.transform(image=img_rgb, mask=mask_bin)
        img_tensor = augmented['image']
        mask_tensor = augmented['mask'].unsqueeze(0)  # Shape [1, H, W]
        
        return img_tensor, mask_tensor



# --- NumPy-based Macenko Stain Normalization Fallback & Simulator Classes ---

def normalize_stain_macenko(img: np.ndarray, io: float = 240.0, beta: float = 0.15, alpha: float = 1.0) -> np.ndarray:
    """NumPy implementation of Macenko Stain Normalization.
    
    Serves as a local fallback when torchstain is not installed.
    """
    img = np.array(img, dtype=np.float32)
    img_clipped = np.clip(img, 1.0, 255.0)
    od = -np.log10(img_clipped / io)
    od_flat = od.reshape(-1, 3)
    mask = np.any(od_flat > beta, axis=1)
    od_hat = od_flat[mask]
    if len(od_hat) < 100:
        return img.astype(np.uint8)
    cov = np.cov(od_hat, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    v = eigvecs[:, [2, 1]]
    proj = np.dot(od_hat, v)
    angles = np.arctan2(proj[:, 1], proj[:, 0])
    min_angle = np.percentile(angles, alpha)
    max_angle = np.percentile(angles, 100 - alpha)
    v_h = np.dot(v, np.array([np.cos(min_angle), np.sin(min_angle)]))
    v_e = np.dot(v, np.array([np.cos(max_angle), np.sin(max_angle)]))
    stain_matrix = np.array([v_e, v_h]).T if v_h[0] < v_e[0] else np.array([v_h, v_e]).T
    stain_matrix_inv = np.linalg.pinv(stain_matrix)
    concentration = np.dot(stain_matrix_inv, od_flat.T)
    max_concentration = np.percentile(concentration, 99, axis=1, keepdims=True)
    ref_stain_matrix = np.array([[0.5626, 0.2137], [0.7201, 0.8010], [0.4062, 0.5580]])
    ref_max_concentration = np.array([[1.9705], [1.0308]])
    normalized_concentration = concentration * (ref_max_concentration / (max_concentration + 1e-8))
    normalized_od = np.dot(ref_stain_matrix, normalized_concentration)
    normalized_img = io * np.power(10, -normalized_od)
    normalized_img = normalized_img.T.reshape(img.shape)
    return np.clip(normalized_img, 0.0, 255.0).astype(np.uint8)


def otsu_threshold(image_gray: np.ndarray) -> int:
    """Computes Otsu's threshold thresholding value for a grayscale image."""
    image_gray = np.clip(image_gray, 0, 255).astype(np.uint8)
    pixel_counts = np.bincount(image_gray.ravel(), minlength=256)
    pixel_probas = pixel_counts / max(len(image_gray.ravel()), 1)
    q_b = np.cumsum(pixel_probas)
    q_f = 1.0 - q_b
    valid = (q_b > 0) & (q_f > 0)
    val_range = np.arange(256)
    m_b = np.cumsum(val_range * pixel_probas) / (q_b + 1e-10)
    m_f = (np.sum(val_range * pixel_probas) - np.cumsum(val_range * pixel_probas)) / (q_f + 1e-10)
    var_between = q_b * q_f * (m_b - m_f) ** 2
    if not np.any(valid):
        return 127
    return int(np.argmax(var_between * valid))


def otsu_tissue_mask(img_rgb: np.ndarray) -> np.ndarray:
    """Computes binary tissue mask for a standard numpy RGB crop using Otsu thresholding."""
    img_gray = np.mean(img_rgb, axis=2).astype(np.uint8)
    threshold = otsu_threshold(img_gray)
    return (img_gray < threshold).astype(np.uint8)


class WSI_Simulator:
    """WSI Simulator.
    
    Generates realistic, synthetic Whole Slide Images (WSIs) to simulate
    stroma, cell nuclei, and tumor boundaries for local verification.
    """
    def __init__(self, size: int = 1024, seed: int = 42):
        self.size = size
        self.seed = seed
        np.random.seed(seed)

    def generate_slide(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generates a synthetic tissue slide and matching tumor mask."""
        bg_noise = np.random.randint(-5, 5, (self.size, self.size, 3))
        bg = np.ones((self.size, self.size, 3), dtype=np.uint8) * np.array([242, 238, 240], dtype=np.uint8)
        bg = np.clip(bg.astype(np.int16) + bg_noise, 0, 255).astype(np.uint8)
        
        grid_size = 16
        low_res = np.random.rand(grid_size, grid_size)
        low_res_img = Image.fromarray((low_res * 255).astype(np.uint8))
        tissue_probability = np.array(low_res_img.resize((self.size, self.size), Image.BILINEAR)) / 255.0
        tissue_mask = (tissue_probability > 0.4).astype(np.uint8)
        
        low_res_tumor = np.random.rand(grid_size, grid_size)
        low_res_tumor_img = Image.fromarray((low_res_tumor * 255).astype(np.uint8))
        tumor_probability = np.array(low_res_tumor_img.resize((self.size, self.size), Image.BILINEAR)) / 255.0
        tumor_mask = ((tumor_probability > 0.55) & (tissue_mask == 1)).astype(np.uint8)
        
        stroma_color = np.array([230, 185, 210])
        tissue_indices = np.where(tissue_mask == 1)
        bg[tissue_indices[0], tissue_indices[1]] = stroma_color
        
        num_nuclei = 4000
        for _ in range(num_nuclei):
            cx = np.random.randint(10, self.size - 10)
            cy = np.random.randint(10, self.size - 10)
            if tissue_mask[cy, cx] == 0:
                continue
            is_tumor_spot = tumor_mask[cy, cx] == 1
            if is_tumor_spot:
                radius = np.random.randint(5, 9)
                n_color = np.array([np.random.randint(20, 60), np.random.randint(15, 45), np.random.randint(100, 140)])
            else:
                if np.random.rand() > 0.4:
                    continue
                radius = np.random.randint(3, 5)
                n_color = np.array([np.random.randint(70, 110), np.random.randint(50, 80), np.random.randint(130, 170)])
            y, x = np.ogrid[-cy:self.size-cy, -cx:self.size-cx]
            mask = x*x + y*y <= radius*radius
            bg[mask] = n_color
            
        return bg, tumor_mask


class TissuePatchDataset(Dataset):
    """Dataset wrapper for extracting patches from simulated WSIs on-the-fly."""
    def __init__(self, images: List[np.ndarray], masks: List[np.ndarray], patch_size: int = 128, stride: int = 64, transform=None, stain_norm: bool = False):
        self.patch_size = patch_size
        self.transform = transform
        self.stain_norm = stain_norm
        self.patches = []
        self.masks = []
        
        for img, mask in zip(images, masks):
            h, w, _ = img.shape
            tissue_mask = otsu_tissue_mask(img)
            for y in range(0, h - patch_size + 1, stride):
                for x in range(0, w - patch_size + 1, stride):
                    tissue_ratio = np.mean(tissue_mask[y:y+patch_size, x:x+patch_size])
                    if tissue_ratio < 0.15:
                        continue
                    self.patches.append(img[y:y+patch_size, x:x+patch_size])
                    self.masks.append(mask[y:y+patch_size, x:x+patch_size])
        print(f"Extracted {len(self.patches)} valid patches of size {patch_size}x{patch_size}.")

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_np = self.patches[idx]
        mask_np = self.masks[idx]
        if self.stain_norm:
            try:
                img_np = normalize_stain_macenko(img_np)
            except:
                pass
        img_pil = Image.fromarray(img_np)
        mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
        
        if self.transform:
            state = torch.get_rng_state()
            img_tensor = self.transform(img_pil)
            torch.set_rng_state(state)
            
            mask_transform_list = []
            for t in self.transform.transforms:
                if isinstance(t, (transforms.RandomHorizontalFlip, transforms.RandomVerticalFlip, transforms.RandomRotation)):
                    mask_transform_list.append(t)
            mask_transform = transforms.Compose(mask_transform_list) if mask_transform_list else lambda x: x
            
            mask_pil_aug = mask_transform(mask_pil)
            mask_np_aug = np.array(mask_pil_aug)
            mask_tensor = torch.tensor(mask_np_aug > 127, dtype=torch.float32).unsqueeze(0)
        else:
            from torchvision.transforms import functional as F
            img_tensor = F.to_tensor(img_pil)
            mask_tensor = torch.tensor(mask_np > 0, dtype=torch.float32).unsqueeze(0)
            
        return img_tensor, mask_tensor



if __name__ == "__main__":
    print("Data Pipeline Module initialized successfully.")
