"""Command-Line Interface (CLI) for Histopathology Tumor Segmentation.

This script coordinates patch extraction, model training, and evaluation.
To help developers and users verify the environment, it features an automatic
data simulation pipeline that generates synthetic gigapixel-like tissue slides
with simulated stroma, nuclei, and tumor clusters.
"""

import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from PIL import Image

# Import configurations and modules
from config import config
from data_pipeline import WSI_Simulator, extract_balanced_patches, MacenkoNormalizer
from models import StandardUNet, AttentionUNet
from train import run_training_pipeline
from evaluate import (
    reconstruct_heatmap, 
    run_ablation_study, 
    run_froc_curve, 
    evaluate_single_crop
)


def simulate_dataset(num_slides: int = 3, size: int = 1024, patch_size: int = 128):
    """Generates synthetic slide and mask TIFF/PNG files to simulate a real WSI dataset.
    
    This is highly useful for dry-runs and automated verification in CPU/GPU environments.
    """
    print(f"--- Simulating WSI Dataset ({num_slides} slides, size {size}x{size}) ---")
    config.setup_directories()
    
    # Generate and save slides
    for idx in range(num_slides):
        seed = 100 + idx
        print(f"Simulating slide {idx+1}/{num_slides} with seed {seed}...")
        sim = WSI_Simulator(size=size, seed=seed)
        slide, mask = sim.generate_slide()
        
        # Save mock slides (OpenCV writes BGR)
        slide_name = f"slide_{seed}.tif"
        mask_name = f"slide_{seed}_mask.png"
        
        slide_path = os.path.join(config.SLIDES_DIR, slide_name)
        mask_path = os.path.join(config.MASKS_DIR, mask_name)
        
        cv2.imwrite(slide_path, cv2.cvtColor(slide, cv2.COLOR_RGB2BGR))
        cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
        
        print(f" -> Saved WSI slide to: {slide_path}")
        print(f" -> Saved WSI mask to: {mask_path}")
        
    print("Simulation dataset generated successfully.")


def extract_patches_pipeline(patch_size: int = 128):
    """Iterates through raw slide files, extracts tissue patches,
    and balances the dataset for training.
    """
    print("--- Extracting Tissue Patches ---")
    config.setup_directories()
    
    # Glob slides
    slide_paths = sorted([os.path.join(config.SLIDES_DIR, f) for f in os.listdir(config.SLIDES_DIR) if f.endswith(('.tif', '.tiff', '.svs'))])
    
    if not slide_paths:
        print("Error: No raw slides found in", config.SLIDES_DIR)
        print("Please run in 'simulate' mode first or populate slides directory.")
        return
        
    normalizer = MacenkoNormalizer()
    
    all_patches = []
    all_labels = []
    
    for slide_path in slide_paths:
        slide_id = os.path.basename(slide_path).split('.')[0]
        # Look for matching mask
        mask_path = os.path.join(config.MASKS_DIR, f"{slide_id}_mask.png")
        if not os.path.exists(mask_path):
            mask_path = None
            
        print(f"Processing slide: {slide_id}")
        paths, masks, labels = extract_balanced_patches(
            slide_path=slide_path,
            mask_path=mask_path,
            output_dir=config.PATCH_DIR,
            patch_size=patch_size,
            patches_per_slide=config.PATCHES_PER_SLIDE,
            tumor_ratio=config.TUMOR_PATCH_RATIO,
            tissue_threshold=config.TISSUE_THRESHOLD,
            tumor_threshold=config.TUMOR_THRESHOLD,
            normalizer=normalizer
        )
        all_patches.extend(paths)
        all_labels.extend(labels)
        
    print(f"Extraction Pipeline Complete. Total patches extracted: {len(all_patches)}")


def main():
    parser = argparse.ArgumentParser(description="Histopathology Tumor Detection Pipeline CLI")
    parser.add_argument(
        "--mode", 
        choices=["all", "simulate", "extract", "train", "evaluate"], 
        default="all", 
        help="Pipeline execution mode. 'all' runs simulate -> extract -> train -> evaluate."
    )
    parser.add_argument(
        "--model_type", 
        choices=["standard_unet", "attention_unet"], 
        default="attention_unet", 
        help="Type of U-Net architecture to train or evaluate."
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=None, 
        help="Number of training epochs (overrides config hyperparameters)."
    )
    parser.add_argument(
        "--slides_count", 
        type=int, 
        default=3, 
        help="Number of mock slides to generate in simulation mode."
    )
    parser.add_argument(
        "--patch_size", 
        type=int, 
        default=128, 
        help="Patch extraction dimensions in pixels."
    )
    
    args = parser.parse_args()
    
    # Dynamic hyperparameter override
    if args.epochs is not None:
        # Patch config values dynamically (since PipelineConfig class is frozen, we do it via module patching if required,
        # or handle inside individual methods. But we can modify global config fields by setting it)
        # Note: config is a dataclass object. We can modify its fields if it is not frozen, but frozen=True is set.
        # We can bypass frozen dataclass restrictions using object.__setattr__
        object.__setattr__(config, "EPOCHS", args.epochs)
        
    object.__setattr__(config, "PATCH_SIZE", args.patch_size)
    
    print("-----------------------------------------------------------------")
    print(f"Running Histopathology Pipeline | Mode: {args.mode} | Model: {args.model_type}")
    print(f"Device Identified: {config.DEVICE}")
    print("-----------------------------------------------------------------")
    
    if args.mode == "simulate":
        simulate_dataset(num_slides=args.slides_count, patch_size=args.patch_size)
        
    elif args.mode == "extract":
        extract_patches_pipeline(patch_size=args.patch_size)
        
    elif args.mode == "train":
        run_training_pipeline(model_type=args.model_type)
        
    elif args.mode == "evaluate":
        # Run evaluation checks
        device = config.DEVICE
        
        # Load best checkpoint if it exists
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"{args.model_type}_best.pth")
        
        print(f"Initializing {args.model_type} for evaluation...")
        if args.model_type == "standard_unet":
            model = StandardUNet()
        else:
            model = AttentionUNet()
            
        if os.path.exists(checkpoint_path):
            print(f"Loading model checkpoint from {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            print("⚠️ Checkpoint file not found. Running evaluations with randomly initialized weights.")
            
        model = model.to(device)
        
        # Setup output paths for evaluation plots
        config.setup_directories()
        ablation_plot = os.path.join(config.OUTPUT_DIR, "stain_normalization_ablation.png")
        froc_plot = os.path.join(config.OUTPUT_DIR, "froc_curve.png")
        crop_plot = os.path.join(config.OUTPUT_DIR, "slide_crop_evaluation.png")
        heatmap_plot = os.path.join(config.HEATMAP_DIR, "slide_heatmap.png")
        
        # Simulate a slide for evaluation tasks
        print("Generating a test slide for evaluation...")
        eval_sim = WSI_Simulator(size=1024, seed=456)
        eval_slide, eval_mask = eval_sim.generate_slide()
        
        # 1. Reconstruct WSI Heatmap
        print("1. Reconstructing whole-slide tumor probability heatmap...")
        heatmap = reconstruct_heatmap(eval_slide, model, device, patch_size=args.patch_size, stride=64)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(eval_slide)
        axes[0].set_title("Test WSI Slide")
        axes[0].axis("off")
        axes[1].imshow(eval_mask, cmap="gray")
        axes[1].set_title("Ground Truth Tumor Mask")
        axes[1].axis("off")
        im = axes[2].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
        axes[2].set_title("Reconstructed Probability Heatmap")
        axes[2].axis("off")
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(heatmap_plot, dpi=150)
        print(f" -> Heatmap stitched plot saved to: {heatmap_plot}")
        plt.close()
        
        # 2. Stain Normalization Ablation Study
        print("2. Running stain normalization ablation study...")
        run_ablation_study(eval_slide, eval_mask, model, device, output_path=ablation_plot)
        
        # 3. FROC Curve
        print("3. Computing Free-Response ROC (FROC)...")
        run_froc_curve(eval_slide, eval_mask, model, device, output_path=froc_plot)
        
        # 4. Crop prediction visualization
        print("4. Visualizing crop-level prediction overlay...")
        evaluate_single_crop(model, device, slide_seed=777, output_path=crop_plot)
        
        print("Evaluation Pipeline completed. Check outputs directory for plots.")
        
    elif args.mode == "all":
        # Complete pipeline
        print("--- Execution Step 1/4: Generating Simulated Data ---")
        simulate_dataset(num_slides=args.slides_count, patch_size=args.patch_size)
        
        print("\n--- Execution Step 2/4: Extracting and Balancing Patches ---")
        extract_patches_pipeline(patch_size=args.patch_size)
        
        print("\n--- Execution Step 3/4: Training Model ---")
        run_training_pipeline(model_type=args.model_type)
        
        # Checkpoint would be written, now evaluate it
        print("\n--- Execution Step 4/4: Evaluating Trained Model ---")
        device = config.DEVICE
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"{args.model_type}_best.pth")
        
        if args.model_type == "standard_unet":
            model = StandardUNet()
        else:
            model = AttentionUNet()
            
        if os.path.exists(checkpoint_path):
            print(f"Loading trained weights from {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            print("Warning: Could not find checkpoint to evaluate, using random weights.")
            
        model = model.to(device)
        
        config.setup_directories()
        ablation_plot = os.path.join(config.OUTPUT_DIR, "stain_normalization_ablation.png")
        froc_plot = os.path.join(config.OUTPUT_DIR, "froc_curve.png")
        crop_plot = os.path.join(config.OUTPUT_DIR, "slide_crop_evaluation.png")
        heatmap_plot = os.path.join(config.HEATMAP_DIR, "slide_heatmap.png")
        
        eval_sim = WSI_Simulator(size=1024, seed=456)
        eval_slide, eval_mask = eval_sim.generate_slide()
        
        # Reconstruct WSI Heatmap
        heatmap = reconstruct_heatmap(eval_slide, model, device, patch_size=args.patch_size, stride=64)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(eval_slide)
        axes[0].set_title("Test WSI Slide")
        axes[0].axis("off")
        axes[1].imshow(eval_mask, cmap="gray")
        axes[1].set_title("Ground Truth Tumor Mask")
        axes[1].axis("off")
        im = axes[2].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
        axes[2].set_title("Reconstructed Probability Heatmap")
        axes[2].axis("off")
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(heatmap_plot, dpi=150)
        plt.close()
        print(f"Heatmap stitched plot saved to: {heatmap_plot}")
        
        # Stain Normalization Ablation Study
        run_ablation_study(eval_slide, eval_mask, model, device, output_path=ablation_plot)
        
        # FROC Curve
        run_froc_curve(eval_slide, eval_mask, model, device, output_path=froc_plot)
        
        # Crop prediction visualization
        evaluate_single_crop(model, device, slide_seed=777, output_path=crop_plot)
        
        print("\nAll pipeline execution steps successfully finalized!")


if __name__ == "__main__":
    main()
