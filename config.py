# %%writefile config.py
"""Configuration Module for Histopathology Tumor Segmentation.

This module defines the directory paths, hardware configurations, and 
hyperparameters tuned specifically for running on a Google Colab instance 
equipped with a 16 GB T4 GPU. It dynamically adjusts paths based on the 
execution environment (detecting Google Colab vs Local) to facilitate 
seamless training, evaluation, and inference.
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, Any

# Detect if the script is running in Google Colab
IS_COLAB = "google.colab" in sys.modules or os.path.exists("/content")

@dataclass(frozen=True)
class PipelineConfig:
    # --- Environment & Storage Paths ---
    # In Colab, we use fast local SSD storage /content/ for active patch processing,
    # and /content/drive/MyDrive/ for persistent model checkpointing and outputs.
    BASE_DIR: str = "/content" if IS_COLAB else os.path.abspath("./workspace")
    DRIVE_DIR: str = "/content/drive/MyDrive/camelyon16" if IS_COLAB else os.path.abspath("./drive_backup")
    
    # Raw Slide and Mask directories
    SLIDES_DIR: str = os.path.join(BASE_DIR, "raw_slides")
    MASKS_DIR: str = os.path.join(BASE_DIR, "raw_masks")
    
    # Fast scratch space for active patch extraction
    PATCH_DIR: str = os.path.join(BASE_DIR, "extracted_patches")
    
    # Persistent output directories
    CHECKPOINT_DIR: str = os.path.join(DRIVE_DIR, "checkpoints")
    OUTPUT_DIR: str = os.path.join(DRIVE_DIR, "outputs")
    HEATMAP_DIR: str = os.path.join(OUTPUT_DIR, "heatmaps")
    LOG_DIR: str = os.path.join(OUTPUT_DIR, "logs")

    # --- Slide Processing & Patch Extraction Settings ---
    TARGET_MAGNIFICATION: int = 20  # Magnification used for patch analysis (usually 20x)
    PATCH_SIZE: int = 256            # Size of patches in pixels (256x256)
    
    # Lower resolution level for tissue detection and mask calculations
    LOWER_RES_LEVEL: int = 4         # Level at which tissue detection is done (Otsu)
    
    # Patch sampling settings
    TISSUE_THRESHOLD: float = 0.3    # Minimum percentage of tissue required in a patch (0.0 to 1.0)
    TUMOR_THRESHOLD: float = 0.5     # Minimum percentage of tumor pixels for tumor label (0.0 to 1.0)
    PATCHES_PER_SLIDE: int = 400     # Maximum patches to extract per slide (to save disk space)
    TUMOR_PATCH_RATIO: float = 0.5   # Ratio of tumor patches in training (balance class distribution)

    # --- Stain Normalization (Macenko Method) ---
    # Reference image path for Macenko stain normalization. If not provided,
    # a default representative histopathology patch will be synthesized/used.
    REF_IMAGE_PATH: str = os.path.join(BASE_DIR, "reference_stain_image.png")

    # --- Model Training Hyperparameters ---
    # 256x256 image with U-Net architecture fits well in 16GB T4 GPU.
    # We choose batch size 16 to be conservative and prevent VRAM spikes.
    BATCH_SIZE: int = 16
    NUM_WORKERS: int = 2 if IS_COLAB else 0  # Colab limits CPU cores, avoid high worker count
    EPOCHS: int = 15
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-5
    
    # --- Hardware Settings ---
    DEVICE: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") or (hasattr(sys, "modules") and "torch" in sys.modules and sys.modules["torch"].cuda.is_available()) else "cpu"
    USE_AMP: bool = True  # Enable Automatic Mixed Precision for VRAM efficiency and speedup

    def setup_directories(self) -> None:
        """Create directories if they do not exist."""
        for path in [
            self.SLIDES_DIR,
            self.MASKS_DIR,
            self.PATCH_DIR,
            self.CHECKPOINT_DIR,
            self.OUTPUT_DIR,
            self.HEATMAP_DIR,
            self.LOG_DIR
        ]:
            os.makedirs(path, exist_ok=True)
            print(f"Directory verified: {path}")

    def get_summary(self) -> Dict[str, Any]:
        """Return config parameters as a dictionary for logging."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("__")}


# Instantiate a global config object
config = PipelineConfig()

if __name__ == "__main__":
    config.setup_directories()
    print("Pipeline configurations successfully initialized.")
    print("Device identified:", config.DEVICE)
