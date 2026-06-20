import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

def otsu_threshold(image_gray):
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
    return np.argmax(var_between * valid)

def otsu_tissue_mask(img_rgb):
    img_gray = np.mean(img_rgb, axis=2).astype(np.uint8)
    threshold = otsu_threshold(img_gray)
    return (img_gray < threshold).astype(np.uint8)

def normalize_stain_macenko(img, io=240, beta=0.15, alpha=1):
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

class WSI_Simulator:
    def __init__(self, size=1024, seed=42):
        self.size = size
        self.seed = seed
        np.random.seed(seed)
    def generate_slide(self):
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
    def __init__(self, images, masks, patch_size=128, stride=64, transform=None, stain_norm=False):
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
    def __len__(self):
        return len(self.patches)
    def __getitem__(self, idx):
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
