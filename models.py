# %%writefile models.py
"""Models Module for Histopathology Tumor Segmentation.

This module defines the deep learning architectures used in the segmentation pipeline:
1. ResNet18 Patch Classifier: A baseline model to screen patch-level slides.
2. Standard U-Net: Pixel-level segmentation network with a ResNet18 encoder.
3. Attention U-Net: Custom segmentation network that integrates Attention Gates 
   in the skip connections to focus on tumor borders and ignore background artifacts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from typing import List, Tuple

# Try importing segmentation-models-pytorch. Fall back to custom code if missing.
try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None
    print("Warning: segmentation_models_pytorch is not installed. StandardUnet will fall back to a custom model.")


class ResNet18Classifier(nn.Module):
    """ResNet18 Baseline Classifier.
    
    Classifies 256x256 histopathology patches as either tumor (1) or normal (0).
    Uses transfer learning with ImageNet pre-trained weights.
    """
    def __init__(self, num_classes: int = 1, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet18(weights=weights)
        
        # Replace the fully connected head for binary classification
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, 3, H, W].
            
        Returns:
            Logits of shape [batch_size, 1].
        """
        return self.resnet(x)


class StandardUNet(nn.Module):
    """Standard U-Net with ResNet18 Encoder.
    
    Uses segmentation-models-pytorch to build a robust encoder-decoder structure.
    Perfect for pixel-wise tumor segmentation.
    """
    def __init__(self, in_channels: int = 3, classes: int = 1):
        super().__init__()
        if smp is not None:
            self.model = smp.Unet(
                encoder_name="resnet18",
                encoder_weights="imagenet",
                in_channels=in_channels,
                classes=classes,
                activation=None  # We output raw logits for numerical stability with BCEWithLogitsLoss
            )
        else:
            # Simple fallback U-Net implementation in case SMP library is not installed
            print("SMP not found, initializing custom standard U-Net.")
            self.model = CustomSimpleUNet(in_channels, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, 3, H, W].
            
        Returns:
            Logits of shape [batch_size, classes, H, W].
        """
        if smp is not None:
            return self.model(x)
        return self.model(x)


class CustomSimpleUNet(nn.Module):
    """A lightweight custom U-Net fallback class."""
    def __init__(self, in_channels: int = 3, classes: int = 1):
        super().__init__()
        # Encoder
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # Pooling
        self.pool = nn.MaxPool2d(2)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(128, 64)
        
        self.final_conv = nn.Conv2d(64, classes, kernel_size=1)

    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        
        # Decoder with skip connections
        d3 = self.up3(x4)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.dec1(d1)
        
        return self.final_conv(d1)


class AttentionGate(nn.Module):
    """Attention Gate for U-Net skip connections.
    
    Mechanics:
        Skip connections transfer high-resolution spatial details directly from encoder
        to decoder, but also carry background noise and non-relevant features.
        The Attention Gate utilizes a coarser gating signal (g) from the decoder to weight 
        the encoder's skip features (x).
        
        Math representation:
            theta_x = Conv1x1(x)
            phi_g = Conv1x1(g)
            psi = Conv1x1( ReLU(theta_x + phi_g) )
            alpha = Sigmoid(psi)
            output = x * alpha
            
    This highlights salient regions (e.g. tumor margins) and suppresses background features.
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        """
        Args:
            F_g: Number of feature channels in gating signal (decoder).
            F_l: Number of feature channels in skip connection (encoder).
            F_int: Intermediate number of channels for projection space.
        """
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: Decoder gating signal, shape [B, F_g, H_g, W_g].
            x: Encoder skip feature map, shape [B, F_l, H_x, W_x].
            
        Returns:
            Weighted encoder feature map, shape [B, F_l, H_x, W_x].
        """
        # Downsample or Upsample gating signal to match the dimensions of skip features
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=True)
            
        # Project skip connection (x) and gating signal (g) to the intermediate space
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Combine projected features, apply non-linearity and map to coefficients (alpha)
        psi = self.relu(g1 + x1)
        alpha = self.psi(psi)
        
        # Element-wise multiply the skip features by the attention coefficients
        return x * alpha


class ResNet18EncoderExtractor(nn.Module):
    """Helper module to extract intermediate feature maps from a ResNet18 encoder.
    
    Extracts multi-resolution skip connection layers for decoding.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        
        self.initial = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Save feature maps for U-Net skip connections
        x0 = self.initial(x)          # [B, 64, H/2, W/2]
        x1 = self.layer1(self.maxpool(x0)) # [B, 64, H/4, W/4]
        x2 = self.layer2(x1)          # [B, 128, H/8, W/8]
        x3 = self.layer3(x2)          # [B, 256, H/16, W/16]
        x4 = self.layer4(x3)          # [B, 512, H/32, W/32]
        
        return x0, x1, x2, x3, x4


class AttentionUNet(nn.Module):
    """Attention U-Net with ResNet18 Encoder.
    
    Combines the fast, pretrained feature extraction of ResNet18 with custom
    decoder layers equipped with Attention Gates. Extremely resource efficient
    on Google Colab's T4 GPU.
    """
    def __init__(self, classes: int = 1, pretrained: bool = True):
        super().__init__()
        self.encoder = ResNet18EncoderExtractor(pretrained=pretrained)
        
        # Decoder upsampling layers
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.attn4 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.dec4 = self._double_conv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.attn3 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.dec3 = self._double_conv(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.attn2 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.dec2 = self._double_conv(128, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.attn1 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.dec1 = self._double_conv(128, 64)
        
        # Final reconstruction layer mapping to classes
        self.final_conv = nn.Conv2d(64, classes, kernel_size=1)

    def _double_conv(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor, shape [B, 3, H, W] (e.g. 256x256).
            
        Returns:
            Logits of shape [B, classes, H, W].
        """
        # Encoder (extract skip connection components)
        x0, x1, x2, x3, x4 = self.encoder(x)
        
        # Decoder Block 4
        d4 = self.up4(x4)
        gated_x3 = self.attn4(g=d4, x=x3)
        d4 = torch.cat([d4, gated_x3], dim=1)
        d4 = self.dec4(d4)
        
        # Decoder Block 3
        d3 = self.up3(d4)
        gated_x2 = self.attn3(g=d3, x=x2)
        d3 = torch.cat([d3, gated_x2], dim=1)
        d3 = self.dec3(d3)
        
        # Decoder Block 2
        d2 = self.up2(d3)
        gated_x1 = self.attn2(g=d2, x=x1)
        d2 = torch.cat([d2, gated_x1], dim=1)
        d2 = self.dec2(d2)
        
        # Decoder Block 1
        d1 = self.up1(d2)
        # ResNet18's conv1 output (x0) is half input dimensions
        gated_x0 = self.attn1(g=d1, x=x0)
        d1 = torch.cat([d1, gated_x0], dim=1)
        d1 = self.dec1(d1)
        
        # Final projection layer to reconstruct original resolution
        out = self.final_conv(d1)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        return out


if __name__ == "__main__":
    # Quick sanity check with dummy input tensor
    model = AttentionUNet()
    dummy_input = torch.randn(2, 3, 256, 256)
    output = model(dummy_input)
    print("Attention U-Net check:")
    print("Input shape: ", dummy_input.shape)
    print("Output shape:", output.shape)
