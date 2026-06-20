import torch
import torch.nn as nn
from torchvision import models

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
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
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = nn.functional.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class ResNetUNet(nn.Module):
    def __init__(self, n_class=1):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        layers = list(resnet.children())
        
        self.layer0 = nn.Sequential(*layers[:3])
        self.layer0_1 = layers[3]
        self.layer1 = layers[4]
        self.layer2 = layers[5]
        self.layer3 = layers[6]
        self.layer4 = layers[7]

        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(256 + 256, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(128 + 128, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(64 + 64, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(64 + 64, 32)
        
        self.up0 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.conv_up0 = DoubleConv(32, 16)
        
        self.out_conv = nn.Conv2d(16, n_class, kernel_size=1)

    def forward(self, x):
        x0 = self.layer0(x)
        x0_1 = self.layer0_1(x0)
        x1 = self.layer1(x0_1)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        y4 = self.up4(x4)
        y4 = torch.cat([y4, x3], dim=1)
        y4 = self.conv_up4(y4)

        y3 = self.up3(y4)
        y3 = torch.cat([y3, x2], dim=1)
        y3 = self.conv_up3(y3)

        y2 = self.up2(y3)
        y2 = torch.cat([y2, x1], dim=1)
        y2 = self.conv_up2(y2)

        y1 = self.up1(y2)
        if y1.shape[2:] != x0.shape[2:]:
            x0 = nn.functional.interpolate(x0, size=y1.shape[2:], mode='bilinear', align_corners=True)
        y1 = torch.cat([y1, x0], dim=1)
        y1 = self.conv_up1(y1)

        y0 = self.up0(y1)
        y0 = self.conv_up0(y0)

        out = self.out_conv(y0)
        return out

class AttentionUNet(nn.Module):
    def __init__(self, n_class=1):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        layers = list(resnet.children())
        
        self.layer0 = nn.Sequential(*layers[:3])
        self.layer0_1 = layers[3]
        self.layer1 = layers[4]
        self.layer2 = layers[5]
        self.layer3 = layers[6]
        self.layer4 = layers[7]

        self.att4 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.att3 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.att2 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)

        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(256 + 256, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(128 + 128, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(64 + 64, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(64 + 64, 32)
        
        self.up0 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.conv_up0 = DoubleConv(32, 16)
        
        self.out_conv = nn.Conv2d(16, n_class, kernel_size=1)

    def forward(self, x):
        x0 = self.layer0(x)
        x0_1 = self.layer0_1(x0)
        x1 = self.layer1(x0_1)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        y4 = self.up4(x4)
        x3_att = self.att4(g=y4, x=x3)
        y4 = torch.cat([y4, x3_att], dim=1)
        y4 = self.conv_up4(y4)

        y3 = self.up3(y4)
        x2_att = self.att3(g=y3, x=x2)
        y3 = torch.cat([y3, x2_att], dim=1)
        y3 = self.conv_up3(y3)

        y2 = self.up2(y3)
        x1_att = self.att2(g=y2, x=x1)
        y2 = torch.cat([y2, x1_att], dim=1)
        y2 = self.conv_up2(y2)

        y1 = self.up1(y2)
        if y1.shape[2:] != x0.shape[2:]:
            x0 = nn.functional.interpolate(x0, size=y1.shape[2:], mode='bilinear', align_corners=True)
        x0_att = self.att1(g=y1, x=x0)
        y1 = torch.cat([y1, x0_att], dim=1)
        y1 = self.conv_up1(y1)

        y0 = self.up0(y1)
        y0 = self.conv_up0(y0)

        out = self.out_conv(y0)
        return out

class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    def forward(self, inputs, targets, smooth=1e-6):
        bce_loss = self.bce(inputs, targets)
        probs = torch.sigmoid(inputs)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        dice_coeff = (2.0 * intersection + smooth) / (probs_flat.sum() + targets_flat.sum() + smooth)
        dice_loss = 1.0 - dice_coeff
        return bce_loss + dice_loss
