import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

import config


class _Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        return self.conv(torch.cat([x, skip], dim=1))


class Colorizer(nn.Module):
    def __init__(self, freeze_encoder=None):
        super().__init__()
        if freeze_encoder is None:
            freeze_encoder = config.FREEZE_ENCODER
        enc = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.in_block = nn.Sequential(enc.conv1, enc.bn1, enc.relu)
        self.maxpool = enc.maxpool
        self.layer1 = enc.layer1
        self.layer2 = enc.layer2
        self.layer3 = enc.layer3
        self.layer4 = enc.layer4
        if freeze_encoder:
            for m in (self.in_block, self.layer1, self.layer2, self.layer3, self.layer4):
                for p in m.parameters():
                    p.requires_grad = False

        self.up3 = _Up(2048, 1024, 512)
        self.up2 = _Up(512, 512, 256)
        self.up1 = _Up(256, 256, 128)
        self.up0 = _Up(128, 64, 64)
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.head = nn.Sequential(nn.Conv2d(32, 2, 1), nn.Tanh())

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, L):
        x = (L / 100.0).repeat(1, 3, 1, 1)
        x = (x - self.mean) / self.std
        s0 = self.in_block(x)
        s1 = self.layer1(self.maxpool(s0))
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        b = self.layer4(s3)
        d = self.up3(b, s3)
        d = self.up2(d, s2)
        d = self.up1(d, s1)
        d = self.up0(d, s0)
        d = torch.relu(self.final_up(d))
        return self.head(d)
