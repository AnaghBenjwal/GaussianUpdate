"""
model.py — DispNet: ResNet-18 over the 11-channel before/after/mask/point-map crop → 3D displacement + uncertainty.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torchvision

from data import C_IN, G_IN, RES


class DispNet(nn.Module):
    def __init__(self, pretrained=True, norm="crop"):
        """norm='crop': the head predicts the move in units of the crop's metric width (g[:, 4] * RES), which keeps
        the regression scale-free; norm='fixed': the head predicts metres directly (run r18)."""
        super().__init__()
        self.norm = norm
        r = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        w = r.conv1.weight.data                                   # (64, 3, 7, 7)
        conv = nn.Conv2d(C_IN, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            conv.weight.zero_()
            conv.weight[:, 0:3] = w                               # BEFORE RGB
            conv.weight[:, 3:6] = w                               # AFTER RGB
            conv.weight[:, 6:] = torch.randn_like(conv.weight[:, 6:]) * 0.01
        r.conv1 = conv
        r.fc = nn.Identity()
        self.backbone = r
        self.head = nn.Sequential(nn.Linear(512 + G_IN, 256), nn.ReLU(inplace=True), nn.Linear(256, 6))

    def forward(self, x, g):
        h = torch.cat([self.backbone(x), g], 1)
        o = self.head(h)
        if self.norm == "crop":
            s = (g[:, 4:5] * RES).clamp(min=0.02)
            return o[:, :3] * s, (o[:, 3:] + s.log()).clamp(-9, 2)
        return o[:, :3], o[:, 3:].clamp(-7, 2)                   # delta (m), log Laplace scale


def laplace_nll(mu, logb, y):
    return ((y - mu).abs() / logb.exp() + logb).sum(1).mean()
