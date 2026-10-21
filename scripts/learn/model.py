"""
model.py — DispNet: ResNet over the 11-channel before/after/mask/point-map crop → 3D displacement + uncertainty.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torchvision

import data as D
from data import C_IN, G_IN

ARCHS = {"r18": (torchvision.models.resnet18, 512), "r34": (torchvision.models.resnet34, 512),
         "r50": (torchvision.models.resnet50, 2048)}
PLANAR_LOGB_Y = -5.3                                          # planar: fixed scale for dy ≈ 0 (b = 5 mm)


class DispNet(nn.Module):
    def __init__(self, pretrained=True, norm="crop", arch="r18", planar=False):
        """norm='crop': the head predicts the move in units of the crop's metric width (g[:, 4] * RES), which keeps
        the regression scale-free; norm='fixed': the head predicts metres directly (run r18).
        planar: predict only the floor-plane components (x, z of the gravity frame); dy = 0 (moves are on the
        support surface, |dy| <= 1 cm by construction)."""
        super().__init__()
        self.norm, self.planar = norm, planar
        make, feat = ARCHS[arch]
        r = make(weights="IMAGENET1K_V1" if pretrained else None)
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
        self.head = nn.Sequential(nn.Linear(feat + G_IN, 256), nn.ReLU(inplace=True), nn.Linear(256, 4 if planar else 6))

    def forward(self, x, g):
        h = torch.cat([self.backbone(x), g], 1)
        o = self.head(h)
        if self.planar:                                           # (mx, mz, bx, bz) -> 3D with dy = 0
            z = torch.zeros_like(o[:, :1])
            o = torch.cat([o[:, :1], z, o[:, 1:2], o[:, 2:3], z + PLANAR_LOGB_Y, o[:, 3:4]], 1)
        if self.norm == "crop":
            s = (g[:, 4:5] * D.RES).clamp(min=0.02)
            mu, logb = o[:, :3] * s, (o[:, 3:] + s.log()).clamp(-9, 2)
            if self.planar:
                logb = torch.cat([logb[:, :1], torch.full_like(logb[:, :1], PLANAR_LOGB_Y), logb[:, 2:]], 1)
            return mu, logb
        return o[:, :3], o[:, 3:].clamp(-7, 2)                   # delta (m), log Laplace scale


def laplace_nll(mu, logb, y, planar=False):
    t = (y - mu).abs() / logb.exp() + logb
    return (t[:, [0, 2]] if planar else t).sum(1).mean()
