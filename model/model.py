import os

import timm
import torch
import torch.nn as nn


class GeoModel(nn.Module):
    def __init__(self, num_coarse, num_fine, num_country):
        super().__init__()

        self.encoder = timm.create_model(
            "vit_base_patch16_224", pretrained=True, num_classes=0
        )

        dim = self.encoder.num_features

        self.gate = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())

        self.coarse = nn.Linear(dim, num_coarse)
        self.fine = nn.Linear(dim, num_fine)
        self.country = nn.Linear(dim, num_country)
        self.residual = nn.Linear(dim, 2)
        self.retrieval = nn.Linear(dim, 256)

    def forward(self, images, view_mask=None):
        B, N, C, H, W = images.shape

        images = images.view(B * N, C, H, W)
        feats = self.encoder(images)
        feats = feats.view(B, N, -1)

        weights = self.gate(feats)
        if view_mask is not None:
            # view_mask: [B, N] with 1 for valid views, 0 for padding
            mask = view_mask.unsqueeze(-1).to(weights.dtype)
            weights = weights * mask
            denom = weights.sum(1) + 1e-6
            z = (feats * weights).sum(1) / denom
        else:
            z = (feats * weights).sum(1) / (weights.sum(1) + 1e-6)

        z = self.head(z)

        retr = nn.functional.normalize(self.retrieval(z), dim=-1)

        return {
            "coarse": self.coarse(z),
            "fine": self.fine(z),
            "country": self.country(z),
            "residual": self.residual(z),
            "retrieval": retr,
        }
