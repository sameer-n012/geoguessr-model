import os

import torch
import torch.nn.functional as F
from cells import CellMapper, compute_residual, latlon_to_cells
from config import cfg
from dataset import CountryEncoder, GeoDataset
from torch.utils.data import DataLoader
from torchvision import transforms

from model import GeoModel


def save_ckpt(model, opt, step, path):
    torch.save(
        {"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, path
    )


def load_ckpt(model, opt, path):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    return ckpt["step"]


def contrastive_loss(embeddings, lat, lon, temp=0.07):
    """
    positives = geographically close (<25km)
    """
    B = embeddings.size(0)

    sim = embeddings @ embeddings.T

    lat = lat.unsqueeze(1)
    lon = lon.unsqueeze(1)

    dist = (lat - lat.T) ** 2 + (lon - lon.T) ** 2
    pos_mask = dist < 0.01  # ~10–30km

    logits = sim / temp

    labels = pos_mask.float()
    labels = labels / (labels.sum(1, keepdim=True) + 1e-6)

    loss = -(labels * F.log_softmax(logits, dim=1)).sum(1).mean()
    return loss


def geo_loss(pred, target, lat, lon):
    loss = 0

    loss += F.cross_entropy(pred["coarse"], target["coarse"])
    loss += 2.0 * F.cross_entropy(pred["fine"], target["fine"])
    loss += 0.5 * F.cross_entropy(pred["country"], target["country"])

    loss += 0.2 * F.smooth_l1_loss(pred["residual"], target["residual"])

    loss += 0.3 * contrastive_loss(pred["retrieval"], lat, lon)

    return loss


transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

cell_mapper = CellMapper()
country_encoder = CountryEncoder()
dataset = GeoDataset(cfg, transform, cell_mapper, country_encoder)
loader = DataLoader(dataset, batch_size=cfg.batch_size)

model = GeoModel(10000, 20000, 300).to(cfg.device)
opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

step = 0

ckpt_path = f"{cfg.ckpt_dir}/last.pt"
if os.path.exists(ckpt_path):
    step = load_ckpt(model, opt, ckpt_path)

for epoch in range(cfg.epochs):
    for batch in loader:
        imgs = batch["images"].to(cfg.device)

        # build targets
        coarse_ids, fine_ids = [], []

        for lat, lon in zip(batch["lat"], batch["lon"]):
            c, f = latlon_to_cells(lat, lon, cfg.coarse_res, cfg.fine_res)
            c_id, f_id = cell_mapper.encode(c, f)
            coarse_ids.append(c_id)
            fine_ids.append(f_id)

        coarse_ids, fine_ids = [], []
        residuals = []

        for lat, lon in zip(batch["lat"], batch["lon"]):
            c, f = latlon_to_cells(lat, lon, cfg.coarse_res, cfg.fine_res)

            c_id, f_id = cell_mapper.encode(c, f)
            coarse_ids.append(c_id)
            fine_ids.append(f_id)

            dx, dy = compute_residual(lat, lon, f)
            residuals.append([dx, dy])

        target = {
            "coarse": torch.tensor(coarse_ids).to(cfg.device),
            "fine": torch.tensor(fine_ids).to(cfg.device),
            "country": torch.tensor(batch["country_id"]).to(cfg.device),
            "residual": torch.tensor(residuals).float().to(cfg.device),
        }

        pred = model(imgs)

        lat_tensor = torch.tensor(batch["lat"]).float().to(cfg.device)
        lon_tensor = torch.tensor(batch["lon"]).float().to(cfg.device)

        loss = geo_loss(pred, target, lat_tensor, lon_tensor)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % cfg.save_every == 0:
            save_ckpt(model, opt, step, ckpt_path)

        print(f"step {step} loss {loss.item():.4f}")
        step += 1
