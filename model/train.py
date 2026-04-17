import json
import os

import torch
import torch.nn.functional as F
from cells import CellMapper, compute_residual, latlon_to_cells
from config import cfg
from dataset import CountryEncoder, GeoDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from model import GeoModel


def save_ckpt(model, opt, step, path, training_data, training_data_path):
    with open(training_data_path, "w") as f:
        json.dump(training_data, f)
    torch.save(
        {"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, path
    )


def load_ckpt(model, opt, path, training_data_path):
    with open(training_data_path, "r") as f:
        training_data = json.load(f)

    ckpt = torch.load(path)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    return ckpt["step"], training_data


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


def collate_variable_views(batch):
    # Pad images to max views in this batch and build a mask.
    max_views = max(x["images"].shape[0] for x in batch)
    C, H, W = batch[0]["images"].shape[1:]
    images = torch.zeros(len(batch), max_views, C, H, W, dtype=batch[0]["images"].dtype)
    view_mask = torch.zeros(len(batch), max_views, dtype=torch.float32)

    out = {
        "images": images,
        "view_mask": view_mask,
        "lat": [],
        "lon": [],
        "country_id": [],
    }

    for i, x in enumerate(batch):
        n = x["images"].shape[0]
        images[i, :n] = x["images"]
        view_mask[i, :n] = 1.0
        out["lat"].append(x["lat"])
        out["lon"].append(x["lon"])
        out["country_id"].append(x["country_id"])

    return out


loader = DataLoader(
    dataset,
    batch_size=cfg.batch_size,
    collate_fn=collate_variable_views,
    pin_memory=cfg.pin_memory,
    num_workers=cfg.num_workers,
    persistent_workers=cfg.persist_workers,
)

num_coarse, num_fine, num_country = 10000, 20000, 300
model = GeoModel(num_coarse, num_fine, num_country).to(cfg.device)
# model = torch.compile(model)
opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

training_data = {
    "device": cfg.device,
    "model_name": cfg.model_name,
    "coarse_res": cfg.coarse_res,
    "fine_res": cfg.fine_res,
    "batch_size": cfg.batch_size,
    "lr": cfg.lr,
    "image_size": cfg.image_size,
    "num_workers": cfg.num_workers,
    "epochs": cfg.epochs,
    "checkpoint_every": cfg.save_every,
    "num_coarse": num_coarse,
    "num_fine": num_fine,
    "num_country": num_country,
    "training_log": [],
}

step, start_step = 0, 0

ckpt_path = f"{cfg.ckpt_dir}/last.pt"
if os.path.exists(ckpt_path):
    start_step, training_data = load_ckpt(model, opt, ckpt_path, cfg.training_data_path)


scaler = torch.cuda.amp.GradScaler()

for epoch in range(cfg.epochs):
    # pbar = tqdm(total=len(loader), desc=f"Epoch {epoch + 1}/{cfg.epochs}", unit="batch")
    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}", unit="batch")

    for batch in pbar:
        while step < start_step:
            step += 1
            continue

        imgs = batch["images"].to(cfg.device, non_blocking=True)
        view_mask = batch["view_mask"].to(cfg.device, non_blocking=True)

        coarse_ids, fine_ids, residuals = [], [], []

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

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            pred = model(imgs, view_mask=view_mask)

            lat_tensor = torch.tensor(batch["lat"]).float().to(cfg.device)
            lon_tensor = torch.tensor(batch["lon"]).float().to(cfg.device)

            loss = geo_loss(pred, target, lat_tensor, lon_tensor)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        training_data["training_log"].append({"step": step, "loss": loss.item()})

        if step % cfg.save_every == 0:
            tqdm.write(f"Saving checkpoint at step {step} with loss {loss.item():.4f}")
            save_ckpt(
                model, opt, step, ckpt_path, training_data, cfg.training_data_path
            )

        # print(f"step {step} loss {loss.item():.4f}")
        pbar.set_postfix(loss=f"{loss.item():.4f}", step=step)
        step += 1
