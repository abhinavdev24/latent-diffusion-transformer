import os
import csv
import uuid
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from apex.optimizers import FusedAdam
from model_structure import DiT, LatentDataset, cosine_alphas_bar
import wandb

# ─── Configuration ─────────────────────────────────────────────────────────────
CONFIG = {
    "data_root": "./dataset/latent",
    "checkpoint_path": "./dit_checkpoint.pt",
    "csv_log_path": "./loss_history.csv",
    "best_models": ["best1.pt", "best2.pt", "best3.pt"],
    "wandb_project": "Latent-DiT-Training",
    "batch_size": 200,
    "lr": 2e-5,
    "total_epochs": 1800,
    "timesteps": 500,
    "patch_size": 2,
    "hidden_dim": 768,
    "num_heads": 8,
    "num_layers": 10,
    "num_workers": 8,
    "pin_memory": True,
}
# ────────────────────────────────────────────────────────────────────────────────


def async_save(state: dict, path: Path):
    """Save model/optimizer state in a background thread."""
    Thread(target=torch.save, args=(state, str(path)), daemon=True).start()


def get_or_create_run_id(path: Path) -> str:
    """Load a WandB run ID if it exists, else generate and save a new one."""
    if path.exists():
        return path.read_text().strip()
    new_id = str(uuid.uuid4())
    path.write_text(new_id)
    return new_id

def build_dataloader(cfg):
    ds = LatentDataset(Path(cfg["data_root"]))
    return DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
        prefetch_factor=2,
        persistent_workers=True,
    )


def setup_wandb(cfg, run_id):
    wandb.init(
        id=run_id,
        resume="allow",
        project=cfg["wandb_project"],
        config=cfg
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def train():
    cfg = CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare directories and IDs
    ckpt_path = Path(cfg["checkpoint_path"])
    csv_path = Path(cfg["csv_log_path"])
    run_id = get_or_create_run_id(Path("wandb_run_id.txt"))

    # Data
    loader = build_dataloader(cfg)

    # Model & optimizer
    model = DiT(
        image_size=cfg["patch_size"] * int(np.sqrt(len(loader.dataset) ** 0)),
        channels_in=loader.dataset[0].shape[0],
        patch_size=cfg["patch_size"],
        hidden_size=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"]
    ).to(device)
    model = torch.compile(model, backend="inductor")
    optimizer = FusedAdam(model.parameters(), lr=cfg["lr"])
    scaler = torch.cuda.amp.GradScaler()

    # Precompute alphas for diffusion
    alphas = torch.flip(cosine_alphas_bar(cfg["timesteps"]), dims=(0,)).to(device)

    # Restore checkpoint if present
    start_epoch = 0
    best_losses = [float("inf")] * 3
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["opt_state"])
        best_losses = ckpt.get("best_losses", best_losses)
        start_epoch = ckpt.get("epoch", 0)
        print(f"> Resuming from epoch {start_epoch}, best_losses={best_losses}")
    else:
        print("> No checkpoint found, starting fresh.")

    # Initialize logs
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "avg_loss"])

    # WandB
    setup_wandb(cfg, run_id)
    wandb.watch(model, log="all", log_freq=100)
    print(f"> Model has ~{count_parameters(model)//1e6:.1f}M parameters")

    # Training loop
    for epoch in range(start_epoch, cfg["total_epochs"]):
        model.train()
        epoch_loss = 0.0

        for batch in loader:
            x = batch.to(device, non_blocking=True)
            bs = x.size(0)
            t = torch.randint(cfg["timesteps"], (bs,), device=device)
            noise = torch.randn_like(x)

            alpha = alphas[t].view(bs, 1, 1, 1)
            noisy_x = alpha.sqrt() * x + (1 - alpha).sqrt() * noise

            with torch.cuda.amp.autocast():
                pred = model(noisy_x, t)
                loss = F.l1_loss(pred, x)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_val = loss.item()
            epoch_loss += loss_val
            wandb.log({"loss_step": loss_val})

        avg_loss = epoch_loss / len(loader)
        wandb.log({"loss_epoch": avg_loss, "epoch": epoch})

        # Update top-3 checkpoints
        for i in range(3):
            if avg_loss < best_losses[i]:
                best_losses.insert(i, avg_loss)
                best_losses.pop()
                torch.save(model.state_dict(), cfg["best_models"][i])
                break

        # Save a rolling checkpoint asynchronously
        async_save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "opt_state": optimizer.state_dict(),
            "best_losses": best_losses
        }, ckpt_path)

        # Append to CSV
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, avg_loss])

        print(f"Epoch {epoch+1:04d} | Avg Loss: {avg_loss:.4f}")

    print("> Training complete.")


if __name__ == "__main__":
    train()