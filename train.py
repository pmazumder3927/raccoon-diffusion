"""
Training script for the tiny raccoon diffusion model.

Supports CUDA (RTX 4080 etc.), Apple Silicon MPS, and CPU. Includes:
    - EMA (exponential moving average) weights for cleaner samples
    - Cosine beta schedule
    - Per-epoch sample grids with a fixed seed (so you can watch raccoons emerge)
    - Loss logging to JSON for plotting
    - Mixed-precision training on CUDA for ~2x speed

Usage:
    python train.py --data_dir ./raccoon_data --epochs 300
"""

import argparse
import copy
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from tqdm import tqdm

from raccoon_diffusion.model import TinyUNet, count_parameters
from raccoon_diffusion.diffusion import GaussianDiffusion


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_transforms(img_size=64):
    """Get image transforms for training.

    Use Resize(short_side) + CenterCrop so non-square inputs do not get stretched.
    Add light color jitter for variety since the dataset is small.
    """
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [-1, 1]
    ])


class EMA:
    """Exponential moving average of model weights with warmup.

    The effective decay starts low and ramps up so the EMA tracks the live model
    quickly at first (otherwise samples stay near random init for a long time
    when the dataset is small and there are few steps per epoch).
    """

    def __init__(self, model, decay=0.9995, warmup=2000):
        self.decay = decay
        self.warmup = warmup
        self.step = 0
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    def _effective_decay(self):
        # Standard warmup formula: matches (1+n)/(10+n) shape but caps at self.decay.
        return min(self.decay, (1 + self.step) / (self.warmup + self.step))

    @torch.no_grad()
    def update(self, model):
        d = self._effective_decay()
        for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.mul_(d).add_(p.detach(), alpha=1 - d)
        for ema_b, b in zip(self.ema_model.buffers(), model.buffers()):
            ema_b.copy_(b)
        self.step += 1


def train(args):
    device = get_device()
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    checkpoints_dir = output_dir / "checkpoints"
    samples_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    transform = get_transforms(args.img_size)
    dataset = ImageFolder(args.data_dir, transform=transform)
    num_workers = 4 if device.type == "cuda" else 0
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    print(f"Dataset size: {len(dataset)} images, {len(dataloader)} batches/epoch")

    # Model
    model = TinyUNet(in_channels=3, out_channels=3, time_emb_dim=128).to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    ema = EMA(model, decay=args.ema_decay)

    # Diffusion
    diffusion = GaussianDiffusion(
        timesteps=args.timesteps,
        schedule=args.schedule,
        device=device,
    )

    # Optimizer + LR schedule
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(dataloader)
    )

    # AMP scaler (CUDA only)
    use_amp = device.type == "cuda" and args.amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Resume
    start_epoch = 0
    history = {"epoch_loss": [], "step_loss": [], "lr": []}
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "ema" in ckpt:
            ema.ema_model.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        history = ckpt.get("history", history)
        print(f"Resumed from epoch {start_epoch}")

    # Fixed seed for sample evolution -- same noise every epoch lets us see the model
    # change rather than sampling variance.
    sample_seed = 1337
    n_samples = args.n_samples

    global_step = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for images, _ in pbar:
            images = images.to(device, non_blocking=True)
            batch_size = images.shape[0]

            t = torch.randint(0, args.timesteps, (batch_size,), device=device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = diffusion.p_losses(model, images, t)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            ema.update(model)

            loss_val = loss.item()
            total_loss += loss_val
            history["step_loss"].append(loss_val)
            global_step += 1
            pbar.set_postfix({"loss": f"{loss_val:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        avg_loss = total_loss / len(dataloader)
        history["epoch_loss"].append(avg_loss)
        history["lr"].append(scheduler.get_last_lr()[0])
        print(f"Epoch {epoch + 1} - avg loss: {avg_loss:.6f}")

        # Save samples EVERY epoch using the LIVE model (so the evolution gif shows real
        # training progress). EMA model is saved separately and used for final inference.
        if (epoch + 1) % args.sample_every == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                samples = diffusion.sample_ddim(
                    model,
                    shape=(n_samples, 3, args.img_size, args.img_size),
                    seed=sample_seed,
                    steps=args.sample_steps,
                )
                samples = ((samples + 1) / 2).clamp(0, 1)
                save_image(
                    samples,
                    samples_dir / f"epoch_{epoch + 1:04d}.png",
                    nrow=int(n_samples ** 0.5),
                )
            model.train()

        # Persist history every epoch so plotting can run mid-training
        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f)

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "ema": ema.ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "args": vars(args),
            }
            torch.save(ckpt, checkpoints_dir / "latest.pt")

    # Final inference model = EMA weights (better samples)
    torch.save(ema.ema_model.state_dict(), output_dir / "raccoon_model.pt")
    # Also keep the raw final weights in case we need them
    torch.save(model.state_dict(), output_dir / "raccoon_model_raw.pt")
    print(f"Training complete. EMA model saved to {output_dir / 'raccoon_model.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train tiny raccoon diffusion model")
    parser.add_argument("--data_dir", type=str, default="./raccoon_data")
    parser.add_argument("--output_dir", type=str, default="./output_model")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--schedule", type=str, default="cosine", choices=["cosine", "linear"])
    parser.add_argument("--ema_decay", type=float, default=0.9995)
    parser.add_argument("--sample_every", type=int, default=1)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()
    train(args)
