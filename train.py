"""
Training script for the tiny raccoon diffusion model.
Optimized for Apple Silicon (M4) with MPS support.

Usage:
    python train.py --data_dir ./raccoon_data --epochs 100
"""

import argparse
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
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_transforms(img_size=64):
    """Get image transforms for training."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # Scale to [-1, 1]
    ])


def train(args):
    # Setup
    device = get_device()
    print(f"Using device: {device}")

    # Create output directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir, "samples").mkdir(exist_ok=True)
    Path(args.output_dir, "checkpoints").mkdir(exist_ok=True)

    # Dataset
    transform = get_transforms(args.img_size)
    dataset = ImageFolder(args.data_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # MPS works better with 0 workers
        pin_memory=False,
    )
    print(f"Dataset size: {len(dataset)} images")

    # Model
    model = TinyUNet(in_channels=3, out_channels=3, time_emb_dim=128).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    # Diffusion
    diffusion = GaussianDiffusion(
        timesteps=args.timesteps,
        device=device,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(dataloader)
    )

    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(device)
            batch_size = images.shape[0]

            # Sample random timesteps
            t = torch.randint(0, args.timesteps, (batch_size,), device=device)

            # Compute loss
            loss = diffusion.p_losses(model, images, t)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.6f}")

        # Save samples
        if (epoch + 1) % args.sample_every == 0:
            model.eval()
            with torch.no_grad():
                samples = diffusion.sample_ddim(
                    model,
                    shape=(16, 3, args.img_size, args.img_size),
                    seed=42,
                    steps=50,
                )
                # Denormalize from [-1, 1] to [0, 1]
                samples = (samples + 1) / 2
                samples = samples.clamp(0, 1)
                save_image(
                    samples,
                    Path(args.output_dir, "samples", f"epoch_{epoch + 1:04d}.png"),
                    nrow=4,
                )
            print(f"Saved samples for epoch {epoch + 1}")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(
                checkpoint,
                Path(args.output_dir, "checkpoints", f"checkpoint_{epoch + 1:04d}.pt"),
            )
            # Also save as latest
            torch.save(
                checkpoint,
                Path(args.output_dir, "checkpoints", "latest.pt"),
            )
            print(f"Saved checkpoint for epoch {epoch + 1}")

    # Save final model (just weights, for inference)
    torch.save(model.state_dict(), Path(args.output_dir, "raccoon_model.pt"))
    print(f"Training complete! Model saved to {args.output_dir}/raccoon_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train tiny raccoon diffusion model")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./raccoon_data",
        help="Path to training data (should have a subfolder with images)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_model",
        help="Output directory for checkpoints and samples",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--img_size", type=int, default=64, help="Image size")
    parser.add_argument("--timesteps", type=int, default=1000, help="Diffusion timesteps")
    parser.add_argument("--sample_every", type=int, default=10, help="Sample every N epochs")
    parser.add_argument("--save_every", type=int, default=20, help="Save checkpoint every N epochs")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    args = parser.parse_args()
    train(args)
