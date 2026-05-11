"""
Generate visualizations for the raccoon-diffusion README.

Produces, under ./assets/:
    - loss_curve.png         per-step and per-epoch training loss
    - sample_evolution.gif   16-grid samples across epochs (same seed each epoch)
    - sample_evolution.png   strip of sample grids at picked epochs
    - denoising.gif          DDIM denoising trajectory (noise -> raccoon)
    - sample_grid.png        large grid of final raccoons (varied seeds)
    - banner.png             single hero raccoon

Run after training finishes (or even mid-training; loss / evolution scripts read
artifacts produced live by train.py).

Usage:
    python visualize.py --output_dir ./output_model --assets_dir ./assets
"""

import argparse
import json
import re
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.utils import make_grid

from raccoon_diffusion.diffusion import GaussianDiffusion
from raccoon_diffusion.model import TinyUNet


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def plot_loss(history_path: Path, out_path: Path):
    with open(history_path) as f:
        history = json.load(f)

    step_loss = np.array(history["step_loss"])
    epoch_loss = np.array(history["epoch_loss"])

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    # Per-step loss in light grey, downsampled if very long
    steps = np.arange(len(step_loss))
    if len(steps) > 4000:
        # Moving-average smoothing for readability
        k = max(1, len(steps) // 2000)
        smoothed = np.convolve(step_loss, np.ones(k) / k, mode="valid")
        ax.plot(steps[: len(smoothed)], smoothed, color="#6e7681", alpha=0.6,
                linewidth=0.8, label="step loss (smoothed)")
    else:
        ax.plot(steps, step_loss, color="#6e7681", alpha=0.5,
                linewidth=0.6, label="step loss")

    # Per-epoch loss on top, positioned at last step of each epoch
    if len(epoch_loss) > 0:
        steps_per_epoch = max(1, len(step_loss) // len(epoch_loss))
        epoch_x = np.arange(1, len(epoch_loss) + 1) * steps_per_epoch
        ax.plot(epoch_x, epoch_loss, color="#58a6ff", linewidth=2.2,
                label="epoch avg", marker="o", markersize=3)

    ax.set_xlabel("training step", color="#c9d1d9")
    ax.set_ylabel("MSE loss", color="#c9d1d9")
    ax.set_title("Raccoon diffusion training loss", color="#c9d1d9", fontsize=14)
    ax.tick_params(colors="#c9d1d9")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.grid(True, alpha=0.15, color="#c9d1d9")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  wrote {out_path}")


def _read_epoch_samples(samples_dir: Path):
    """Return [(epoch_int, PIL.Image), ...] sorted by epoch."""
    pattern = re.compile(r"epoch_(\d+)\.png$")
    items = []
    for p in samples_dir.glob("epoch_*.png"):
        m = pattern.search(p.name)
        if m:
            items.append((int(m.group(1)), p))
    items.sort(key=lambda x: x[0])
    return [(e, Image.open(p).convert("RGB")) for e, p in items]


def _annotate(img: Image.Image, text: str) -> Image.Image:
    """Add a small epoch label to the top-left of a sample grid."""
    from PIL import ImageDraw, ImageFont

    img = img.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    pad = 6
    bbox = draw.textbbox((pad, pad), text, font=font)
    draw.rectangle([bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2], fill=(0, 0, 0, 220))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return img


def make_evolution_gif(samples_dir: Path, out_path: Path, max_frames: int = 60,
                       upscale: int = 1, fps: int = 8):
    items = _read_epoch_samples(samples_dir)
    if not items:
        print("  no epoch samples found, skipping evolution gif")
        return

    if len(items) > max_frames:
        idx = np.linspace(0, len(items) - 1, max_frames).astype(int)
        items = [items[i] for i in idx]

    frames = []
    for epoch, img in items:
        if upscale != 1:
            img = img.resize((img.width * upscale, img.height * upscale), Image.NEAREST)
        img = _annotate(img, f"epoch {epoch}")
        frames.append(np.array(img))

    # Hold the last frame for a couple seconds
    frames.extend([frames[-1]] * fps * 2)

    # Palette-mode GIF keeps file size down (256 colors is plenty for these grids)
    imageio.mimsave(out_path, frames, fps=fps, loop=0, palettesize=128)
    print(f"  wrote {out_path} ({len(frames)} frames)")


def make_evolution_strip(samples_dir: Path, out_path: Path, n_picks: int = 6):
    """A static horizontal strip of N evenly-spaced epoch grids for the README."""
    items = _read_epoch_samples(samples_dir)
    if not items:
        return
    if len(items) < n_picks:
        picks = items
    else:
        idx = np.linspace(0, len(items) - 1, n_picks).astype(int)
        picks = [items[i] for i in idx]

    annotated = [_annotate(img, f"epoch {e}") for e, img in picks]
    w, h = annotated[0].size
    strip = Image.new("RGB", (w * len(annotated), h), color=(13, 17, 23))
    for i, img in enumerate(annotated):
        strip.paste(img, (i * w, 0))
    strip.save(out_path)
    print(f"  wrote {out_path}")


def make_denoising_gif(model_path: Path, out_path: Path, device, img_size=64,
                       steps=50, batch=8, fps: int = 12, upscale: int = 3):
    model = TinyUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    diffusion = GaussianDiffusion(timesteps=1000, schedule="cosine", device=device)

    with torch.no_grad():
        _, trajectory = diffusion.sample_ddim(
            model,
            shape=(batch, 3, img_size, img_size),
            seed=7,
            steps=steps,
            return_trajectory=True,
        )

    frames = []
    for x in trajectory:
        x = ((x + 1) / 2).clamp(0, 1)
        grid = make_grid(x, nrow=batch, padding=2, pad_value=0.05)
        arr = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img = Image.fromarray(arr)
        if upscale != 1:
            img = img.resize((img.width * upscale, img.height * upscale), Image.NEAREST)
        frames.append(np.array(img))

    # Hold final frame
    frames.extend([frames[-1]] * fps * 2)
    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    print(f"  wrote {out_path} ({len(frames)} frames)")


def make_final_grid(model_path: Path, out_path: Path, device, img_size=64,
                    n=64, seed=0, steps=100):
    model = TinyUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    diffusion = GaussianDiffusion(timesteps=1000, schedule="cosine", device=device)

    with torch.no_grad():
        x = diffusion.sample_ddim(model, shape=(n, 3, img_size, img_size),
                                  seed=seed, steps=steps)
    x = ((x + 1) / 2).clamp(0, 1).cpu()
    grid = make_grid(x, nrow=int(n ** 0.5), padding=2, pad_value=0.05)
    arr = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    # Upscale for crispness in the README
    img = img.resize((img.width * 3, img.height * 3), Image.NEAREST)
    img.save(out_path)
    print(f"  wrote {out_path}")


def make_banner(model_path: Path, out_path: Path, device, img_size=64, seed=42, steps=100):
    model = TinyUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    diffusion = GaussianDiffusion(timesteps=1000, schedule="cosine", device=device)
    with torch.no_grad():
        x = diffusion.sample_ddim(model, shape=(1, 3, img_size, img_size),
                                  seed=seed, steps=steps)
    x = ((x + 1) / 2).clamp(0, 1).cpu()
    arr = (x[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img = Image.fromarray(arr).resize((img_size * 6, img_size * 6), Image.NEAREST)
    img.save(out_path)
    print(f"  wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./output_model")
    parser.add_argument("--assets_dir", type=str, default="./assets")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Skip: loss, evolution, denoising, grid, banner")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    assets_dir = Path(args.assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    history_path = output_dir / "history.json"
    samples_dir = output_dir / "samples"
    model_path = output_dir / "raccoon_model.pt"

    if "loss" not in args.skip and history_path.exists():
        plot_loss(history_path, assets_dir / "loss_curve.png")

    if "evolution" not in args.skip and samples_dir.exists():
        make_evolution_gif(samples_dir, assets_dir / "sample_evolution.gif")
        make_evolution_strip(samples_dir, assets_dir / "sample_evolution.png")

    if "denoising" not in args.skip and model_path.exists():
        make_denoising_gif(model_path, assets_dir / "denoising.gif", device)

    if "grid" not in args.skip and model_path.exists():
        make_final_grid(model_path, assets_dir / "sample_grid.png", device)

    if "banner" not in args.skip and model_path.exists():
        make_banner(model_path, assets_dir / "banner.png", device)

    print("done.")


if __name__ == "__main__":
    main()
