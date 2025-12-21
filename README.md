# Raccoon Diffusion

A tiny diffusion model that generates unique raccoon images. Train your own model and generate raccoons seeded by your git commit history.

## Features

- **Tiny U-Net architecture** (~8M parameters) - trains quickly on consumer hardware
- **DDPM training** with linear beta schedule
- **DDIM sampling** for fast inference (50 steps)
- **Deterministic generation** - same seed produces the same raccoon

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare training data

Download raccoon images automatically:
```bash
pip install ddgs
python prepare_data.py --download --num_images 500
```

Or use your own images:
```bash
python prepare_data.py --source_dir ./my_raccoons --output_dir ./raccoon_data
```

### 3. Train the model

```bash
python train.py --data_dir ./raccoon_data --epochs 100
```

Training options:
- `--epochs 100`: Number of training epochs (100-200 recommended)
- `--batch_size 32`: Adjust based on your RAM
- `--lr 1e-4`: Learning rate
- `--sample_every 10`: Generate samples every N epochs

### 4. Generate raccoons

```bash
python generate_raccoon.py
```

This generates `raccoon.png` using your commit history as the seed.

## Model Architecture

The model is a tiny U-Net:
- 4 resolution levels: 64 -> 32 -> 16 -> 8 -> 4
- Channel progression: 64 -> 128 -> 256 -> 256
- Self-attention at 16x16 and 8x8 resolutions
- ~8M parameters

## How the Seed Works

Each generation:
1. Hashes all your commit SHAs
2. Combines with the current date
3. Creates a deterministic random seed

This means:
- Same day + same commit history = same raccoon
- New commits = new seed = new raccoon
- New day = new seed = new raccoon

## Requirements

- Python 3.10+
- PyTorch 2.0+
- ~500+ raccoon images for training

See [TRAINING.md](TRAINING.md) for detailed instructions.

## License

MIT
