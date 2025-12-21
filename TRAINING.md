# Training Your Raccoon Diffusion Model

This guide walks you through training a tiny diffusion model that generates unique raccoon images.

## Requirements

- Python 3.10+
- M4 Mac (or any machine with GPU)
- ~500+ raccoon images for training

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare training data

**Option A: Download images automatically**
```bash
pip install duckduckgo_search
python prepare_data.py --download --num_images 500
```

**Option B: Use your own images**
```bash
# Put your raccoon images in a folder, then:
python prepare_data.py --source_dir ./my_raccoons --output_dir ./raccoon_data
```

**Option C: Use a dataset from Kaggle/HuggingFace**
- Download a raccoon dataset
- Extract to a folder
- Run prepare_data.py to resize and organize

### 3. Verify your dataset

```bash
python prepare_data.py --verify --output_dir ./raccoon_data
```

You should see:
```
raccoons/: 500 images
Total: 500 images
Dataset size looks good!
```

### 4. Train the model

```bash
python train.py --data_dir ./raccoon_data --epochs 100
```

Training options:
- `--epochs 100`: Number of training epochs (100-200 recommended)
- `--batch_size 32`: Adjust based on your RAM (16 for 8GB, 64 for 32GB+)
- `--lr 1e-4`: Learning rate
- `--sample_every 10`: Generate samples every N epochs
- `--save_every 20`: Save checkpoint every N epochs

**Expected training time on M4 Mac:**
- 500 images, 100 epochs: ~1-2 hours
- 1000 images, 200 epochs: ~4-6 hours

### 5. Monitor training

Check `output_model/samples/` to see generated samples during training.
Early epochs will look like noise, but you should see raccoon-like shapes emerge around epoch 30-50.

### 6. Test generation locally

```bash
python generate_raccoon.py
```

This will generate `raccoon.png` using your commit history as the seed.

## Deploying to GitHub Actions

### 1. Upload your trained model

Create a GitHub release and upload `output_model/raccoon_model.pt`:

```bash
# Create a release and upload the model
gh release create v1.0 output_model/raccoon_model.pt --title "Raccoon Model v1.0"
```

### 2. Trigger the workflow

The workflow runs daily at midnight UTC, or you can trigger it manually:
- Go to Actions → Update Raccoon Image → Run workflow

## Troubleshooting

### "Model not found" error
Make sure `raccoon_model.pt` exists in `output_model/` or was uploaded to a GitHub release.

### Poor image quality
- Train for more epochs (try 200+)
- Use more training images (1000+)
- Check that your training images are good quality raccoon close-ups

### Out of memory
- Reduce `--batch_size` to 16 or 8
- Ensure no other heavy processes are running

### Training loss not decreasing
- Check your dataset has actual raccoon images
- Try lowering the learning rate to `1e-5`

## Model Architecture

The model is a tiny U-Net (~2.5M parameters):
- 4 resolution levels: 64 → 32 → 16 → 8 → 4
- Channel progression: 64 → 128 → 256 → 256
- Self-attention at 16x16 and 8x8 resolutions
- DDPM training with linear beta schedule
- DDIM sampling for fast inference (50 steps)

## How the Seed Works

Each day, the script:
1. Hashes all your commit SHAs
2. Combines with the current date
3. Creates a deterministic random seed

This means:
- Same day + same commit history = same raccoon
- New commits = new seed = new raccoon
- New day = new seed = new raccoon
