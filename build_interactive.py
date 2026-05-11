"""
Build the interactive blog visualizations.

Outputs into ./assets/interactive/:
    trajectory.json   — 50 DDIM step frames (one continuous denoising) as base64 PNGs
    epoch_grids.json  — saved sample grid per epoch (training timeline)
    seed_grid.json    — 64 different seeds, one raccoon each (seed explorer)
    interp.json       — 32-frame slerp between two seeds (latent interpolation)
    *.html            — standalone, single-file viewers that the blog can iframe

You may embed these in a Next.js / static site by copying the .html files into
/public, or pull the JSONs and write your own React components.

Run after training:
    python build_interactive.py
"""

import argparse
import base64
import io
import json
import re
from pathlib import Path

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


def tensor_to_png_b64(x: torch.Tensor, scale: int = 4) -> str:
    """3xHxW [-1,1] tensor → upscaled base64 PNG string."""
    img = ((x + 1) / 2).clamp(0, 1)
    arr = (img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    pil = Image.fromarray(arr)
    if scale != 1:
        pil = pil.resize((pil.width * scale, pil.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def grid_to_png_b64(x: torch.Tensor, nrow: int, scale: int = 1) -> str:
    """BxCxHxW [-1,1] → upscaled base64 PNG of a grid."""
    img = ((x + 1) / 2).clamp(0, 1).cpu()
    grid = make_grid(img, nrow=nrow, padding=2, pad_value=0.05)
    arr = (grid.permute(1, 2, 0).numpy() * 255).astype("uint8")
    pil = Image.fromarray(arr)
    if scale != 1:
        pil = pil.resize((pil.width * scale, pil.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_trajectory(model, diffusion, out_dir: Path, img_size=64, steps=50, seed=7):
    """A single DDIM trajectory, 50 steps from pure noise to raccoon."""
    with torch.no_grad():
        _, traj = diffusion.sample_ddim(
            model, shape=(1, 3, img_size, img_size),
            seed=seed, steps=steps, return_trajectory=True,
        )

    frames = [tensor_to_png_b64(x[0], scale=4) for x in traj]
    data = {
        "img_size": img_size,
        "scale": 4,
        "steps": steps,
        "seed": seed,
        "schedule": "cosine",
        "frames": frames,
    }
    (out_dir / "trajectory.json").write_text(json.dumps(data))
    print(f"  trajectory.json: {len(frames)} frames")


def build_seed_grid(model, diffusion, out_dir: Path, img_size=64, n=64, steps=100):
    """Render N independent raccoons, one per seed."""
    entries = []
    # batch them for speed
    batch_size = 16
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        bsz = end - start
        # Manually seed each so each entry is reproducible from its own seed
        all_samples = []
        for s in range(start, end):
            with torch.no_grad():
                x = diffusion.sample_ddim(
                    model, shape=(1, 3, img_size, img_size), seed=s, steps=steps
                )
            all_samples.append(x[0])
        for s, x in zip(range(start, end), all_samples):
            entries.append({"seed": s, "png": tensor_to_png_b64(x, scale=3)})
        print(f"  seed batch {end}/{n}")

    data = {"img_size": img_size, "scale": 3, "steps": steps, "items": entries}
    (out_dir / "seed_grid.json").write_text(json.dumps(data))
    print(f"  seed_grid.json: {len(entries)} raccoons")


def build_interpolation(model, diffusion, out_dir: Path, img_size=64, n_frames=32,
                        steps=100, seed_a=3, seed_b=42):
    """Slerp two noise tensors and decode each blend with DDIM."""
    g_a = torch.Generator(device="cpu").manual_seed(seed_a)
    g_b = torch.Generator(device="cpu").manual_seed(seed_b)
    z_a = torch.randn(1, 3, img_size, img_size, generator=g_a)
    z_b = torch.randn(1, 3, img_size, img_size, generator=g_b)

    device = next(model.parameters()).device

    def slerp(t, a, b):
        a_flat = a.flatten()
        b_flat = b.flatten()
        omega = torch.acos((a_flat * b_flat).sum() /
                            (a_flat.norm() * b_flat.norm() + 1e-8))
        so = torch.sin(omega)
        if so.abs() < 1e-6:
            return (1 - t) * a + t * b
        return (torch.sin((1 - t) * omega) / so) * a + (torch.sin(t * omega) / so) * b

    frames = []
    ts = torch.linspace(0, 1, n_frames)
    for t in ts:
        z = slerp(t.item(), z_a, z_b).to(device)
        # Run DDIM starting from this exact noise tensor
        # (re-implement quickly here so we control the initial noise)
        x = z.clone()
        step_size = diffusion.timesteps // steps
        timesteps = list(range(0, diffusion.timesteps, step_size))[::-1]
        with torch.no_grad():
            for i, ti in enumerate(timesteps):
                t_batch = torch.full((1,), ti, device=device, dtype=torch.long)
                eps = model(x, t_batch.float())
                ac = diffusion.alphas_cumprod[ti]
                ac_prev = (diffusion.alphas_cumprod[timesteps[i + 1]]
                           if i + 1 < len(timesteps)
                           else torch.tensor(1.0, device=device))
                x0 = ((x - torch.sqrt(1 - ac) * eps) / torch.sqrt(ac)).clamp(-1, 1)
                x = torch.sqrt(ac_prev) * x0 + torch.sqrt(1 - ac_prev) * eps
        frames.append({"t": float(t), "png": tensor_to_png_b64(x[0], scale=4)})

    data = {
        "img_size": img_size, "scale": 4,
        "seed_a": seed_a, "seed_b": seed_b,
        "frames": frames,
    }
    (out_dir / "interp.json").write_text(json.dumps(data))
    print(f"  interp.json: {len(frames)} frames")


def build_epoch_grids(samples_dir: Path, out_dir: Path):
    """Dump every saved epoch grid as base64 (no inference needed)."""
    pattern = re.compile(r"epoch_(\d+)\.png$")
    entries = []
    for p in sorted(samples_dir.glob("epoch_*.png")):
        m = pattern.search(p.name)
        if not m:
            continue
        epoch = int(m.group(1))
        png_bytes = p.read_bytes()
        entries.append({
            "epoch": epoch,
            "png": base64.b64encode(png_bytes).decode("ascii"),
        })
    data = {"items": entries}
    (out_dir / "epoch_grids.json").write_text(json.dumps(data))
    print(f"  epoch_grids.json: {len(entries)} epochs")


# ---- HTML viewers --------------------------------------------------------

VIEWER_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Instrument+Serif&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg: #0a0a0a;            /* charcoal-black */
  --surface: #141414;
  --border: rgba(255,255,255,0.08);
  --text: rgba(255,255,255,0.9);
  --muted: rgba(255,255,255,0.5);
  --accent: #ff6b3d;        /* accent-orange */
  --accent-soft: rgba(255,107,61,0.18);
  --sans: 'Inter', system-ui, -apple-system, sans-serif;
  --serif: 'Instrument Serif', ui-serif, Georgia, serif;
  --mono: 'JetBrains Mono', 'SF Mono', Monaco, Consolas, monospace;
}

* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text);
       font-family: var(--sans); font-size: 14px;
       padding: 24px;
       display: flex; flex-direction: column; align-items: center; gap: 16px; }

h2 { font-family: var(--serif); font-weight: 400;
     font-size: clamp(24px, 4vw, 34px);
     letter-spacing: -0.01em; margin: 0; color: var(--text); }

canvas, img.frame { image-rendering: pixelated; border-radius: 8px;
                    background: #050505;
                    box-shadow: 0 1px 0 rgba(255,255,255,0.04) inset,
                                0 20px 40px -16px rgba(0,0,0,0.6); }

.row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
       justify-content: center; }

.meta { font-family: var(--mono); font-size: 12px;
        color: var(--muted); letter-spacing: 0.01em; }
.meta b, .meta strong { color: var(--text); font-weight: 500; }

button { background: transparent; border: 1px solid var(--border); color: var(--text);
         padding: 7px 14px; border-radius: 999px; cursor: pointer;
         font: 500 13px var(--sans); letter-spacing: 0.01em;
         transition: 0.18s ease; }
button:hover:not(:disabled) { border-color: var(--accent);
                              color: var(--accent); background: var(--accent-soft); }
button:disabled { opacity: 0.4; cursor: default; }

input[type=range] { -webkit-appearance: none; appearance: none;
                    width: min(420px, 86vw); height: 22px;
                    background: transparent; accent-color: var(--accent); }
input[type=range]::-webkit-slider-runnable-track {
  height: 2px; background: var(--border); border-radius: 999px;
}
input[type=range]::-moz-range-track {
  height: 2px; background: var(--border); border-radius: 999px;
}
input[type=range]::-webkit-slider-thumb {
  -webkit-appearance: none; appearance: none;
  width: 14px; height: 14px; border-radius: 999px;
  background: var(--accent); margin-top: -6px;
  box-shadow: 0 0 0 4px rgba(255,107,61,0.18);
}
input[type=range]::-moz-range-thumb {
  width: 14px; height: 14px; border: none; border-radius: 999px;
  background: var(--accent);
  box-shadow: 0 0 0 4px rgba(255,107,61,0.18);
}

input[type=number] { background: transparent; border: 1px solid var(--border);
                     color: var(--text); padding: 5px 10px; border-radius: 6px;
                     font: 500 13px var(--mono); width: 110px; }
input[type=number]:focus { outline: none; border-color: var(--accent); }

.grid { display: grid; gap: 4px; grid-template-columns: repeat(8, 1fr); }
.grid img { width: 100%; cursor: pointer; image-rendering: pixelated;
            border-radius: 4px; border: 1.5px solid transparent;
            transition: 0.15s ease; opacity: 0.92; }
.grid img:hover { border-color: var(--accent); transform: translateY(-1px);
                  opacity: 1; }
.grid img.selected { border-color: var(--accent);
                     box-shadow: 0 0 0 2px rgba(255,107,61,0.35);
                     opacity: 1; }

a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; text-decoration-color: rgba(255,107,61,0.5); }
ul { padding-left: 20px; }
li { margin: 6px 0; }
"""


def write_html(out_dir: Path, name: str, title: str, body: str, script: str):
    html = f"""<!doctype html><html><head>
<meta charset='utf-8'><title>{title}</title>
<style>{VIEWER_CSS}</style></head><body>
<h2>{title}</h2>
{body}
<script>{script}</script>
</body></html>"""
    (out_dir / name).write_text(html)


def write_viewers(out_dir: Path):
    # Trajectory scrubber
    write_html(out_dir, "trajectory.html", "denoising trajectory",
        """
        <img id='frame' class='frame' width='256' height='256'/>
        <input id='slider' type='range' min='0' max='0' value='0'/>
        <div class='row meta'>
          <span>step <span id='step'>0</span> / <span id='total'>0</span></span>
          <button id='play' class='primary'>play</button>
          <button id='reset'>reset</button>
        </div>
        """,
        """
        (async () => {
          const data = await fetch('trajectory.json').then(r => r.json());
          const img = document.getElementById('frame');
          const slider = document.getElementById('slider');
          const stepEl = document.getElementById('step');
          const totalEl = document.getElementById('total');
          slider.max = data.frames.length - 1;
          totalEl.textContent = data.frames.length - 1;
          const render = i => {
            img.src = 'data:image/png;base64,' + data.frames[i];
            stepEl.textContent = i;
            slider.value = i;
          };
          render(0);
          slider.oninput = e => render(+e.target.value);
          let playTimer = null;
          document.getElementById('play').onclick = () => {
            if (playTimer) { clearInterval(playTimer); playTimer = null;
                              document.getElementById('play').textContent = 'play'; return; }
            document.getElementById('play').textContent = 'pause';
            let i = +slider.value;
            playTimer = setInterval(() => {
              i = (i + 1) % data.frames.length;
              render(i);
            }, 80);
          };
          document.getElementById('reset').onclick = () => render(0);
        })();
        """)

    # Epoch scrubber
    write_html(out_dir, "epochs.html", "training timeline",
        """
        <img id='frame' class='frame' width='384' height='384'/>
        <input id='slider' type='range' min='0' max='0' value='0'/>
        <div class='row meta'>
          <span>epoch <span id='epoch'>0</span></span>
          <button id='play' class='primary'>play</button>
        </div>
        """,
        """
        (async () => {
          const data = await fetch('epoch_grids.json').then(r => r.json());
          const items = data.items;
          const img = document.getElementById('frame');
          const slider = document.getElementById('slider');
          const epochEl = document.getElementById('epoch');
          slider.max = items.length - 1;
          const render = i => {
            img.src = 'data:image/png;base64,' + items[i].png;
            epochEl.textContent = items[i].epoch;
            slider.value = i;
          };
          render(items.length - 1);
          slider.value = items.length - 1;
          slider.oninput = e => render(+e.target.value);
          let playTimer = null;
          document.getElementById('play').onclick = () => {
            if (playTimer) { clearInterval(playTimer); playTimer = null;
                              document.getElementById('play').textContent = 'play'; return; }
            document.getElementById('play').textContent = 'pause';
            let i = 0; render(i);
            playTimer = setInterval(() => {
              i = (i + 1) % items.length;
              render(i);
              if (i === items.length - 1) {
                clearInterval(playTimer); playTimer = null;
                document.getElementById('play').textContent = 'play';
              }
            }, 100);
          };
        })();
        """)

    # Seed explorer
    write_html(out_dir, "seeds.html", "seed explorer",
        """
        <img id='zoom' class='frame' width='256' height='256'/>
        <div class='meta'>seed: <span id='seed'>0</span> · click a thumbnail to zoom</div>
        <div id='grid' class='grid' style='max-width: 640px;'></div>
        """,
        """
        (async () => {
          const data = await fetch('seed_grid.json').then(r => r.json());
          const grid = document.getElementById('grid');
          const zoom = document.getElementById('zoom');
          const seedEl = document.getElementById('seed');
          let selected = null;
          data.items.forEach(item => {
            const im = document.createElement('img');
            im.src = 'data:image/png;base64,' + item.png;
            im.title = 'seed ' + item.seed;
            im.onclick = () => {
              if (selected) selected.classList.remove('selected');
              im.classList.add('selected'); selected = im;
              zoom.src = im.src; seedEl.textContent = item.seed;
            };
            grid.appendChild(im);
          });
          if (data.items.length) grid.children[0].click();
        })();
        """)

    # Interpolation
    write_html(out_dir, "interp.html", "latent interpolation",
        """
        <img id='frame' class='frame' width='320' height='320'/>
        <input id='slider' type='range' min='0' max='0' value='0' step='1'/>
        <div class='meta'>t = <span id='t'>0.00</span>
                          · seeds <span id='sa'></span> ↔ <span id='sb'></span></div>
        """,
        """
        (async () => {
          const data = await fetch('interp.json').then(r => r.json());
          const img = document.getElementById('frame');
          const slider = document.getElementById('slider');
          const tEl = document.getElementById('t');
          document.getElementById('sa').textContent = data.seed_a;
          document.getElementById('sb').textContent = data.seed_b;
          slider.max = data.frames.length - 1;
          const render = i => {
            img.src = 'data:image/png;base64,' + data.frames[i].png;
            tEl.textContent = data.frames[i].t.toFixed(2);
            slider.value = i;
          };
          render(0);
          slider.oninput = e => render(+e.target.value);
        })();
        """)

    # Index page that links everything
    (out_dir / "index.html").write_text(f"""<!doctype html><html><head>
<meta charset='utf-8'><title>raccoon diffusion · interactive</title>
<style>{VIEWER_CSS}</style></head><body>
<h2>raccoon diffusion</h2>
<div class='meta'>interactive demos &nbsp;·&nbsp; standalone html, ready to iframe</div>
<ul>
  <li><a href='trajectory.html'>denoising trajectory</a> &nbsp;<span class='meta'>scrub 50 ddim steps</span></li>
  <li><a href='epochs.html'>training timeline</a> &nbsp;<span class='meta'>watch raccoons emerge</span></li>
  <li><a href='seeds.html'>seed explorer</a> &nbsp;<span class='meta'>64 raccoons, 64 seeds</span></li>
  <li><a href='interp.html'>latent interpolation</a> &nbsp;<span class='meta'>slerp between two seeds</span></li>
  <li><a href='inference.html'>live inference</a> &nbsp;<span class='meta'>onnx in your browser</span></li>
</ul>
</body></html>""")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./output_model/raccoon_model.pt")
    parser.add_argument("--samples_dir", type=str, default="./output_model/samples")
    parser.add_argument("--out_dir", type=str, default="./assets/interactive")
    parser.add_argument("--seed_n", type=int, default=64)
    parser.add_argument("--interp_frames", type=int, default=32)
    args = parser.parse_args()

    device = get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = TinyUNet().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    diffusion = GaussianDiffusion(timesteps=1000, schedule="cosine", device=device)

    print("building trajectory ...")
    build_trajectory(model, diffusion, out_dir)

    print("building seed grid ...")
    build_seed_grid(model, diffusion, out_dir, n=args.seed_n)

    print("building interpolation ...")
    build_interpolation(model, diffusion, out_dir, n_frames=args.interp_frames)

    print("building epoch grids ...")
    build_epoch_grids(Path(args.samples_dir), out_dir)

    print("writing viewer HTML ...")
    write_viewers(out_dir)

    print(f"\ndone. open {out_dir / 'index.html'} in a browser.")


if __name__ == "__main__":
    main()
