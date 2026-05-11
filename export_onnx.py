"""
Export the trained U-Net to ONNX for in-browser inference.

The DDIM sampling loop runs in JavaScript; this script exports just the noise
prediction step (a single forward through the U-Net) plus the schedule constants
the loop needs.

Outputs in ./assets/interactive/:
    raccoon_unet.onnx     - the noise predictor
    schedule.json         - {alphas_cumprod, sqrt_one_minus_alphas_cumprod, timesteps}
                            for a 1000-step cosine schedule
    inference.html        - standalone "click to generate a raccoon" demo using
                            onnxruntime-web from a CDN
"""

import argparse
import json
from pathlib import Path

import torch

from raccoon_diffusion.diffusion import GaussianDiffusion
from raccoon_diffusion.model import TinyUNet


VIEWER_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Instrument+Serif&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg: #0a0a0a;
  --surface: #141414;
  --border: rgba(255,255,255,0.08);
  --text: rgba(255,255,255,0.9);
  --muted: rgba(255,255,255,0.5);
  --accent: #ff6b3d;
  --accent-soft: rgba(255,107,61,0.18);
  --sans: 'Inter', system-ui, -apple-system, sans-serif;
  --serif: 'Instrument Serif', ui-serif, Georgia, serif;
  --mono: 'JetBrains Mono', 'SF Mono', Monaco, Consolas, monospace;
}

* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text);
       font-family: var(--sans); font-size: 14px;
       padding: 28px;
       display: flex; flex-direction: column; align-items: center; gap: 16px; }

h2 { font-family: var(--serif); font-weight: 400;
     font-size: clamp(24px, 4vw, 34px);
     letter-spacing: -0.01em; margin: 0; }

canvas#out { image-rendering: pixelated; border-radius: 10px;
             width: 384px; height: 384px; background: #050505;
             box-shadow: 0 1px 0 rgba(255,255,255,0.04) inset,
                         0 20px 40px -16px rgba(0,0,0,0.6); }

.row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
       justify-content: center; }

.meta { font-family: var(--mono); font-size: 12px;
        color: var(--muted); letter-spacing: 0.01em; }

button { background: transparent; border: 1px solid var(--border); color: var(--text);
         padding: 8px 16px; border-radius: 999px; cursor: pointer;
         font: 500 13px var(--sans); letter-spacing: 0.01em;
         transition: 0.18s ease; }
button:hover:not(:disabled) { border-color: var(--accent);
                              color: var(--accent); background: var(--accent-soft); }
button:disabled { opacity: 0.4; cursor: default; }

button.primary { background: var(--accent); border-color: var(--accent);
                 color: #0a0a0a; font-weight: 600; }
button.primary:hover:not(:disabled) { background: #ff8559; border-color: #ff8559;
                                       color: #0a0a0a; }

input[type=number] { background: transparent; border: 1px solid var(--border);
                     color: var(--text); padding: 6px 10px; border-radius: 6px;
                     font: 500 13px var(--mono); width: 130px; }
input[type=number]:focus { outline: none; border-color: var(--accent); }
"""


INFERENCE_HTML = """<!doctype html><html><head>
<meta charset='utf-8'><title>Generate a raccoon</title>
<style>__CSS__</style>
<script src='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.min.js'></script>
</head><body>
<h2>generate a raccoon, live in your browser</h2>
<canvas id='out' width='64' height='64'></canvas>
<div class='row'>
  <span class='meta'>seed</span>
  <input id='seed' type='number' value='0'/>
  <button id='go' class='primary' disabled>loading…</button>
  <button id='rand'>random</button>
</div>
<div class='meta'>
  steps <input id='steps' type='number' value='25' min='5' max='200'/>
  &nbsp;·&nbsp; <span id='status'>fetching model…</span>
</div>
<script>
const TIMESTEPS = 1000;
const IMG_SIZE = 64;
const CHANNELS = 3;

let session = null;
let schedule = null;
let seed = 0;

// Mulberry32 PRNG so the seed is reproducible in the browser
function rngFromSeed(s) {
  let a = (s + 0x6d2b79f5) >>> 0;
  return function() {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Box-Muller normal samples from our PRNG
function gaussian(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function noiseTensor(rng) {
  const len = CHANNELS * IMG_SIZE * IMG_SIZE;
  const buf = new Float32Array(len);
  for (let i = 0; i < len; i++) buf[i] = gaussian(rng);
  return buf;
}

function drawTo(canvas, x) {
  // x: Float32Array length CHANNELS*IMG_SIZE*IMG_SIZE, range roughly [-1, 1]
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(IMG_SIZE, IMG_SIZE);
  for (let p = 0; p < IMG_SIZE * IMG_SIZE; p++) {
    const r = Math.max(0, Math.min(1, (x[0 * IMG_SIZE * IMG_SIZE + p] + 1) / 2));
    const g = Math.max(0, Math.min(1, (x[1 * IMG_SIZE * IMG_SIZE + p] + 1) / 2));
    const b = Math.max(0, Math.min(1, (x[2 * IMG_SIZE * IMG_SIZE + p] + 1) / 2));
    img.data[4 * p + 0] = (r * 255) | 0;
    img.data[4 * p + 1] = (g * 255) | 0;
    img.data[4 * p + 2] = (b * 255) | 0;
    img.data[4 * p + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

async function init() {
  schedule = await fetch('schedule.json').then(r => r.json());
  session = await ort.InferenceSession.create('raccoon_unet.onnx',
    { executionProviders: ['wasm'] });
  document.getElementById('go').disabled = false;
  document.getElementById('go').textContent = 'generate';
  document.getElementById('status').textContent = 'ready';
}

async function generate() {
  const btn = document.getElementById('go');
  const status = document.getElementById('status');
  btn.disabled = true; btn.textContent = 'denoising…';
  seed = parseInt(document.getElementById('seed').value) | 0;
  const steps = Math.max(5, parseInt(document.getElementById('steps').value) | 0);
  const rng = rngFromSeed(seed);
  let x = noiseTensor(rng);

  const stepSize = (TIMESTEPS / steps) | 0;
  const timesteps = [];
  for (let t = 0; t < TIMESTEPS; t += stepSize) timesteps.push(t);
  timesteps.reverse();

  const canvas = document.getElementById('out');
  const xShape = [1, CHANNELS, IMG_SIZE, IMG_SIZE];
  const tShape = [1];

  for (let i = 0; i < timesteps.length; i++) {
    const ti = timesteps[i];
    const xT = new ort.Tensor('float32', x, xShape);
    const tT = new ort.Tensor('float32', new Float32Array([ti]), tShape);
    const out = await session.run({ x: xT, t: tT });
    const eps = out[Object.keys(out)[0]].data;
    const ac = schedule.alphas_cumprod[ti];
    const acPrev = (i + 1 < timesteps.length)
        ? schedule.alphas_cumprod[timesteps[i + 1]] : 1.0;
    const sqA = Math.sqrt(ac), sqB = Math.sqrt(1 - ac);
    const sqAp = Math.sqrt(acPrev), sqBp = Math.sqrt(1 - acPrev);
    for (let j = 0; j < x.length; j++) {
      let x0 = (x[j] - sqB * eps[j]) / sqA;
      if (x0 < -1) x0 = -1; else if (x0 > 1) x0 = 1;
      x[j] = sqAp * x0 + sqBp * eps[j];
    }
    if (i % Math.max(1, Math.floor(steps / 12)) === 0 || i === timesteps.length - 1) {
      drawTo(canvas, x);
      status.textContent = `step ${i + 1}/${timesteps.length}`;
      await new Promise(r => setTimeout(r, 0));  // yield to browser to paint
    }
  }
  drawTo(canvas, x);
  status.textContent = `seed ${seed} · ${steps} steps · done`;
  btn.disabled = false; btn.textContent = 'generate';
}

document.getElementById('go').onclick = generate;
document.getElementById('rand').onclick = () => {
  document.getElementById('seed').value = (Math.random() * 1e9) | 0;
};
init();
</script>
</body></html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./output_model/raccoon_model.pt")
    parser.add_argument("--out_dir", type=str, default="./assets/interactive")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = TinyUNet()
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.eval()

    dummy_x = torch.randn(1, 3, 64, 64)
    dummy_t = torch.tensor([500.0])

    onnx_path = out_dir / "raccoon_unet.onnx"
    torch.onnx.export(
        model,
        (dummy_x, dummy_t),
        onnx_path.as_posix(),
        input_names=["x", "t"],
        output_names=["eps"],
        opset_version=17,
        dynamo=False,
        dynamic_axes={"x": {0: "batch"}, "t": {0: "batch"}, "eps": {0: "batch"}},
    )
    print(f"wrote {onnx_path}")

    # Schedule constants for the JS DDIM loop
    diffusion = GaussianDiffusion(timesteps=1000, schedule="cosine", device="cpu")
    schedule = {
        "timesteps": 1000,
        "schedule": "cosine",
        "alphas_cumprod": diffusion.alphas_cumprod.tolist(),
    }
    (out_dir / "schedule.json").write_text(json.dumps(schedule))
    print(f"wrote {out_dir / 'schedule.json'}")

    (out_dir / "inference.html").write_text(
        INFERENCE_HTML.replace("__CSS__", VIEWER_CSS)
    )
    print(f"wrote {out_dir / 'inference.html'}")


if __name__ == "__main__":
    main()
