# CSS ML: autoencoder

A full autoencoder, encoder **and** decoder, running entirely in CSS. **Zero bytes of JavaScript.**

Draw a digit on the 8×8 grid:

1. the **encoder** (64 → 64 → 32 → 2) squeezes your 64 pixels into just **2 numbers**, shown as a dot on a 20×20 map;
2. the **decoder** (2 → 32 → 64 → 256) draws a 16×16 digit back from those 2 numbers alone.

The redrawn digit is what the network *kept* from your drawing. Click (or tap) anywhere else on the map and the decoder draws what lives there instead.

- **Network:** 25,154 int8 weights, both halves quantized to integers
- **Page:** one self-contained ~770 KB HTML file with no `<script>`, no event handlers and no external requests
- **Browsers:** tested in Chromium, Firefox and WebKit, where every encoder unit, map cell, decoder unit and pixel matches the Python integer model exactly

## Encoder vs decoder, simply

- The **encoder** turns data into meaning: a drawing becomes a short code (here, a point on a map). It is the "understanding" half.
- The **decoder** turns meaning back into data: a code becomes a drawing. It is the "generating" half.

Trained together, they can only succeed if the code keeps what matters (the shape of the digit) and drops the rest (the exact pixels you clicked), so similar digits end up close together on the map.

## How it works

| Stage | CSS |
|---|---|
| **Drawing** | Copied from the digit demo: press and drag (`:active` + `:hover`, with a ~115-day transition keeping pixels painted) on devices that hover; on touch screens each cell is a `<label>` for a hidden checkbox. **Clear** is a `<button type="reset">`. |
| **Normalisation** | Also copied: crop to the bounding box, square it, resample to 8×8 with `clamp()` one-hot selectors, bit-exact with `scripts/norm.py`. |
| **Encoder** | `--e0_k: max(0, calc((bias + Σ w·q) / R))` integer ReLUs, then two linear outputs `--ex`, `--ey`. |
| **Map cell** | `--dcol: clamp(0, calc((ex − origin − (W−1)/2) / W), 19)`. With an odd cell width `W`, rounding that is exactly `floor`, never a `.5` tie. Each map label compares its own column and row to it to show the dot. |
| **Which cell is drawn** | The drawing's cell by default. `.app:has(.ix7:checked)` (a pinned map cell, a radio) overrides it, and on devices that hover `.app:has(.cx7:hover)` overrides both. **Back to my drawing** checks a separate radio in the same group, and **Clear** resets the whole form, so both release the pin. |
| **Decoder** | Copied from `latent-space/`: integer ReLUs from the cell centre, then a hard sigmoid `clamp(0, calc((bias + Σ) / R), 100)` per pixel. |

All intermediate values stay below 2²⁴ (the peak is about 83,000), and every division is by an odd integer, so engines doing CSS math in 32-bit floats give the same answer.

## Honest numbers

The encoder only sees a coarse 8×8 black-and-white drawing, and a plain 2D VAE mixes similar digits (4/9, 3/5/8) on the map. So during training **only**, a small linear digit classifier also read the 2 numbers and pushed each digit into its own region. It is not in the page.

| Test | Lands on its own digit's colour |
|---|---|
| Hand-drawn 8×8 digits from the digit demo (`../tests/fixtures.json`, never trained on) | 120 / 146 (82%) |
| MNIST training drawings (random size and position) | 77% |

When a drawing lands in the wrong region, the decoder draws that region's digit: you can see exactly what the network "understood".

Recalculating the page after one painted pixel takes about 9 ms in Chromium, 16 ms in Firefox and 18 ms in WebKit (desktop, median). It has not been measured on a real phone.

## Rebuild

From the repository root (see the root README for the `.venv` and `npm install` setup):

```sh
.venv/bin/python autoencoder/scripts/train.py   # ~1 min: trains on MNIST (../data/) and system fonts, writes model.json
.venv/bin/python autoencoder/scripts/build.py   # writes index.html and tests/fixtures.json
node autoencoder/tests/browser.test.mjs         # draws, pins, hovers and taps in Chromium, Firefox and WebKit
```

The font-rendered digits come from the fonts installed on the machine, so retraining elsewhere gives a slightly different model.

| File | Purpose |
|---|---|
| `index.html` | The whole demo |
| `model.json` | Quantized encoder and decoder, map geometry and colours |
| `scripts/train.py` | Data, numpy VAE training, int8 quantization, map |
| `scripts/norm.py` | Bounding-box normalisation (copied from the digit demo) |
| `scripts/build.py` | Compiles `model.json` into CSS |
| `scripts/page.html` | Page template |
| `tests/` | Browser test and fixtures |
