# CSS ML: latent space

A neural network that **draws** digits, running entirely in CSS. **Zero bytes of JavaScript.**

Open `index.html` and move over the map (or tap it). Every cell is a point in a 2D latent space, and a decoder generates the digit that lives there, pixel by pixel, in the browser's style engine. Walk from a 7 to a 9 and watch it bend.

- **Network:** the decoder half of a variational autoencoder, 2 → 32 → 64 → 256 (a 16×16 image), ReLU, 18,848 integer weights
- **Page:** one self-contained ~480 KB HTML file with no `<script>`, no event handlers, and no external requests
- **Browsers:** tested in Chromium, Firefox and WebKit, where every pixel and hidden unit matches the Python reference exactly

## How it works

Every value is a [registered custom property](https://developer.mozilla.org/docs/Web/CSS/@property) (`@property --x { syntax: "<integer>" }`) on one element. The browser recomputes them whenever the position changes.

| Stage | CSS |
|---|---|
| **Position** | The map is 20×20 `<label>`s for radio buttons, so a click or a tap pins a cell. `.app:has(.ix7:checked) { --zx: 14 }` sets x from the checked column and y from the checked row: 40 rules, not 400. Under `@media (hover: hover)` the same rules on `:hover` preview the cell under the pointer. |
| **Layers** | `--h0_k: max(0, calc((bias + w₀·var(--zx) + w₁·var(--zy)) / R))`, an integer ReLU. `<integer>` properties round, which gives the quantization. |
| **Pixels** | `--yN: clamp(0, calc((bias + Σ w·h) / R), 100)`, a hard sigmoid `(logit + 2) / 4` folded into the weights, as an intensity 0–100. |
| **Drawing** | Each `<i>` of the image reads its pixel: `background: color-mix(in oklab, paper, ink calc(var(--v) * 1%))`. |
| **Coordinates** | Sign, whole part and two decimals from `clamp()` and `round`, printed with CSS counters. |

All intermediate values are integers below 2²⁴, so engines that do CSS math in 32-bit floats give the same answer. Each layer divides by an odd number, so rounding never lands on an exact `.5`.

The map colours come from the training set: every MNIST digit is encoded, and each cell takes the most common label among its 50 nearest codes. Brighter cells hold more training digits. The colours are static; the digit is not.

Moving the pointer costs one style recalculation of the whole decoder: about 20 ms in Chromium and Firefox and about 45 ms in WebKit on a recent laptop.

## Rebuild

From the repository root:

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # numpy, Pillow
npm install                       # Playwright, only needed for tests

python3 latent-space/scripts/train.py   # ~30 s, reuses MNIST in ../data/, writes model.json and tests/fixtures.json
python3 latent-space/scripts/build.py   # writes index.html
node latent-space/tests/browser.test.mjs  # clicks, hovers and taps the map in Chromium, Firefox and WebKit
```

| File | Purpose |
|---|---|
| `index.html` | The whole app |
| `model.json` | Quantized decoder weights and the map (coordinates, colours, labels) |
| `scripts/train.py` | Data, numpy VAE training, int8 quantization, digit map |
| `scripts/build.py` | Compiles `model.json` into the decoder's CSS |
| `scripts/page.html` | Page template: layout and styles |
| `tests/` | Browser test and fixtures (the integer decoder's output for every cell) |
