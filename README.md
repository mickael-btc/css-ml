# CSS ML

A neural network that recognizes hand-drawn digits, running entirely in CSS. **Zero bytes of JavaScript.**

**Live: https://css-ml.vercel.app** · or open `index.html`, draw a digit on the 8×8 grid, and the browser's style engine classifies it.

- **Network:** 64 → 64 → 32 → 10 MLP with ReLU, 6,570 integer weights
- **Page:** one self-contained 300 KB HTML file with no `<script>`, no event handlers, and no external requests
- **Browsers:** tested in Chromium, Firefox and WebKit, where the prediction matches the Python reference exactly

## More demos

Same idea, zero JavaScript, one folder each:

| Demo | What the CSS does |
|---|---|
| [Latent space](https://css-ml.vercel.app/latent-space) · [`latent-space/`](latent-space) | A VAE decoder draws a 16×16 digit for any point you pick on a 2D map |
| [Tic-tac-toe](https://css-ml.vercel.app/tic-tac-toe) · [`tic-tac-toe/`](tic-tac-toe) | A neural network plays O and never loses (checked against every possible game) |
| [Rock paper scissors](https://css-ml.vercel.app/rock-paper-scissors) · [`rock-paper-scissors/`](rock-paper-scissors) | An online model learns your habits while you play and counters them |

`npm run build:all` and `npm run test:all` build and test all four.

## How it works

Every value is a [registered custom property](https://developer.mozilla.org/docs/Web/CSS/@property) (`@property --x { syntax: "<integer>" }`) on one element. The browser recomputes them whenever the drawing changes.

| Stage | CSS |
|---|---|
| **Drawing** | `.app:has(.grid:active):has(#c5:hover)` sets `--p5: 1` with a `0s` transition. Going back to 0 takes a ~115-day transition, so the pixel stays painted. **Clear** sets `transition: none`, which cancels every transition at once. On touch screens each cell is a `<label>` for a hidden checkbox, so a tap toggles it, and **Clear** is a `<button type="reset">`. |
| **Normalization** | Finds the bounding box with `clamp()` and running products, makes it square, and resamples it to 8×8 with one-hot `clamp(0, 1 - (src - i)², 1)` selectors. You can draw at any size, anywhere. |
| **Layers** | `--a0_k: max(0, calc((bias + w₀·var(--q0) + … ) / R))`, an integer ReLU. `<integer>` properties round, which gives the quantization. |
| **Argmax** | `--w3: calc(clamp(0, o3 - o0, 1) * clamp(0, o3 - o1, 1) * …)` is 1 for the winning digit only. |
| **Confidence** | `exp((oᵢ - max) / scale) / Σ`, a real softmax. |
| **Output** | `counter-reset: d var(--pred); content: counter(d)` |

All intermediate values are integers below 2²⁴, so engines that do CSS math in 32-bit floats give the same answer. Each layer divides by an odd number, so rounding never lands on an exact `.5`.

**Safari:** WebKit freezes `:hover` while a mouse button is held, so press-and-drag paints only the first cell. Click **Hover to draw** under the grid to paint by just moving the pointer.

**Phones and tablets:** CSS `:hover` never follows a finger, so dragging can't be done without JavaScript. Tap cells to paint them instead, and tap a cell again to erase it.

## Accuracy

| Test | Accuracy |
|---|---|
| MNIST test set, random size (5–8 cells) and position | 85.7% |
| Hand-drawn 8×8 digits (`scripts/handdrawn.py`) | 87.2% |

8×8 black-and-white input is very coarse. Digits drawn clearly and roughly upright work best.

## Rebuild

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # numpy, Pillow
npm install                       # Playwright, only needed for tests

npm run train   # python3 scripts/train.py: downloads MNIST to data/, ~5 min, writes model.json
npm run build   # python3 scripts/build.py: writes index.html
npm test        # draws every fixture in Chromium, Firefox and WebKit, compares against Python
```

| File | Purpose |
|---|---|
| `index.html` | The whole app. Deploy this file. |
| `model.json` | Quantized weights |
| `scripts/train.py` | Data, augmentation, numpy MLP training, int8 quantization |
| `scripts/norm.py` | Bounding-box normalization, bit-exact with the CSS version |
| `scripts/build.py` | Compiles `model.json` into the network's CSS |
| `scripts/page.html` | Page template: layout and styles |
| `scripts/handdrawn.py` | Hand-drawn evaluation digits |
| `tests/` | Browser test and fixtures |

Font-rendered training digits come from the fonts installed on the machine (macOS, Linux or Windows paths).

Inspired by [zero-js](https://github.com/shuding/zero-js) by Shu Ding.

## Deploy

The repo is ready for [Vercel](https://vercel.com/new): import it, no settings needed. `vercel.json`:

- publishes **only `index.html`**: the build copies it to `dist/`, the only folder served. Scripts, weights and tests stay private to the repo
- skips `npm install` (Playwright is only for tests)
- redirects every other path to `/`, so there is no 404 page

Any static host works the same way: `index.html` is the whole site.

## License

[MIT](LICENSE)
