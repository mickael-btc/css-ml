# CSS ML: playground

A classifier that **trains itself in CSS**. Place red and blue points on a plane. The browser's style engine runs 24 epochs of training and colours the background with what the model learned. **Zero bytes of JavaScript**, and no weights shipped: the page holds only the training rule.

Open `index.html`, pick a brush, click to place points (tap on a phone), click a point again to remove it. Try a circle of red inside blue, or four corners (XOR). Click a bar in the chart to see the model after that epoch, or hover one to preview it.

## How it trains

Each of the 10×10 point positions is a group of three radios (empty, red, blue). Only the label for the current brush is clickable, so a click sets the colour, and a second click with the same brush picks "empty". Each position gets a label `t` of +1 (red), −1 (blue) or 0 (empty), and six fixed features:

```
f = (45, 9x, 9y, x² − 33, y² − 33, xy)        x, y in −9..9, every feature at most 81 in size
```

The squared terms let the boundary curve into a circle, and `xy` lets it split four corners. Training is a full-batch **margin perceptron**, written out once per epoch as registered `<integer>` custom properties on `.app`:

| Step | CSS |
|---|---|
| **Point colour** | `.app:has(#r12:checked) { --t12: 1 }`, and `-1` for blue |
| **Score** | `--s3_12: calc(45*var(--w3_0) + -27*var(--w3_1) + …)`, one per point per epoch |
| **Update** | `--g3_12: calc(t * clamp(0, 3001 - t * s, 1))`: the colour if `t·s ≤ 3000` (wrong, or too close to the line), else 0 |
| **Weights** | `--w4_k: clamp(-24000, calc(w3_k + (Σ f_k · g) / 15), 24000)`, all points at once |
| **Errors** | `--err3: calc(Σ clamp(0, 1 - t·s, 1) + n - 100)`: points with `t·s ≤ 0` |
| **Shown epoch** | the picked chart bar sets `--u0 … --u5` to that epoch's weights. Hovering a bar overrides it |
| **Plane** | each of the 19×19 background cells carries its own features inline and computes `--q: calc(45*var(--u0) + var(--fx)*var(--u1) + …)` |

That is 5,186 properties, recomputed on every click. Only the few the page reads further down (`--u`, `--err`, `--n`, …) inherit. The rest are `inherits: false`, so the browser doesn't copy thousands of values into every element.

**Exact in every engine:**
- `/ 15` is odd, so an integer divided by it never lands on `.5`, and CSS's rounding matches Python's `floor((2a + 15) / 30)`.
- The weight clamp keeps every score under 9.3M, below 2²⁴, so engines that do CSS math in 32-bit floats give the same numbers.

## Numbers

- `index.html`: 1.25 MB (114 KB gzipped), 24 epochs, 100 point positions, 361 plane cells
- One click, from checking a point to the last epoch and a plane cell being recomputed: about 34 ms in Chromium, 21 ms in Firefox, 44 ms in WebKit (desktop, measured in the test)
- The demo sets in `reference.py` (linear, circle, XOR, one class) end at 0 errors. Circle and XOR take 3 to 10 epochs to get there.

## Rebuild

From the repo root:

```sh
.venv/bin/python playground/scripts/reference.py --fixtures   # runs the rule in numpy, writes tests/fixtures.json
.venv/bin/python playground/scripts/build.py                  # writes index.html
node playground/tests/browser.test.mjs                        # Chromium, Firefox, WebKit
```

The test clicks every fixture set in. For each one it checks every weight of every epoch, the error counts, the accuracy and all 361 plane cells against `reference.py`. It also checks Clear, removing and repainting a point, picking and hovering an epoch, and a set placed by tap on an emulated phone.

| File | Purpose |
|---|---|
| `index.html` | The whole demo |
| `scripts/reference.py` | The training rule in numpy, and the fixtures |
| `scripts/build.py` | Writes the rule out as CSS, epoch by epoch |
| `scripts/page.html` | Layout and styles |
| `tests/` | Browser test and fixtures |
