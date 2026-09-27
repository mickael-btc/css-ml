# Connect Four against CSS

Play Connect Four against a neural network that runs entirely in CSS. **Zero bytes of JavaScript.**

Open `index.html` and tap a column. You are red and play first; the network answers as yellow straight away.

- **Network:** 84 → 128 → 7 MLP with ReLU, 11,783 integer weights, trained to imitate a depth-7 alpha-beta search
- **Three hard-coded rules on top:** win now if possible, else block your immediate four, else avoid a column that lets you win right on top of its piece. The page says when a rule made the move.
- **Page:** one self-contained ~390 KB HTML file with no `<script>`, no event handlers, and no external requests
- **Exact:** every browser game in the test matches the Python integer model move for move, in Chromium, Firefox and WebKit

## How strong is it?

Measured in Python with the integer model, as the CSS plays it (network second, 400 games against random and greedy, 200 against each search):

| Opponent (plays first) | Network wins | Draws | Network loses |
|---|---|---|---|
| Random moves | 100% | 0% | 0% |
| Greedy: win, else block, else random | 97% | 2% | 2% |
| Alpha-beta depth 2, first 2 moves random | 34% | 14% | 52% |
| Alpha-beta depth 4, first 2 moves random | 24% | 14% | 62% |

It beats casual play easily but loses to anything that searches a few moves ahead. Moving second is also a real handicap: with perfect play, the first player wins Connect Four. Without the three rules, the network alone beats greedy 83% of the time instead of 97%.

## How it works

The game is 21 turns, and each turn is two nested elements, so the stylesheet describes **one** turn and the browser applies it 21 times:

```html
<div class="turn">   <!-- your 7 radios for this turn; copies the previous state: --PH_i: var(--H_i) -->
  <div class="step"> <!-- computes this turn's state:  --H_i: calc(var(--PH_i) + …) -->
    <div class="turn"> … the next turn, nested inside, inherits everything
```

Custom properties inherit, so each turn reads the one before it. A property can't refer to itself, which is why the copy lives on its own element. The weights appear once in the stylesheet, not 21 times.

| Stage | CSS |
|---|---|
| **Your move** | Each turn is a radio group (`t0`…`t20`, 7 radios each). Each column holds one `<label>` per turn, and `:has()` shows only the current turn's. `.turn:has(> .c3:checked)` sets `--hm3: 1` for that turn only. **New game** is a `<button type="reset">`. |
| **Gravity** | Column heights `--hg_c` are carried from turn to turn. Your piece lands on cell `row * 7 + col` where `row = height`, via `clamp(0, 1 - (h - r)², 1)`. |
| **Network** | `--h_k: max(0, calc((bias + w·inputs) / R))`, an integer ReLU over your 42 cells and its 42 cells, then 7 logits. |
| **Tactics** | For the square each column would fill: does it complete one of the 69 lines for the network (win), for you (block), or does the square above it complete one for you (danger)? |
| **Choosing a column** | `--s_c: 8·logit + tie-break + 2²¹·win + 2²⁰·block − 2¹⁹·danger − 2²²·full`. The tie-break (centre first) makes every score distinct. `--w_c` is a product of `clamp(0, s_c − s_d, 1)` masks, so exactly one column gets 1. |
| **Result** | 69 lines of four, `clamp(0, a + b + c + d − 3, 1)` each. A win or a full board sets `--over`, which shrinks every label to zero width so the board locks. |
| **Bars** | The latest decision's scores through `exp()`, a real softmax, with a softened temperature for display. |

Every value is an integer below 2²⁴, so engines that do CSS math in 32-bit floats give exactly the Python answer. The hidden layer divides by an odd number, so rounding never lands on an exact `.5`.

Recalculating one move takes about 20 ms in Chromium, 35 ms in WebKit and 120 ms in Firefox on a laptop, measured in the page from Playwright.

## Training

`scripts/train.py`:

1. Plays 3,000 games between the network's seat and first players of every strength (random, greedy, alpha-beta depth 2–5 with noise), keeping the positions where the network is to move.
2. Labels each position with a depth-7 alpha-beta search (window heuristic, centre bonus), using all cores. Search values are cached in `data/labels.json` (git-ignored), so reruns skip this.
3. Trains a numpy MLP on a softmax of those values, with left-right mirrored boards for twice the data.
4. Quantizes to int8 weights and integer activations.
5. Four rounds of DAgger: the integer model plays, and the positions it reaches get labelled and added.
6. Evaluates the integer model with the tactics against the four opponents above and writes test fixtures.

It takes about 18 minutes on 12 cores the first time.

## Rebuild

From the repository root:

```sh
.venv/bin/python connect-four/scripts/train.py   # ~18 min first time; writes model.json and tests/fixtures.json
.venv/bin/python connect-four/scripts/build.py   # writes index.html
node connect-four/tests/browser.test.mjs         # plays the fixture games in Chromium, Firefox and WebKit
```

| File | Purpose |
|---|---|
| `index.html` | The whole game |
| `model.json` | Quantized weights and the evaluation above |
| `scripts/game.py` | Rules, alpha-beta search, and the network's move exactly as the CSS computes it |
| `scripts/train.py` | Positions, search labels, numpy training, quantization, DAgger, evaluation |
| `scripts/build.py` | Compiles `model.json` into the game's CSS |
| `scripts/page.html` | Page template: layout and styles |
| `tests/` | Browser test and fixture games |
