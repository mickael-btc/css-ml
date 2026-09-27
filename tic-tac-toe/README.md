# Tic-tac-toe against CSS

Play tic-tac-toe against a neural network that runs entirely in CSS. **Zero bytes of JavaScript.** It never loses.

Open `index.html` and tap a square. You are X, and the network answers as O straight away.

- **Network:** 18 → 64 → 9 MLP with ReLU, 1,801 integer weights, written out four times (once per O move)
- **Page:** one self-contained ~270 KB HTML file with no `<script>`, no event handlers, and no external requests
- **Never loses:** checked in Python against every possible X strategy (489 distinct games), and every browser game in the test matches the Python integer model move for move

## How it works

| Stage | CSS |
|---|---|
| **Your moves** | Each of your five moves is its own radio group (`t0`…`t4`, 9 radios each). Every square holds one `<label>` per move, and `:has()` shows only the labels for the move you're on: `.app:has([name=t0]:checked):not(:has([name=t1]:checked)) label.t1`. **New game** is a `<button type="reset">`. |
| **Board** | `--X{t}_{i}: 1` when radio `m{t}_{i}` is checked. The board O sees before its move t is `--xb{t}_i` = your moves 0..t and `--ob{t}_i` = its own moves 0..t-1. |
| **Network** | `--h{t}_k: max(0, calc((bias + w·inputs) / R))`, an integer ReLU, then 9 logits `--g{t}_c`. Four copies, one per O move, each reading the previous copies' moves. |
| **Choosing a square** | `--s{t}_c: 16·logit + (8 - c) - 2²⁰·occupied`. The tie-break term makes every legal score distinct, and occupied squares sink far below. `--w{t}_c` is a product of `clamp(0, s_c - s_d, 1)` masks, so exactly one square gets 1. |
| **O's move** | `--O{t}_c = w · played(t) · (1 - you just won)`. |
| **Result** | Each of the 8 lines is `clamp(0, a + b + c - 2, 1)`. A win or a full board sets `--over`, which shrinks every label to zero width so the board locks. |
| **Heat map** | The latest decision's scores through `exp()`, a real softmax. The network is almost certain of every move, so the display uses a softened temperature where the favourite move typically shows around 60%. |

Every value is an integer below 2²⁴, so engines that do CSS math in 32-bit floats give exactly the Python answer. Each layer divides by an odd number, so rounding never lands on an exact `.5`.

## Training

`scripts/train.py` runs minimax over the whole game tree. For each of the 2,097 reachable positions where O is to move, the targets are the best moves, where winning sooner beats winning later and a draw beats a loss. A numpy MLP learns to put its probability mass on that set, then it is quantized to int8 weights and integer activations. The script plays the integer model against every X strategy and only saves it if it never loses, never picks a taken square, and never ties two legal squares.

## Rebuild

From the repository root:

```sh
.venv/bin/python tic-tac-toe/scripts/train.py   # a few seconds; writes model.json and tests/fixtures.json
.venv/bin/python tic-tac-toe/scripts/build.py   # writes index.html
node tic-tac-toe/tests/browser.test.mjs         # plays the fixture games in Chromium, Firefox and WebKit
```

| File | Purpose |
|---|---|
| `index.html` | The whole game |
| `model.json` | Quantized weights |
| `scripts/train.py` | Minimax targets, numpy training, quantization, the never-loses sweep |
| `scripts/build.py` | Compiles `model.json` into the game's CSS |
| `scripts/page.html` | Page template: layout and styles |
| `tests/` | Browser test and fixture games |
