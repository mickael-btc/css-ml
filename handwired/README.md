# Handwired: code as neural networks, in pure CSS

Regular code rewritten as ReLU networks, after Cheng Lou's [handwired](https://github.com/chenglou/handwired): `(a > 4) ? 10 : 12` becomes `above4 = relu(a − 4) − relu(a − 5)`, then `above4·10 + (1 − above4)·12`. No training, exact, and still differentiable. Here the networks run in the browser's style engine like the rest of this repo. **Zero bytes of JavaScript.**

**Live:** https://css-ml.vercel.app/handwired

| Step | Page | What it shows |
|---|---|---|
| 1 | [`tic-tac-toe/`](tic-tac-toe) | The trained tic-tac-toe policy **plus its rules** (free squares, the argmax, "stop once X has won", who won) as one 7-layer ReLU net. Every computed value in the CSS is `max(0, calc(…))` or a plain sum. |
| 1 | [`connect-four/`](connect-four) | A whole connect-four turn is one 13-layer net of 4,451 units: your piece falling, the trained policy, the three tactics, the choice, its piece falling. Applied 21 times by nested elements. |
| 2 | [`interpreter/`](interpreter) | **A network that runs a network.** One fixed net of 13,608 units takes a board *and the 1,801 weights of any 18 → 64 → 9 int8 model* as inputs, and plays like that model. The page loads the trained model, a hand-written one, or random weights into the same CSS. |
| 3 | [`tower/`](tower) | **A network running a copy of itself.** `interpret(net)` compiles a net that runs `net` from its numbers. I1 = interpret(N) runs a tiny 2 → 2 → 1 net; I2 = interpret(I1) runs I1 running N. |
| 4 | [`unfrozen/`](unfrozen) | **A rule that keeps learning.** "Red inside the box" compiled into a 2 → 4 → 1 ReLU net, then all 17 numbers trained by backpropagation in CSS, 16 epochs, on the points you place. |
| 5 | [`cpu/`](cpu) | **A computer made of neurons.** A tiny 8-bit CPU (accumulator, 8 bytes of RAM, 16 instructions of program, a 16-byte output tape) whose clock cycle is one 811-unit ReLU net, run 128 times by nested elements. Programs are data: count to n, Fibonacci, multiply, sum 1…n, powers of two. |

## The compiler

`scripts/hw.py` is the whole thing. A `Prog` is built from inputs, constants, `+`, constant `×` and `relu(v, R)` (a registered `<integer>` unit, `round(relu(v) / R)`). On top of that it has integer logic:

| Code | ReLU form |
|---|---|
| `x ≥ t`, x an integer | `relu(x − t + 1) − relu(x − t)` |
| `x == t` | `relu(x − t + 1) − 2·relu(x − t) + relu(x − t − 1)` |
| `a && b && c` (0/1) | `relu(a + b + c − 2)` |
| `a \|\| b \|\| …` | `s − relu(s − 1)` |
| `bit ? v : 0`, 0 ≤ v ≤ M | `relu(v − M·(1 − bit))` |
| argmax of distinct integers | `relu(1 − Σ_d [s_d > s_c])` |
| lexicographic choice | d beats c if `key_d > key_c`, or the keys tie and `base_d > base_c`. This replaces score tricks like `2²¹·win + 2²⁰·block + …` that don't fit under 2²⁴ |
| `⌊u / 2^j⌋` | `round(relu(2u + 1 − 2^j) / 2^(j+1))`, one unit, never an exact .5 |
| `w * x`, both inputs | the bits of w, each gating x, summed with powers of two |

`compile()` puts every ReLU at 1 + the depth of its deepest input, carries values needed further down through identity units (two for a signed value), and returns a layered, sparse integer MLP (`Net`). The same `Net` runs exactly in numpy (`forward`), reports the largest partial sum any unit can see (`max_mass`, which must stay below 2²⁴ for 32-bit-float CSS engines), and compiles to CSS (`to_css`).

## Step by step

**1. Rules as weights** (`scripts/rules.py`). The players are checked against the original demos' Python:
- **Tic-tac-toe:** exhaustively, on all 5,478 reachable boards. The move equals `int_move` on all 2,097 positions where O is to move, and the judge (winner, draw, free squares, winning lines) is right on every board.
- **Connect four:** the game tree is too big to enumerate. The move equals `int_move` on 24,708 positions (the original test games plus 3,000 games against random and greedy players), and the turn net replays 400 random games turn by turn.

The original pages compute tactics and the argmax with hand-written `clamp()` formulas; here they are weights in the same network as the policy.

**2. Interpreter** (`scripts/interpreter.py`). The model's weights come in as unsigned bytes, `u = w + 128`.
- **Layer 1:** inputs are 0/1, so `x·u = relu(u − 256·(1 − x))`.
- **Layer 2:** all 8 bits of each weight come out at once, `bit_j = q_j − 2·q_(j+1)`. Each bit gates the hidden activation, and `w·h = −128·(sign·h) + Σ 2^j·(bit_j·h)`.

It is exact for any 18 → 64 → 9 model with int8 weights, hidden biases within ±1024 and R = 1, as long as its sums stay under 2²⁴. That limit is checked for every model the page loads, on every position. It was checked on all 25,893 (position, move, model) cases, and on 300 random models with weights at the edges of the int8 range (in numpy, since those can pass 2²⁴).

**3. Tower** (`scripts/tower.py`). `interpret(guest)` knows the guest's wiring (which unit feeds which, each unit's divisor) but takes every weight and bias as an input. The gates need activation bounds; they come from interval arithmetic over the guest's input ranges, so they hold for every input.

| Net | inputs | units | weights | |
|---|---|---|---|---|
| N | 2 | (2 hidden) | 9 numbers | the program |
| I1 = interpret(N) | 11 | 34 | 82 | in CSS |
| I2 = interpret(I1) | 114 | 1,866 | 3,324 | in CSS |
| I3 = interpret(I2) | 4,795 | 191,288 | 270,246 | numpy only: its bounds reach 10¹⁸ |

I1 and I2 equal N exactly for the 5 presets and 400 random N (every weight −4…3, bias −4…4) on all four inputs. I3 is exact too, on 100 cases.

**4. Unfrozen** (`scripts/unfrozen.py` is the integer reference).
- **Training:** a full-batch margin perceptron, backpropagated through the ReLUs (`f = clamp(0, h, 1)` is the derivative). The first layer gets a 5× smaller step, because with a big step a box edge overshoots, flips sign, and its ReLU dies.
- **From the rule:** the box set has 0 errors at epoch 0. The "wider", "shifted" and "stripe" sets bend the rule to 0 errors within a few epochs.
- **From random:** the same sets end at 3–8 errors after 16 epochs.

**5. Neural CPU** (`scripts/cpu.py`). One clock cycle as ReLUs:
- **Fetch:** `[PC = i]` gates line i's opcode and argument.
- **Decode:** `[op = k]`, AND not halted.
- **Read:** `arg mod 8` is one integer unit, and `[addr = j]` gates RAM cell j.
- **ALU:** `A + m`, `A − m` and `A + n`, each wrapped mod 256 with one comparison.
- **Write:** the result of whichever opcode is on, gated; the same for each RAM cell and the tape's next slot.
- **Next PC:** the jump target, `PC + 1`, or `PC` (halted).

The program's 16 opcodes and 16 arguments are inputs, so the same net runs any program. Checked against a plain Python emulator on 3,066 programs (the demo programs with every input, and 3,000 random programs), every register at every one of the 128 cycles. In the browser a full run recomputes in about 2–3 ms in Chromium and WebKit, and about 130 ms in Firefox.

## Rebuild and test

From the repository root:

```sh
.venv/bin/python handwired/scripts/build.py --nets   # recompile and re-check every net (~2 min), then build the pages
.venv/bin/python handwired/scripts/build.py          # pages only, from handwired/nets/
node handwired/tests/browser.test.mjs                # every page in Chromium, Firefox and WebKit
node handwired/tests/browser.test.mjs tower          # one page
```

The browser test:
- **Games:** replays the original fixture games (tic-tac-toe, connect four) and each interpreter model's games, move by move.
- **Tower:** checks both levels for every preset and for custom N's set by clicking.
- **Unfrozen:** compares all 17 weights at every epoch, and every error count, with the numpy reference.
- **CPU:** runs every demo program with a spread of inputs and compares every tick's A and the final registers, RAM and tape.
- **Static:** checks that the step 1, 2 and 5 stylesheets only contain inputs, sums and ReLU units, with no `clamp()` and no product of two values.

| Page | Size | Properties | One move (Chromium / Firefox / WebKit) |
|---|---|---|---|
| tic-tac-toe | 0.9 MB | 5,340 | instant |
| connect-four | 1.0 MB | 5,426 | ~90 / ~400 / ~60 ms |
| interpreter | 2.4 MB | 15,673 | ~110 / ~310 / ~190 ms |
| tower | 1.0 MB | 7,746 | instant |
| unfrozen | 2.2 MB | 10,956 | ~40 / ~50 / ~50 ms |
| cpu | 0.3 MB | 918 (×128 cycles) | ~2 / ~130 / ~3 ms per full run |

| File | Purpose |
|---|---|
| `scripts/hw.py` | The compiler: `Prog`, `Net`, numpy forward, CSS output |
| `scripts/rules.py` | Step 1 nets and their exhaustive checks |
| `scripts/interpreter.py` | Step 2 net, the three models, checks, fixtures |
| `scripts/tower.py` | `interpret()`, interval bounds, the tower, checks |
| `scripts/unfrozen.py` | Step 4 training rule in numpy, demo sets, fixtures |
| `scripts/cpu.py` | Step 5: the emulator, the cycle net, the programs, checks, fixtures |
| `scripts/build*.py`, `common.py`, `shell.html` | The pages |
| `nets/` | Compiled nets as JSON |
| `tests/` | Browser test and fixtures |
