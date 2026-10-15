"""Step 2: a network that runs a network. One fixed ReLU net takes a tic-tac-toe board AND the weights
of any 18 -> 64 -> 9 int8 model as inputs, and plays that model's move.

    .venv/bin/python handwired/scripts/interpreter.py

The interpreter's own weights are fixed and know nothing about any model. A model's weights arrive
as 1,801 input values, so the product w * x of two inputs has to be built out of ReLUs:

  layer 1   x is 0 or 1, and v = w + 128 is in 0..255:  x * v = relu(v - 256 * (1 - x))
            so sum x_i w_i = sum relu(...) - 128 * sum x_i
  layer 2   h is a hidden activation in 0..H, u = w + 128 a byte. All its bits at once, with one
            integer unit each: q_j = floor(u / 2^j) = round(relu(2u + 1 - 2^j) / 2^(j+1)), and
            bit_j = q_j - 2 q_(j+1). Then bit * h = relu(h - H * (1 - bit)), and
            w * h = -128 * (sign * h) + sum_j<7 2^j (bit_j * h), sign = 1 - bit_7 (two's complement).

Weights come in as unsigned bytes, w + 128 (the usual int8 zero point), so they are never negative.

Then the rules from step 1 pick the move: the best free square, only if you moved and the game is on.
Exact for any model with int8 weights (-128..127), hidden biases within +-B1, a first layer with R = 1,
and sums below 2^24 (asserted for every model the page loads, on every position).

Writes handwired/nets/interp_player.json and handwired/nets/models.json (the three models the page
can load: the trained one, a hand-written one, random weights).
"""
import json

import numpy as np

from common import NETS, REPO, ROOT
from hw import LIMIT, Prog, total
from rules import ttt, ttt_states

N_IN, N_HID, N_OUT = 18, 64, 9
B1 = 1024                                   # hidden biases allowed in -B1..B1
H = N_IN * 127 + B1                         # no hidden activation can exceed this
GB = 1 << 20                                # |logit| the page can carry from turn to turn


def interp_player():
    """Inputs: the board before this turn (PX, PO), your move (9, one-hot or 0), the model (1,801:
    weights as unsigned bytes w + 128, biases as they are). Outputs: the board after both moves
    (X, O) and the model's 9 logits."""
    p = Prog()
    PX = [p.input(f"PX{i}") for i in range(9)]
    PO = [p.input(f"PO{i}") for i in range(9)]
    m = [p.input(f"m{i}") for i in range(9)]
    V1 = [[p.input(f"v1_{i}_{k}") for k in range(N_HID)] for i in range(N_IN)]   # w + 128, 0..255
    b1 = [p.input(f"b1_{k}", signed=True) for k in range(N_HID)]
    V2 = [[p.input(f"v2_{k}_{c}") for c in range(N_OUT)] for k in range(N_HID)]
    b2 = [p.input(f"b2_{c}", signed=True) for c in range(N_OUT)]
    PG = [p.input(f"PG{c}", signed=True) for c in range(N_OUT)]   # the logits of its last decision

    x = [PX[i] + m[i] for i in range(9)]                    # the board the model sees (your move is legal)
    o = PO
    inputs = x + o

    # layer 1: inputs are 0/1, so one gate per weight
    h = []
    for k in range(N_HID):
        gated = total(p.relu(V1[i][k] - 256 * (1 - inputs[i])) for i in range(N_IN))
        h.append(p.relu(gated - 128 * total(inputs) + b1[k]))

    # layer 2: every weight's bits at once, then h gated by each bit
    g = []
    for c in range(N_OUT):
        terms = []
        for k in range(N_HID):
            u = p.relu(V2[k][c])                             # one unit, so the bits below sit beside h
            q = {0: u, 8: 0}
            for j in range(1, 8):                            # q_j = floor(u / 2^j), never an exact .5
                q[j] = p.relu(2 * u + 1 - (1 << j), R=1 << (j + 1))
            bit = {j: q[j] - 2 * q[j + 1] for j in range(8)}
            sign = 1 - bit[7]                                # two's complement: w < 0
            terms.append(-128 * p.gate(sign, h[k], H))
            terms += [(1 << j) * p.gate(bit[j], h[k], H) for j in range(7)]
        g.append(p.split(total(terms) + b2[c]))

    # the rules, as in step 1: the best free square, only if you moved, didn't win, and it isn't full
    base = [ttt.TIE_SCALE * g[c] + (8 - c) for c in range(9)]
    best = p.pick(base, [[1 - x[c] - o[c] for c in range(9)]])
    xwon = p.OR(*[p.relu(x[a] + x[b] + x[cc] - 2) for a, b, cc in ttt.LINES])
    full = p.relu(total(x) + total(o) - 8)
    move = [p.AND(best[c], total(m), 1 - xwon, 1 - full) for c in range(9)]

    out = {f"X{i}": x[i] for i in range(9)}
    out.update({f"O{i}": o[i] + move[i] for i in range(9)})
    out.update({f"g{c}": g[c] for c in range(9)})
    go = total(m)                                  # this turn's logits if you moved, else the last ones
    out.update({f"G{c}": p.gate(go, g[c], GB, signed=True) + p.gate(1 - go, PG[c], GB, signed=True)
                for c in range(9)})
    return p.compile(out)


# ---------------------------------------------------------------- the models it can run

def flatten(layers):
    """A {W, b, R} 18-64-9 model as the interpreter's 1,801 weight inputs, in its input order."""
    W1, b1 = np.array(layers[0]["W"]), np.array(layers[0]["b"])
    W2, b2 = np.array(layers[1]["W"]), np.array(layers[1]["b"])
    assert W1.shape == (N_IN, N_HID) and W2.shape == (N_HID, N_OUT) and layers[0].get("R", 1) == 1
    assert W1.min() >= -128 and W1.max() <= 127 and W2.min() >= -128 and W2.max() <= 127   # int8
    assert np.abs(b1).max() <= B1
    return np.r_[W1.ravel() + 128, b1, W2.ravel() + 128, b2].astype(np.int64)


def hand_model():
    """Written by hand, no training: win if you can, else block, else centre, corner, edge.
    Hidden unit (c, line): 2 of your pieces on the line and c empty. 48 of the 64 units are used."""
    W1, b1 = np.zeros((N_IN, N_HID), dtype=int), np.zeros(N_HID, dtype=int)
    W2, b2 = np.zeros((N_HID, N_OUT), dtype=int), np.zeros(N_OUT, dtype=int)
    k = 0
    for who, value in ((9, 100), (0, 10)):          # O's own lines (offset 9) to win, X's lines to block
        for c in range(9):
            for line in ttt.LINES:
                if c not in line:
                    continue
                a, b = [i for i in line if i != c]
                W1[who + a, k] = W1[who + b, k] = 1
                W1[c, k] = W1[9 + c, k] = -1               # c must be empty
                b1[k] = -1
                W2[k, c] = value
                k += 1
    b2[:] = [2, 1, 2, 1, 3, 1, 2, 1, 2]                    # centre, then corners, then edges
    return [{"W": W1.tolist(), "b": b1.tolist(), "R": 1}, {"W": W2.tolist(), "b": b2.tolist()}]


def random_model(seed=3):
    """Untrained: int8 weights at random."""
    rng = np.random.default_rng(seed)
    return [{"W": rng.integers(-60, 61, (N_IN, N_HID)).tolist(), "b": rng.integers(-60, 61, N_HID).tolist(), "R": 1},
            {"W": rng.integers(-60, 61, (N_HID, N_OUT)).tolist(), "b": rng.integers(-60, 61, N_OUT).tolist()}]


def guest_move(layers, board):
    """What the model plays, computed directly: its logits, then step 1's rule for the move."""
    return int(np.argmax(ttt.int_scores(layers, [ttt.encode(board)])[0]))


# ---------------------------------------------------------------- check

def check(net, models):
    """Every position with O to move, for every model: the interpreter's move and logits equal the model's."""
    states = ttt_states()
    positions = [b for b, who in states.items() if who == "X" and not ttt.winner(b) and not all(b)]
    rows, want_o, want_g = [], [], []
    for name, layers in models.items():
        w = flatten(layers)
        for b in positions:
            for i in range(9):                     # every legal X move from here
                if b[i]:
                    continue
                after = ttt.play(b, i, "X")
                m = np.zeros(9, dtype=np.int64); m[i] = 1
                PX = [1 if v == "X" else 0 for v in b]
                PO = [1 if v == "O" else 0 for v in b]
                rows.append(np.r_[PX, PO, m, w, np.arange(9) * 1000 - 4000])
                o = np.array(PO)
                if not ttt.winner(after) and not all(after):
                    o[guest_move(layers, after)] += 1
                want_o.append(o)
                want_g.append(ttt.int_forward(layers, [ttt.encode(after)])[0])
    X = np.array(rows)
    out = net.forward(X)
    col = {n: k for k, n in enumerate(net.out_names)}
    O = out[:, [col[f"O{i}"] for i in range(9)]]
    G = out[:, [col[f"g{c}"] for c in range(9)]]
    assert (O == np.array(want_o)).all(), "interpreter's move differs from the model's"
    assert (G == np.array(want_g)).all(), "interpreter's logits differ from the model's"
    assert (out[:, [col[f"G{c}"] for c in range(9)]] == G).all(), "last logits not taken from this turn"
    assert np.abs(G).max() < GB
    idle = X.copy(); idle[:, 18:27] = 0                       # nobody moved: the old logits stay
    held = net.forward(idle[:200])[:, [col[f"G{c}"] for c in range(9)]]
    assert (held == np.arange(9) * 1000 - 4000).all(), "last logits not carried"
    mass = net.max_mass(X)
    assert mass < LIMIT, mass
    return f"{len(X):,} (position, X move, model) cases over {len(models)} models, max |partial sum| {mass:,}"


def check_extremes(net, n=300, seed=0):
    """Random weights at the edges of the allowed range, random boards: logits equal the model's."""
    rng = np.random.default_rng(seed)
    rows, want = [], []
    for _ in range(n):
        layers = [{"W": rng.choice([-128, 127, 0, -1, 1, 64], (N_IN, N_HID)).tolist(),
                   "b": rng.integers(-B1, B1 + 1, N_HID).tolist(), "R": 1},
                  {"W": rng.choice([-128, 127, 0, -1, 1, -64], (N_HID, N_OUT)).tolist(),
                   "b": rng.integers(-1000, 1001, N_OUT).tolist()}]
        cells = rng.permutation(9)[:rng.integers(0, 8)]
        PX, PO = np.zeros(9, dtype=np.int64), np.zeros(9, dtype=np.int64)
        for t, c in enumerate(cells):
            (PX if t % 2 == 0 else PO)[c] = 1
        rows.append(np.r_[PX, PO, np.zeros(9, dtype=np.int64), flatten(layers), np.zeros(9, dtype=np.int64)])
        want.append(ttt.int_forward(layers, [np.r_[PX, PO]])[0])
    out = net.forward(np.array(rows))
    col = {n: k for k, n in enumerate(net.out_names)}
    assert (out[:, [col[f"g{c}"] for c in range(9)]] == np.array(want)).all(), "extreme weights"
    return f"{n} random extreme-weight models exact in numpy (their sums can pass 2^24, so not for CSS)"


def fixtures(models, n=6, seed=2):
    """Random games for the browser test: X plays at random, each model answers."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, layers in models.items():
        games = []
        for _ in range(n):
            b, xs, os = (0,) * 9, [], []
            while not ttt.winner(b) and not all(b):
                i = int(rng.choice([c for c in range(9) if not b[c]]))
                b = ttt.play(b, i, "X"); xs.append(i)
                if ttt.winner(b) or all(b):
                    break
                o = guest_move(layers, b)
                b = ttt.play(b, o, "O"); os.append(o)
            games.append({"x": xs, "o": os, "result": ttt.winner(b) or "draw"})
        out[name] = games
    return out


def main():
    net = interp_player()
    learned = json.loads((REPO / "tic-tac-toe" / "model.json").read_text())["layers"]
    models = {"learned": learned, "hand": hand_model(), "random": random_model()}
    t = net.count_by_tag()
    print(f"interpreter: {net.depth} layers, {net.n_units:,} units, {net.n_weights:,} weights, "
          f"{len(net.in_names):,} inputs ({t})")
    print(check(net, models))
    print(check_extremes(net))
    (NETS / "interp_player.json").write_text(json.dumps(net.to_json()))
    (NETS / "models.json").write_text(json.dumps({k: {"layers": v, "flat": flatten(v).tolist()}
                                                   for k, v in models.items()}))
    (ROOT / "tests" / "interpreter.json").write_text(json.dumps(fixtures(models)))
    print("wrote interp_player.json, models.json, tests/interpreter.json")


if __name__ == "__main__":
    main()
