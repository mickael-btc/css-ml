"""Train the tic-tac-toe network and save it as integers to model.json.

    .venv/bin/python tic-tac-toe/scripts/train.py

  1. minimax over the whole game tree: for every reachable position where O is to move,
     the set of best moves (a win sooner beats a win later, a draw beats a loss)
  2. a small numpy MLP learns to put its probability mass on that set
  3. quantised to int8 weights and integer activations, like the digit network
  4. checked exhaustively: against every possible X strategy the integer model never
     loses and never plays an occupied cell
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
HIDDEN = [64]
TIE_SCALE = 16      # score = TIE_SCALE * logit + tie-break, so two cells never score the same
MASK = 1 << 20      # subtracted from occupied cells


# ---------------------------------------------------------------- the game

def winner(board):
    """board: tuple of 9 in {0, 'X', 'O'}."""
    for a, b, c in LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    return None


@lru_cache(maxsize=None)
def value(board, to_move):
    """Score for O: +(10 - moves) for an O win, -(10 - moves) for an X win, 0 draw."""
    w = winner(board)
    empties = [i for i in range(9) if not board[i]]
    if w or not empties:
        depth = 9 - len(empties)
        return 0 if not w else (10 - depth) * (1 if w == "O" else -1)
    scores = [value(play(board, i, to_move), other(to_move)) for i in empties]
    return max(scores) if to_move == "O" else min(scores)


def play(board, i, who):
    return board[:i] + (who,) + board[i + 1:]


def other(who):
    return "O" if who == "X" else "X"


def best_moves(board):
    empties = [i for i in range(9) if not board[i]]
    scores = {i: value(play(board, i, "O"), "X") for i in empties}
    top = max(scores.values())
    return [i for i in empties if scores[i] == top], scores


def o_positions():
    """Every reachable position with O to move and the game still on."""
    seen, stack, out = set(), [(0,) * 9], []
    while stack:
        board = stack.pop()
        if board in seen:
            continue
        seen.add(board)
        if winner(board) or all(board):
            continue
        n = sum(1 for v in board if v)
        who = "X" if n % 2 == 0 else "O"
        if who == "O":
            out.append(board)
        for i in range(9):
            if not board[i]:
                stack.append(play(board, i, who))
    return out


def encode(board):
    """18 inputs: 9 for X, then 9 for O."""
    return [1 if v == "X" else 0 for v in board] + [1 if v == "O" else 0 for v in board]


# ---------------------------------------------------------------- float training

def forward(weights, biases, x):
    activations, pre = [x], []
    for l, (W, b) in enumerate(zip(weights, biases)):
        z = activations[-1] @ W + b
        pre.append(z)
        activations.append(np.maximum(z, 0) if l < len(weights) - 1 else z)
    return pre, activations


def train(X, target, legal, steps=6000, seed=0):
    """Maximise log(sum of softmax mass on the best moves), with illegal cells masked out."""
    rng = np.random.default_rng(seed)
    sizes = [X.shape[1], *HIDDEN, 9]
    weights = [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes, sizes[1:])]
    biases = [np.zeros(b) for b in sizes[1:]]
    params = weights + biases
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    lr = 0.01

    for step in range(1, steps + 1):
        pre, act = forward(weights, biases, X)
        z = act[-1] - 1e9 * (1 - legal)
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z) * legal
        p /= p.sum(axis=1, keepdims=True)
        mass = (p * target).sum(axis=1, keepdims=True)
        # d(-log mass)/dz = p - p * target / mass
        grad = (p - p * target / mass) / len(X)

        grads_w, grads_b = [], []
        for l in reversed(range(len(weights))):
            grads_w.insert(0, act[l].T @ grad)
            grads_b.insert(0, grad.sum(axis=0))
            if l:
                grad = (grad @ weights[l].T) * (pre[l - 1] > 0)

        for k, g in enumerate(grads_w + grads_b):
            m[k] = 0.9 * m[k] + 0.1 * g
            v[k] = 0.999 * v[k] + 0.001 * g * g
            params[k] -= lr * (m[k] / (1 - 0.9 ** step)) / (np.sqrt(v[k] / (1 - 0.999 ** step)) + 1e-8)

        if step % 1000 == 0:
            picks = np.argmax(np.where(legal > 0, act[-1], -np.inf), axis=1)
            good = target[np.arange(len(X)), picks].mean()
            print(f"step {step}: loss {-np.log(mass).mean():.4f}  best move {good:.2%}")

    return weights, biases


# ---------------------------------------------------------------- integer model

def css_round(x):
    """A registered <integer> custom property rounds to nearest, halves toward +infinity."""
    return np.floor(x + 0.5).astype(np.int64)


def quantize(weights, biases, calibration):
    layers, scale = [], 1.0
    a = calibration.astype(np.int64)
    for l, (W, b) in enumerate(zip(weights, biases)):
        w_scale = 127 / np.abs(W).max()
        Wq = np.round(W * w_scale).astype(int)
        bq = np.round(b * scale * w_scale).astype(int)
        z = a @ Wq + bq
        if l < len(weights) - 1:
            R = max(1, int(np.percentile(np.maximum(z, 0), 99.9) / 255))
            R += 1 - R % 2  # odd, so z / R never lands on an exact .5
            layers.append({"W": Wq.tolist(), "b": bq.tolist(), "R": R})
            a = css_round(np.maximum(z, 0) / R)
            scale *= w_scale / R
        else:
            layers.append({"W": Wq.tolist(), "b": bq.tolist()})
            scale *= w_scale
    return layers, float(scale)


def int_forward(layers, X):
    """The network exactly as the CSS computes it: logits."""
    a = np.asarray(X, dtype=np.int64)
    for layer in layers:
        z = a @ np.array(layer["W"]) + np.array(layer["b"])
        a = css_round(np.maximum(z, 0) / layer["R"]) if "R" in layer else z
    return a


def int_scores(layers, X):
    """Logits scaled, tie-broken (earlier cell wins) and masked, as in the CSS."""
    X = np.asarray(X, dtype=np.int64)
    occupied = X[..., :9] + X[..., 9:]
    return TIE_SCALE * int_forward(layers, X) + (8 - np.arange(9)) - MASK * occupied


def int_move(layers, board):
    return int(np.argmax(int_scores(layers, [encode(board)])[0]))


# ---------------------------------------------------------------- exhaustive check

def sweep(layers):
    """Play the integer model against every X strategy. Returns (games, losses, illegal, results)."""
    games = []

    def go(board, xs, os):
        w = winner(board)
        if w or all(board):
            games.append((xs, os, w or "draw"))
            return
        for i in range(9):
            if board[i]:
                continue
            b = play(board, i, "X")
            if winner(b) or all(b):
                games.append((xs + [i], os, winner(b) or "draw"))
                continue
            o = int_move(layers, b)
            if b[o]:
                games.append((xs + [i], os + [o], "illegal"))
                continue
            go(play(b, o, "O"), xs + [i], os + [o])

    go((0,) * 9, [], [])
    return games


def main():
    positions = o_positions()
    X = np.array([encode(b) for b in positions], dtype=float)
    legal = np.array([[0.0 if v else 1.0 for v in b] for b in positions])
    target = np.zeros((len(positions), 9))
    for n, b in enumerate(positions):
        target[n, best_moves(b)[0]] = 1
    print(f"positions with O to move: {len(positions)}")

    for seed in range(20):
        weights, biases = train(X, target, legal, seed=seed)
        layers, scale = quantize(weights, biases, X)

        # every position: the integer model plays a best move, and legal scores never tie
        scores = int_scores(layers, X.astype(np.int64))
        picks = scores.argmax(axis=1)
        best = target[np.arange(len(X)), picks].mean()
        logits = int_forward(layers, X.astype(np.int64))
        ties = sum(len(set(l[legal[n] > 0])) < int(legal[n].sum()) for n, l in enumerate(logits))
        games = sweep(layers)
        results = {r: sum(1 for g in games if g[2] == r) for r in ("O", "draw", "X", "illegal")}
        print(f"seed {seed}: integer best move {best:.2%}, logit ties {ties}, sweep {results}, "
              f"max |score| {np.abs(TIE_SCALE * logits).max()}")
        if results["X"] == 0 and results["illegal"] == 0:
            break
    else:
        raise SystemExit("no seed gave a network that never loses")

    assert np.abs(TIE_SCALE * logits).max() + MASK < 2 ** 24

    (ROOT / "model.json").write_text(json.dumps({"layers": layers, "scale": scale}))
    fixtures(layers, games)
    print(f"model.json: {sum(np.array(l['W']).size for l in layers)} weights, "
          f"{len(games)} games against every X strategy, none lost")


def fixtures(layers, games):
    """A spread of complete games for the browser test, including every O win."""
    rng = np.random.default_rng(1)
    o_wins = [g for g in games if g[2] == "O"]
    draws = [g for g in games if g[2] == "draw"]
    pick = [o_wins[i] for i in rng.choice(len(o_wins), min(12, len(o_wins)), replace=False)]
    pick += [draws[i] for i in rng.choice(len(draws), min(12, len(draws)), replace=False)]
    (ROOT / "tests").mkdir(exist_ok=True)
    (ROOT / "tests" / "fixtures.json").write_text(json.dumps(
        [{"x": xs, "o": os, "result": r} for xs, os, r in pick]))


if __name__ == "__main__":
    main()
