"""Train the Connect Four network and save it as integers to model.json.

    .venv/bin/python connect-four/scripts/train.py

  1. positions where the network (yellow, second player) is to move, from games between
     opponents of every strength: random, tactical, and alpha-beta searches with noise
  2. every position labelled by a depth-7 alpha-beta search: the value of each column
  3. a numpy MLP learns a softmax over those values (board mirrored left-right for twice the data)
  4. quantised to int8 weights and integer activations, like the digit network
  5. four rounds of DAgger: the integer model plays, and the positions it reaches get labelled too
  6. evaluated with the hard-coded tactics on, as the CSS plays, against four opponents
"""
import json
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

from game import (BLOCK, CELLS, COLS, DANGER, MASK, ROWS, TIE_SCALE, WIN, Board, bb_from, bb_won,  # noqa: E402
                  can_play, cell, css_round, encode, int_forward, int_move, int_scores, move_values,
                  negamax, play_bit, tactics, WIN_SCORE)

HIDDEN = [128]
DEPTH = 7          # search depth for the labels
TEMPERATURE = 12   # softmax over search values (heuristic units) for the targets
ROUNDS = 5         # 1 + 4 rounds of DAgger
CLIP = 400         # values beyond this are all "winning" or "losing"


# ---------------------------------------------------------------- players

def immediate(b, who):
    """(winning columns, columns that stop the other side's immediate win) for `who`."""
    mine, theirs = (b.me, b.you) if who == "me" else (b.you, b.me)
    from game import completes
    wins = [c for c in b.legal() if completes(mine, cell(b.height[c], c))]
    blocks = [c for c in b.legal() if completes(theirs, cell(b.height[c], c))]
    return wins, blocks


def random_player(b, who, rng):
    return rng.choice(b.legal())


def greedy_player(b, who, rng):
    """Wins if it can, blocks if it must, else random."""
    wins, blocks = immediate(b, who)
    return rng.choice(wins or blocks or b.legal())


def search_player(depth, noise=0.0):
    def play(b, who, rng):
        if rng.random() < noise:
            return rng.choice(b.legal())
        v = move_values(b, who, depth)
        top = max(v.values())
        return rng.choice([c for c in v if v[c] == top])
    return play


def model_player(layers, use_tactics=True):
    def play(b, who, rng):
        assert who == "me"
        return int_move(layers, b, use_tactics)
    return play


def play_game(you, me, rng, record=None):
    """`you` moves first. Returns (your columns, its columns, winner or 'draw')."""
    b, ys, ms = Board(), [], []
    while True:
        c = you(b, "you", rng)
        b.play(c, "you"); ys.append(c)
        if b.won("you"):
            return ys, ms, "you"
        if record is not None:
            record.append(b.copy())
        c = me(b, "me", rng)
        b.play(c, "me"); ms.append(c)
        if b.won("me"):
            return ys, ms, "me"
        if b.full():
            return ys, ms, "draw"


def opponent(rng):
    """A first player of random strength, for varied positions."""
    k = rng.random()
    if k < 0.15:
        return random_player
    if k < 0.35:
        return greedy_player
    return search_player(rng.choice([2, 3, 4, 5]), noise=rng.choice([0.0, 0.1, 0.25]))


# ---------------------------------------------------------------- labels

def label(key):
    you, me = key
    b = Board()
    b.you, b.me = list(you), list(me)
    b.height = [sum(you[cell(r, c)] + me[cell(r, c)] for r in range(ROWS)) for c in range(COLS)]
    v = move_values(b, "me", DEPTH)
    return key, v


CACHE = ROOT / "data" / "labels.json"  # git-ignored: search values, so reruns skip the slow part


def load_cache():
    if not CACHE.exists():
        return {}
    return {(tuple(k[0]), tuple(k[1])): {int(c): x for c, x in v.items()} for k, v in json.loads(CACHE.read_text())}


def save_cache(cache):
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps([[list(k), v] for k, v in cache.items()]))


def label_all(boards, cache):
    keys = {(tuple(b.you), tuple(b.me)) for b in boards} - cache.keys()
    t = time.time()
    with Pool() as pool:
        for key, v in pool.imap_unordered(label, sorted(keys), chunksize=16):
            cache[key] = v
    print(f"labelled {len(keys)} new positions in {time.time() - t:.0f}s ({len(cache)} total)")


def dataset(cache):
    X, T, L = [], [], []
    for (you, me), v in cache.items():
        for mirror in (False, True):
            if mirror:
                flip = [cell(r, COLS - 1 - c) for r in range(ROWS) for c in range(COLS)]
                you_, me_ = [you[i] for i in flip], [me[i] for i in flip]
                v_ = {COLS - 1 - c: x for c, x in v.items()}
            else:
                you_, me_, v_ = list(you), list(me), v
            legal = np.zeros(COLS)
            vals = np.full(COLS, -np.inf)
            for c, x in v_.items():
                legal[c] = 1
                vals[c] = np.clip(x, -CLIP, CLIP) / TEMPERATURE
            p = np.exp(vals - vals.max()) * legal
            X.append(you_ + me_); T.append(p / p.sum()); L.append(legal)
    return np.array(X, dtype=float), np.array(T), np.array(L)


# ---------------------------------------------------------------- float training

def forward(weights, biases, x):
    activations, pre = [x], []
    for l, (W, b) in enumerate(zip(weights, biases)):
        z = activations[-1] @ W + b
        pre.append(z)
        activations.append(np.maximum(z, 0) if l < len(weights) - 1 else z)
    return pre, activations


def train(X, target, legal, epochs=60, seed=0, batch=256):
    rng = np.random.default_rng(seed)
    sizes = [X.shape[1], *HIDDEN, COLS]
    weights = [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes, sizes[1:])]
    biases = [np.zeros(b) for b in sizes[1:]]
    params = weights + biases
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    step = 0
    for epoch in range(epochs):
        lr = 0.003 * (0.5 * (1 + np.cos(np.pi * epoch / epochs))) + 1e-4
        order = rng.permutation(len(X))
        for s in range(0, len(X), batch):
            idx = order[s:s + batch]
            x, t, lg = X[idx], target[idx], legal[idx]
            pre, act = forward(weights, biases, x)
            z = act[-1] - 1e9 * (1 - lg)
            z = z - z.max(axis=1, keepdims=True)
            p = np.exp(z) * lg
            p /= p.sum(axis=1, keepdims=True)
            grad = (p - t) / len(idx)
            grads_w, grads_b = [], []
            for l in reversed(range(len(weights))):
                grads_w.insert(0, act[l].T @ grad + 1e-5 * weights[l])
                grads_b.insert(0, grad.sum(axis=0))
                if l:
                    grad = (grad @ weights[l].T) * (pre[l - 1] > 0)
            step += 1
            for k, g in enumerate(grads_w + grads_b):
                m[k] = 0.9 * m[k] + 0.1 * g
                v[k] = 0.999 * v[k] + 0.001 * g * g
                params[k] -= lr * (m[k] / (1 - 0.9 ** step)) / (np.sqrt(v[k] / (1 - 0.999 ** step)) + 1e-8)
        if (epoch + 1) % 20 == 0:
            _, act = forward(weights, biases, X)
            pick = np.argmax(np.where(legal > 0, act[-1], -np.inf), axis=1)
            best = (target[np.arange(len(X)), pick] >= target.max(axis=1) - 1e-9).mean()
            print(f"  epoch {epoch + 1}: top move matches the search {best:.1%}")
    return weights, biases


def quantize(weights, biases, calibration):
    layers, a = [], calibration.astype(np.int64)
    for l, (W, b) in enumerate(zip(weights, biases)):
        w_scale = 127 / np.abs(W).max()
        Wq = np.round(W * w_scale).astype(int)
        if l == 0:
            bq = np.round(b * w_scale).astype(int)
        else:
            bq = np.round(b * prev_scale * w_scale).astype(int)
        z = a @ Wq + bq
        if l < len(weights) - 1:
            R = max(1, int(np.percentile(np.maximum(z, 0), 99.9) / 255))
            R += 1 - R % 2  # odd, so z / R never lands on an exact .5
            layers.append({"W": Wq.tolist(), "b": bq.tolist(), "R": R})
            a = css_round(np.maximum(z, 0) / R)
            prev_scale = w_scale / R if l == 0 else prev_scale * w_scale / R
        else:
            layers.append({"W": Wq.tolist(), "b": bq.tolist()})
    return layers


# ---------------------------------------------------------------- positions and evaluation

def sample_positions(n, seed=3):
    """Positions the network faces, for the bar temperature."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        record = []
        play_game(opponent(rng), search_player(3, noise=0.2), rng, record)
        out += record
    return out[:n]


def collect(n_games, me_factory, seed):
    rng = random.Random(seed)
    boards = []
    for _ in range(n_games):
        play_game(opponent(rng), me_factory(rng), rng, boards)
    return boards


def varied(player, opening=2):
    """A deterministic search always plays the same game, so its first moves are random."""
    def play(b, who, rng):
        return rng.choice(b.legal()) if sum(b.you) < opening else player(b, who, rng)
    return play


def evaluate(layers, games=200, seed=7, use_tactics=True):
    opponents = {
        "random": (random_player, games),
        "greedy (win, else block, else random)": (greedy_player, games),
        "alpha-beta depth 2, 2 random opening moves": (varied(search_player(2)), games // 2),
        "alpha-beta depth 4, 2 random opening moves": (varied(search_player(4)), games // 2),
    }
    me = model_player(layers, use_tactics)
    report = {}
    for name, (opp, n) in opponents.items():
        rng = random.Random(seed)
        results = [play_game(opp, me, rng)[2] for _ in range(n)]
        report[name] = {r: results.count(r) / n for r in ("me", "draw", "you")}
    return report


def fmt(report):
    return "\n".join(f"  vs {k:40s} network wins {v['me']:.0%}  draws {v['draw']:.0%}  loses {v['you']:.0%}"
                     for k, v in report.items())


def fixtures(layers):
    """Complete games for the browser test: wins for each side against several opponents."""
    rng = random.Random(11)
    me = model_player(layers)
    d2, d4 = varied(search_player(2)), varied(search_player(4))
    want = [(random_player, "me"), (greedy_player, "me"), (d2, "me"), (d4, "me"), (d4, "you"), (d2, "you"),
            (greedy_player, "you"), (d4, "draw"), (random_player, "me"), (d2, "me")]
    games = []
    for opp, result in want:
        for _ in range(400):
            ys, ms, r = play_game(opp, me, rng)
            if r == result and all(g["you"] != ys for g in games):
                games.append({"you": ys, "me": ms, "result": r})
                break
    (ROOT / "tests").mkdir(exist_ok=True)
    (ROOT / "tests" / "fixtures.json").write_text(json.dumps(games))
    return games


def main():
    t0 = time.time()
    cache = load_cache()
    boards = collect(3000, lambda rng: search_player(4, noise=0.15), seed=1)
    label_all(boards, cache)
    save_cache(cache)

    layers = None
    for round_ in range(ROUNDS):
        X, T, L = dataset(cache)
        print(f"round {round_}: training on {len(X)} positions (with mirrors)")
        weights, biases = train(X, T, L, seed=round_)
        layers = quantize(weights, biases, X)
        report = evaluate(layers, games=100)
        print(fmt(report))
        if round_ < ROUNDS - 1:  # DAgger: label the positions the integer model itself reaches
            boards = collect(2500, lambda rng: model_player(layers), seed=100 + round_)
            label_all(boards, cache)
            save_cache(cache)

    # the integer model's scores stay far from the tactic weights and below 2^24
    logits = int_forward(layers, X.astype(np.int64))
    peak = int(np.abs(TIE_SCALE * logits).max())
    print(f"max |8 * logit| = {peak}")
    assert peak < DANGER // 2, "logits would overlap the tactic weights"
    assert MASK + WIN + BLOCK + DANGER + peak < 2 ** 24

    report = evaluate(layers, games=400)
    bare = evaluate(layers, games=400, use_tactics=False)
    print("final, with the hard-coded tactics (as the CSS plays):\n" + fmt(report))
    print("the network alone, no tactics:\n" + fmt(bare))

    (ROOT / "model.json").write_text(json.dumps({"layers": layers, "eval": report, "eval_network_only": bare}))
    games = fixtures(layers)
    print(f"fixtures: {len(games)} games, results {[g['result'] for g in games]}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
