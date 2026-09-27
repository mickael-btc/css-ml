"""The integer training rule the CSS runs, in Python: the reference the browser test compares against.

    .venv/bin/python playground/scripts/reference.py        # train on the demo datasets, print the boundaries
    .venv/bin/python playground/scripts/reference.py --fixtures   # also write tests/fixtures.json

Points sit on a POINTS x POINTS grid at odd coordinates -9, -7, ..., 9. The background plane is
PLANE x PLANE cells at every integer -9..9, so every point is also a plane cell.

Each position has a label t in {-1 (blue), 0 (empty), +1 (red)} and six fixed integer features:
    45, 9x, 9y, x² - 33, y² - 33, xy        (all at most 81 in size, so none dominates)

Training is a full-batch margin perceptron, all integers, unrolled for EPOCHS epochs from w = 0:
    s_p   = sum_k w_k f_k(p)                          score of position p
    g_p   = t_p if t_p * s_p <= MARGIN else 0         points still wrong, or too close to the line
    w_k  <- clamp(-WMAX, w_k + round(sum_p g_p f_k(p) / RATE), WMAX)
RATE is odd, so the division never lands on .5 and CSS's round-to-nearest agrees with Python.
Clamping the weights keeps every score below 2^24, so 32-bit float CSS engines stay exact.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

POINTS = 10
PLANE = 19
EPOCHS = 24
MARGIN = 3000
RATE = 15
WMAX = 24000
FEATURES = ["1", "x", "y", "x²", "y²", "xy"]


def features(x, y):
    """Six integer features of a plane coordinate, x and y in -9..9."""
    return [45, 9 * x, 9 * y, x * x - 33, y * y - 33, x * y]


POINT_XY = [(2 * c - 9, 9 - 2 * r) for r in range(POINTS) for c in range(POINTS)]  # row-major, y up
PLANE_XY = [(c - 9, 9 - r) for r in range(PLANE) for c in range(PLANE)]
F_POINTS = np.array([features(x, y) for x, y in POINT_XY], dtype=np.int64)  # (100, 6)
F_PLANE = np.array([features(x, y) for x, y in PLANE_XY], dtype=np.int64)    # (361, 6)
FMAX = int(np.abs(F_PLANE).max())
BOUND = WMAX * int(np.abs(F_PLANE).sum(axis=1).max())
assert BOUND < 2 ** 24, BOUND


def divide(a, d):
    """Round a / d to nearest. d is odd, so there are no ties."""
    assert d % 2 == 1
    return np.floor_divide(2 * a + d, 2 * d)


def train(t):
    """t: 100 labels in {-1, 0, 1}. Returns every epoch's weights, scores and error counts."""
    t = np.asarray(t, dtype=np.int64)
    w = np.zeros(len(FEATURES), dtype=np.int64)
    weights, errors = [w.copy()], []
    for _ in range(EPOCHS):
        s = F_POINTS @ w
        errors.append(int(np.sum((t != 0) & (t * s <= 0))))
        g = np.where(t * s <= MARGIN, t, 0)
        w = np.clip(w + divide(g @ F_POINTS, RATE), -WMAX, WMAX)
        weights.append(w.copy())
    s = F_POINTS @ w
    errors.append(int(np.sum((t != 0) & (t * s <= 0))))
    for w in weights:
        assert np.abs(F_PLANE @ w).max() < 2 ** 24
    return {"w": [w.tolist() for w in weights], "errors": errors, "n": int(np.sum(t != 0))}


def plane(w):
    return (F_PLANE @ np.asarray(w, dtype=np.int64)).tolist()


# ---------------------------------------------------------------- demo datasets (by grid position)

def dataset(rule):
    """Label every position with rule(x, y) -> -1/0/1."""
    return [rule(x, y) for x, y in POINT_XY]


def sparse(t, keep):
    """Keep only positions whose index satisfies keep(i), to look hand-placed."""
    return [v if keep(i) else 0 for i, v in enumerate(t)]


DATASETS = {
    "linear": sparse(dataset(lambda x, y: 1 if y > x / 2 + 1 else -1), lambda i: (i * 7) % 3 == 0),
    "circle": sparse(dataset(lambda x, y: 1 if x * x + y * y < 30 else (-1 if x * x + y * y > 45 else 0)),
                     lambda i: (i * 5) % 2 == 0),
    "xor": sparse(dataset(lambda x, y: 1 if x * y > 0 else -1), lambda i: (i * 3) % 4 == 1),
    "empty": [0] * (POINTS * POINTS),
    "one-class": sparse(dataset(lambda x, y: 1 if x < -3 else 0), lambda i: i % 3 == 0),
    "few": [1 if i in (22, 23) else -1 if i in (76, 77, 67) else 0 for i in range(POINTS * POINTS)],
}
# the test reaches this one from "few" by clicking: point 22 removed, point 23 repainted blue
DATASETS["few-edited"] = [0 if i == 22 else -1 if i == 23 else v for i, v in enumerate(DATASETS["few"])]


def show(name, t, result):
    s = np.array(plane(result["w"][-1])).reshape(PLANE, PLANE)
    tp = np.array(t).reshape(POINTS, POINTS)
    print(f"{name}: errors per epoch {result['errors']}  final w {result['w'][-1]}")
    for r in range(PLANE):
        row = ""
        for c in range(PLANE):
            if r % 2 == 0 and c % 2 == 0 and tp[r // 2, c // 2]:
                row += "R" if tp[r // 2, c // 2] > 0 else "B"
            else:
                row += "+" if s[r, c] > 0 else ("-" if s[r, c] < 0 else "0")
        print("   " + row)


def main():
    sets = []
    for name, t in DATASETS.items():
        result = train(t)
        show(name, t, result)
        sets.append({"name": name, "t": t, **result})
    fixtures = {"epochs": EPOCHS, "features": F_PLANE.tolist(), "sets": sets}
    print(f"max |score| bound {BOUND:,} (< 2^24 = {2 ** 24:,})")
    if "--fixtures" in sys.argv:
        path = ROOT / "tests" / "fixtures.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(fixtures))
        print(f"wrote {path.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    main()
