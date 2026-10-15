"""Step 4: unfreeze a hand-written rule and keep training it. The reference the CSS is tested against.

    .venv/bin/python handwired/scripts/unfrozen.py        # train on the demo sets, print what happens
    .venv/bin/python handwired/scripts/unfrozen.py --fixtures

The rule is "red inside the box |x| <= 3 and |y| <= 3", compiled by hand into a 2 -> 4 -> 1 ReLU net:
each hidden unit measures how far a point is past one side of the box, and the output says red unless
any of them is positive:

    h1 = relu(8x - 32)   h2 = relu(-8x - 32)   h3 = relu(8y - 32)   h4 = relu(-8y - 32)
    s  = 64 - 16 (h1 + h2 + h3 + h4)           red if s > 0

Then every one of its 17 numbers is trained, integers only, full batch, for EPOCHS epochs: a margin
perceptron on s, backpropagated through the ReLUs (the derivative of relu(z) is [z > 0]):

    g_p  = t_p if t_p * s_p <= MARGIN else 0                 points still wrong, or too close
    v_k += round(sum_p g_p h_kp / RV)                        output weights
    d   += round(sum_p g_p * DS / RD)                        output bias
    a_k += round(v_k * sum_p g_p [z_kp > 0] x_p / RAB)       first layer, through the ReLU
    b_k += round(v_k * sum_p g_p [z_kp > 0] y_p / RAB)
    c_k += round(v_k * sum_p g_p [z_kp > 0] * CS / RA)

All divisors are odd, so CSS's rounding agrees with Python's, and every weight is clamped so every
sum stays below 2^24. Starting from random weights instead shows what the rule is worth.
"""
import json
import sys

import numpy as np

from common import ROOT

POINTS = 8                      # points at odd coordinates -7, -5, ..., 7
PLANE = 15                      # the background at every integer -7..7
EPOCHS = 16
H = 4
MARGIN = 56
RV, RD, RA, RAB = 31, 1, 31, 151
DS, CS = 4, 2
WMAX = 255                      # first layer, output weights
DMAX = 2047                     # output bias

POINT_XY = [(2 * c - 7, 7 - 2 * r) for r in range(POINTS) for c in range(POINTS)]   # row-major, y up
PLANE_XY = [(c - 7, 7 - r) for r in range(PLANE) for c in range(PLANE)]
PX = np.array([x for x, _ in POINT_XY], dtype=np.int64)
PY = np.array([y for _, y in POINT_XY], dtype=np.int64)

RULE = {"a": [8, -8, 0, 0], "b": [0, 0, 8, -8], "c": [-32] * 4, "v": [-16] * 4, "d": 64}


def random_start(seed=5):
    rng = np.random.default_rng(seed)
    return {"a": rng.integers(-8, 9, H).tolist(), "b": rng.integers(-8, 9, H).tolist(),
            "c": rng.integers(-32, 33, H).tolist(), "v": rng.integers(-16, 17, H).tolist(), "d": 0}


STARTS = {"rule": RULE, "random": random_start()}


def divide(a, d):
    """Round a / d to nearest, halves up. d is odd, so there are no halves."""
    assert d % 2 == 1
    return np.floor_divide(2 * a + d, 2 * d)


def scores(w, xs, ys):
    a, b, c, v = (np.array(w[k], dtype=np.int64) for k in "abcv")
    z = np.outer(xs, a) + np.outer(ys, b) + c                     # (P, H)
    return z, np.maximum(z, 0), np.maximum(z, 0) @ v + w["d"]


def train(t, start):
    """t: 64 labels in {-1, 0, 1}. Returns every epoch's weights and errors."""
    t = np.asarray(t, dtype=np.int64)
    w = {k: (np.array(v, dtype=np.int64) if k != "d" else int(v)) for k, v in start.items()}
    hist, errors = [], []
    for e in range(EPOCHS + 1):
        z, h, s = scores(w, PX, PY)
        hist.append({k: (v.tolist() if k != "d" else int(v)) for k, v in w.items()})
        errors.append(int(np.sum((t != 0) & (t * s <= 0))))
        if e == EPOCHS:
            break
        g = np.where(t * s <= MARGIN, t, 0)
        f = (z > 0).astype(np.int64)
        gf = g[:, None] * f                                        # (P, H)
        v = w["v"]
        new = {
            "v": np.clip(v + divide(g @ h, RV), -WMAX, WMAX),
            "d": int(np.clip(w["d"] + divide(int(g.sum()) * DS, RD), -DMAX, DMAX)),
            "a": np.clip(w["a"] + divide(v * (PX @ gf), RAB), -WMAX, WMAX),
            "b": np.clip(w["b"] + divide(v * (PY @ gf), RAB), -WMAX, WMAX),
            "c": np.clip(w["c"] + divide(v * gf.sum(axis=0) * CS, RA), -WMAX, WMAX),
        }
        w = new
    return {"w": hist, "errors": errors, "n": int(np.sum(t != 0))}


# worst cases, for 2^24: |z| <= 255*7*2 + 255, h the same, |s| <= 4 * 255 * |h| + DMAX
ZMAX = WMAX * 7 * 2 + WMAX
assert 4 * WMAX * ZMAX + DMAX < 2 ** 24
assert WMAX * POINTS * POINTS * 7 * 2 < 2 ** 24        # v * sum(g f x): the first-layer updates


# ---------------------------------------------------------------- demo sets

def label(rule, keep=lambda i: True):
    return [rule(x, y) if keep(i) else 0 for i, (x, y) in enumerate(POINT_XY)]


def box(x, y):
    return 1 if abs(x) <= 3 and abs(y) <= 3 else -1


DATASETS = {
    "box": label(box, lambda i: i % 2 == 0 or abs(POINT_XY[i][0]) <= 5),
    "wider": label(lambda x, y: 1 if abs(x) <= 5 and abs(y) <= 3 else -1, lambda i: i % 2 == 0 or abs(POINT_XY[i][0]) <= 5),
    "shifted": label(lambda x, y: 1 if -1 <= x <= 5 and -1 <= y <= 5 else -1, lambda i: (i * 3) % 4 != 1),
    "stripe": label(lambda x, y: 1 if abs(x) <= 3 else -1, lambda i: (i * 5) % 3 != 0),
    "few": [1 if i in (27, 28) else -1 if i in (0, 7, 56, 63, 31) else 0 for i in range(POINTS * POINTS)],
    "empty": [0] * (POINTS * POINTS),
}


def show(name, start, t, result):
    w = result["w"][-1]
    _, _, s = scores(w, np.array([x for x, _ in PLANE_XY]), np.array([y for _, y in PLANE_XY]))
    s = s.reshape(PLANE, PLANE)
    tp = np.array(t).reshape(POINTS, POINTS)
    print(f"{name:8} from {start:6}: errors {result['errors']}")
    for r in range(PLANE):
        row = ""
        for c in range(PLANE):
            if r % 2 == 0 and c % 2 == 0 and tp[r // 2, c // 2]:
                row += "R" if tp[r // 2, c // 2] > 0 else "B"
            else:
                row += "+" if s[r, c] > 0 else "·"
        print("      " + row)


def main():
    sets = []
    for name, t in DATASETS.items():
        for start in STARTS:
            result = train(t, STARTS[start])
            if "-q" not in sys.argv:
                show(name, start, t, result)
            else:
                print(f"{name:8} from {start:6}: errors {result['errors']}")
            sets.append({"name": name, "start": start, "t": t, **result})
    if "--fixtures" in sys.argv:
        path = ROOT / "tests" / "unfrozen.json"
        path.write_text(json.dumps({"epochs": EPOCHS, "sets": sets}))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
