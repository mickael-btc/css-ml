"""Step 3: a network that runs a copy of the network that runs a network.

    .venv/bin/python handwired/scripts/tower.py

`interpret(guest)` compiles a ReLU net whose inputs are the guest's inputs plus every one of the
guest's weights and biases, and whose outputs are the guest's outputs. It knows the guest's wiring
(which unit feeds which, each unit's rounding divisor) but none of its numbers. Then we apply it to
its own output:

  N    a tiny 2 -> 2 -> 1 net (XOR, AND, OR, ... depending on its 9 numbers)
  I1 = interpret(N)     inputs: x, N's 9 numbers
  I2 = interpret(I1)    inputs: x, N's 9 numbers, and every number of I1

so I2(x, N, I1) = I1(x, N) = N(x). I2 is a network running a copy of its own compiler's output.

Products of two inputs are built like step 2: a weight comes in as u = w - lo, an unsigned integer
of B bits; its bits all at once, q_j = floor(u / 2^j) = round(relu(2u + 1 - 2^j) / 2^(j+1)),
bit_j = q_j - 2 q_(j+1); then bit * a = relu(a - A (1 - bit)) for an activation 0 <= a <= A, and
w * a = lo * a + sum 2^j (bit_j * a). Signed activations are split into relu(a) and relu(-a) first.

The contract: weights within each layer's range [lo, lo + 2^B), and activations within the bounds
the interpreter was built for. Those come from interval arithmetic over the guest's input ranges, so
they hold for every input; for I1 they also hold for every N in range.

Writes handwired/nets/tower.json and handwired/tests/tower.json.
"""
import json

import numpy as np

from common import NETS, ROOT
from hw import LIMIT, Layer, Net, Prog, Val, css_round, total

# ---------------------------------------------------------------- the guest N

N_LO, N_BITS = -4, 3          # N's weights: -4..3
N_BIAS = 4                    # N's biases: -4..4
PRESETS = {                   # x1, x2 -> 2 hidden ReLUs -> 1 output
    "XOR":  {"W1": [[1, 1], [1, 1]], "b1": [0, -1], "W2": [1, -2], "b2": 0},
    "AND":  {"W1": [[1, 0], [1, 0]], "b1": [-1, 0], "W2": [1, 0], "b2": 0},
    "OR":   {"W1": [[1, 1], [1, 1]], "b1": [0, -1], "W2": [1, -1], "b2": 0},
    "NAND": {"W1": [[1, 0], [1, 0]], "b1": [-1, 0], "W2": [-1, 0], "b2": 1},
    "x1 − x2": {"W1": [[1, -1], [-1, 1]], "b1": [0, 0], "W2": [1, -1], "b2": 0},
}


def n_topology():
    """N's wiring with every slot present, its numbers set to zero (they are inputs, not part of it)."""
    l1 = Layer(2, [0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 0, 0], [0, 0], [1, 1], ["learned"] * 2)
    l2 = Layer(2, [0, 1], [0, 0], [0, 0], [0], None, ["out"])
    return Net([l1, l2], ["x1", "x2"], ["y"])


def n_numbers(preset):
    """N's 9 numbers in the order interpret() reads them: per layer, edge weights (in the layer's
    edge order) then biases."""
    p = PRESETS[preset]
    W1, l1 = np.array(p["W1"]), n_topology().layers[0]
    return [int(W1[s, t]) for s, t in zip(l1.src, l1.dst)] + list(p["b1"]) + list(p["W2"]) + [p["b2"]]


def n_forward(preset, x):
    p = PRESETS[preset]
    h = np.maximum(np.array(x) @ np.array(p["W1"]) + np.array(p["b1"]), 0)
    return int(h @ np.array(p["W2"]) + p["b2"])


# ---------------------------------------------------------------- the compiler

def interval_bounds(net, in_lo, in_hi, w_lo=None, w_hi=None, b_lo=None, b_hi=None):
    """[lo, hi] of every unit's activation, layer by layer, by interval arithmetic. Weights and biases
    are either the net's own (fixed) or ranges per layer (w_lo[d] .. w_hi[d])."""
    lo, hi = [np.asarray(in_lo, dtype=np.int64)], [np.asarray(in_hi, dtype=np.int64)]
    for d, layer in enumerate(net.layers):
        a_lo, a_hi = lo[-1][layer.src], hi[-1][layer.src]
        if w_lo is None:
            wl = wh = layer.w
        else:
            wl, wh = np.full(len(layer.w), w_lo[d]), np.full(len(layer.w), w_hi[d])
        corners = np.stack([wl * a_lo, wl * a_hi, wh * a_lo, wh * a_hi])
        z_lo = np.zeros(layer.n_out, dtype=np.int64); z_hi = np.zeros(layer.n_out, dtype=np.int64)
        np.add.at(z_lo, layer.dst, corners.min(axis=0))
        np.add.at(z_hi, layer.dst, corners.max(axis=0))
        if b_lo is None:
            z_lo += layer.b; z_hi += layer.b
        else:
            z_lo += b_lo[d]; z_hi += b_hi[d]
        if layer.R is not None:
            z_lo, z_hi = css_round(np.maximum(z_lo, 0) / layer.R), css_round(np.maximum(z_hi, 0) / layer.R)
        lo.append(z_lo); hi.append(z_hi)
    return lo, hi


def interpret(guest, ranges, bias_ranges, in_lo, in_hi, act_lo, act_hi, prefix):
    """A Prog net running `guest`: inputs are the guest's inputs, then per layer its edge weights
    (as u = w - lo, ranges[d] = (lo, bits)) and biases (signed, within bias_ranges[d]). Activation
    bounds act_lo/act_hi[d] (from interval_bounds) are the contract the gates rely on.
    Returns (net, names of the number inputs)."""
    p = Prog()
    a = []
    for i, name in enumerate(guest.in_names):
        a.append(p.input(name, signed=in_lo[i] < 0))
    numbers = []
    for d, layer in enumerate(guest.layers):
        lo_w, bits = ranges[d]
        us = [p.input(f"{prefix}w{d}_{e}") for e in range(len(layer.w))]
        bs = [p.input(f"{prefix}b{d}_{k}", signed=True) for k in range(layer.n_out)]
        numbers += [f"{prefix}w{d}_{e}" for e in range(len(layer.w))] + [f"{prefix}b{d}_{k}" for k in range(layer.n_out)]

        # each source activation, as nonnegative parts with their bounds
        parts = {}
        for s in set(layer.src.tolist()):
            lo_a, hi_a = int(act_lo[d][s]), int(act_hi[d][s])
            if lo_a >= 0:
                parts[s] = [(1, a[s], hi_a)]
            else:
                pos, neg = (p.relu(a[s]) if hi_a > 0 else 0), p.relu(-a[s])
                parts[s] = ([(1, pos, hi_a)] if hi_a > 0 else []) + [(-1, neg, -lo_a)]

        z = [Val() + bs[k] for k in range(layer.n_out)]
        for e, (s, t) in enumerate(zip(layer.src.tolist(), layer.dst.tolist())):
            u = us[e]
            bit = None
            for sign, v, bound in parts[s]:
                if bound == 0:
                    continue
                if bound == 1:                       # a 0/1 activation: one gate for the whole weight
                    top = (1 << bits) - 1
                    z[t] = z[t] + sign * (p.relu(u - (top + 1) * (1 - v)) + lo_w * v)
                    continue
                if bit is None:
                    uu = p.relu(u)
                    q = {0: uu, bits: 0}
                    for j in range(1, bits):
                        q[j] = p.relu(2 * uu + 1 - (1 << j), R=1 << (j + 1))
                    bit = [q[j] - 2 * q[j + 1] for j in range(bits)]
                prod = lo_w * v + total((1 << j) * p.gate(bit[j], v, bound) for j in range(bits))
                z[t] = z[t] + sign * prod
        if layer.R is None:
            a = z
        else:
            a = [p.relu(z[k], R=int(layer.R[k])) for k in range(layer.n_out)]
    net = p.compile({n: v for n, v in zip(guest.out_names, a)})
    return net, numbers


def bits_for(span):
    return max(1, int(span).bit_length())


def layer_ranges(net):
    """Per layer: (lo, bits) covering every weight in it, and a symmetric bias range."""
    ranges, biases = [], []
    for layer in net.layers:
        M = int(np.abs(layer.w).max()) if len(layer.w) else 0
        lo = -M
        ranges.append((lo, bits_for(2 * M)))
        B = int(np.abs(layer.b).max()) if len(layer.b) else 0
        biases.append((-B, B))
    return ranges, biases


def encode(net, ranges):
    """A net's numbers as the inputs interpret() expects: per layer, u = w - lo per edge, then biases."""
    out = []
    for layer, (lo, bits) in zip(net.layers, ranges):
        u = layer.w - lo
        assert (u >= 0).all() and (u < (1 << bits)).all()
        out += u.tolist() + layer.b.tolist()
    return out


# ---------------------------------------------------------------- build the tower

def number_bounds(net, ranges, biases):
    """Input ranges of a net's numbers as interpret() takes them: u in 0..2^bits - 1, biases in range."""
    lo, hi = [], []
    for (_, bits), (bl, bh), layer in zip(ranges, biases, net.layers):
        lo += [0] * len(layer.w) + [bl] * layer.n_out
        hi += [(1 << bits) - 1] * len(layer.w) + [bh] * layer.n_out
    return lo, hi


def build(levels=2):
    """[(net, ranges)] for I1 .. I_levels. I_(k+1) = interpret(I_k)."""
    N = n_topology()
    n_ranges, n_bias = [(N_LO, N_BITS)] * 2, [(-N_BIAS, N_BIAS)] * 2
    in_lo, in_hi = [0, 0], [1, 1]
    lo, hi = interval_bounds(N, in_lo, in_hi, [N_LO] * 2, [N_LO + (1 << N_BITS) - 1] * 2, [-N_BIAS] * 2, [N_BIAS] * 2)
    guest, ranges, biases = N, n_ranges, n_bias
    tower = []
    for k in range(levels):
        net, _ = interpret(guest, ranges, biases, in_lo, in_hi, lo, hi, "nijklm"[k])
        nl, nh = number_bounds(guest, ranges, biases)
        in_lo, in_hi = in_lo + nl, in_hi + nh            # the new net's inputs: the old inputs + the old numbers
        lo, hi = interval_bounds(net, in_lo, in_hi)      # its activations, for every input in range
        ranges, biases = layer_ranges(net)
        tower.append((net, ranges, int(max(h.max() for h in hi))))
        guest = net
    return N, tower


def numbers_of_n(preset):
    """N's 9 numbers as I1 reads them (weights offset by -lo)."""
    vals = n_numbers(preset)
    return [v - N_LO for v in vals[:4]] + vals[4:6] + [v - N_LO for v in vals[6:8]] + vals[8:]


def random_n(rng):
    return {"W1": rng.integers(N_LO, N_LO + 8, (2, 2)).tolist(), "b1": rng.integers(-N_BIAS, N_BIAS + 1, 2).tolist(),
            "W2": rng.integers(N_LO, N_LO + 8, 2).tolist(), "b2": int(rng.integers(-N_BIAS, N_BIAS + 1))}


def check(tower, n_random=400, seed=0, upto=None):
    """Every level on every input, for the presets and random N's: I_k(x, N, I1, ..) == N(x)."""
    rng = np.random.default_rng(seed)
    names = list(PRESETS) + [f"_r{n}" for n in range(n_random)]
    for n in names[len(PRESETS):]:
        PRESETS[n] = random_n(rng)
    try:
        rows = np.array([list(x) + numbers_of_n(n) for n in names for x in X4])
        want = np.array([n_forward(n, x) for n in names for x in X4])
    finally:
        for n in names[len(PRESETS) - n_random:]:
            PRESETS.pop(n, None)
    out = []
    for k, (net, _, _) in enumerate(tower[:upto]):
        if k:  # this level also takes the numbers of the one below
            prev, ranges, _ = tower[k - 1]
            rows = np.hstack([rows, np.tile(encode(prev, ranges), (len(rows), 1))])
        assert (net.forward(rows)[:, 0] == want).all(), f"I{k + 1} differs from N"
        out.append((net.max_mass(rows), len(rows)))
    return out


X4 = [(0, 0), (0, 1), (1, 0), (1, 1)]


def main():
    N, tower = build(levels=3)
    for k, (net, _, bound) in enumerate(tower):
        print(f"I{k + 1}: {net.depth} layers, {net.n_units:,} units, {net.n_weights:,} weights, "
              f"{len(net.in_names):,} inputs, activations up to {bound:,}")
    two = check(tower[:2])
    for k, (mass, n) in enumerate(two):
        assert mass < LIMIT, mass
        print(f"I{k + 1} == N on {n:,} cases (5 presets and 400 random N, every input), max |partial sum| {mass:,}")
    mass3, n3 = check(tower, n_random=20)[2]
    print(f"I3 == N on {n3} cases in numpy; max |partial sum| {mass3:,}, past 2^24, so it stays out of the CSS")

    (I1, r1, _), (I2, r2, _) = tower[0], tower[1]
    (NETS / "tower.json").write_text(json.dumps({
        "I1": I1.to_json(), "I2": I2.to_json(), "I1_numbers": encode(I1, r1),
        "presets": {k: numbers_of_n(k) for k in PRESETS},
        "truth": {k: [n_forward(k, x) for x in X4] for k in PRESETS},
        "sizes": [{"units": net.n_units, "weights": net.n_weights, "inputs": len(net.in_names), "layers": net.depth}
                  for net, _, _ in tower],
    }))
    (ROOT / "tests" / "tower.json").write_text(json.dumps({k: [n_forward(k, x) for x in X4] for k in PRESETS}))
    print("wrote tower.json")


if __name__ == "__main__":
    main()
