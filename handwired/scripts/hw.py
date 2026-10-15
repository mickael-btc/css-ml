"""A tiny handwiring compiler: write a program out of +, constant * and relu, get a layered integer net.

    p = Prog()
    a = p.input("a")
    above4 = p.ge(a, 5)                     # relu(a - 4) - relu(a - 5): 1 when a > 4
    p.compile({"result": above4 * 10 + (1 - above4) * 12})

A program is a graph of **nodes**. Every node is one ReLU unit, `relu(affine) / R` rounded like a
registered CSS <integer>. Everything between nodes is an affine form (`Val`): a sum of
constant * node plus a constant. Affine forms cost nothing, they fold into the next layer's weights.

`compile` lowers the graph to strict layers: a node sits at 1 + the depth of its deepest input, and
a value needed further down is carried through the layers in between by an identity unit (two for a
signed input: relu(x) and relu(-x)). The last layer is linear and holds the outputs. The result is an
ordinary MLP, `Net`, that runs exactly in numpy and compiles to CSS.
"""
from collections import defaultdict

import numpy as np

LIMIT = 1 << 24  # engines that do CSS math in 32-bit floats are exact below this


def css_round(x):
    """A registered <integer> custom property rounds to nearest, halves toward +infinity."""
    return np.floor(x + 0.5).astype(np.int64)


# ---------------------------------------------------------------- affine forms

class Val:
    """sum(coef * value) + const, where each value is a program input or a node."""

    __slots__ = ("terms", "const")

    def __init__(self, terms=None, const=0):
        self.terms = {k: v for k, v in (terms or {}).items() if v}
        self.const = int(const)

    @staticmethod
    def of(x):
        return x if isinstance(x, Val) else Val({}, int(x))

    def __add__(self, other):
        other = Val.of(other)
        terms = dict(self.terms)
        for k, v in other.terms.items():
            terms[k] = terms.get(k, 0) + v
        return Val(terms, self.const + other.const)

    __radd__ = __add__

    def __neg__(self):
        return Val({k: -v for k, v in self.terms.items()}, -self.const)

    def __sub__(self, other):
        return self + (-Val.of(other))

    def __rsub__(self, other):
        return Val.of(other) - self

    def __mul__(self, c):
        assert not isinstance(c, Val), "only constant * value: the product of two values is not affine"
        c = int(c)
        return Val({k: v * c for k, v in self.terms.items()}, self.const * c)

    __rmul__ = __mul__


def total(vals):
    out = Val()
    for v in vals:
        out = out + v
    return out


# ---------------------------------------------------------------- programs

class Prog:
    def __init__(self):
        self.inputs = []          # ids, in input order
        self.signed = set()       # inputs that may be negative (nodes never are)
        self.names = {}           # input id -> name
        self.expr = {}            # node id -> Val
        self.R = {}               # node id -> divisor
        self.tag = {}             # node id -> label, for counting where the weights come from
        self.depth = {}
        self._tag = "rules"

    def tagging(self, tag):
        """Nodes made inside `with p.tagging("learned"):` are counted as learned."""
        prog = self

        class Ctx:
            def __enter__(self):
                self.old, prog._tag = prog._tag, tag

            def __exit__(self, *a):
                prog._tag = self.old

        return Ctx()

    def input(self, name, signed=False):
        i = len(self.depth)
        self.inputs.append(i)
        self.names[i] = name
        self.depth[i] = 0
        if signed:
            self.signed.add(i)
        return Val({i: 1})

    def relu(self, v, R=1):
        v = Val.of(v)
        i = len(self.depth)
        self.expr[i] = v
        self.R[i] = int(R)
        self.tag[i] = self._tag
        self.depth[i] = 1 + max([self.depth[k] for k in v.terms] + [0])
        return Val({i: 1})

    # ---- integer logic. Every argument is an integer-valued Val; bits are 0 or 1.

    def ge(self, v, t):
        """[v >= t] for integer v."""
        return self.relu(v - t + 1) - self.relu(v - t)

    def eq(self, v, t):
        """[v == t] for integer v."""
        return self.relu(v - t + 1) - 2 * self.relu(v - t) + self.relu(v - t - 1)

    def AND(self, *bits):
        return self.relu(total(bits) - (len(bits) - 1))

    def OR(self, *bits):
        s = total(bits)
        return s - self.relu(s - 1)

    def gate(self, bit, v, bound, signed=False):
        """bit * v, for 0 <= v <= bound (or |v| <= bound when signed)."""
        off = bound * (1 - bit)
        if not signed:
            return self.relu(v - off)
        return self.relu(v - off) - self.relu(-v - off)

    def materialize(self, v, lo):
        """The same value, but as one unit: affine forms with a big fan-in get read many times."""
        return self.relu(v - lo) + lo

    def argmax(self, scores, lo=None):
        """One-hot of the largest score. Scores must be distinct integers (and >= lo, if given:
        each score then becomes one unit first, so every comparison reads 2 values, not the fan-in)."""
        if lo is not None:
            scores = [self.materialize(s, lo) for s in scores]
        out = []
        for c, s in enumerate(scores):
            beaten = total(self.ge(scores[d] - s, 1) for d in range(len(scores)) if d != c)
            out.append(self.relu(1 - beaten))
        return out

    def split(self, v):
        """The same signed value as two units, relu(v) - relu(-v): later reads cost 2 weights, not v's fan-in."""
        return self.relu(v) - self.relu(-v)

    def pick(self, base, tiers):
        """One-hot choice, lexicographic: the option with the most important tier bits set, then the
        largest `base`. tiers: lists of 0/1 per option, most important first. Same result as argmax of
        sum(BIG_t * tier_t) + base with every BIG far above the base range, without the big numbers:
        option d beats c when key_d > key_c, or the keys are equal and base_d > base_c (bases distinct)."""
        n, T = len(base), len(tiers)
        key = [self.relu(total((1 << (T - 1 - t)) * tiers[t][c] for t in range(T))) for c in range(n)]  # one unit each
        out = []
        for c in range(n):
            beats = []
            for d in range(n):
                if d == c:
                    continue
                gap = key[d] - key[c]
                beats.append(self.ge(gap, 1) + self.AND(self.eq(gap, 0), self.ge(base[d] - base[c], 1)))
            out.append(self.relu(1 - total(beats)))
        return out

    def dense(self, layers, inputs):
        """Paste in a trained integer MLP ({W, b, R} layers, last one linear). Returns its outputs."""
        with self.tagging("learned"):
            a = list(inputs)
            for layer in layers:
                W = np.asarray(layer["W"])
                z = [total(int(W[i, k]) * a[i] for i in range(len(a)) if W[i, k]) + int(layer["b"][k])
                     for k in range(W.shape[1])]
                a = [self.relu(v, layer["R"]) for v in z] if "R" in layer else z
            return a

    # ---- lowering

    def compile(self, outputs):
        """outputs: {name: Val} (or a list of Vals). Returns a Net."""
        if not isinstance(outputs, dict):
            outputs = {f"y{k}": v for k, v in enumerate(outputs)}
        names = list(outputs)
        outs = [Val.of(outputs[n]) for n in names]

        # only keep nodes the outputs depend on
        live, stack = set(), [k for v in outs for k in v.terms]
        while stack:
            k = stack.pop()
            if k in live:
                continue
            live.add(k)
            if k in self.expr:
                stack.extend(self.expr[k].terms)

        D = max([self.depth[k] for v in outs for k in v.terms] + [0])
        last_use = defaultdict(int)  # value -> deepest layer that reads it
        for k in live:
            if k in self.expr:
                for u in self.expr[k].terms:
                    last_use[u] = max(last_use[u], self.depth[k] - 1)
        for v in outs:
            for u in v.terms:
                last_use[u] = D

        # units of each layer: (value id, part) with part "v", or "+"/"-" for a carried signed input
        layers_units = [[("in", i) for i in range(len(self.inputs))]]
        where = [{(i, "v"): n for n, i in enumerate(self.inputs)}]
        for d in range(1, D + 1):
            units = []
            for k in sorted(live):
                if k in self.expr and self.depth[k] == d:
                    units.append((k, "v"))
                elif self.depth[k] < d <= last_use[k]:
                    units.extend([(k, "+"), (k, "-")] if k in self.signed else [(k, "v")])
            layers_units.append(units)
            where.append({u: n for n, u in enumerate(units)})

        def read(val, d):
            """Weights of `val` read from layer d: {unit: coef}."""
            w = defaultdict(int)
            for k, c in val.terms.items():
                if (k, "v") in where[d]:
                    w[where[d][(k, "v")]] += c
                else:
                    w[where[d][(k, "+")]] += c
                    w[where[d][(k, "-")]] -= c
            return w

        layers = []
        for d in range(1, D + 1):
            src, dst, ws, b, R, tags = [], [], [], [], [], []
            for n, (k, part) in enumerate(layers_units[d]):
                if part == "v" and k in self.expr and self.depth[k] == d:
                    w, bias, r, tag = read(self.expr[k], d - 1), self.expr[k].const, self.R[k], self.tag[k]
                else:  # carry
                    sign = -1 if part == "-" else 1
                    if d - 1 == 0:
                        w = {where[0][(k, "v")]: sign}
                    elif (k, "v") in where[d - 1]:
                        w = {where[d - 1][(k, "v")]: 1}
                    else:
                        w = {where[d - 1][(k, part)]: 1}
                    bias, r, tag = 0, 1, "carry"
                for s, c in w.items():
                    if c:
                        src.append(s); dst.append(n); ws.append(c)
                b.append(bias); R.append(r); tags.append(tag)
            layers.append(Layer(len(layers_units[d - 1]), src, dst, ws, b, R, tags))

        src, dst, ws, b = [], [], [], []
        for n, v in enumerate(outs):
            for s, c in read(v, D).items():
                if c:
                    src.append(s); dst.append(n); ws.append(c)
            b.append(v.const)
        layers.append(Layer(len(layers_units[D]), src, dst, ws, b, None, ["out"] * len(outs)))
        return Net(layers, [self.names[i] for i in self.inputs], names, sorted(self.signed))


# ---------------------------------------------------------------- nets

class Layer:
    """Sparse integer layer: z = b + sum over edges w * x[src] into dst. R None means linear."""

    def __init__(self, n_in, src, dst, w, b, R, tags):
        order = np.lexsort((np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)))
        self.n_in = n_in
        self.src = np.asarray(src, dtype=np.int64)[order]
        self.dst = np.asarray(dst, dtype=np.int64)[order]
        self.w = np.asarray(w, dtype=np.int64)[order]
        self.b = np.asarray(b, dtype=np.int64)
        self.R = None if R is None else np.asarray(R, dtype=np.int64)
        self.tags = tags

    @property
    def n_out(self):
        return len(self.b)

    def pre(self, x):
        """z for a batch x of shape (N, n_in), in int64."""
        N = x.shape[0]
        z = np.tile(self.b, (N, 1))
        if not len(self.w):
            return z
        starts = np.flatnonzero(np.r_[1, np.diff(self.dst)])
        chunk = max(1, int(4e7 // max(len(self.w), 1)))
        for a in range(0, N, chunk):
            prod = x[a:a + chunk][:, self.src] * self.w
            z[a:a + chunk][:, self.dst[starts]] += np.add.reduceat(prod, starts, axis=1)
        return z

    def absmass(self, x):
        """sum |w * x| per unit: an upper bound on every partial sum CSS computes."""
        N = x.shape[0]
        z = np.tile(np.abs(self.b), (N, 1))
        if not len(self.w):
            return z
        starts = np.flatnonzero(np.r_[1, np.diff(self.dst)])
        chunk = max(1, int(4e7 // max(len(self.w), 1)))
        for a in range(0, N, chunk):
            prod = np.abs(x[a:a + chunk][:, self.src] * self.w)
            z[a:a + chunk][:, self.dst[starts]] += np.add.reduceat(prod, starts, axis=1)
        return z

    def act(self, z):
        if self.R is None:
            return z
        return css_round(np.maximum(z, 0) / self.R)


class Net:
    def __init__(self, layers, in_names, out_names, signed_inputs=()):
        self.layers = layers
        self.in_names = in_names
        self.out_names = out_names
        self.signed_inputs = list(signed_inputs)

    def forward(self, X, trace=False):
        a = np.atleast_2d(np.asarray(X, dtype=np.int64))
        acts = [a]
        for layer in self.layers:
            a = layer.act(layer.pre(a))
            acts.append(a)
        return acts if trace else a

    def max_mass(self, X):
        """Largest sum |w * x| any unit sees over the batch. Must stay below 2**24 for CSS."""
        acts = self.forward(X, trace=True)
        return max(int(layer.absmass(a).max()) for layer, a in zip(self.layers, acts))

    # ---- stats

    @property
    def depth(self):
        return len(self.layers)

    @property
    def n_weights(self):
        """Nonzero weights plus biases."""
        return sum(len(l.w) + int(np.count_nonzero(l.b)) for l in self.layers)

    @property
    def n_units(self):
        return sum(l.n_out for l in self.layers[:-1])

    def count_by_tag(self):
        out = defaultdict(int)
        for l in self.layers:
            tags = np.asarray(l.tags)
            out_w = np.bincount(l.dst, minlength=l.n_out) + (l.b != 0)
            for t, n in zip(tags, out_w):
                out[t] += int(n)
        return dict(out)

    def widths(self):
        return [len(self.in_names)] + [l.n_out for l in self.layers]

    # ---- serialisation: the {W, b, R} layout the other demos use, sparse

    def to_json(self):
        return {
            "inputs": self.in_names, "outputs": self.out_names, "signed_inputs": self.signed_inputs,
            "layers": [{"n_in": l.n_in, "src": l.src.tolist(), "dst": l.dst.tolist(), "w": l.w.tolist(),
                        "b": l.b.tolist(), "R": None if l.R is None else l.R.tolist(), "tags": l.tags}
                       for l in self.layers],
        }

    @staticmethod
    def from_json(d):
        layers = [Layer(l["n_in"], l["src"], l["dst"], l["w"], l["b"], l["R"], l["tags"]) for l in d["layers"]]
        return Net(layers, d["inputs"], d["outputs"], d.get("signed_inputs", []))

    @staticmethod
    def from_dense(layers, in_names=None, out_names=None, tag="learned"):
        """A trained {W, b, R} MLP (last layer linear) as a Net."""
        out = []
        for l in layers:
            W = np.asarray(l["W"])
            src, dst = np.nonzero(W)
            R = [l["R"]] * W.shape[1] if "R" in l else None
            out.append(Layer(W.shape[0], src, dst, W[src, dst], l["b"], R, [tag] * W.shape[1]))
        n_in = out[0].n_in
        return Net(out, in_names or [f"x{i}" for i in range(n_in)],
                   out_names or [f"y{k}" for k in range(out[-1].n_out)])


# ---------------------------------------------------------------- CSS

def css_affine(bias, terms):
    """bias + sum c * var(name), grouping names that share a coefficient."""
    groups = defaultdict(list)
    for c, name in terms:
        groups[int(c)].append(f"var(--{name})")
    parts = [str(int(bias))] if bias or not groups else []
    for c, vs in sorted(groups.items(), key=lambda g: -abs(g[0])):
        inner = vs[0] if len(vs) == 1 else "(" + " + ".join(vs) + ")"
        parts.append(inner if c == 1 else f"-{inner}" if c == -1 else f"{c}*{inner}")
    s = " + ".join(parts).replace("+ -", "- ")
    # calc() has no unary minus in front of var() or a parenthesis: -(a + b) must be 0 - (a + b)
    return "0 - " + s[1:] if s.startswith("-") else s


def to_css(net, input_names, prefix, output_names=None):
    """[(property, expression)] computing the net in CSS. Inputs are existing property names.

    Hidden units are `max(0, calc(...))` (with `/ R` when R > 1), outputs `calc(...)`; nothing else.
    """
    names = list(input_names)
    rules = []
    for d, layer in enumerate(net.layers):
        last = d == len(net.layers) - 1
        if last:
            out = output_names or [f"{prefix}o{k}" for k in range(layer.n_out)]
        else:
            out = [f"{prefix}{d}_{k}" for k in range(layer.n_out)]
        by_dst = defaultdict(list)
        for s, t, w in zip(layer.src, layer.dst, layer.w):
            by_dst[int(t)].append((int(w), names[int(s)]))
        for k in range(layer.n_out):
            body = css_affine(layer.b[k], by_dst[k])
            if last:
                expr = f"calc({body})"
            elif layer.R[k] == 1:
                expr = f"max(0, calc({body}))"
            else:
                expr = f"max(0, calc(({body}) / {layer.R[k]}))"
            rules.append((out[k], expr))
        names = out
    return rules
