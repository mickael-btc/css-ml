"""Compile model.json into a single self-contained index.html with zero JavaScript.

    .venv/bin/python tic-tac-toe/scripts/build.py

You are X and O is a neural network. Every value is a registered custom property on `.app`:

  your moves     --Xt_i   radio group t (your t-th move) has cell i checked
  board seen     --xbt_i  the board O looks at before its t-th move
  --obt_i
  network        --ht_k   max(0, calc((bias + sum(w * input)) / R))      integer ReLU
                 --gt_c   logits
  decision       --st_c   16 * logit + tie-break - 2^20 * occupied       every legal score distinct
                 --wt_c   product of clamp(0, s_c - s_d, 1): 1 for the highest score only
  O's move       --Ot_c   wt_c, if you played move t and did not just win
  result         --xwin, --owin, --full, --over

The network is unrolled four times, once per O move. The page lives in page.html.
"""
import json
from pathlib import Path
from string import Template

import numpy as np

from train import LINES, MASK, TIE_SCALE

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent

X_TURNS = 5   # X plays at most five times
O_TURNS = 4   # O answers the first four


def var(name):
    return f"var(--{name})"


def total(terms):
    terms = list(terms)
    return "calc(" + " + ".join(terms) + ")" if terms else "0"


class Stylesheet:
    """Collects @property registrations, standalone rules, and the values computed on .app."""

    def __init__(self):
        self.registrations = []
        self.rules = []
        self.values = []

    def register(self, name, syntax="<integer>", initial=0):
        self.registrations.append(
            f'@property --{name} {{ syntax: "{syntax}"; inherits: true; initial-value: {initial}; }}'
        )

    def define(self, name, expression, syntax="<integer>", initial=0):
        self.register(name, syntax, initial)
        self.values.append(f"--{name}: {expression};")

    def rule(self, selector, body):
        self.rules.append(f"{selector} {{ {body} }}")

    def render(self):
        app = ".app {\n  " + "\n  ".join(self.values) + "\n}"
        return "\n".join([*self.registrations, *self.rules, app])


def weighted_sum(bias, weights, inputs):
    terms = [str(bias)] + [f"{w} * {var(x)}" for w, x in zip(weights, inputs) if w]
    return " + ".join(terms).replace("+ -", "- ")


def any_line(cells):
    """1 if any of the 8 lines is full in `cells` (a list of 9 property names)."""
    lines = [f"clamp(0, {var(cells[a])} + {var(cells[b])} + {var(cells[c])} - 2, 1)" for a, b, c in LINES]
    return f"clamp(0, {' + '.join(lines)}, 1)"


# ---------------------------------------------------------------- the game, stage by stage

def add_moves(css):
    """Your moves come straight from the checked radios."""
    for t in range(X_TURNS):
        for i in range(9):
            css.register(f"X{t}_{i}")
            css.rule(f".app:has(#m{t}_{i}:checked)", f"--X{t}_{i}: 1;")
        css.define(f"played{t}", total(var(f"X{t}_{i}") for i in range(9)))


def add_network(css, layers, temperature):
    """One copy of the network per O move; each sees the board just after your move t."""
    for t in range(O_TURNS):
        for i in range(9):
            css.define(f"xb{t}_{i}", total(var(f"X{u}_{i}") for u in range(t + 1)))
            css.define(f"ob{t}_{i}", total(var(f"O{u}_{i}") for u in range(t)))

        inputs = [f"xb{t}_{i}" for i in range(9)] + [f"ob{t}_{i}" for i in range(9)]
        for l, layer in enumerate(layers):
            W = np.array(layer["W"])
            outputs = []
            for k, bias in enumerate(layer["b"]):
                s = weighted_sum(bias, W[:, k], inputs)
                if "R" in layer:
                    name = f"h{t}_{l}_{k}"
                    css.define(name, f"max(0, calc(({s}) / {layer['R']}))")
                else:
                    name = f"g{t}_{k}"
                    css.define(name, f"calc({s})")
                outputs.append(name)
            inputs = outputs

        # scores: scaled logits, a tie-break that favours earlier cells, occupied cells pushed far down
        for c in range(9):
            occupied = f"{var(f'xb{t}_{c}')} + {var(f'ob{t}_{c}')}"
            css.define(f"s{t}_{c}", f"calc({TIE_SCALE} * {var(f'g{t}_{c}')} + {8 - c} - {MASK} * ({occupied}))")

        # argmax: every legal score is distinct, so exactly one mask is 1
        for c in range(9):
            masks = [f"clamp(0, {var(f's{t}_{c}')} - {var(f's{t}_{d}')}, 1)" for d in range(9) if d != c]
            css.define(f"w{t}_{c}", f"calc({' * '.join(masks)})")

        # O only moves if you played move t and it did not win you the game
        css.define(f"xwon{t}", any_line([f"xb{t}_{i}" for i in range(9)]))
        for c in range(9):
            css.define(f"O{t}_{c}", f"calc({var(f'w{t}_{c}')} * {var(f'played{t}')} * (1 - {var(f'xwon{t}')}))")

        # the network's view as probabilities, for the heat map
        css.define(f"smax{t}", f"max({', '.join(var(f's{t}_{c}') for c in range(9))})")
        for c in range(9):
            free = f"(1 - {var(f'xb{t}_{c}')} - {var(f'ob{t}_{c}')})"
            css.define(f"e{t}_{c}", f"calc({free} * exp(calc(({var(f's{t}_{c}')} - {var(f'smax{t}')}) / {temperature})))",
                       "<number>")
        css.define(f"esum{t}", total(var(f"e{t}_{c}") for c in range(9)), "<number>", initial=1)
        for c in range(9):
            css.define(f"pct{t}_{c}", f"calc({var(f'e{t}_{c}')} / {var(f'esum{t}')} * 100)")


def heat_temperature(layers):
    """The trained network is nearly certain, so its softmax says 100% everywhere. For the heat
    map only, pick a temperature where the favourite move typically gets about 60%."""
    from train import encode, int_scores, o_positions
    boards = o_positions()
    S = int_scores(layers, [encode(b) for b in boards]).astype(float)
    legal = np.array([[0 if v else 1 for v in b] for b in boards])
    lo, hi = 1.0, 1e7
    for _ in range(60):
        T = (lo * hi) ** 0.5
        z = np.where(legal > 0, (S - S.max(axis=1, keepdims=True)) / T, -np.inf)
        top = np.median(1 / np.exp(z).sum(axis=1))
        lo, hi = (T, hi) if top > 0.6 else (lo, T)
    return round(T)


def add_result(css):
    for i in range(9):
        css.define(f"x{i}", total(var(f"X{t}_{i}") for t in range(X_TURNS)))
        css.define(f"o{i}", total(var(f"O{t}_{i}") for t in range(O_TURNS)))

    css.define("xwin", any_line([f"x{i}" for i in range(9)]))
    css.define("owin", any_line([f"o{i}" for i in range(9)]))
    css.define("filled", total(f"{var(f'x{i}')} + {var(f'o{i}')}" for i in range(9)))
    css.define("full", f"clamp(0, {var('filled')} - 8, 1)")
    css.define("over", f"clamp(0, {var('xwin')} + {var('owin')} + {var('full')}, 1)")
    css.define("draw", f"calc({var('full')} * (1 - {var('xwin')}) * (1 - {var('owin')}))")
    css.define("playing", f"calc(1 - {var('over')})")
    css.define("started", var("played0"))

    # the network copy that made the latest O move (all 0 before the first one)
    for t in range(O_TURNS):
        later = f" * (1 - {var(f'played{t + 1}')})" if t + 1 < O_TURNS else ""
        css.define(f"last{t}", f"calc({var(f'played{t}')} * (1 - {var(f'xwon{t}')}){later})")
    css.define("thought", total(var(f"last{t}") for t in range(O_TURNS)))


def add_visuals(css, layers):
    """Point each piece of UI at the property it displays."""
    css.register("cell-pct")
    css.register("heat", "<number>")
    for i in range(9):
        on_line = [f"clamp(0, {var(f'o{a}')} + {var(f'o{b}')} + {var(f'o{c}')} - 2, 1) + "
                   f"clamp(0, {var(f'x{a}')} + {var(f'x{b}')} + {var(f'x{c}')} - 2, 1)"
                   for a, b, c in LINES if i in (a, b, c)]
        pct = total(f"{var(f'last{t}')} * {var(f'pct{t}_{i}')}" for t in range(O_TURNS))
        css.rule(f"#c{i}", f"--x: var(--x{i}); --o: var(--o{i}); --win: clamp(0, {' + '.join(on_line)}, 1); "
                           f"--free: calc((1 - var(--x{i}) - var(--o{i})) * var(--playing)); "
                           f"--cell-pct: {pct}; --heat: calc({pct} / 100);")

    # inside the network: the latest copy's input, hidden units and scores
    for i in range(9):
        for side, prefix in (("x", "xb"), ("o", "ob")):
            val = total(f"{var(f'last{t}')} * {var(f'{prefix}{t}_{i}')}" for t in range(O_TURNS))
            css.rule(f".seen.{side} i:nth-child({i + 1})", f"--v: {val};")
        pct = total(f"{var(f'last{t}')} * {var(f'pct{t}_{i}')}" for t in range(O_TURNS))
        css.rule(f".seen.p i:nth-child({i + 1})", f"--v: calc({pct} / 100);")

    fixtures = json.loads((ROOT / "tests" / "fixtures.json").read_text())
    from train import encode, int_forward, play
    boards = []
    for g in fixtures:
        board = (0,) * 9
        for n, xm in enumerate(g["x"]):
            board = play(board, xm, "X")
            if n < len(g["o"]):
                boards.append(encode(board))
                board = play(board, g["o"][n], "O")
    X = np.array(boards)
    for l in range(len(layers) - 1):
        peak = np.maximum(np.percentile(int_forward(layers[:l + 1], X), 90, axis=0), 1).astype(int)
        for k, p in enumerate(peak):
            a = total(f"{var(f'last{t}')} * {var(f'h{t}_{l}_{k}')}" for t in range(O_TURNS))
            css.rule(f".layer{l} i:nth-child({k + 1})", f"--a: min(1, calc({a} / {p}));")


# ---------------------------------------------------------------- page

def cells_html():
    rows = []
    for i in range(9):
        labels = "".join(f'<label for="m{t}_{i}" class="t{t}" aria-label="cell {i + 1}"></label>' for t in range(X_TURNS))
        rows.append(f'      <div class="cell" id="c{i}"><span class="pct"></span>{labels}</div>')
    return "\n".join(rows)


def radios_html():
    return "\n".join(
        "      " + "".join(f'<input type="radio" name="t{t}" id="m{t}_{i}" tabindex="-1">' for i in range(9))
        for t in range(X_TURNS)
    )


def turn_css():
    """Only the labels of your current move are on the board: move t once moves 0..t-1 are in."""
    rules = [".cell label { display: none; }",
             ".app:not(:has([name=t0]:checked)) .cell label.t0 { display: block; }"]
    for t in range(1, X_TURNS):
        rules.append(f".app:has([name=t{t - 1}]:checked):not(:has([name=t{t}]:checked)) .cell label.t{t} "
                     f"{{ display: block; }}")
    return "\n".join(rules)


def main():
    model = json.loads((ROOT / "model.json").read_text())
    layers, scale = model["layers"], model["scale"]
    hidden = [len(l["b"]) for l in layers[:-1]]
    n_params = sum(np.array(l["W"]).size + len(l["b"]) for l in layers)
    arch = "→".join(str(n) for n in [18, *hidden, 9])

    css = Stylesheet()
    add_moves(css)
    temperature = heat_temperature(layers)
    add_network(css, layers, temperature)
    add_result(css)
    add_visuals(css, layers)

    units = "".join(
        f'<div class="units layer{l}" style="--cols: 8">{"<i></i>" * n}</div>' for l, n in enumerate(hidden)
    )
    template = Template((SCRIPTS / "page.html").read_text())
    html = template.substitute(
        network_css=css.render() + "\n" + turn_css(),
        arch=arch,
        n_params=f"{n_params:,}",
        radios=radios_html(),
        cells=cells_html(),
        hidden_units=units,
        n_hidden=hidden[0],
    )

    assert "<script" not in html.lower() and "javascript:" not in html.lower()
    (ROOT / "index.html").write_text(html)
    print(f"index.html: {len(html) / 1024:.0f} KB, {arch}, {n_params:,} params, heat temperature {temperature}")


if __name__ == "__main__":
    main()
