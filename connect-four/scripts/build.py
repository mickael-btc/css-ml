"""Compile model.json into a single self-contained index.html with zero JavaScript.

    .venv/bin/python connect-four/scripts/build.py

You are red and play first; yellow is a neural network. The game is 21 turns, and each turn is
two nested elements, so the stylesheet describes one turn and the browser applies it 21 times:

  <div class="turn">      your radio group for this turn (7 columns), and a copy of the
                          previous turn's state:     --PH_i: var(--H_i)  …
    <div class="step">    this turn's new state:     --H_i: calc(var(--PH_i) + …)  …
      <div class="turn">  the next turn, nested inside, inherits everything
        …

Custom properties inherit, so each turn reads the one before it. A property can't refer to
itself, which is why the copy lives on its own element. The game's UI sits inside the deepest
turn, where the state is final.

Per turn, computed on .step (every value an integer custom property):

  your move      --hm_c    1 for the column you picked this turn
  board          --H_i     your pieces after your move (cell i = row * 7 + col, row 0 at the bottom)
                 --A_i     the network's pieces after its move
                 --hg_c    column heights
  network        --h_k     max(0, calc((bias + sum(w * input)) / R))      integer ReLU
                 --g_c     7 logits
  tactics        --win_c   the network's piece would complete a line of four here
                 --blk_c   yours would
                 --dng_c   yours would, one row above where the network would land
  decision       --s_c     8 * logit + tie-break + 2^21 win + 2^20 block - 2^19 danger - 2^22 full
                 --w_c     product of clamp(0, s_c - s_d, 1): 1 for the highest score only
  its move       --am_c    w_c, if you played this turn and did not just win

The page lives in page.html.
"""
import json
import sys
from pathlib import Path
from string import Template

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

from game import (BLOCK, CELLS, COLS, DANGER, LINES, MASK, ROWS, THROUGH, TIE, TIE_SCALE, WIN,  # noqa: E402
                  cell)

TURNS = 21  # you play at most 21 times; the network answers every one


def var(name):
    return f"var(--{name})"


def total(terms):
    terms = list(terms)
    return "calc(" + " + ".join(terms) + ")" if terms else "0"


def eq(x, k):
    """1 if the integer property x equals k, else 0."""
    return f"clamp(0, 1 - ({var(x)} - {k}) * ({var(x)} - {k}), 1)"


def any_of(terms):
    return f"clamp(0, {' + '.join(terms)}, 1)"


def line_full(plane, l):
    return f"clamp(0, {' + '.join(var(f'{plane}{i}') for i in l)} - 3, 1)"


def completes(plane, i):
    """1 if a piece on cell i finishes a line: some line through i already has the other three."""
    return any_of(f"clamp(0, {' + '.join(var(f'{plane}{j}') for j in l if j != i)} - 2, 1)" for l in THROUGH[i])


class Stylesheet:
    """@property registrations plus rule blocks, each a selector with its declarations."""

    def __init__(self):
        self.names = set()
        self.registrations = []
        self.blocks = {}

    def register(self, name, syntax="<integer>", initial=0):
        assert name not in self.names, f"--{name} registered twice"
        self.names.add(name)
        self.registrations.append(
            f'@property --{name} {{ syntax: "{syntax}"; inherits: true; initial-value: {initial}; }}'
        )

    def set(self, selector, name, expression):
        self.blocks.setdefault(selector, []).append(f"--{name}: {expression};")

    def define(self, selector, name, expression, syntax="<integer>", initial=0):
        self.register(name, syntax, initial)
        self.set(selector, name, expression)

    def rule(self, selector, body):
        self.blocks.setdefault(selector, []).append(body)

    def render(self):
        blocks = [f"{sel} {{\n  " + "\n  ".join(decls) + "\n}" for sel, decls in self.blocks.items()]
        return "\n".join([*self.registrations, *blocks])


# ---------------------------------------------------------------- one turn

CARRIED = {  # computed on .step, copied to P<name> on the next .turn
    **{f"H{i}": 0 for i in range(CELLS)},
    **{f"A{i}": 0 for i in range(CELLS)},
    **{f"hg{c}": 0 for c in range(COLS)},
    **{f"dp{c}": 0 for c in range(COLS)},
    "lastA": 0, "lastH": 0, "why": 0, "acol": 0,
}


def add_turn(css, layers, temperature):
    T, S = ".turn", ".step"

    # the previous turn's state, under another name
    for name in CARRIED:
        css.define(T, f"P{name}", var(name))

    # your move this turn: reset, then set by the checked radio (a direct child of this .turn)
    for c in range(COLS):
        css.define(T, f"hm{c}", "0")
        css.rule(f".turn:has(> .c{c}:checked)", f"--hm{c}: 1;")

    for name in CARRIED:
        css.register(name)
    css.set(S, "pl", total(var(f"hm{c}") for c in range(COLS)))
    css.register("pl")

    # your piece lands on top of its column
    for r in range(ROWS):
        for c in range(COLS):
            i = cell(r, c)
            css.define(S, f"land{i}", eq(f"Phg{c}", r))
            css.set(S, f"H{i}", f"calc({var(f'PH{i}')} + {var(f'hm{c}')} * {var(f'land{i}')})")
    for c in range(COLS):
        css.define(S, f"hy{c}", f"calc({var(f'Phg{c}')} + {var(f'hm{c}')})")

    css.define(S, "hwin", any_of(line_full("H", l) for l in LINES))
    css.define(S, "go", f"calc({var('pl')} * (1 - {var('hwin')}))")

    # the network looks at your pieces and its own
    inputs = [f"H{i}" for i in range(CELLS)] + [f"PA{i}" for i in range(CELLS)]
    for l, layer in enumerate(layers):
        W = np.array(layer["W"])
        outputs = []
        for k, bias in enumerate(layer["b"]):
            terms = [str(bias)] + [f"{w} * {var(x)}" for w, x in zip(W[:, k], inputs) if w]
            s = " + ".join(terms).replace("+ -", "- ")
            if "R" in layer:
                name = f"h{l}_{k}"
                css.define(S, name, f"max(0, calc(({s}) / {layer['R']}))")
            else:
                name = f"g{k}"
                css.define(S, name, f"calc({s})")
            outputs.append(name)
        inputs = outputs

    # tactics: where its piece would land now, and what that square completes
    for r in range(ROWS):
        for c in range(COLS):
            i = cell(r, c)
            css.define(S, f"lnd{i}", eq(f"hy{c}", r))
            css.define(S, f"cA{i}", completes("PA", i))
            css.define(S, f"cH{i}", completes("H", i))
    for c in range(COLS):
        col = [cell(r, c) for r in range(ROWS)]
        css.define(S, f"win{c}", total(f"{var(f'lnd{i}')} * {var(f'cA{i}')}" for i in col))
        css.define(S, f"blk{c}", total(f"{var(f'lnd{i}')} * {var(f'cH{i}')}" for i in col))
        css.define(S, f"dng{c}", total(f"{var(f'lnd{i}')} * {var(f'cH{i + COLS}')}" for i in col[:-1]))
        css.define(S, f"full{c}", f"clamp(0, {var(f'hy{c}')} - {ROWS - 1}, 1)")
        css.define(S, f"s{c}", f"calc({TIE_SCALE} * {var(f'g{c}')} + {TIE[c]} + {WIN} * {var(f'win{c}')} + "
                               f"{BLOCK} * {var(f'blk{c}')} - {DANGER} * {var(f'dng{c}')} - {MASK} * {var(f'full{c}')})")

    # argmax: every score is distinct, so exactly one mask is 1
    for c in range(COLS):
        masks = [f"clamp(0, {var(f's{c}')} - {var(f's{d}')}, 1)" for d in range(COLS) if d != c]
        css.define(S, f"w{c}", f"calc({' * '.join(masks)})")
        css.define(S, f"am{c}", f"calc({var(f'w{c}')} * {var('go')})")

    for r in range(ROWS):
        for c in range(COLS):
            i = cell(r, c)
            css.set(S, f"A{i}", f"calc({var(f'PA{i}')} + {var(f'am{c}')} * {var(f'lnd{i}')})")
    for c in range(COLS):
        css.set(S, f"hg{c}", f"calc({var(f'hy{c}')} + {var(f'am{c}')})")

    # for display: the latest decision as percentages, the two newest pieces, and why it moved
    css.define(S, "smax", f"max({', '.join(var(f's{c}') for c in range(COLS))})")
    for c in range(COLS):
        css.define(S, f"e{c}", f"calc((1 - {var(f'full{c}')}) * exp(calc(({var(f's{c}')} - {var('smax')}) / {temperature})))",
                   "<number>")
    css.define(S, "esum", total(var(f"e{c}") for c in range(COLS)), "<number>", initial=1)
    for c in range(COLS):
        css.set(S, f"dp{c}", f"calc({var('go')} * {var(f'e{c}')} / {var('esum')} * 100 + (1 - {var('go')}) * {var(f'Pdp{c}')})")

    placed_a = total(f"{var(f'am{i % COLS}')} * {var(f'lnd{i}')} * {i + 1}" for i in range(CELLS))
    placed_h = total(f"{var(f'hm{i % COLS}')} * {var(f'land{i}')} * {i + 1}" for i in range(CELLS))
    css.set(S, "lastA", f"calc({placed_a} + (1 - {var('go')}) * {var('PlastA')})")
    css.set(S, "lastH", f"calc({placed_h} + (1 - {var('pl')}) * {var('PlastH')})")
    column = total(f"{var(f'am{c}')} * {c + 1}" for c in range(COLS))
    css.set(S, "acol", f"calc({column} + (1 - {var('go')}) * {var('Pacol')})")
    chose_win = total(f"{var(f'am{c}')} * {var(f'win{c}')}" for c in range(COLS))
    chose_blk = total(f"{var(f'am{c}')} * {var(f'blk{c}')}" for c in range(COLS))
    css.define(S, "cw", chose_win)
    css.define(S, "cb", chose_blk)
    # 1 the network chose, 2 a block, 3 a win
    css.set(S, "why", f"calc({var('go')} * (1 + 2 * {var('cw')} + (1 - {var('cw')}) * {var('cb')}) + "
                      f"(1 - {var('go')}) * {var('Pwhy')})")


# ---------------------------------------------------------------- the final state, for the UI

def add_ui(css):
    U = ".ui"
    css.define(U, "awin", any_of(line_full("A", l) for l in LINES))
    css.define(U, "filled", total(var(f"hg{c}") for c in range(COLS)))
    css.define(U, "full", f"clamp(0, {var('filled')} - {CELLS - 1}, 1)")
    css.define(U, "over", f"clamp(0, {var('hwin')} + {var('awin')} + {var('full')}, 1)")
    css.define(U, "draw", f"calc({var('full')} * (1 - {var('hwin')}) * (1 - {var('awin')}))")
    css.define(U, "playing", f"calc(1 - {var('over')})")
    css.define(U, "started", f"clamp(0, {var('lastH')}, 1)")
    css.define(U, "thought", f"clamp(0, {var('lastA')}, 1)")
    for w in (1, 2, 3):
        css.define(U, f"why{w}", eq("why", w))

    # "--free" is 0 on full columns and after the game, which shrinks their labels to nothing
    for c in range(COLS):
        css.rule(f"#col{c}", f"--free: calc((1 - clamp(0, {var(f'hg{c}')} - {ROWS - 1}, 1)) * {var('playing')}); "
                             f"--pct: {var(f'dp{c}')};")

    for i in range(CELLS):
        on_line = [line_full(p, l) for l in THROUGH[i] for p in ("H", "A")]
        css.rule(f"#q{i}", f"--you: {var(f'H{i}')}; --me: {var(f'A{i}')}; --win: {any_of(on_line)}; "
                           f"--new: calc({eq('lastA', i + 1)} + {eq('lastH', i + 1)});")


def turn_css():
    """Only the labels of your current turn are active: turn t once turns 0..t-1 are in."""
    rules = [".col label { display: none; }",
             ".app:not(:has([name=t0]:checked)) .col label.t0 { display: block; }"]
    for t in range(1, TURNS):
        rules.append(f".app:has([name=t{t - 1}]:checked):not(:has([name=t{t}]:checked)) .col label.t{t} "
                     f"{{ display: block; }}")
    return "\n".join(rules)


def heat_temperature(layers):
    """For the bars only: a temperature where the favourite column typically shows about 60%."""
    from train import sample_positions
    from game import int_scores
    S = []
    for b in sample_positions(400):
        s = np.array(int_scores(layers, b), dtype=float)
        if s.max() < BLOCK:  # tactics make a move certain anyway; look at the network's own calls
            S.append(np.where(s > -MASK // 2, s, -np.inf))
    lo, hi = 1.0, 1e6
    for _ in range(60):
        T = (lo * hi) ** 0.5
        top = np.median([1 / np.exp((s - s.max()) / T).sum() for s in S])
        lo, hi = (T, hi) if top > 0.6 else (lo, T)
    return round(T)


# ---------------------------------------------------------------- page

def chain_html():
    """21 nested turns, each with its radio group, around the UI."""
    open_, close = [], []
    for t in range(TURNS):
        radios = "".join(f'<input type="radio" name="t{t}" id="m{t}_{c}" class="c{c}" tabindex="-1">' for c in range(COLS))
        open_.append(f'<div class="turn">{radios}<div class="step">')
        close.append("</div></div>")
    return "\n".join(open_), "".join(close)


def board_html():
    cells = []
    for r in reversed(range(ROWS)):
        for c in range(COLS):
            cells.append(f'<i id="q{cell(r, c)}"></i>')
    columns = []
    for c in range(COLS):
        labels = "".join(f'<label for="m{t}_{c}" class="t{t}" aria-label="column {c + 1}"></label>' for t in range(TURNS))
        columns.append(f'<div class="col" id="col{c}">{labels}</div>')
    bars = "".join(f'<div class="bar" id="bar{c}"><span></span><b></b></div>' for c in range(COLS))
    return bars, "".join(cells), "".join(columns)


def main():
    model = json.loads((ROOT / "model.json").read_text())
    layers = model["layers"]
    hidden = [len(l["b"]) for l in layers[:-1]]
    n_params = sum(np.array(l["W"]).size + len(l["b"]) for l in layers)
    arch = "→".join(str(n) for n in [2 * CELLS, *hidden, COLS])

    temperature = model.get("temperature") or heat_temperature(layers)
    css = Stylesheet()
    add_turn(css, layers, temperature)
    add_ui(css)
    # the bars read --dp through #bar{c}
    for c in range(COLS):
        css.rule(f"#bar{c}", f"--pct: {var(f'dp{c}')};")

    chain_open, chain_close = chain_html()
    bars, cells, columns = board_html()
    template = Template((SCRIPTS / "page.html").read_text())
    html = template.substitute(
        network_css=css.render() + "\n" + turn_css(),
        arch=arch,
        n_params=f"{n_params:,}",
        n_hidden=hidden[0],
        chain_open=chain_open,
        chain_close=chain_close,
        bars=bars,
        cells=cells,
        columns=columns,
    )

    assert "<script" not in html.lower() and "javascript:" not in html.lower()
    (ROOT / "index.html").write_text(html)
    print(f"index.html: {len(html) / 1024:.0f} KB, {arch}, {n_params:,} params, bar temperature {temperature}")


if __name__ == "__main__":
    main()
