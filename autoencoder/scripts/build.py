"""Compile model.json into a single self-contained index.html with zero JavaScript.

    .venv/bin/python autoencoder/scripts/build.py

Every value is a registered CSS custom property (@property) on `.app`:

  painting       --pN        press+drag (or hover mode) paints a pixel; on touch, each cell is a checkbox
  normalisation  --qN        crop to the bounding box, square it, resample to 8x8 (bit-exact with norm.py)
  encoder        --eL_K      max(0, calc((bias + sum(w * input)) / R))    integer ReLU
                 --ex --ey   calc((bias + sum) / R): the drawing's 2 numbers
  map cell       --dcol --drow   which cell the drawing lands in (floor to the cell grid, clamped)
                 --col --row     the cell the decoder draws: the drawing's, or a pinned / hovered one
  decoder        --hL_K      integer ReLU from the cell centre (--zx, --zy)
                 --yN        clamp(0, calc((bias + sum) / R), LEVELS)    hard sigmoid pixels

The painting and normalisation code is copied from the digit demo (scripts/build.py at the repo root),
the map and decoder from latent-space/. The page itself lives in page.html.
"""
import json
from pathlib import Path
from string import Template

import numpy as np

from norm import GRID, normalize
from train import cell_of, int_forward

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent

N_PIXELS = GRID * GRID
FOREVER = "9999999s"  # ~115 days: long enough that a painted pixel never fades


# ---------------------------------------------------------------- css helpers

def var(name):
    return f"var(--{name})"


def total(terms):
    return "calc(" + " + ".join(terms) + ")"


class Stylesheet:
    """Collects @property registrations, standalone rules, and the values computed on .app."""

    def __init__(self):
        self.registrations = []
        self.rules = []
        self.values = []
        self.names = set()

    def register(self, name, syntax="<integer>", initial=0):
        assert name not in self.names, f"--{name} defined twice"
        self.names.add(name)
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


# ---------------------------------------------------------------- drawing (copied from the digit demo)

def add_painting(css):
    """Press and drag to paint, or tap a cell, with no JavaScript.

    Drag (--dN): while the mouse is down on the grid, the hovered cell jumps to 1 (0s transition).
    When the pointer moves on, it heads back to 0 over FOREVER, so the stroke stays.
    Clear removes `transition` altogether, which cancels every running transition: all pixels 0.
    Tap (--kN): touch screens never move :hover with the finger, so each cell is also a <label>
    for a hidden checkbox. --kN is not transitioned, so Clear (a form reset) unchecks it at once.
    Drag is limited to (hover: hover): a tap would otherwise also fire it.
    """
    for i in range(N_PIXELS):
        css.register(f"d{i}")
        css.register(f"k{i}")

    css.rule(".app", f"transition-property: {', '.join(f'--d{i}' for i in range(N_PIXELS))}; "
                     f"transition-duration: {FOREVER}; transition-timing-function: linear;")

    for i in range(N_PIXELS):
        durations = ", ".join("0s" if j == i else FOREVER for j in range(N_PIXELS))
        pressed = f".app:has(.grid:active):has(#c{i}:hover)"
        hovered = f".app:has(#pen:checked):has(#c{i}:hover)"

        css.rule(f"@media (hover: hover) {{ {pressed}, {hovered}",
                 f"--d{i}: 1; transition-duration: {durations}; }}")
        css.rule(f".app:has(#k{i}:checked)", f"--k{i}: 1;")
        css.rule(f"#c{i}", f"--v: var(--p{i});")

    css.rule(".app:has(.clear:active)", "transition: none !important;")

    pixels = [f"p{i}" for i in range(N_PIXELS)]
    for i, pixel in enumerate(pixels):
        css.define(pixel, f"max(var(--d{i}), var(--k{i}))")

    css.define("ink", total(var(p) for p in pixels))
    css.define("live", "clamp(0, var(--ink), 1)")  # 0 while the grid is empty


def add_normalization(css):
    """Bounding-box crop + square + resample, bit-exact with norm.py.

    floor(x) is written as round(x - 0.49): a registered <integer> rounds to nearest.
    """
    def pixel(axis, line, k):
        return f"p{line * GRID + k}" if axis == "r" else f"p{k * GRID + line}"

    for axis in "rc":
        for line in range(GRID):
            ink = total(var(pixel(axis, line, k)) for k in range(GRID))
            css.define(f"any{axis}{line}", f"clamp(0, {ink}, 1)")

        for line in range(GRID):
            previous = f"{var(f'lead{axis}{line - 1}')} * " if line else ""
            css.define(f"lead{axis}{line}", f"calc({previous}(1 - {var(f'any{axis}{line}')}))")

        for line in reversed(range(GRID)):
            previous = f"{var(f'trail{axis}{line + 1}')} * " if line < GRID - 1 else ""
            css.define(f"trail{axis}{line}", f"calc({previous}(1 - {var(f'any{axis}{line}')}))")

        leading = total(var(f"lead{axis}{n}") for n in range(GRID))
        trailing = total(var(f"trail{axis}{n}") for n in range(GRID))
        css.define(f"start{axis}", leading)
        css.define(f"size{axis}", f"calc({GRID} - var(--start{axis}) - {trailing})")

    css.define("side", "max(var(--sizer), var(--sizec))")

    for axis in "rc":
        css.define(f"pad{axis}", f"calc((var(--side) - var(--size{axis})) / 2 - 0.49)")
        css.define(f"org{axis}", f"calc(var(--start{axis}) - var(--pad{axis}))")

        for out in range(GRID):
            css.define(f"src{axis}{out}", f"calc(var(--org{axis}) + ({out} + 0.5) * var(--side) / {GRID} - 0.49)")

            for src in range(GRID):
                distance = f"(var(--src{axis}{out}) - {src})"
                css.define(f"eq{axis}{out}_{src}", f"clamp(0, calc(1 - {distance} * {distance}), 1)")

    for r in range(GRID):
        for c in range(GRID):
            css.define(f"rs{r}_{c}", total(f"var(--eqr{r}_{s}) * var(--p{s * GRID + c})" for s in range(GRID)))

    for r in range(GRID):
        for c in range(GRID):
            css.define(f"q{r * GRID + c}", total(f"var(--eqc{c}_{s}) * var(--rs{r}_{s})" for s in range(GRID)))


# ---------------------------------------------------------------- encoder, map, decoder

def add_encoder(css, layers):
    """Hidden layers are integer ReLUs; the last one is linear and gives the drawing's --ex, --ey."""
    inputs = [f"q{i}" for i in range(N_PIXELS)]

    for l, layer in enumerate(layers):
        W = np.array(layer["W"])
        outputs = []
        for k, bias in enumerate(layer["b"]):
            s = weighted_sum(bias, W[:, k], inputs)
            if layer.get("linear"):
                name = f"e{'xy'[k]}"
                css.define(name, f"calc(({s}) / {layer['R']})")
            else:
                name = f"e{l}_{k}"
                css.define(name, f"max(0, calc(({s}) / {layer['R']}))")
            outputs.append(name)

        inputs = outputs


def add_cell(css, model):
    """Snap the drawing's code to a map cell, then pick which cell the decoder draws.

    floor(z / W) for an odd cell width W is round((z - (W - 1) / 2) / W), never a .5 tie.
    Precedence: a hovered map cell (devices that hover) > a pinned map cell > the drawing's cell.
    One rule per column and one per row, keyed on classes the radios and labels carry.
    """
    side = model["map"]
    W, origin = model["grid"]["width"], model["grid"]["origin"]
    shift = origin + (W - 1) // 2

    css.define("dcol", f"clamp(0, calc((var(--ex) - {shift}) / {W}), {side - 1})")
    css.define("drow", f"calc({side - 1} - clamp(0, calc((var(--ey) - {shift}) / {W}), {side - 1}))")

    css.define("col", "var(--dcol)")
    css.define("row", "var(--drow)")
    css.define("pinned", "0")
    for c in range(side):
        css.rule(f".app:has(.ix{c}:checked)", f"--col: {c}; --pinned: 1;")
    for r in range(side):
        css.rule(f".app:has(.iy{r}:checked)", f"--row: {r};")

    hover = [f".app:has(.cx{c}:hover) {{ --col: {c}; --pinned: 1; }}" for c in range(side)]
    hover += [f".app:has(.cy{r}:hover) {{ --row: {r}; }}" for r in range(side)]
    css.rules.append("@media (hover: hover) {\n  " + "\n  ".join(hover) + "\n}")

    # the decoder reads the centre of the chosen cell
    css.define("zx", f"calc({shift} + var(--col) * {W})")
    css.define("zy", f"calc({shift} + ({side - 1} - var(--row)) * {W})")
    css.define("show", "max(var(--live), var(--pinned))")  # nothing to draw: empty grid and no cell picked

    # each map label knows its own column and row, to place the drawing's dot
    for c in range(side):
        css.rule(f".map .cx{c}", f"--hx: {c};")
    for r in range(side):
        css.rule(f".map .cy{r}", f"--hy: {r};")

    # coordinates as text, in latent units with two decimals
    units = model["units"]
    for axis in "xy":
        css.define(f"cent{axis}", f"calc(var(--z{axis}) * {100 / units:.6f})")
        z = var(f"cent{axis}")
        css.define(f"neg{axis}", f"clamp(0, calc(-1 * {z}), 1)")
        css.define(f"abs{axis}", f"max({z}, calc(-1 * {z}))")
        css.define(f"whole{axis}", f"calc(({var(f'abs{axis}')} - 49.5) / 100)")
        css.define(f"frac{axis}", f"calc({var(f'abs{axis}')} - 100 * {var(f'whole{axis}')})")


def add_decoder(css, layers):
    inputs = ["zx", "zy"]

    for l, layer in enumerate(layers):
        W = np.array(layer["W"])
        is_output = "top" in layer
        outputs = []
        for k, bias in enumerate(layer["b"]):
            name = f"y{k}" if is_output else f"h{l}_{k}"
            s = weighted_sum(bias, W[:, k], inputs)
            if is_output:
                css.define(name, f"clamp(0, calc(({s}) / {layer['R']}), {layer['top']})")
            else:
                css.define(name, f"max(0, calc(({s}) / {layer['R']}))")
            outputs.append(name)

        inputs = outputs


def add_visuals(css, model, drawings, cells):
    """Pixels and units light up relative to how strongly each one usually fires."""
    n_out = len(model["decoder"][-1]["b"])
    for i in range(n_out):
        css.rule(f".img i:nth-child({i + 1})", f"--v: calc(var(--y{i}) * var(--show));")
    for i in range(N_PIXELS):
        css.rule(f".seen i:nth-child({i + 1})", f"--v: var(--q{i});")

    enc = int_forward(model["encoder"], drawings)
    for l in range(len(model["encoder"]) - 1):
        typical = np.maximum(np.percentile(enc[l], 90, axis=0), 1).astype(int)
        for k, peak in enumerate(typical):
            css.rule(f".enc{l} i:nth-child({k + 1})", f"--a: min(1, var(--e{l}_{k}) / {peak});")

    dec = int_forward(model["decoder"], cells)
    for l in range(len(model["decoder"]) - 1):
        typical = np.maximum(np.percentile(dec[l], 90, axis=0), 1).astype(int)
        for k, peak in enumerate(typical):
            css.rule(f".dec{l} i:nth-child({k + 1})", f"--a: calc(min(1, var(--h{l}_{k}) / {peak}) * var(--show));")


# ---------------------------------------------------------------- page

def map_html(model):
    side = model["map"]
    peak = np.log1p(max(model["density"]))
    anchors = {cell: digit for digit, cell in model["anchors"].items()}

    cells = []
    for i in range(side * side):
        r, c = divmod(i, side)
        crowd = np.log1p(model["density"][i]) / peak
        style = f"--d: {model['classes'][i]}; --n: {crowd:.2f}"
        label = f"<b>{anchors[i]}</b>" if i in anchors else ""
        cells.append(
            f'      <input type="radio" name="z" form="pad" id="m{i}" class="cell ix{c} iy{r}" tabindex="-1">'
            f'<label for="m{i}" class="cx{c} cy{r}" style="{style}">{label}</label>'
        )

    return "\n".join(cells)


def stage(title, subtitle, viz):
    return (f'      <figure class="stage">\n'
            f'        <div class="viz">{viz}</div>\n'
            f'        <figcaption><span>{title}</span><small>{subtitle}</small></figcaption>\n'
            f'      </figure>')


def pipeline_html(model):
    enc_sizes = [len(layer["b"]) for layer in model["encoder"][:-1]]
    dec_sizes = [len(layer["b"]) for layer in model["decoder"][:-1]]
    side = int(round(len(model["decoder"][-1]["b"]) ** 0.5))
    arrow = '      <div class="link"></div>'
    code = '<div class="zin"><span class="zbar x"><i></i></span><span class="zbar y"><i></i></span></div>'

    def units(cls, n):
        return f'<div class="units {cls}" style="--cols: 8">{"<i></i>" * n}</div>'

    encoder = [
        stage("Your drawing", "cropped to 8×8", f'<div class="seen">{"<i></i>" * N_PIXELS}</div>'), arrow,
        stage("Encoder layer 1", f"{enc_sizes[0]} ReLU units", units("enc0", enc_sizes[0])), arrow,
        stage("Encoder layer 2", f"{enc_sizes[1]} ReLU units", units("enc1", enc_sizes[1])), arrow,
        stage("The code", "2 numbers → a map cell", code),
    ]
    decoder = [
        stage("Cell centre", "2 numbers", code), arrow,
        stage("Decoder layer 1", f"{dec_sizes[0]} ReLU units", units("dec0", dec_sizes[0])), arrow,
        stage("Decoder layer 2", f"{dec_sizes[1]} ReLU units", units("dec1", dec_sizes[1])), arrow,
        stage("Pixels", f"{side}×{side} hard sigmoid", f'<div class="img small">{"<i></i>" * side * side}</div>'),
    ]
    return "\n".join(encoder), "\n".join(decoder)


def fixtures(model):
    """Hand-drawn digits and map cells, with every integer the page must show for them."""
    hand = json.loads((ROOT.parent / "tests" / "fixtures.json").read_text())
    X = np.array(hand["X"], np.int64)
    picks = [0, 3, 7, 12, 21, 30, 44, 58, 71, 90]  # a spread of digits and styles
    grid, side = model["grid"], model["map"]
    xs, ys = model["xs"], model["ys"]

    def decode(col, row):
        out = int_forward(model["decoder"], np.array([[xs[col], ys[row]]]))
        return [v for layer in out for v in layer[0].tolist()]

    drawn = []
    for n in picks:
        enc = int_forward(model["encoder"], normalize(X[n:n + 1]))
        col, row = (int(v[0]) for v in cell_of(enc[-1], grid))
        drawn.append({
            "X": X[n].tolist(), "label": int(hand["label"][n]),
            "encoder": [v for layer in enc for v in layer[0].tolist()],
            "col": col, "row": row, "decoder": decode(col, row),
        })

    cells = [0, side - 1, 57, 150, 222, 333, side * side - 1]
    return {
        "drawn": drawn,
        "cells": {str(i): {"col": i % side, "row": i // side, "decoder": decode(i % side, i // side)} for i in cells},
    }


def main():
    model = json.loads((ROOT / "model.json").read_text())
    n_params = sum(np.array(layer["W"]).size + len(layer["b"]) for layer in model["encoder"] + model["decoder"])
    enc_arch = "→".join(str(n) for n in [N_PIXELS, *(len(l["b"]) for l in model["encoder"])])
    dec_arch = "→".join(str(n) for n in [2, *(len(l["b"]) for l in model["decoder"])])
    side = int(round(len(model["decoder"][-1]["b"]) ** 0.5))

    hand = np.array(json.loads((ROOT.parent / "tests" / "fixtures.json").read_text())["X"], np.int64)
    cells = np.array([(x, y) for y in model["ys"] for x in model["xs"]])

    css = Stylesheet()
    add_painting(css)
    add_normalization(css)
    add_encoder(css, model["encoder"])
    add_cell(css, model)
    add_decoder(css, model["decoder"])
    add_visuals(css, model, normalize(hand), cells)

    encoder_pipe, decoder_pipe = pipeline_html(model)
    hits, total_hand = model["on_map"]["handdrawn"]
    template = Template((SCRIPTS / "page.html").read_text())
    html = template.substitute(
        network_css=css.render(),
        enc_arch=enc_arch,
        dec_arch=dec_arch,
        n_params=f"{n_params:,}",
        map=model["map"],
        side=side,
        levels=model["levels"],
        hand_hits=hits,
        hand_total=total_hand,
        grid_cells="\n".join(
            f'      <input type="checkbox" id="k{i}" tabindex="-1"><label class="px" id="c{i}" for="k{i}"></label>'
            for i in range(N_PIXELS)
        ),
        map_cells=map_html(model),
        image_cells="<i></i>" * side * side,
        encoder_pipe=encoder_pipe,
        decoder_pipe=decoder_pipe,
    )

    assert "<script" not in html.lower() and "javascript:" not in html.lower()
    (ROOT / "index.html").write_text(html)
    (ROOT / "tests" / "fixtures.json").write_text(json.dumps(fixtures(model)))
    print(f"index.html: {len(html) / 1024:.0f} KB, encoder {enc_arch}, decoder {dec_arch}, {n_params:,} params")


if __name__ == "__main__":
    main()
