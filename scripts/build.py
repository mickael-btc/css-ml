"""Compile model.json into a single self-contained index.html with zero JavaScript.

    python3 scripts/build.py

Every value is a registered CSS custom property (@property) on `.app`, evaluated by
the browser's style engine each time a pixel changes:

  painting       --pN     press+drag (or hover mode) sets a pixel to 1; a ~115-day
                          transition back to 0 keeps it painted
  normalisation  --qN     crop to the bounding box, square it, resample to 8x8
  hidden layers  --aL_K   max(0, calc((bias + sum(w * input)) / R))    integer ReLU
  logits         --oC
  argmax         --wC     product of clamp(0, oC - oD, 1) masks
  softmax        --sC     exp((oC - max) / scale) / sum

The page itself (layout and styles) lives in page.html.
"""
import json
from pathlib import Path
from string import Template

import numpy as np

from norm import GRID, normalize
from train import int_forward

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent

N_PIXELS = GRID * GRID
N_CLASSES = 10
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


# ---------------------------------------------------------------- the network, stage by stage

def add_painting(css):
    """Press and drag to paint, with no JavaScript.

    While the mouse is down on the grid, the hovered cell's pixel jumps to 1 (0s transition).
    When the pointer moves on, it heads back to 0 over FOREVER, so the stroke stays.
    Clear removes `transition` altogether, which cancels every running transition: all pixels 0.
    Two separate :has() are needed for WebKit; hover mode exists because WebKit freezes
    :hover while a mouse button is held.
    """
    pixels = [f"p{i}" for i in range(N_PIXELS)]
    for pixel in pixels:
        css.register(pixel)

    css.rule(".app", f"transition-property: {', '.join('--' + p for p in pixels)}; "
                     f"transition-duration: {FOREVER}; transition-timing-function: linear;")

    for i in range(N_PIXELS):
        durations = ", ".join("0s" if j == i else FOREVER for j in range(N_PIXELS))
        pressed = f".app:has(.grid:active):has(#c{i}:hover)"
        hovered = f".app:has(#pen:checked):has(#c{i}:hover)"

        css.rule(f"{pressed}, {hovered}", f"--p{i}: 1; transition-duration: {durations};")
        css.rule(f"#c{i}", f"--v: var(--p{i});")

    css.rule(".app:has(.clear:active)", "transition: none !important;")

    css.define("ink", total(var(p) for p in pixels))
    css.define("live", "clamp(0, var(--ink), 1)")  # 0 while the grid is empty


def add_normalization(css):
    """Bounding-box crop + square + resample, bit-exact with norm.py.

    floor(x) is written as round(x - 0.49): a registered <integer> rounds to nearest.
    """
    def pixel(axis, line, k):
        return f"p{line * GRID + k}" if axis == "r" else f"p{k * GRID + line}"

    # extent of the ink along rows (r) and columns (c)
    for axis in "rc":
        for line in range(GRID):
            ink = total(var(pixel(axis, line, k)) for k in range(GRID))
            css.define(f"any{axis}{line}", f"clamp(0, {ink}, 1)")

        # lead{n} stays 1 while lines 0..n are all empty; trail{n} the same from the far end
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

    # which source line each output line samples, as one-hot selectors
    for axis in "rc":
        css.define(f"pad{axis}", f"calc((var(--side) - var(--size{axis})) / 2 - 0.49)")
        css.define(f"org{axis}", f"calc(var(--start{axis}) - var(--pad{axis}))")

        for out in range(GRID):
            css.define(f"src{axis}{out}", f"calc(var(--org{axis}) + ({out} + 0.5) * var(--side) / {GRID} - 0.49)")

            for src in range(GRID):
                distance = f"(var(--src{axis}{out}) - {src})"
                css.define(f"eq{axis}{out}_{src}", f"clamp(0, calc(1 - {distance} * {distance}), 1)")

    # resample: pick source rows, then source columns
    for r in range(GRID):
        for c in range(GRID):
            css.define(f"rs{r}_{c}", total(f"var(--eqr{r}_{s}) * var(--p{s * GRID + c})" for s in range(GRID)))

    for r in range(GRID):
        for c in range(GRID):
            css.define(f"q{r * GRID + c}", total(f"var(--eqc{c}_{s}) * var(--rs{r}_{s})" for s in range(GRID)))


def weighted_sum(bias, weights, inputs):
    terms = [str(bias)] + [f"{w} * {var(x)}" for w, x in zip(weights, inputs) if w]
    return " + ".join(terms).replace("+ -", "- ")


def add_layers(css, layers):
    """Dense layers; hidden ones are integer ReLUs, the last one gives the logits --oC."""
    inputs = [f"q{i}" for i in range(N_PIXELS)]

    for l, layer in enumerate(layers):
        W = np.array(layer["W"])
        is_output = l == len(layers) - 1
        outputs = []

        for k, bias in enumerate(layer["b"]):
            name = f"o{k}" if is_output else f"a{l}_{k}"
            s = weighted_sum(bias, W[:, k], inputs)

            if is_output:
                css.define(name, f"calc({s})")
            else:
                css.define(name, f"max(0, calc(({s}) / {layer['R']}))")
            outputs.append(name)

        inputs = outputs


def add_decision(css, scale):
    """Argmax, softmax and the winner's confidence."""
    classes = range(N_CLASSES)

    # --wC is 1 only for the winner: it beats lower digits strictly and ties-or-beats higher ones
    for c in classes:
        masks = [f"clamp(0, var(--o{c}) - var(--o{d}){' + 1' if d > c else ''}, 1)" for d in classes if d != c]
        css.define(f"w{c}", f"calc({' * '.join(masks)})")

    css.define("pred", total(f"{c} * var(--w{c})" for c in classes if c))
    css.define("omax", f"max({', '.join(var(f'o{c}') for c in classes)})")

    for c in classes:
        css.define(f"e{c}", f"exp(calc((var(--o{c}) - var(--omax)) / {scale:.4f}))", "<number>")
    css.define("esum", total(var(f"e{c}") for c in classes), "<number>", initial=1)

    for c in classes:
        css.define(f"s{c}", f"calc(var(--e{c}) / var(--esum))", "<number>")
        css.define(f"pct{c}", f"calc(var(--s{c}) * 100)")

    css.define("conf", total(f"var(--w{c}) * var(--pct{c})" for c in classes))


def add_visuals(css, layers):
    """Point each piece of UI at the property it displays."""
    for c in range(N_CLASSES):
        css.rule(f".row:nth-child({c + 1}), .score:nth-child({c + 1})",
                 f"--s: var(--s{c}); --w: calc(var(--w{c}) * var(--live)); --pct: var(--pct{c});")

    for i in range(N_PIXELS):
        css.rule(f".seen i:nth-child({i + 1})", f"--v: var(--q{i});")

    # hidden units light up relative to how strongly they usually fire
    fixtures = json.loads((ROOT / "tests" / "fixtures.json").read_text())
    X = normalize(np.array(fixtures["X"]))

    for l in range(len(layers) - 1):
        typical_peak = np.maximum(np.percentile(int_forward(layers[:l + 1], X), 90, axis=0), 1).astype(int)
        for k, peak in enumerate(typical_peak):
            css.rule(f".layer{l} i:nth-child({k + 1})",
                     f"--a: calc(min(1, var(--a{l}_{k}) / {peak}) * var(--live));")


# ---------------------------------------------------------------- page

def stage(title, subtitle, viz):
    return (f'      <figure class="stage">\n'
            f'        <div class="viz">{viz}</div>\n'
            f'        <figcaption><span>{title}</span><small>{subtitle}</small></figcaption>\n'
            f'      </figure>')


def pipeline_html(hidden_sizes):
    input_grid = f'<div class="seen">{"<i></i>" * N_PIXELS}</div>'
    hidden = [f'<div class="units layer{l}" style="--cols: 8">{"<i></i>" * n}</div>'
              for l, n in enumerate(hidden_sizes)]
    scores = "".join(f'<span class="score"><i></i><b>{c}</b></span>' for c in range(N_CLASSES))

    stages = [
        stage("Scaled to fit", "8×8 input", input_grid),
        stage("Layer 1", f"{hidden_sizes[0]} units", hidden[0]),
        stage("Layer 2", f"{hidden_sizes[1]} units", hidden[1]),
        stage("Scores", "one per digit", f'<div class="scores">{scores}</div>'),
    ]
    link = '\n      <span class="link" aria-hidden="true"></span>\n'

    return link.join(stages)


def main():
    model = json.loads((ROOT / "model.json").read_text())
    layers = model["layers"]
    hidden_sizes = [len(layer["b"]) for layer in layers[:-1]]
    n_params = sum(np.array(layer["W"]).size + len(layer["b"]) for layer in layers)
    arch = "→".join(str(n) for n in [N_PIXELS, *hidden_sizes, N_CLASSES])

    css = Stylesheet()
    add_painting(css)
    add_normalization(css)
    add_layers(css, layers)
    add_decision(css, model["scale"])
    add_visuals(css, layers)

    template = Template((SCRIPTS / "page.html").read_text())
    html = template.substitute(
        network_css=css.render(),
        arch=arch,
        n_params=f"{n_params:,}",
        grid_cells="\n".join(f'      <span class="px" id="c{i}"></span>' for i in range(N_PIXELS)),
        score_rows="\n".join(
            f'      <div class="row"><span class="lbl">{c}</span><span class="bar"></span><span class="pct"></span></div>'
            for c in range(N_CLASSES)
        ),
        pipeline=pipeline_html(hidden_sizes),
    )

    assert "<script" not in html.lower() and "javascript:" not in html.lower()
    (ROOT / "index.html").write_text(html)
    print(f"index.html: {len(html) / 1024:.0f} KB, {arch}, {n_params:,} params")


if __name__ == "__main__":
    main()
