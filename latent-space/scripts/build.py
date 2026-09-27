"""Compile model.json into a single self-contained index.html with zero JavaScript.

    .venv/bin/python latent-space/scripts/build.py

Every value is a registered CSS custom property (@property) on `.app`:

  position       --zx --zy   set by the checked map cell (a radio), or the hovered one
  hidden layers  --hL_K      max(0, calc((bias + sum(w * input)) / R))    integer ReLU
  pixels         --yN        clamp(0, calc((bias + sum(w * input)) / R), LEVELS)   hard sigmoid

The page itself (layout and styles) lives in page.html.
"""
import json
import sys
from pathlib import Path
from string import Template

import numpy as np

from train import int_forward

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.append(str(ROOT.parent / "scripts"))  # shared demo nav

from nav import nav_css, nav_html  # noqa: E402


# ---------------------------------------------------------------- css helpers

def var(name):
    return f"var(--{name})"


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


# ---------------------------------------------------------------- the decoder, stage by stage

def add_position(css, model):
    """The checked radio sets --zx and --zy; on devices that hover, the hovered cell wins while hovered.

    One rule per column and one per row (not per cell), keyed on classes the radios and labels carry.
    """
    css.register("zx")
    css.register("zy")

    for c, x in enumerate(model["xs"]):
        css.rule(f".app:has(.ix{c}:checked)", f"--zx: {x};")
    for r, y in enumerate(model["ys"]):
        css.rule(f".app:has(.iy{r}:checked)", f"--zy: {y};")

    hover = [f".app:has(.cx{c}:hover) {{ --zx: {x}; }}" for c, x in enumerate(model["xs"])]
    hover += [f".app:has(.cy{r}:hover) {{ --zy: {y}; }}" for r, y in enumerate(model["ys"])]
    css.rules.append("@media (hover: hover) {\n  " + "\n  ".join(hover) + "\n}")

    # coordinates as text: sign, whole part, two decimals
    for axis in "xy":
        z = var(f"z{axis}")
        css.define(f"neg{axis}", f"clamp(0, calc(-1 * {z}), 1)")
        css.define(f"abs{axis}", f"max({z}, calc(-1 * {z}))")
        css.define(f"whole{axis}", f"calc(({var(f'abs{axis}')} - 49.5) / {model['z_scale']})")
        css.define(f"frac{axis}", f"calc({var(f'abs{axis}')} - {model['z_scale']} * {var(f'whole{axis}')})")


def add_layers(css, layers):
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


def add_visuals(css, model):
    layers = model["layers"]
    n_pixels = len(layers[-1]["b"])
    for i in range(n_pixels):
        css.rule(f".img i:nth-child({i + 1})", f"--v: var(--y{i});")

    # hidden units light up relative to how strongly they usually fire across the map
    grid = np.array([(x, y) for y in model["ys"] for x in model["xs"]])
    outputs = int_forward(layers, grid)
    for l in range(len(layers) - 1):
        typical_peak = np.maximum(np.percentile(outputs[l], 90, axis=0), 1).astype(int)
        for k, peak in enumerate(typical_peak):
            css.rule(f".layer{l} i:nth-child({k + 1})", f"--a: min(1, var(--h{l}_{k}) / {peak});")


# ---------------------------------------------------------------- page

def map_html(model):
    side = model["map"]
    peak = np.log1p(max(model["density"]))
    anchors = {cell: digit for digit, cell in model["anchors"].items()}
    default = model["anchors"]["8"]

    cells = []
    for i in range(side * side):
        r, c = divmod(i, side)
        crowd = np.log1p(model["density"][i]) / peak
        style = f"--d: {model['classes'][i]}; --n: {crowd:.2f}"
        checked = " checked" if i == default else ""
        label = f"<b>{anchors[i]}</b>" if i in anchors else ""
        cells.append(
            f'      <input type="radio" name="z" id="m{i}" class="ix{c} iy{r}"{checked} aria-label="x {c + 1}, y {r + 1}">'
            f'<label for="m{i}" class="cx{c} cy{r}" style="{style}">{label}</label>'
        )

    return "\n".join(cells)


def stage(title, subtitle, viz):
    return (f'      <figure class="stage">\n'
            f'        <div class="viz">{viz}</div>\n'
            f'        <figcaption><span>{title}</span><small>{subtitle}</small></figcaption>\n'
            f'      </figure>')


def pipeline_html(hidden_sizes, side):
    position = ('<div class="zin"><span class="zbar x"><i></i></span><span class="zbar y"><i></i></span></div>')
    hidden = [f'<div class="units layer{l}" style="--cols: 8">{"<i></i>" * n}</div>'
              for l, n in enumerate(hidden_sizes)]
    image = f'<div class="img small">{"<i></i>" * side * side}</div>'
    arrow = '      <div class="link"></div>'

    return "\n".join([
        stage("Position", "2 numbers", position), arrow,
        stage("Hidden layer 1", f"{hidden_sizes[0]} ReLU units", hidden[0]), arrow,
        stage("Hidden layer 2", f"{hidden_sizes[1]} ReLU units", hidden[1]), arrow,
        stage("Pixels", f"{side}×{side} hard sigmoid", image),
    ])


def main():
    model = json.loads((ROOT / "model.json").read_text())
    layers = model["layers"]
    hidden_sizes = [len(layer["b"]) for layer in layers[:-1]]
    n_pixels = len(layers[-1]["b"])
    side = int(round(n_pixels ** 0.5))
    n_params = sum(np.array(layer["W"]).size + len(layer["b"]) for layer in layers)
    arch = "→".join(str(n) for n in [2, *hidden_sizes, n_pixels])

    css = Stylesheet()
    add_position(css, model)
    add_layers(css, layers)
    add_visuals(css, model)

    template = Template((SCRIPTS / "page.html").read_text())
    html = template.substitute(
        nav_css=nav_css(),
        nav=nav_html("latent-space"),
        network_css=css.render(),
        arch=arch,
        n_params=f"{n_params:,}",
        map=model["map"],
        side=side,
        levels=model["levels"],
        extent=f"{model['extent']:.1f}",
        map_cells=map_html(model),
        image_cells="<i></i>" * n_pixels,
        pipeline=pipeline_html(hidden_sizes, side),
    )

    assert "<script" not in html.lower() and "javascript:" not in html.lower()
    (ROOT / "index.html").write_text(html)
    print(f"index.html: {len(html) / 1024:.0f} KB, {arch}, {n_params:,} params")


if __name__ == "__main__":
    main()
