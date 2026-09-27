"""Write index.html: a classifier that trains itself in CSS, with zero JavaScript.

    .venv/bin/python playground/scripts/build.py

No weights come from Python. The page holds the training *rule*, unrolled for every epoch, as
registered custom properties (@property) on `.app`; the browser recomputes the whole run each
time a point is placed. reference.py is the same rule in numpy, for the test.

  points         --tP        +1 red, -1 blue, 0 empty: set by the checked radio of position P
  scores         --sE_P      sum_k wE_k * f_k(P)                    at epoch E
  updates        --gE_P      tP if tP * sE_P <= MARGIN else 0        clamp() as a step function
  weights        --wE_K      clamp(-WMAX, w(E-1)_K + round(sum_P g * f_K(P) / RATE), WMAX)
  errors         --errE      points with tP * sE_P <= 0
  shown epoch    --uK        the weights of the epoch picked in the chart (or hovered)
  plane          --q         each background cell's own score, from --uK and its features

The page itself (layout and styles) lives in page.html.
"""
from collections import defaultdict
from pathlib import Path
from string import Template

from reference import (EPOCHS, F_PLANE, F_POINTS, FEATURES, MARGIN, PLANE, PLANE_XY, POINT_XY, POINTS,
                       RATE, WMAX)

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
N = POINTS * POINTS
K = len(FEATURES)


# ---------------------------------------------------------------- css helpers

def var(name):
    return f"var(--{name})"


class Stylesheet:
    """Collects @property registrations, standalone rules, and the values computed on .app."""

    def __init__(self):
        self.names = set()
        self.registrations = []
        self.rules = []
        self.values = []

    def register(self, name, syntax="<integer>", initial=0, inherits=True):
        assert name not in self.names, f"--{name} defined twice"
        self.names.add(name)
        self.registrations.append(
            f'@property --{name}{{syntax:"{syntax}";inherits:{"true" if inherits else "false"};initial-value:{initial}}}'
        )

    def define(self, name, expression, syntax="<integer>", initial=0, inherits=True):
        self.register(name, syntax, initial, inherits)
        self.values.append(f"--{name}: {expression.replace(' * ', '*')};")

    def rule(self, selector, body):
        self.rules.append(f"{selector} {{ {body} }}")

    def render(self):
        app = ".app {\n  " + "\n  ".join(self.values) + "\n}"
        return "\n".join([*self.registrations, *self.rules, app])


def total(terms):
    return "calc(" + " + ".join(terms) + ")"


def linear(coefficients, names):
    """sum c * var(name), grouping names that share a coefficient: 9 * (a + b) is shorter than 9 * a + 9 * b."""
    groups = defaultdict(list)
    for c, name in zip(coefficients, names):
        if c:
            groups[int(c)].append(var(name))
    terms = []
    for c, vs in sorted(groups.items()):
        inner = vs[0] if len(vs) == 1 else "(" + " + ".join(vs) + ")"
        terms.append(inner if c == 1 else f"{c} * {inner}")
    return " + ".join(terms).replace("+ -", "- ") if terms else "0"


# ---------------------------------------------------------------- the training run

def add_points(css):
    for p in range(N):
        css.register(f"t{p}", inherits=False)
        css.rule(f".app:has(#r{p}:checked)", f"--t{p}: 1;")
        css.rule(f".app:has(#b{p}:checked)", f"--t{p}: -1;")
    css.define("n", total(f"{var(f't{p}')} * {var(f't{p}')}" for p in range(N)))


def add_training(css):
    for k in range(K):
        css.define(f"w0_{k}", "0", inherits=False)

    for e in range(EPOCHS + 1):
        weights = [f"w{e}_{k}" for k in range(K)]
        for p in range(N):
            css.define(f"s{e}_{p}", f"calc({linear(F_POINTS[p], weights)})", inherits=False)
        # a point counts as an error when t * s <= 0; the 100 - n empty positions (t = 0) always pass that test
        css.define(f"err{e}", total(
            [f"clamp(0, 1 - {var(f't{p}')} * {var(f's{e}_{p}')}, 1)" for p in range(N)] + [f"{var('n')} - {N}"]))
        if e == EPOCHS:
            break
        for p in range(N):
            t = var(f"t{p}")
            css.define(f"g{e}_{p}", f"calc({t} * clamp(0, {MARGIN + 1} - {t} * {var(f's{e}_{p}')}, 1))",
                       inherits=False)
        for k in range(K):
            step = linear(F_POINTS[:, k], [f"g{e}_{p}" for p in range(N)])
            css.define(f"w{e + 1}_{k}", f"clamp({-WMAX}, calc({var(f'w{e}_{k}')} + ({step}) / {RATE}), {WMAX})",
                       inherits=False)


def add_display(css):
    """--uK and --ue: the weights and error count of the epoch picked in the chart, or hovered."""
    for k in range(K):
        css.register(f"u{k}")
    css.register("ue")
    css.register("ep")

    def show(e):
        return " ".join(f"--u{k}: {var(f'w{e}_{k}')};" for k in range(K)) + f" --ue: {var(f'err{e}')}; --ep: {e};"

    for e in range(EPOCHS + 1):
        css.rule(f".app:has(.ep{e}:checked)", show(e))  # a class, not #id: the hover rules below must win
    hover = [f".app:has(.bar{e}:hover) {{ {show(e)} }}" for e in range(EPOCHS + 1)]
    css.rules.append("@media (hover: hover) {\n  " + "\n  ".join(hover) + "\n}")

    css.define("acc", "calc(100 * (var(--n) - var(--ue)) / max(var(--n), 1))")

    # each plane cell scores itself: its features are inline (--x --y --xx --yy --xy), the weights inherited
    css.register("q")
    f = ["45", "var(--fx)", "var(--fy)", "var(--fxx)", "var(--fyy)", "var(--fxy)"]
    css.rule(".plane i", "--q: calc(" + " + ".join(f"{a} * var(--u{k})" for k, a in enumerate(f)) + ");")


# ---------------------------------------------------------------- html

def plane_html():
    cells = []
    for (x, y), f in zip(PLANE_XY, F_PLANE):
        cells.append(f'<i style="--fx:{f[1]};--fy:{f[2]};--fxx:{f[3]};--fyy:{f[4]};--fxy:{f[5]}"></i>')
    return "".join(cells)


def points_html():
    rows = []
    for p, (x, y) in enumerate(POINT_XY):
        r, c = divmod(p, POINTS)
        rows.append(
            f'      <span class="pt" id="pt{p}" style="grid-area:{2 * r + 1}/{2 * c + 1}">'
            f'<input type="radio" name="p{p}" id="n{p}" class="n" tabindex="-1" checked>'
            f'<input type="radio" name="p{p}" id="r{p}" class="r" tabindex="-1">'
            f'<input type="radio" name="p{p}" id="b{p}" class="b" tabindex="-1">'
            f'<label for="n{p}" class="ln"></label><label for="r{p}" class="lr"></label>'
            f'<label for="b{p}" class="lb"></label></span>'
        )
    return "\n".join(rows)


def epochs_html():
    radios = "".join(f'<input type="radio" name="epoch" id="e{e}" class="ep{e}" tabindex="-1"{" checked" if e == EPOCHS else ""}>'
                     for e in range(EPOCHS + 1))
    bars = "".join(f'<label for="e{e}" class="bar bar{e}" style="--err:var(--err{e});--k:{e}"><i></i></label>'
                   for e in range(EPOCHS + 1))
    return radios, bars


def weights_html():
    names = ["1", "x", "y", "x²", "y²", "xy"]
    return "\n".join(
        f'      <div class="wrow" style="--w:var(--u{k})"><span>{n}</span><span class="wbar"><i></i></span><output></output></div>'
        for k, n in enumerate(names))


def main():
    css = Stylesheet()
    add_points(css)
    add_training(css)
    add_display(css)

    radios, bars = epochs_html()
    html = Template((SCRIPTS / "page.html").read_text()).substitute(
        network_css=css.render(),
        epochs=EPOCHS,
        bars=EPOCHS + 1,
        last=EPOCHS,
        plane=PLANE,
        margin=MARGIN,
        points=points_html(),
        plane_cells=plane_html(),
        epoch_radios=radios,
        epoch_bars=bars,
        weight_rows=weights_html(),
        n_props=f"{len(css.names):,}",
    )

    lowered = html.lower()
    assert "<script" not in lowered and "javascript:" not in lowered
    (ROOT / "index.html").write_text(html)
    print(f"index.html: {len(html) / 1024:.0f} KB, {len(css.names):,} properties, {EPOCHS} epochs")


if __name__ == "__main__":
    main()
