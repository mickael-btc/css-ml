"""Shared by the handwired page builds: the stylesheet collector, net-to-CSS, and the page shell."""
import json
import sys
from pathlib import Path
from string import Template

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
REPO = ROOT.parent
NETS = ROOT / "nets"
sys.path.insert(0, str(SCRIPTS))
sys.path.append(str(REPO / "scripts"))  # shared demo nav

from hw import Net, to_css  # noqa: E402
from nav import nav_css, nav_html  # noqa: E402

STEPS = [  # (folder, nav title, step label)
    ("tic-tac-toe", "Tic-tac-toe", "1"),
    ("connect-four", "Connect four", "1"),
    ("interpreter", "Swap the brain", "2"),
    ("tower", "Nesting dolls", "3"),
    ("unfrozen", "Rule that learns", "4"),
]


def var(name):
    return f"var(--{name})"


def load_net(name):
    return Net.from_json(json.loads((NETS / f"{name}.json").read_text()))


class Stylesheet:
    """@property registrations plus rule blocks. Values that only the stylesheet reads don't inherit,
    so the browser doesn't copy thousands of them into every element below."""

    def __init__(self):
        self.names = set()
        self.registrations = []
        self.blocks = {}

    def register(self, name, syntax="<integer>", initial=0, inherits=True):
        assert name not in self.names, f"--{name} registered twice"
        self.names.add(name)
        self.registrations.append(
            f'@property --{name}{{syntax:"{syntax}";inherits:{"true" if inherits else "false"};initial-value:{initial}}}')

    def set(self, selector, name, expression):
        self.blocks.setdefault(selector, []).append(f"--{name}: {expression};")

    def define(self, selector, name, expression, inherits=True, syntax="<integer>", initial=0):
        self.register(name, syntax, initial, inherits)
        self.set(selector, name, expression)

    def rule(self, selector, body):
        self.blocks.setdefault(selector, []).append(body)

    def net(self, selector, net, inputs, prefix, outputs=None, inherits_outputs=True):
        """Writes `net` out as CSS reading the properties `inputs`. Returns the output names."""
        rules = to_css(net, inputs, prefix, outputs)
        n_out = net.layers[-1].n_out
        for k, (name, expr) in enumerate(rules):
            self.define(selector, name, expr, inherits=inherits_outputs and k >= len(rules) - n_out)
        return [name for name, _ in rules[-n_out:]]

    def render(self):
        blocks = [f"{sel} {{\n  " + "\n  ".join(decls) + "\n}" for sel, decls in self.blocks.items()]
        return "\n".join([*self.registrations, *blocks])


def steps_html(current):
    """The handwired sub-nav. Pages live in handwired/<folder>/, the overview in handwired/."""
    items = []
    for folder, title, step in STEPS:
        label = f"<small>{step}</small>{title}"
        if folder == current:
            items.append(f'<a aria-current="page">{label}</a>')
        else:
            href = f"{folder}/" if current == "" else f"../{folder}/"
            items.append(f'<a href="{href}">{label}</a>')
    over = '<a href="./" aria-current="page">Overview</a>' if current == "" else '<a href="../">Overview</a>'
    return f'<nav class="steps" aria-label="Handwired steps">{over}{"".join(items)}</nav>'


def page(folder, title, description, css, styles, body):
    """Writes handwired/<folder>/index.html (folder "" is the overview) and returns its size in KB."""
    depth = 1 if folder == "" else 2
    html = Template((SCRIPTS / "shell.html").read_text()).substitute(
        title=title, description=description, network_css=css, styles=styles, body=body,
        steps=steps_html(folder), nav_css=nav_css(), nav=nav_html("handwired", depth))
    lowered = html.lower()
    assert "<script" not in lowered and "javascript:" not in lowered
    out = ROOT / folder / "index.html" if folder else ROOT / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html)
    return len(html) / 1024


def fmt(n):
    return f"{n:,}"


def anatomy_html(net):
    """One bar per layer: how many of its units are learned, hand-written, or just carrying a value."""
    widest = max(l.n_out for l in net.layers)
    rows = []
    for d, layer in enumerate(net.layers):
        n = {t: layer.tags.count(t) for t in ("learned", "rules", "carry", "out")}
        name = "output" if d == len(net.layers) - 1 else f"layer {d + 1}"
        bar = "".join(f'<i class="{c}" style="width:{100 * n[t] / widest:.2f}%"></i>'
                      for t, c in (("learned", "l"), ("rules", "r"), ("out", "r"), ("carry", "c")) if n[t])
        rows.append(f'<div class="row"><span>{name}</span><div class="bar">{bar}</div><span class="n">{layer.n_out:,}</span></div>')
    legend = ('<div class="legend"><span><i style="background:var(--accent)"></i>learned</span>'
              '<span><i style="background:var(--hand)"></i>hand-written</span>'
              '<span><i style="background:var(--line)"></i>carry (passes a value down a layer)</span></div>')
    return f'<div class="anatomy">{"".join(rows)}</div>{legend}'


def weight_counts(net):
    """(learned, hand-written, carry) weights, counting nonzero weights and biases."""
    t = net.count_by_tag()
    return t.get("learned", 0), t.get("rules", 0) + t.get("out", 0), t.get("carry", 0)
