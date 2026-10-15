"""Step 3 page: N, I1 running N, and I2 running I1 running N, all in CSS on all four inputs.

    .venv/bin/python handwired/scripts/build_tower.py

N's 9 numbers come from a preset or from 9 radio groups (custom). I1's 103 numbers are constants on
the page, and they are inputs to I2 exactly like N's numbers are inputs to I1.
"""
import json

from common import NETS, Net, Stylesheet, fmt, page, var
from tower import N_BIAS, N_LO, PRESETS, X4

N_SLOTS = [("w₁₁", "W1", 0, 0), ("w₂₁", "W1", 1, 0), ("w₁₂", "W1", 0, 1), ("w₂₂", "W1", 1, 1),
           ("b₁", "b1", 0, None), ("b₂", "b1", 1, None), ("v₁", "W2", 0, None), ("v₂", "W2", 1, None), ("c", "b2", None, None)]


def build():
    t = json.loads((NETS / "tower.json").read_text())
    I1, I2 = Net.from_json(t["I1"]), Net.from_json(t["I2"])
    n_names = I1.in_names[2:]                        # N's 9 numbers, as I1 reads them
    i_names = I2.in_names[len(I1.in_names):]         # I1's 103 numbers, as I2 reads them
    assert len(n_names) == 9 and len(i_names) == len(t["I1_numbers"])
    css = Stylesheet()
    A = ".app"
    keys = [k for k in PRESETS]

    # N: a preset, or the custom radios. Values are what I1 reads: weights offset by -lo
    for n in n_names:
        css.register(n)
    for k, key in enumerate(keys):
        css.rule(f".app:has(#p{k}:checked)", " ".join(f"--{n}: {v};" for n, v in zip(n_names, t["presets"][key])))
    custom = []
    for s, n in enumerate(n_names):
        weight = n.startswith("nw")
        values = range(N_LO, N_LO + 8) if weight else range(-N_BIAS, N_BIAS + 1)
        css.register(f"c{s}")
        for v in values:
            css.rule(f".app:has(#c{s}_{v + 10}:checked)", f"--c{s}: {v - N_LO if weight else v};")
        custom.append((s, n, weight, list(values)))
    css.rule(".app:has(#pcustom:checked)", " ".join(f"--{n}: var(--c{s});" for s, n in enumerate(n_names)))
    # what N's numbers mean, for display (weights back to signed)
    for s, n in enumerate(n_names):
        css.define(A, f"shown{s}", f"calc(var(--{n}) - {-N_LO})" if n.startswith("nw") else var(n))

    # I1's numbers: constants
    for n, v in zip(i_names, t["I1_numbers"]):
        css.define(A, n, str(v))  # inherits: the strip below shows them

    for a, (x1, x2) in enumerate(X4):
        css.define(A, f"xa{a}", str(x1), inherits=False)
        css.define(A, f"xb{a}", str(x2), inherits=False)
        base = [f"xa{a}", f"xb{a}"] + n_names
        css.net(A, I1, base, f"one{a}_", [f"y1_{a}"])
        css.net(A, I2, base + i_names, f"two{a}_", [f"y2_{a}"])

    # page pieces
    presets = "".join(f'<input type="radio" name="preset" id="p{k}"{" checked" if k == 0 else ""}>' for k in range(len(keys)))
    presets += '<input type="radio" name="preset" id="pcustom">'
    preset_labels = "".join(f'<label for="p{k}" class="btn">{key}</label>' for k, key in enumerate(keys))
    preset_labels += '<label for="pcustom" class="btn">Custom</label>'
    xor = t["presets"]["XOR"]
    rows, radios = [], []
    for (s, n, weight, values), (label, *_rest) in zip(custom, N_SLOTS):
        start = xor[s] + N_LO if weight else xor[s]
        for v in values:
            radios.append(f'<input type="radio" name="c{s}" id="c{s}_{v + 10}"{" checked" if v == start else ""}>')
        chips = "".join(f'<label for="c{s}_{v + 10}">{v}</label>' for v in values)
        rows.append(f'<div class="num"><span>{label}</span><b style="--v: var(--shown{s})"></b><div class="chips c{s}">{chips}</div></div>')
    for s, n, weight, values in custom:
        for v in values:
            css.rule(f".app:has(#c{s}_{v + 10}:checked) .chips.c{s} label[for=c{s}_{v + 10}]",
                     "border-color: var(--hand); color: var(--hand);")
    for k in range(len(keys)):
        css.rule(f".app:has(#p{k}:checked) .presets label[for=p{k}]", "border-color: var(--hand); color: var(--hand);")
    css.rule(".app:has(#pcustom:checked) .presets label[for=pcustom]", "border-color: var(--hand); color: var(--hand);")
    css.rule(".app:not(:has(#pcustom:checked)) .chips", "opacity: .35; pointer-events: none;")

    table = "".join(f'<tr><td>{x1}</td><td>{x2}</td><td class="n"><b style="--v: var(--y1_{a})"></b></td>'
                    f'<td class="n"><b style="--v: var(--y2_{a})"></b></td></tr>' for a, (x1, x2) in enumerate(X4))
    i1_strip = "".join(f'<i style="--v: var(--{n})"></i>' for n in i_names)
    s1, s2, s3 = t["sizes"]
    sizes = f"""
    <tr><td>N</td><td>the program</td><td class="n">2</td><td class="n">—</td><td class="n">9</td><td></td></tr>
    <tr><td>I1 = interpret(N)</td><td>runs any N</td><td class="n">{s1['inputs']}</td><td class="n">{fmt(s1['units'])}</td><td class="n">{fmt(s1['weights'])}</td><td>in CSS, ×4</td></tr>
    <tr><td>I2 = interpret(I1)</td><td>runs I1 running N</td><td class="n">{s2['inputs']}</td><td class="n">{fmt(s2['units'])}</td><td class="n">{fmt(s2['weights'])}</td><td>in CSS, ×4</td></tr>
    <tr><td>I3 = interpret(I2)</td><td>runs I2 running I1 running N</td><td class="n">{fmt(s3['inputs'])}</td><td class="n">{fmt(s3['units'])}</td><td class="n">{fmt(s3['weights'])}</td><td>numpy only</td></tr>"""

    body = f"""
<header>
  <h1>Nesting dolls</h1>
  <p class="lead">A console that runs a console, which runs a tiny network.</p>
</header>
<div class="explain"><ul><li><b>N</b> is a tiny network: 9 numbers that turn two inputs (0 or 1) into one answer. The buttons give it different jobs: XOR, AND, OR…</li><li><b>Console 1</b> runs N, reading its 9 numbers.</li><li><b>Console 2</b> runs console 1, reading console 1's {len(i_names)} numbers. So it runs N too, one level further down.</li><li><b>Try:</b> click the buttons. Both columns of the table always give the same answers. Pick <i>Custom</i> to set N's numbers yourself.</li></ul></div>
<form class="hidden-inputs">{presets}{"".join(radios)}</form>
<div class="main">
  <div class="left">
    <h2>N, the tiny network</h2>
    <div class="presets">{preset_labels}</div>
    <div class="nums">{"".join(rows)}</div>
    <p class="hint">Its 9 numbers. Pick Custom to change them.</p>
  </div>
  <div class="right">
    <h2>The answers</h2>
    <table class="truth"><tr><th>x₁</th><th>x₂</th><th class="n">console 1</th><th class="n">console 2</th></tr>{table}</table>
    <p class="hint">Computed live by your browser, for all four inputs at once.</p>
  </div>
</div>
<details class="deep"><summary>How it works (for the curious)</summary>
<section>
  <h2>I1, as I2 sees it</h2>
  <p>The {len(i_names)} numbers of I1 (weights offset to unsigned, then biases), fed to I2 as inputs. I2 knows I1's wiring, which unit feeds which, but none of these values.</p>
  <div class="strip">{i1_strip}</div>
</section>
<section>
  <h2>The tower</h2>
  <div class="scroll"><table><tr><th>Net</th><th></th><th class="n">inputs</th><th class="n">units</th><th class="n">weights</th><th></th></tr>{sizes}</table></div>
  <p>Each level costs roughly the previous one's weights times their bit width. I3 is exact too (checked in numpy), but its activation bounds, which it needs for the products, reach 10¹⁸, far past what CSS computes exactly. So the tower stops at two in the browser.</p>
</section>
<footer>Checked in numpy for the five presets and 400 random N (every weight −4…3, every bias −4…4) on all four inputs: I1 and I2 equal N exactly, and every sum stays under 2²⁴. The browser test checks every preset and some custom N's in Chromium, Firefox and WebKit. Built by <code>handwired/scripts/tower.py</code>.</footer>
</details>
"""
    styles = """
.main { display: grid; grid-template-columns: 1fr 1fr; gap: 36px; align-items: start; }
@media (max-width: 720px) { .main { grid-template-columns: 1fr; gap: 24px; } }
.main h2 { font-size: 15px; margin: 0 0 12px; }
.presets { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 14px; }
.presets .btn { font-size: 13px; padding: 4px 10px; }
.nums { display: grid; gap: 4px; }
.num { display: grid; grid-template-columns: 34px 30px 1fr; gap: 8px; align-items: center; font-size: 13px; }
.num span { color: var(--muted); }
.num b, .truth b { font-weight: 700; }
.num b::after, .truth b::after, .strip i::after { counter-reset: v var(--v); content: counter(v); }
.chips { display: flex; gap: 2px; flex-wrap: wrap; transition: opacity .2s; }
.chips label { font-size: 11px; min-width: 24px; text-align: center; border: 1px solid var(--line); border-radius: 4px; padding: 1px 3px; cursor: pointer; }
.hint { font-size: 12px; color: var(--muted); margin: 12px 0 0; max-width: 52ch; }
.truth { max-width: 380px; }
.truth td, .truth th { font-size: 15px; }
.truth b { color: var(--hand); font-size: 18px; }
.strip { display: flex; flex-wrap: wrap; gap: 3px; font-size: 11px; color: var(--muted); }
.strip i { font-style: normal; border: 1px solid var(--line); border-radius: 3px; padding: 0 4px; min-width: 22px; text-align: center; }
"""
    kb = page("tower", "Interpreter Tower",
              "A ReLU network in pure CSS that runs a copy of the interpreter network, which runs a tiny net.",
              css.render(), styles, body)
    print(f"tower/index.html: {kb:.0f} KB, {len(css.names):,} properties")


if __name__ == "__main__":
    build()
