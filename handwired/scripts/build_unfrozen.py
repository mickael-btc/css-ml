"""Step 4 page: a hand-written rule, unfrozen and trained live in CSS by backpropagation.

    .venv/bin/python handwired/scripts/build_unfrozen.py

Every epoch is written out as registered integer properties on .app (see unfrozen.py for the rule):

  weights   --aE_k --bE_k --cE_k --vE_k --dE     epoch E; epoch 0 is the rule (or random)
  forward   --hE_p_k   max(0, calc(x_p * a + y_p * b + c))      the hidden ReLUs
            --sE_p     calc(sum v * h + d)                       the score
  backward  --gE_p     t if t * s <= MARGIN else 0
            --fE_p_k   clamp(0, h, 1): the ReLU's derivative, [z > 0]
  update    a(E+1)_k = clamp(-WMAX, a + v * sum_p x_p g f / RAB, WMAX)  and so on
"""
from common import Stylesheet, page, var
from unfrozen import (CS, DMAX, DS, EPOCHS, H, MARGIN, PLANE, PLANE_XY, POINT_XY, POINTS, RA, RAB, RD, RULE, RV,
                      STARTS, WMAX)

N = POINTS * POINTS
NAMES = {"a": "x", "b": "y", "c": "1"}


def lin(coefs, names):
    """sum c * var(name) for constant c, skipping zeros."""
    terms = [f"{c} * {var(n)}" if c not in (1, -1) else ("" if c == 1 else "-1 * ") + var(n)
             for c, n in zip(coefs, names) if c]
    return " + ".join(terms).replace("+ -", "- ") or "0"


def build():
    css = Stylesheet()
    A = ".app"

    # labels from the point radios
    for p in range(N):
        css.register(f"t{p}", inherits=False)
        css.rule(f".app:has(#r{p}:checked)", f"--t{p}: 1;")
        css.rule(f".app:has(#b{p}:checked)", f"--t{p}: -1;")
    css.define(A, "n", "calc(" + " + ".join(f"var(--t{p}) * var(--t{p})" for p in range(N)) + ")")

    # epoch 0: the rule, or random
    w0 = [f"a0_{k}" for k in range(H)] + [f"b0_{k}" for k in range(H)] + [f"c0_{k}" for k in range(H)] + \
         [f"v0_{k}" for k in range(H)] + ["d0"]
    for n in w0:
        css.register(n, inherits=False)
    for key, start in STARTS.items():
        vals = start["a"] + start["b"] + start["c"] + start["v"] + [start["d"]]
        css.rule(f".app:has(#start-{key}:checked)", " ".join(f"--{n}: {v};" for n, v in zip(w0, vals)))

    xs = [x for x, _ in POINT_XY]
    ys = [y for _, y in POINT_XY]
    for e in range(EPOCHS + 1):
        a, b, c, v = ([f"{w}{e}_{k}" for k in range(H)] for w in "abcv")
        d = f"d{e}"
        for p in range(N):
            for k in range(H):
                css.define(A, f"h{e}_{p}_{k}", f"max(0, calc({lin([xs[p], ys[p], 1], [a[k], b[k], c[k]])}))", inherits=False)
            css.define(A, f"s{e}_{p}", "calc(" + " + ".join(f"var(--{v[k]}) * var(--h{e}_{p}_{k})" for k in range(H))
                       + f" + var(--{d}))", inherits=False)
        css.define(A, f"err{e}", "calc(" + " + ".join(f"clamp(0, 1 - var(--t{p}) * var(--s{e}_{p}), 1)" for p in range(N))
                   + f" + var(--n) - {N})")
        if e == EPOCHS:
            break
        for p in range(N):
            t = f"var(--t{p})"
            css.define(A, f"g{e}_{p}", f"calc({t} * clamp(0, {MARGIN + 1} - {t} * var(--s{e}_{p}), 1))", inherits=False)
            for k in range(H):
                css.define(A, f"f{e}_{p}_{k}", f"clamp(0, var(--h{e}_{p}_{k}), 1)", inherits=False)
        nxt = e + 1
        for k in range(H):
            gh = " + ".join(f"var(--g{e}_{p}) * var(--h{e}_{p}_{k})" for p in range(N))
            css.define(A, f"v{nxt}_{k}", f"clamp({-WMAX}, calc(var(--{v[k]}) + ({gh}) / {RV}), {WMAX})", inherits=False)
            for w, coef, R in (("a", xs, RAB), ("b", ys, RAB), ("c", [CS] * N, RA)):
                s = " + ".join(f"{cf} * var(--g{e}_{p}) * var(--f{e}_{p}_{k})" for p, cf in enumerate(coef) if cf)
                cur = f"{w}{e}_{k}"
                css.define(A, f"{w}{nxt}_{k}", f"clamp({-WMAX}, calc(var(--{cur}) + var(--{v[k]}) * ({s}) / {R}), {WMAX})",
                           inherits=False)
        gsum = " + ".join(f"var(--g{e}_{p})" for p in range(N))
        css.define(A, f"d{nxt}", f"clamp({-DMAX}, calc(var(--{d}) + {DS} * ({gsum}) / {RD}), {DMAX})", inherits=False)

    # the shown epoch: picked in the chart, or hovered
    shown = [f"u{w}{k}" for w in "abcv" for k in range(H)] + ["ud"]
    for n in shown:
        css.register(n)
    css.register("ue")
    css.register("ep")

    def show(e):
        decl = " ".join(f"--u{w}{k}: var(--{w}{e}_{k});" for w in "abcv" for k in range(H))
        return decl + f" --ud: var(--d{e}); --ue: var(--err{e}); --ep: {e};"

    for e in range(EPOCHS + 1):
        css.rule(f".app:has(.ep{e}:checked)", show(e))
    hover = [f".app:has(.bar{e}:hover) {{ {show(e)} }}" for e in range(EPOCHS + 1)]
    css.rules_extra = "@media (hover: hover) {\n  " + "\n  ".join(hover) + "\n}"
    css.define(A, "acc", "calc(100 * (var(--n) - var(--ue)) / max(var(--n), 1))")

    # each plane cell scores itself from the shown weights
    for k in range(H):
        css.register(f"q{k}", inherits=False)
    css.register("q", inherits=False)
    hk = " ".join(f"--q{k}: max(0, calc(var(--fx) * var(--ua{k}) + var(--fy) * var(--ub{k}) + var(--uc{k})));" for k in range(H))
    css.rule(".plane i", hk + " --q: calc(" + " + ".join(f"var(--uv{k}) * var(--q{k})" for k in range(H)) + " + var(--ud));")

    # what changed: each number against the rule
    for w in "abcv":
        for k in range(H):
            r = RULE[w][k]
            css.rule(f"#n{w}{k}", f"--w: var(--u{w}{k}); --moved: min(1, max(calc(var(--u{w}{k}) - {r}), calc({r} - var(--u{w}{k}))));")
    css.rule("#nd", f"--w: var(--ud); --moved: min(1, max(calc(var(--ud) - {RULE['d']}), calc({RULE['d']} - var(--ud))));")

    # html
    plane = "".join(f'<i style="--fx:{x};--fy:{y}"></i>' for x, y in PLANE_XY)
    pts = []
    for p in range(N):
        r, c = divmod(p, POINTS)
        pts.append(f'<span class="pt" id="pt{p}" style="grid-area:{2 * r + 1}/{2 * c + 1}">'
                   f'<input type="radio" name="p{p}" id="n{p}" class="n" tabindex="-1" checked>'
                   f'<input type="radio" name="p{p}" id="r{p}" class="r" tabindex="-1">'
                   f'<input type="radio" name="p{p}" id="b{p}" class="b" tabindex="-1">'
                   f'<label for="n{p}" class="ln"></label><label for="r{p}" class="lr"></label>'
                   f'<label for="b{p}" class="lb"></label></span>')
    radios = "".join(f'<input type="radio" name="epoch" id="e{e}" class="ep{e}" tabindex="-1"{" checked" if e == EPOCHS else ""}>'
                     for e in range(EPOCHS + 1))
    bars = "".join(f'<label for="e{e}" class="bar bar{e}" style="--err:var(--err{e});--k:{e}"><i></i></label>' for e in range(EPOCHS + 1))
    rows = []
    for k in range(H):
        cells = "".join(f'<td class="n w" id="n{w}{k}"><output></output><small>{RULE[w][k]}</small></td>' for w in "abcv")
        rows.append(f"<tr><td>h{k + 1}</td>{cells}</tr>")
    rows.append(f'<tr><td>out</td><td></td><td></td><td></td><td class="n w" id="nd"><output></output><small>{RULE["d"]} (bias)</small></td></tr>')

    body = f"""
<header>
  <h1>Unfrozen: a rule that keeps learning</h1>
  <p>Start from code: <code>red if |x| ≤ 3 and |y| ≤ 3</code>. Compiled by hand into a 2 → 4 → 1 ReLU net, each hidden unit measures how far past one side of the box a point is. Then all 17 of its numbers are <span class="learned">unfrozen</span>: the browser's style engine trains them by backpropagation, {EPOCHS} epochs, every time you place a point. <b>0 bytes of JavaScript.</b></p>
</header>
<form id="pts" class="main">
  <div class="pad">
    <div class="tools">
      <span class="seg"><label for="br-r" class="br">Red</label><label for="br-b" class="bb">Blue</label></span>
      <span class="seg"><label for="start-rule" class="sr">From the rule</label><label for="start-random" class="sx">From random</label></span>
      <input type="radio" name="brush" id="br-r" tabindex="-1" checked><input type="radio" name="brush" id="br-b" tabindex="-1">
      <input type="radio" name="start" id="start-rule" tabindex="-1" checked><input type="radio" name="start" id="start-random" tabindex="-1">
    </div>
    <div class="board"><div class="plane">{plane}</div><div class="pts">{"".join(pts)}</div></div>
    <div class="tools"><button type="reset">Clear</button><small>Click to place a point, click again to remove it.</small></div>
    <p class="hint">Try: fill the box with red and the outside with blue, and the rule already gets them all right at epoch 0. Then put red points just right of the box, and watch the rule's numbers move to take them in. Switch to “From random” to see what the rule was worth.</p>
  </div>
  <div class="result">
    <div class="big"></div>
    <div class="sub"></div>
    <div class="chart-wrap"><h3><span>errors per epoch</span><span>click or hover an epoch</span></h3>
      {radios}<div class="chart">{bars}</div><div class="axis"><span>0 (the rule)</span><span>{EPOCHS}</span></div></div>
    <div class="scroll"><table class="nums"><tr><th></th><th class="n">x</th><th class="n">y</th><th class="n">1</th><th class="n">out</th></tr>{"".join(rows)}</table></div>
    <p class="hint">The 17 numbers at the shown epoch. Small: the hand-written value. <span class="hand">Blue</span>: moved by training.</p>
  </div>
</form>
<section>
  <h2>How it trains</h2>
  <div class="scroll"><table>
    <tr><th>Step</th><th>CSS, per point p and hidden unit k</th></tr>
    <tr><td>forward</td><td><code>h = max(0, calc(x·a + y·b + c))</code>, <code>s = calc(Σ v·h + d)</code></td></tr>
    <tr><td>loss</td><td><code>g = t · clamp(0, {MARGIN + 1} − t·s, 1)</code>: the point's colour if it's wrong or within the margin</td></tr>
    <tr><td>through the ReLU</td><td><code>f = clamp(0, h, 1)</code>, the derivative [z &gt; 0]</td></tr>
    <tr><td>update</td><td><code>v += Σ g·h / {RV}</code>, <code>a += v · Σ g·f·x / {RAB}</code>, <code>b += v · Σ g·f·y / {RAB}</code>, <code>c += v · Σ {CS}·g·f / {RA}</code>, <code>d += {DS}·Σ g</code>, each clamped</td></tr>
  </table></div>
  <p>Integers throughout. Every divisor is odd, so CSS's rounding matches Python's exactly, and the clamps keep every sum under 2²⁴. The first layer's step is five times smaller than the rest: with a big step a box edge overshoots, flips, and its ReLU dies.</p>
</section>
<footer>The browser test places the demo sets from <code>handwired/scripts/unfrozen.py</code> from both starts and checks all 17 numbers at every epoch, and every error count, against the numpy reference in Chromium, Firefox and WebKit.</footer>
"""
    styles = f"""
.main {{ display: grid; grid-template-columns: auto 1fr; gap: 36px; align-items: start; }}
@media (max-width: 760px) {{ .main {{ grid-template-columns: 1fr; gap: 24px; }} }}
.pad {{ display: grid; gap: 12px; width: min(400px, calc(100vw - 32px)); }}
.app {{ --red: #d9481c; --blue: #2c6bd1; --ring: var(--bg); --live: clamp(0, var(--n), 1); }}
.tools {{ display: flex; flex-wrap: wrap; gap: 8px; justify-content: space-between; align-items: center; }}
.tools small {{ font-size: 11px; color: var(--muted); }}
.seg {{ display: inline-flex; border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }}
.seg label {{ padding: 5px 10px; font-size: 13px; cursor: pointer; color: var(--muted); }}
.seg label + label {{ border-left: 1px solid var(--line); }}
.br::before, .bb::before {{ content: ""; display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; background: var(--c); }}
.br {{ --c: var(--red); }} .bb {{ --c: var(--blue); }}
.app:has(#br-r:checked) .br, .app:has(#br-b:checked) .bb, .app:has(#start-rule:checked) .sr, .app:has(#start-random:checked) .sx {{ color: var(--fg); background: var(--cell); }}
.board {{ position: relative; padding: 12px; background: var(--line); border-radius: 8px; user-select: none; -webkit-user-select: none; touch-action: manipulation; -webkit-tap-highlight-color: transparent; }}
.plane {{ display: grid; grid-template-columns: repeat({PLANE}, 1fr); aspect-ratio: 1; border-radius: 3px; overflow: hidden; }}
.plane i {{ --pos: clamp(0, var(--q) / {MARGIN}, 1); --neg: clamp(0, var(--q) / -{MARGIN}, 1);
  background: color-mix(in oklab, color-mix(in oklab, var(--cell), var(--red) calc(var(--pos) * 55%)), var(--blue) calc(var(--neg) * 55%)); }}
.pts {{ position: absolute; inset: 12px; display: grid; grid-template: repeat({PLANE}, 1fr) / repeat({PLANE}, 1fr); }}
.pt {{ position: relative; }}
.pt::after {{ content: ""; position: absolute; inset: 38%; border-radius: 50%; background: var(--muted); opacity: .35; pointer-events: none; }}
.pt:has(.r:checked)::after, .pt:has(.b:checked)::after {{ inset: 4%; opacity: 1; box-shadow: 0 0 0 2px var(--ring); }}
.pt:has(.r:checked)::after {{ background: var(--red); }}
.pt:has(.b:checked)::after {{ background: var(--blue); }}
.pt label {{ display: none; position: absolute; inset: -50%; border-radius: 50%; cursor: pointer; z-index: 1; }}
.app:has(#br-r:checked) .pt:not(:has(.r:checked)) .lr, .app:has(#br-r:checked) .pt:has(.r:checked) .ln,
.app:has(#br-b:checked) .pt:not(:has(.b:checked)) .lb, .app:has(#br-b:checked) .pt:has(.b:checked) .ln {{ display: block; }}
@media (hover: hover) {{ .pt label:hover {{ box-shadow: inset 0 0 0 2px var(--fg); }} }}
.hint {{ font-size: 12px; color: var(--muted); margin: 0; }}
.result {{ display: grid; gap: 18px; min-width: 0; align-content: start; }}
.big {{ font-size: 56px; line-height: .9; font-weight: 700; display: grid; }}
.big::before, .big::after {{ grid-area: 1 / 1; }}
.big::after {{ counter-reset: a var(--acc); content: counter(a) "%"; color: var(--accent); opacity: var(--live); }}
.big::before {{ content: "–"; color: var(--line); opacity: calc(1 - var(--live)); }}
.sub {{ font-size: 13px; color: var(--muted); }}
.sub::before {{ counter-reset: r calc(var(--n) - var(--ue)) n var(--n) e var(--ep); content: counter(r) " of " counter(n) " points right after epoch " counter(e); }}
.chart-wrap {{ display: grid; gap: 6px; }}
.chart-wrap h3 {{ font-size: 12px; font-weight: 400; color: var(--muted); margin: 0; display: flex; justify-content: space-between; }}
.chart {{ display: grid; grid-template-columns: repeat({EPOCHS + 1}, 1fr); gap: 2px; height: 70px; border-bottom: 1px solid var(--line); }}
.bar {{ display: grid; align-items: end; cursor: pointer; height: 100%; position: relative; }}
.bar i {{ height: calc(var(--err) / max(var(--n), 1) * 100%); min-height: 2px; border-radius: 1px 1px 0 0;
  background: color-mix(in oklab, var(--muted), var(--accent) calc(clamp(0, 1 - (var(--ep) - var(--k)) * (var(--ep) - var(--k)), 1) * 100%)); }}
@media (hover: hover) {{ .bar:hover {{ background: var(--cell); }} }}
.axis {{ display: flex; justify-content: space-between; font-size: 11px; color: var(--muted); }}
.nums td.w {{ color: color-mix(in oklab, var(--fg), var(--hand) calc(var(--moved) * 100%)); }}
.nums output::after {{ counter-reset: v var(--w); content: counter(v); }}
.nums small {{ display: block; font-size: 10px; color: var(--muted); }}
"""
    out = css.render() + "\n" + css.rules_extra
    kb = page("unfrozen", "Unfrozen Rule",
              "A hand-written rule compiled into a ReLU net, then trained live by backpropagation in pure CSS.",
              out, styles, body)
    print(f"unfrozen/index.html: {kb:.0f} KB, {len(css.names):,} properties")


if __name__ == "__main__":
    build()
