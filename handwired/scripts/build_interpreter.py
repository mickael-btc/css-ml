"""Step 2 page: one fixed ReLU network in CSS that runs whichever tic-tac-toe model you load into it.

    .venv/bin/python handwired/scripts/build_interpreter.py

The model is data: picking one sets its 1,801 numbers as custom properties on .app,
`.app:has(#model-hand:checked) { --v1_0_0: 128; … }`. The interpreter's stylesheet never changes and
contains none of them. Five nested turns, as in connect four: .turn copies the board it inherits and
holds your radios, .step runs the interpreter on it.
"""
import json

from common import NETS, Stylesheet, anatomy_html, fmt, load_net, page, var

TURNS = 5
MODELS = [("learned", "Trained", "the tic-tac-toe demo's network, trained on every position"),
          ("hand", "Hand-written", "48 hand-set hidden units: win if you can, else block, else centre, corner, edge"),
          ("random", "Random", "untrained: random int8 weights")]


def build():
    net, judge = load_net("interp_player"), load_net("ttt_judge")
    models = json.loads((NETS / "models.json").read_text())
    css = Stylesheet()
    T, S, U = ".turn", ".step", ".ui"

    # the model: 1,801 inputs, set by the picked radio, inherited all the way down
    weight_names = [n for n in net.in_names if n[:2] in ("v1", "b1", "v2", "b2")]
    for n in weight_names:
        css.register(n)
    for key, _, _ in MODELS:
        flat = models[key]["flat"]
        assert len(flat) == len(weight_names)
        css.rule(f".app:has(#model-{key}:checked)", " ".join(f"--{n}:{v};" for n, v in zip(weight_names, flat)))

    for i in range(9):
        css.define(T, f"PX{i}", var(f"X{i}"))
        css.define(T, f"PO{i}", var(f"O{i}"))
        css.define(T, f"m{i}", "0")
        css.rule(f".turn:has(> .c{i}:checked)", f"--m{i}: 1;")
    for c in range(9):
        css.define(T, f"PG{c}", var(f"G{c}"))
    css.net(S, net, net.in_names, "i", net.out_names)
    css.net(U, judge, [f"X{i}" for i in range(9)] + [f"O{i}" for i in range(9)], "j", judge.out_names)

    css.rule(".app:has(.turn input:checked) .ui", "--started: 1;")
    css.register("started")
    for i in range(9):
        css.rule(f"#c{i}", f"--x: var(--X{i}); --o: var(--O{i}); --win: var(--win{i}); --free: var(--free{i}); "
                           f"--g: var(--G{i});")
    turn_rules = [".cell label { display: none; }",
                  ".app:not(:has([name=t0]:checked)) .cell label.t0 { display: block; }"]
    for t in range(1, TURNS):
        turn_rules.append(f".app:has([name=t{t - 1}]:checked):not(:has([name=t{t}]:checked)) .cell label.t{t} "
                          "{ display: block; }")
    for key, _, _ in MODELS:
        turn_rules.append(f".app:has(#model-{key}:checked) .pick label[for=model-{key}] "
                          "{ border-color: var(--hand); color: var(--hand); }")

    # the loaded weights, drawn: each square reads one inherited weight
    w1 = "".join(f'<i style="--v:var(--v1_{i}_{k})"></i>' for k in range(64) for i in range(18))
    w2 = "".join(f'<i style="--v:var(--v2_{k}_{c})"></i>' for c in range(9) for k in range(64))

    opening = "\n".join('<div class="turn">' + "".join(
        f'<input type="radio" name="t{t}" id="m{t}_{i}" class="c{i}" tabindex="-1">' for i in range(9)) +
        '<div class="step">' for t in range(TURNS))
    closing = "</div></div>" * TURNS
    cells = "\n".join(f'<div class="cell" id="c{i}"><span class="g"></span>' + "".join(
        f'<label for="m{t}_{i}" class="t{t}" aria-label="cell {i + 1}"></label>' for t in range(TURNS)) + "</div>"
        for i in range(9))
    model_radios = "".join(f'<input type="radio" name="model" id="model-{k}"{" checked" if n == 0 else ""}>'
                           for n, (k, _, _) in enumerate(MODELS))
    model_labels = "".join(f'<label for="model-{k}" class="btn" title="{d}">{t}</label>' for k, t, d in MODELS)
    model_rows = "".join(f"<tr><td>{t}</td><td>{d}</td></tr>" for _, t, d in MODELS)

    n_model = len(weight_names)
    body = f"""
<header>
  <h1>A network that runs a network</h1>
  <p>O is played by one fixed ReLU network, written into the stylesheet once. Its inputs are the board <em>and the {fmt(n_model)} weights of a tic-tac-toe model</em>. Its own {fmt(net.n_weights)} weights know the game's rules (free squares, lines, when to move) but not any model: the model's numbers come in as data, and the interpreter multiplies, adds and compares whatever arrives. <b>0 bytes of JavaScript.</b></p>
  <p>Load a different model and the same interpreter plays like it.</p>
</header>
<form id="models" class="hidden-inputs">{model_radios}</form>
<div class="pick">{model_labels}</div>
<form id="game" class="game">
{opening}
<div class="ui">
  <div class="main">
    <div class="pad">
      <div class="board">
{cells}
      </div>
      <div class="tools"><button type="reset" form="game">New game</button><small>The numbers: the model's logits at its last move.</small></div>
    </div>
    <div class="result">
      <div class="status" aria-live="polite"><span class="start">Your move</span><span class="turn-s">Your turn</span><span class="owin">The model wins</span><span class="xwin">You win</span><span class="draw">Draw</span></div>
      <figure class="weights"><div class="w1">{w1}</div><figcaption>layer 1 of the loaded model: 18 × 64 weights, as inputs</figcaption></figure>
      <figure class="weights"><div class="w2">{w2}</div><figcaption>layer 2: 64 × 9 <span class="learned">positive</span> <span class="hand">negative</span></figcaption></figure>
    </div>
  </div>
</div>
{closing}
</form>
<section>
  <h2>The models</h2>
  <div class="scroll"><table><tr><th>Model</th><th></th></tr>{model_rows}</table></div>
  <p>Each one is an 18 → 64 → 9 network with int8 weights. The page keeps them as three CSS rules of {fmt(n_model)} numbers each and nothing else. Change the numbers and the same interpreter plays something else.</p>
</section>
<section>
  <h2>Multiplying two inputs with ReLUs</h2>
  <p>A ReLU network can only add up its inputs with fixed weights, and here the model's weights are inputs too. So every <code>weight × activation</code> is built from ReLUs. The weights come in as unsigned bytes <code>u = w + 128</code>.</p>
  <div class="scroll"><table>
    <tr><th>Step</th><th>As ReLU</th></tr>
    <tr><td>layer 1: x is 0 or 1</td><td><code>x · u = relu(u − 256·(1 − x))</code>, then subtract <code>128·Σx</code></td></tr>
    <tr><td>a byte's bits, all at once</td><td><code>q_j = ⌊u / 2^j⌋ = round(relu(2u + 1 − 2^j) / 2^(j+1))</code>, <code>bit_j = q_j − 2·q_(j+1)</code></td></tr>
    <tr><td>bit × activation, 0 ≤ h ≤ {fmt(18 * 127 + 1024)}</td><td><code>relu(h − {fmt(18 * 127 + 1024)}·(1 − bit))</code></td></tr>
    <tr><td>w × h</td><td><code>−128·(sign·h) + Σ_j&lt;7 2^j·(bit_j·h)</code>, two's complement</td></tr>
    <tr><td>then</td><td>step 1's rules: the best free square, if you moved and the game is on</td></tr>
  </table></div>
  {anatomy_html(net)}
</section>
<footer>Checked in numpy: for all three models, on every position and every X move (25,893 cases), the interpreter's logits and move equal the model's own forward pass. It is also exact on 300 random models with weights at the edges of the int8 range; those can push sums past 2²⁴, so the page only loads models that are checked to stay under it. The browser test plays each model's games in Chromium, Firefox and WebKit.</footer>
"""
    styles = """
.game, .turn, .step, .ui { display: contents; }
.pick { display: flex; gap: 8px; flex-wrap: wrap; margin-top: -14px; }
.main { display: grid; grid-template-columns: auto 1fr; gap: 36px; align-items: start; }
@media (max-width: 720px) { .main { grid-template-columns: 1fr; gap: 24px; } }
.pad { display: grid; gap: 12px; width: min(320px, calc(100vw - 32px)); }
.board { display: grid; grid-template-columns: repeat(3, 1fr); gap: 4px; padding: 4px; background: var(--line); border-radius: 8px; -webkit-tap-highlight-color: transparent; }
.cell { position: relative; aspect-ratio: 1; border-radius: 5px; display: grid; place-items: center; background: var(--cell);
  outline: calc(var(--win) * 2px) solid var(--accent); outline-offset: -2px; }
.cell::before, .cell::after { grid-area: 1 / 1; font-size: clamp(44px, 14vw, 60px); font-weight: 700; line-height: 1; }
.cell::before { content: "X"; opacity: var(--x); }
.cell::after { content: "O"; color: var(--accent); opacity: var(--o); transition: opacity .2s .2s; }
.cell .g { position: absolute; left: 6px; top: 3px; font-size: 10px; color: var(--muted); pointer-events: none; }
.cell .g::after { counter-reset: g var(--g); content: counter(g); }
.cell label { position: absolute; inset: 0; z-index: 1; cursor: pointer; border-radius: 5px; width: calc(var(--free) * 100%); }
@media (hover: hover) { .cell label:hover { box-shadow: inset 0 0 0 2px color-mix(in oklab, var(--cell), var(--ink-c) 35%); } }
.tools { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
.tools small { font-size: 11px; color: var(--muted); text-align: right; }
.result { display: grid; gap: 16px; align-content: start; min-width: 0; }
.status { display: grid; font-size: clamp(26px, 5vw, 36px); font-weight: 700; line-height: 1.1; }
.status span { grid-area: 1 / 1; transition: opacity .2s .3s; }
.status .start { opacity: calc(1 - var(--started)); color: var(--line); }
.status .turn-s { opacity: calc(var(--started) * (1 - var(--over))); }
.status .owin { opacity: var(--owin); color: var(--accent); }
.status .xwin { opacity: var(--xwin); color: var(--accent); }
.status .draw { opacity: var(--draw); }
.weights { margin: 0; display: grid; gap: 6px; }
.weights figcaption { font-size: 11px; color: var(--muted); }
.w1, .w2 { display: grid; gap: 1px; }
.w1 { grid-template-columns: repeat(64, 1fr); grid-template-rows: repeat(18, 5px); grid-auto-flow: column; max-width: 448px; }
.w2 { grid-template-columns: repeat(64, 1fr); grid-template-rows: repeat(9, 5px); grid-auto-flow: row; max-width: 448px; }
.w1 i, .w2 i { display: block; background: color-mix(in oklab, color-mix(in oklab, var(--cell), var(--accent) calc(max(0, var(--v) - 128) / 127 * 100%)), var(--hand) calc(max(0, 128 - var(--v)) / 128 * 100%)); }
"""
    kb = page("interpreter", "Network Interpreter",
              "One fixed ReLU network in pure CSS takes a tic-tac-toe model's weights as inputs and plays like that model.",
              css.render() + "\n" + "\n".join(turn_rules), styles, body)
    print(f"interpreter/index.html: {kb:.0f} KB, {len(css.names):,} properties")


if __name__ == "__main__":
    build()
