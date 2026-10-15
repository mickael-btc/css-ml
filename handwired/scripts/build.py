"""Build every handwired page, and the overview.

    .venv/bin/python handwired/scripts/build.py           # pages only, from the nets already in handwired/nets
    .venv/bin/python handwired/scripts/build.py --nets    # also recompile and re-check every net first (~2 min)
"""
import json
import subprocess
import sys

from common import NETS, SCRIPTS, fmt, load_net, page
import build_games
import build_interpreter
import build_tower
import build_unfrozen


def overview():
    ttt, turn, interp = load_net("ttt_player"), load_net("c4_turn"), load_net("interp_player")
    tower = json.loads((NETS / "tower.json").read_text())
    s = tower["sizes"]
    cards = [
        ("tic-tac-toe", "1", "Rules as weights: tic-tac-toe",
         f"The trained policy plus its rules (free squares, argmax, “stop once X has won”, who won) as one {ttt.depth}-layer ReLU net. Every computed value in the CSS is <code>max(0, calc(…))</code> or a sum.",
         "plays exactly like the original on all 2,097 positions"),
        ("connect-four", "1", "Rules as weights: connect four",
         f"A whole turn is one {turn.depth}-layer net of {fmt(turn.n_units)} units: gravity, the trained policy, the three tactics, the choice. Applied 21 times by nested elements.",
         "matches the original on 24,708 positions"),
        ("interpreter", "2", "A network that runs a network",
         f"One fixed net of {fmt(interp.n_units)} units takes a board and the 1,801 weights of any 18 → 64 → 9 model as inputs. Load the trained model, a hand-written one or random weights: same CSS.",
         "equals each model's forward pass in all 25,893 cases"),
        ("tower", "3", "A network running a copy of itself",
         f"<code>interpret(net)</code> applied to its own output: I1 runs a tiny net N, I2 runs I1 running N. {s[0]['units']} → {fmt(s[1]['units'])} units; I3 would be {fmt(s[2]['units'])}.",
         "exact for every N in range, at both levels"),
        ("unfrozen", "4", "Unfrozen: a rule that keeps learning",
         "“Red inside the box” compiled to a 2 → 4 → 1 ReLU net, then all 17 numbers trained by backpropagation in CSS on points you place.",
         "every weight at every epoch equals numpy"),
    ]
    items = "".join(f'<a class="card" href="{f}/"><small>step {n}</small><h2>{t}</h2><p>{d}</p><span>{c}</span></a>'
                    for f, n, t, d, c in cards)
    body = f"""
<header>
  <h1>Handwired</h1>
  <p>Regular code, rewritten as a neural network. Ideas from Cheng Lou's <a href="https://github.com/chenglou/handwired">handwired</a>: an <code>if</code> becomes ReLUs, <code>(a &gt; 4) ? 10 : 12</code> becomes <code>above4 = relu(a − 4) − relu(a − 5)</code>, <code>above4·10 + (1 − above4)·12</code>. No training, exact, and still differentiable.</p>
  <p>Here it meets the rest of this repo, where networks run in the browser's style engine. Every page is pure CSS, <b>0 bytes of JavaScript</b>, and each one is tested to match its Python reference exactly in Chromium, Firefox and WebKit.</p>
</header>
<div class="cards">{items}</div>
<section>
  <h2>The compiler</h2>
  <p><code>handwired/scripts/hw.py</code> is about 450 lines. A program is built from inputs, constants, <code>+</code>, constant <code>×</code> and <code>relu</code>; helpers give integer logic (<code>ge</code>, <code>eq</code>, <code>AND</code>, <code>OR</code>, gating, a lexicographic argmax). <code>compile()</code> places each ReLU at 1 + the depth of its deepest input, carries values that are needed further down through identity units, and returns an ordinary layered integer MLP. The same net runs in numpy, and <code>to_css()</code> writes it out as registered <code>@property</code> integers.</p>
  <div class="scroll"><table>
    <tr><th>Code</th><th>Becomes</th></tr>
    <tr><td><code>x ≥ t</code> (integer x)</td><td><code>relu(x − t + 1) − relu(x − t)</code></td></tr>
    <tr><td><code>x == t</code></td><td><code>relu(x − t + 1) − 2·relu(x − t) + relu(x − t − 1)</code></td></tr>
    <tr><td><code>a &amp;&amp; b &amp;&amp; c</code></td><td><code>relu(a + b + c − 2)</code></td></tr>
    <tr><td><code>a || b || …</code></td><td><code>s − relu(s − 1)</code></td></tr>
    <tr><td><code>bit ? v : 0</code>, 0 ≤ v ≤ M</td><td><code>relu(v − M·(1 − bit))</code></td></tr>
    <tr><td><code>⌊u / 2^j⌋</code></td><td><code>round(relu(2u + 1 − 2^j) / 2^(j+1))</code>, one integer unit</td></tr>
    <tr><td><code>w * x</code>, both inputs</td><td>the bits of w, each gating x, summed with powers of two</td></tr>
  </table></div>
</section>
<footer>Rebuild with <code>.venv/bin/python handwired/scripts/build.py --nets</code>, test with <code>node handwired/tests/browser.test.mjs</code>. See <code>handwired/README.md</code>.</footer>
"""
    styles = """
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 12px; }
.card { display: grid; gap: 8px; align-content: start; border: 1px solid var(--line); border-radius: 8px; padding: 16px; color: var(--fg); text-decoration: none; }
@media (hover: hover) { .card:hover { border-color: var(--hand); } }
.card small { color: var(--hand); font-size: 11px; }
.card h2 { font-size: 15px; margin: 0; line-height: 1.3; }
.card p { margin: 0; font-size: 13px; color: var(--muted); }
.card span { font-size: 12px; color: var(--accent); }
header a { color: var(--fg); text-underline-offset: 3px; }
"""
    kb = page("", "Handwired", "Regular code rewritten as neural networks, running in pure CSS: rules as weights, "
              "a network that runs networks, a tower of interpreters, and a rule that keeps learning.", "", styles, body)
    print(f"index.html: {kb:.0f} KB")


def main():
    if "--nets" in sys.argv:
        for script in ("rules.py", "interpreter.py", "tower.py"):
            subprocess.run([sys.executable, str(SCRIPTS / script)], check=True)
        subprocess.run([sys.executable, str(SCRIPTS / "unfrozen.py"), "-q", "--fixtures"], check=True)
    build_games.build_ttt()
    build_games.build_c4()
    build_interpreter.build()
    build_tower.build()
    build_unfrozen.build()
    overview()


if __name__ == "__main__":
    main()
