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
        ("tic-tac-toe", "1", "Tic-tac-toe, one network",
         "The computer player and all its rules in a single network. It plays exactly like before.", "Play →"),
        ("connect-four", "1", "Connect four, one network per turn",
         "One network drops the pieces, thinks, blocks and plays: a whole turn.", "Play →"),
        ("interpreter", "2", "Swap the brain",
         "A fixed “console” network plays whichever strategy you load into it, like swapping cartridges.", "Swap →"),
        ("tower", "3", "Nesting dolls",
         "A console running a console running a tiny network. Both give the same answers.", "Look →"),
        ("unfrozen", "4", "A rule that keeps learning",
         "A rule written by hand becomes a network, then keeps learning from the points you place.", "Teach it →"),
    ]
    items = "".join(f'<a class="card" href="{f}/"><small>step {n}</small><h2>{t}</h2><p>{d}</p><span>{c}</span></a>'
                    for f, n, t, d, c in cards)
    body = f"""
<header>
  <h1>Handwired</h1>
  <p class="lead">Normal code, turned into a neural network. Then networks that run other networks. All in CSS, with 0 bytes of JavaScript.</p>
</header>
<div class="explain"><ul><li><b>A neural network</b> is just thousands of tiny steps of “add these numbers, and if the result is negative, use 0”.</li><li><b>That's enough to write rules.</b> “Is this square free?” or “did someone win?” each become a few of those steps. So code can become a network, with no training.</li><li><b>And a network can run another network</b>, if you give it that network's numbers: like a console and its cartridge.</li><li>Idea from Cheng Lou's <a href="https://github.com/chenglou/handwired">handwired</a>. Each page is tested to give exactly the same answers as Python, in Chrome, Firefox and Safari.</li></ul></div>
<div class="cards">{items}</div>
<details class="deep"><summary>How it works (for the curious)</summary>
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
</details>
"""
    styles = """
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 12px; }
.card { display: grid; gap: 8px; align-content: start; border: 1px solid var(--line); border-radius: 8px; padding: 16px; color: var(--fg); text-decoration: none; }
@media (hover: hover) { .card:hover { border-color: var(--hand); } }
.card small { color: var(--hand); font-size: 11px; }
.card h2 { font-size: 15px; margin: 0; line-height: 1.3; }
.card p { margin: 0; font-size: 13px; color: var(--muted); }
.card span { font-size: 12px; color: var(--accent); }
.explain a { color: var(--fg); }
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
