"""Step 1 pages: tic-tac-toe and connect four where the whole player is one handwired net.

    .venv/bin/python handwired/scripts/build_games.py

The CSS has three kinds of computed values and nothing else:
  inputs     set by a checked radio: `.app:has(#m0_4:checked) { --X0_4: 1 }`
  sums       which board each copy of the net sees: `calc(var(--X0_4) + var(--X1_4))`
  units      the nets, every one `max(0, calc(bias + w * var(--…) + …))`, outputs `calc(…)`
No clamp(), no product of two values, no argmax written in CSS: the rules are weights.
"""
import json

import numpy as np

from common import REPO, Stylesheet, anatomy_html, fmt, load_net, page, var, weight_counts

TTT_X, TTT_O = 5, 4      # you play at most 5 times, the net answers the first 4
C4_TURNS = 21
ROWS, COLS = 6, 7


def trained_params(path):
    layers = json.loads(path.read_text())["layers"]
    return sum(np.array(l["W"]).size + len(l["b"]) for l in layers)


# ---------------------------------------------------------------- tic-tac-toe

def build_ttt():
    player, judge = load_net("ttt_player"), load_net("ttt_judge")
    css = Stylesheet()
    A = ".app"

    for t in range(TTT_X):
        for i in range(9):
            css.register(f"X{t}_{i}", inherits=False)
            css.rule(f".app:has(#m{t}_{i}:checked)", f"--X{t}_{i}: 1;")
        css.define(A, f"played{t}", "calc(" + " + ".join(var(f"X{t}_{i}") for i in range(9)) + ")", inherits=False)

    # one copy of the player per O move, each looking at the board right after your move t
    for t in range(TTT_O):
        for i in range(9):
            css.define(A, f"xb{t}_{i}", "calc(" + " + ".join(var(f"X{u}_{i}") for u in range(t + 1)) + ")",
                       inherits=False)
            css.define(A, f"ob{t}_{i}", "calc(" + " + ".join(var(f"O{u}_{i}") for u in range(t)) + ")" if t else "0",
                       inherits=False)
        inputs = [f"xb{t}_{i}" for i in range(9)] + [f"ob{t}_{i}" for i in range(9)] + [f"played{t}"]
        css.net(A, player, inputs, f"p{t}_", [f"O{t}_{c}" for c in range(9)], inherits_outputs=False)

    # the final board, and the judge net on it
    for i in range(9):
        css.define(A, f"x{i}", "calc(" + " + ".join(var(f"X{t}_{i}") for t in range(TTT_X)) + ")")
        css.define(A, f"o{i}", "calc(" + " + ".join(var(f"O{t}_{i}") for t in range(TTT_O)) + ")")
    css.net(A, judge, [f"x{i}" for i in range(9)] + [f"o{i}" for i in range(9)], "j_", judge.out_names)

    turn_rules = [".cell label { display: none; }",
                  ".app:not(:has([name=t0]:checked)) .cell label.t0 { display: block; }"]
    for t in range(1, TTT_X):
        turn_rules.append(f".app:has([name=t{t - 1}]:checked):not(:has([name=t{t}]:checked)) .cell label.t{t} "
                          "{ display: block; }")
    for i in range(9):
        css.rule(f"#c{i}", f"--x: var(--x{i}); --o: var(--o{i}); --win: var(--win{i}); --free: var(--free{i});")
    css.rule(".app:has([name=t0]:checked)", "--started: 1;")
    css.register("started")

    radios = "\n".join("      " + "".join(f'<input type="radio" name="t{t}" id="m{t}_{i}" tabindex="-1">' for i in range(9))
                       for t in range(TTT_X))
    cells = "\n".join(
        f'      <div class="cell" id="c{i}">' +
        "".join(f'<label for="m{t}_{i}" class="t{t}" aria-label="cell {i + 1}"></label>' for t in range(TTT_X)) + "</div>"
        for i in range(9))

    n_trained = trained_params(REPO / "tic-tac-toe" / "model.json")
    body = f"""
<header>
  <h1>Tic-tac-toe, one network</h1>
  <p class="lead">You play X. The computer plays O, and it never loses.</p>
</header>
<div class="explain"><ul><li><b>What's new:</b> in the original demo, a trained network picked the moves, and rules written by hand sat around it: “this square is taken”, “X has won”. Here those rules became part of the network. The whole player is <b>one network</b>.</li><li><b>What you'll notice:</b> nothing different, and that's the point. It plays exactly like the original, move for move. The rules turned into numbers without changing anything.</li><li><b>Try:</b> play a game. Every move goes through that one network.</li></ul></div>
<div class="main">
  <div class="pad">
    <form id="game">
{radios}
    </form>
    <div class="board">
{cells}
    </div>
    <div class="tools"><button type="reset" form="game">New game</button><small>You always start.</small></div>
  </div>
  <div class="result">
    <div class="status" aria-live="polite"><span class="start">Your move</span><span class="turn">Your turn</span><span class="owin">The network wins</span><span class="xwin">You win?!</span><span class="draw">Draw</span></div>
    <p class="note">No rule is written in the CSS: the network decides everything, including when the game is over.</p>
  </div>
</div>
<details class="deep"><summary>How it works (for the curious)</summary>
<section>
  <h2>The player, layer by layer</h2>
  <p>{fmt(player.n_units)} units and {fmt(player.n_weights)} nonzero weights. The trained layers come first; everything after the logits was written by hand. Each layer only reads the one before, so values needed later pass down through carry units, <code>max(0, x) = x</code>.</p>
  {anatomy_html(player)}
</section>
<section>
  <h2>How a rule becomes weights</h2>
  <div class="scroll"><table>
    <tr><th>Rule</th><th>As ReLU</th></tr>
    <tr><td>x ≥ t, for an integer x</td><td><code>relu(x − t + 1) − relu(x − t)</code></td></tr>
    <tr><td>a AND b AND c</td><td><code>relu(a + b + c − 2)</code></td></tr>
    <tr><td>a OR b OR …</td><td><code>s − relu(s − 1)</code>, with s = a + b + …</td></tr>
    <tr><td>a line of three X</td><td><code>relu(x₀ + x₁ + x₂ − 2)</code></td></tr>
    <tr><td>square c is the best free one</td><td><code>relu(1 − Σ_d beats(d, c))</code>, where d beats c if it is free and c isn't, or both are and d's score is higher</td></tr>
    <tr><td>O moves only if you moved, didn't win, and the board isn't full</td><td><code>relu(best_c + moved − x_won − full − 1)</code></td></tr>
  </table></div>
</section>
<footer>Checked in numpy on all 5,478 reachable boards: the network's move equals the original demo's integer model on all 2,097 positions where O is to move, and the judge net's winner, draw and free squares are right on every board. The browser test replays the original fixture games in Chromium, Firefox and WebKit. Built by <code>handwired/scripts/rules.py</code> and <code>build_games.py</code>.</footer>
</details>
"""
    styles = """
.main { display: grid; grid-template-columns: auto 1fr; gap: 40px; align-items: start; }
@media (max-width: 660px) { .main { grid-template-columns: 1fr; gap: 24px; } }
.pad { display: grid; gap: 12px; width: min(320px, calc(100vw - 32px)); }
.board { display: grid; grid-template-columns: repeat(3, 1fr); gap: 4px; padding: 4px; background: var(--line); border-radius: 8px; -webkit-tap-highlight-color: transparent; }
.cell { position: relative; aspect-ratio: 1; border-radius: 5px; display: grid; place-items: center; background: var(--cell);
  outline: calc(var(--win) * 2px) solid var(--accent); outline-offset: -2px; }
.cell::before, .cell::after { grid-area: 1 / 1; font-size: clamp(44px, 14vw, 60px); font-weight: 700; line-height: 1; }
.cell::before { content: "X"; opacity: var(--x); }
.cell::after { content: "O"; color: var(--accent); opacity: var(--o); scale: calc(0.6 + 0.4 * var(--o)); transition: opacity .2s .2s, scale .2s .2s; }
.cell label { position: absolute; inset: 0; z-index: 1; cursor: pointer; border-radius: 5px; width: calc(var(--free) * 100%); }
@media (hover: hover) { .cell label:hover { box-shadow: inset 0 0 0 2px color-mix(in oklab, var(--cell), var(--ink-c) 35%); } }
.tools { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
.tools small { font-size: 12px; color: var(--muted); }
.result { display: grid; gap: 14px; align-content: start; }
.status { display: grid; font-size: clamp(28px, 6vw, 40px); font-weight: 700; line-height: 1.1; }
.status span { grid-area: 1 / 1; transition: opacity .2s .3s; }
.status .start { opacity: calc(1 - var(--started)); color: var(--line); }
.status .turn { opacity: calc(var(--started) * (1 - var(--over))); }
.status .owin { opacity: var(--owin); color: var(--accent); }
.status .xwin { opacity: var(--xwin); color: var(--accent); }
.status .draw { opacity: var(--draw); }
.note { font-size: 13px; color: var(--muted); max-width: 44ch; margin: 0; }
"""
    kb = page("tic-tac-toe", "Handwired Tic-Tac-Toe",
              "Tic-tac-toe against one ReLU network in pure CSS: a trained policy plus its rules compiled into weights.",
              css.render() + "\n" + "\n".join(turn_rules), styles, body)
    print(f"tic-tac-toe/index.html: {kb:.0f} KB, {len(css.names):,} properties")


# ---------------------------------------------------------------- connect four

def build_c4():
    """21 nested turns, each `<div class="turn"><div class="step">`: .turn renames the board it
    inherits (--PY_i: var(--Y_i)) and holds this turn's radios; .step runs the turn net on it."""
    turn, judge = load_net("c4_turn"), load_net("c4_judge")
    cells = ROWS * COLS
    css = Stylesheet()
    T, S, U = ".turn", ".step", ".ui"

    for i in range(cells):
        css.define(T, f"PY{i}", var(f"Y{i}"))
        css.define(T, f"PM{i}", var(f"M{i}"))
    for c in range(COLS):
        css.define(T, f"h{c}", "0")
        css.rule(f".turn:has(> .c{c}:checked)", f"--h{c}: 1;")
    inputs = [f"PY{i}" for i in range(cells)] + [f"PM{i}" for i in range(cells)] + [f"h{c}" for c in range(COLS)]
    outs = turn.out_names  # Y0.., M0.., a0..
    css.net(S, turn, inputs, "t", outs)

    css.net(U, judge, [f"Y{i}" for i in range(cells)] + [f"M{i}" for i in range(cells)], "j", judge.out_names)
    css.rule(".app:has(.turn input:checked) .ui", "--started: 1;")
    css.register("started")
    for c in range(COLS):
        css.rule(f"#col{c}", f"--free: var(--free{c});")
    for i in range(cells):
        css.rule(f"#q{i}", f"--you: var(--Y{i}); --me: var(--M{i}); --win: var(--win{i});")

    turn_rules = [".col label { display: none; }",
                  ".app:not(:has([name=t0]:checked)) .col label.t0 { display: block; }"]
    for t in range(1, C4_TURNS):
        turn_rules.append(f".app:has([name=t{t - 1}]:checked):not(:has([name=t{t}]:checked)) .col label.t{t} "
                          "{ display: block; }")

    opening = "\n".join(
        '<div class="turn">' + "".join(f'<input type="radio" name="t{t}" id="m{t}_{c}" class="c{c}" tabindex="-1">'
                                      for c in range(COLS)) + '<div class="step">' for t in range(C4_TURNS))
    closing = "</div></div>" * C4_TURNS
    board = "".join(f'<i id="q{r * COLS + c}"></i>' for r in reversed(range(ROWS)) for c in range(COLS))
    cols = "".join(f'<div class="col" id="col{c}">' +
                   "".join(f'<label for="m{t}_{c}" class="t{t}" aria-label="column {c + 1}"></label>' for t in range(C4_TURNS))
                   + "</div>" for c in range(COLS))

    n_trained = trained_params(REPO / "connect-four" / "model.json")
    body = f"""
<header>
  <h1>Connect four, one network per turn</h1>
  <p class="lead">You play red. Yellow is one network that does a whole turn.</p>
</header>
<div class="explain"><ul><li><b>One turn, one network:</b> it drops your piece, thinks, checks whether it can win or must block you, and drops its own piece. All of that is a single network.</li><li><b>What you'll notice:</b> it plays exactly like the original connect-four demo. Before, its tactics were rules written by hand next to the network. Now they are part of it.</li><li><b>Try:</b> click a column to play.</li></ul></div>
<form id="game" class="game">
{opening}
<div class="ui">
  <div class="main">
    <div class="pad">
      <div class="board-wrap"><div class="board">{board}</div><div class="cols">{cols}</div></div>
      <div class="tools"><button type="reset" form="game">New game</button><small>You play first.</small></div>
    </div>
    <div class="result">
      <div class="status" aria-live="polite"><span class="start">Your move</span><span class="turn-s">Your turn</span><span class="mewin">The network wins</span><span class="youwin">You win!</span><span class="draw">Draw</span></div>
      <p class="note">The page runs the same network 21 times, once for each turn of the game.</p>
    </div>
  </div>
</div>
{closing}
</form>
<details class="deep"><summary>How it works (for the curious)</summary>
<section>
  <h2>One turn, layer by layer</h2>
  <p>{fmt(turn.n_units)} units and {fmt(turn.n_weights)} nonzero weights. Your column comes in as 7 inputs (one-hot, or all zero when you haven't played that turn yet); the board after both moves goes out. Each layer reads only the one before, so values needed later pass down through carry units.</p>
  {anatomy_html(turn)}
</section>
<section>
  <h2>The rules inside</h2>
  <div class="scroll"><table>
    <tr><th>Rule</th><th>As ReLU</th></tr>
    <tr><td>a piece in column c lands on row r</td><td><code>relu(filled(r − 1, c) − filled(r, c))</code></td></tr>
    <tr><td>it lands there and finishes a line</td><td><code>relu(lands + three_others − 3)</code></td></tr>
    <tr><td>win, block, danger in column c</td><td>OR over rows and lines: <code>s − relu(s − 1)</code></td></tr>
    <tr><td>the move</td><td>column d beats c if its tiers (legal, win, block, safe) say so, else if its score is higher: <code>[key_d &gt; key_c] + [key_d = key_c]·[score_d &gt; score_c]</code>, each part a pair of ReLUs</td></tr>
    <tr><td>the original's <code>2²¹·win + 2²⁰·block − 2¹⁹·danger − 2²²·full</code></td><td>the same order, without numbers anywhere near 2²⁴</td></tr>
  </table></div>
</section>
<footer>Checked in numpy: the move matches the original demo's integer model on 24,708 positions (its browser-test games plus 3,000 games against random and greedy players), and the turn net replays 400 random games turn by turn. The browser test replays the original fixture games in Chromium, Firefox and WebKit.</footer>
</details>
"""
    styles = """
.game, .turn, .step, .ui { display: contents; }
.main { display: grid; grid-template-columns: auto 1fr; gap: 36px; align-items: start; }
@media (max-width: 720px) { .main { grid-template-columns: 1fr; gap: 24px; } }
.pad { display: grid; gap: 12px; width: min(440px, calc(100vw - 32px)); }
.board-wrap { position: relative; -webkit-tap-highlight-color: transparent; }
.board { display: grid; grid-template-columns: repeat(7, 1fr); gap: 6px; padding: 6px; background: var(--hand); border-radius: 10px; }
.board i { aspect-ratio: 1; border-radius: 50%;
  background: color-mix(in oklab, color-mix(in oklab, var(--bg), #d93a2b calc(var(--you) * 100%)), #f2b01e calc(var(--me) * 100%));
  box-shadow: inset 0 2px 3px rgb(0 0 0 / .25); outline: calc(var(--win) * 3px) solid var(--fg); outline-offset: -3px; transition: background-color .25s; }
.cols { position: absolute; inset: 0; display: grid; grid-template-columns: repeat(7, 1fr); padding: 0 3px; }
.col { position: relative; }
.col label { position: absolute; inset: 0; cursor: pointer; border-radius: 8px; width: calc(var(--free) * 100%); }
@media (hover: hover) { .col label:hover { background: rgb(255 255 255 / .12); box-shadow: inset 0 0 0 2px rgb(255 255 255 / .35); } }
.tools { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
.tools small { font-size: 12px; color: var(--muted); }
.result { display: grid; gap: 14px; align-content: start; }
.status { display: grid; font-size: clamp(26px, 5vw, 36px); font-weight: 700; line-height: 1.1; }
.status span { grid-area: 1 / 1; transition: opacity .2s .3s; }
.status .start { opacity: calc(1 - var(--started)); color: var(--line); }
.status .turn-s { opacity: calc(var(--started) * (1 - var(--over))); }
.status .mewin { opacity: var(--mewin); color: #c98a00; }
.status .youwin { opacity: var(--youwin); color: #d93a2b; }
.status .draw { opacity: calc(var(--full) * (1 - var(--mewin)) * (1 - var(--youwin))); }
.note { font-size: 13px; color: var(--muted); max-width: 44ch; margin: 0; }
"""
    kb = page("connect-four", "Handwired Connect Four",
              "Connect four in pure CSS where one whole turn, rules and trained policy, is a single ReLU network.",
              css.render() + "\n" + "\n".join(turn_rules), styles, body)
    print(f"connect-four/index.html: {kb:.0f} KB, {len(css.names):,} properties")


if __name__ == "__main__":
    build_ttt()
    build_c4()
