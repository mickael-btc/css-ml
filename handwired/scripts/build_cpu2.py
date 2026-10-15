"""Step 6 page: a game console made of neurons. Snake on a CPU whose clock tick is one ReLU network.

    .venv/bin/python handwired/scripts/build_cpu2.py

60 nested turns. Each turn is `<div class="turn">` (it copies the state it inherits, --PA: var(--A) …,
wakes the CPU with --PH: 0, and holds that move's four radios), then 3 nested steps, each one net
of 12 clock ticks (`<div class="step">`, with `<div class="cp">` copying the state between them).
A turn whose radios are all unchecked is display: none, so the browser computes nothing in it or
below it: a click only computes that one turn, 36 ticks. Browsers stop nesting HTML at 512 levels,
which is why 12 ticks share one net: 60 turns x 6 elements stay well under it.

The screen for the state before move t sits in turn t, after its steps, so it reads the state the
previous turn left; it hides as soon as move t is played. Only its four arrows are for move t.
"""
from common import Stylesheet, anatomy_html, fmt, page, var
from cpu2 import (BUTTONS, CONSTS, DELTA, FOOD, HEAD, K, NEW, NO_ARG, OPS, RAM, ROM, SCORE, SIDE, STATE, STATUS,
                  STRIDE, TAIL, TURNS, U, W, Dag, NETS, cell, cpu2_net, is_border, snake_cartridge)
import json

ARROWS = {1: "↑", 2: "→", 3: "↓", 4: "←"}
WHAT = {
    "WAIT": "stop here for good", "SYNC": "wait for the next button, then go on at t",
    "LDI": "A = n", "LD": "A = memory[m]", "ST": "memory[m] = A",
    "LDX": "A = memory[B + n]", "STX": "memory[B + n] = A", "SETX": "memory[B] = n",
    "LDB": "B = memory[m]", "TAB": "B = A", "TBA": "A = B",
    "ADD": "A = A + memory[m]", "SUB": "A = A − memory[m]", "ADDI": "A = A + n",
    "WRAP": "if A ≥ n: A = A − n", "NEXT": "B = B + n (around the 90 cells)",
    "JMP": "go to t", "JZ": "go to t if A is 0", "JNZ": "go to t if A is not 0",
    "JXNZ": "go to t if memory[B] is not 0", "IN": "A = the button pressed (1 to 4)",
}
ARGS = {"LDI": "n", "LD": "m", "ST": "m", "LDX": "n", "STX": "n", "SETX": "n", "LDB": "m", "ADD": "m", "SUB": "m",
        "ADDI": "n", "WRAP": "n", "NEXT": "n", "JMP": "t", "JZ": "t", "JNZ": "t", "JXNZ": "t", "SYNC": "t"}


def memory_map():
    """The 9 x 10 framebuffer as a small grid: walls, the variables they hold, the playfield."""
    names = {SCORE: "score", HEAD: "head", TAIL: "tail", FOOD: "food", NEW: "new", STATUS: "state"}
    names.update({DELTA + b: ARROWS[b] for b in BUTTONS})
    cells = []
    for a in range(RAM):
        if a in names:
            cells.append(f'<i class="v"><small>{a}</small>{names[a]}</i>')
        elif is_border(a):
            cells.append(f'<i class="w"><small>{a}</small></i>')
        else:
            cells.append(f'<i><small>{a}</small></i>')
    return f'<div class="memmap">{"".join(cells)}</div>'


def build():
    d = json.loads((NETS / "cpu2.json").read_text())
    net = Dag.from_json(d)
    one = cpu2_net(1)
    ops, args, ram, listing = snake_cartridge()
    css = Stylesheet()

    # the cartridge and the machine at power-on, on the form
    G = ".game"
    for i in range(ROM):
        css.register(f"op{i}")
        css.register(f"arg{i}")
        css.set(G, f"op{i}", str(ops[i]))
        css.set(G, f"arg{i}", str(args[i]))

    # copies: a turn copies and wakes the CPU; between steps, a plain copy
    for k in STATE:
        css.define(".turn", f"P{k}", "0" if k in ("H", "CY") else var(k))
        css.set(".cp", f"P{k}", var(k))
    css.register("KEY")
    for b in BUTTONS:
        css.rule(f".turn > .k{b}:checked ~ .step", f"--KEY: {b};")

    inputs = {k: f"P{k}" for k in STATE}
    inputs.update({"KEY": "KEY"}, **{f"op{i}": f"op{i}" for i in range(ROM)}, **{f"arg{i}": f"arg{i}" for i in range(ROM)})
    rules = net.css(inputs, "u", STATE)
    for n, (name, expr) in enumerate(rules):
        css.define(".step", name, expr, inherits=n >= len(net.nodes))
    s0 = {k: 0 for k in STATE}
    s0.update({f"M{j}": v for j, v in enumerate(ram)})
    for k in STATE:
        css.set(G, k, str(s0[k]))

    # the screen: one copy per turn
    screen_cells = "".join(f'<i style="--v: var(--M{cell(r, c)})"></i>' for r in range(1, SIDE + 1) for c in range(1, SIDE + 1))

    def screen(t):
        if t < TURNS:
            pad = "".join(f'<label for="m{t}_{b}" class="b{b}" aria-label="{BUTTONS[b]}">{ARROWS[b]}</label>' for b in BUTTONS)
            left = "1 move left" if TURNS - t == 1 else f"{TURNS - t} moves left"
        else:
            pad, left = "", "no moves left"
        return f"""<div class="disp" id="s{t}">
  <div class="screen">{screen_cells}</div>
  <div class="side">
    <div class="score"><span>score</span><b></b></div>
    <div class="status"><span class="play">{"Your move" if t < TURNS else "Out of moves"}</span><span class="over">Game over</span></div>
    <div class="pad">{pad}</div>
    <div class="tools"><button type="reset">New game</button><small>{left}</small></div>
    <small class="ran">{'The console has not run yet.' if t == 0 else 'Your last move took <b></b> instructions.'}</small>
  </div>
</div>"""

    steps = K // U
    nest = []
    for t in range(TURNS):
        radios = "".join(f'<input type="radio" name="t{t}" id="m{t}_{b}" class="k{b}" tabindex="-1">' for b in BUTTONS)
        nest.append(f'<div class="turn">{radios}<div class="step">' + '<div class="cp"><div class="step">' * (steps - 1))
    closing = []
    for t in reversed(range(TURNS)):
        closing.append("</div></div>" * (steps - 1) + "</div>" + screen(t) + "</div>")
    machine = "\n".join(nest) + "\n" + screen(TURNS) + "\n" + "\n".join(closing)

    ops_rows = []
    names = [o for o in OPS]
    half = (len(names) + 1) // 2
    for r in range(half):
        cells = []
        for o in (names[r], names[r + half] if r + half < len(names) else None):
            if o is None:
                cells.append("<td></td><td>other codes do nothing</td>")
            else:
                cells.append(f"<td><code>{o}{' ' + ARGS[o] if o in ARGS else ''}</code></td><td>{WHAT[o]}</td>")
        ops_rows.append("<tr>" + "".join(cells) + "</tr>")
    listing_rows = "".join(
        f'<tr><td class="n">{i}</td><td>{lab}</td><td><code>{txt}</code></td><td>{com}</td></tr>'
        for i, (lab, txt, com) in enumerate(listing))
    per_tick = net.n_units // U

    body = f"""
<header>
  <h1>A game console made of neurons</h1>
  <p class="lead">Snake, played on a tiny computer whose every clock tick is one neural network. Pure CSS, with 0 bytes of JavaScript.</p>
</header>
<div class="explain"><ul>
  <li><b>The console:</b> a small 8-bit computer with 90 bytes of memory. The 8 × 8 screen is part of that memory: whatever the program writes there is what you see.</li>
  <li><b>One tick, one network:</b> {fmt(per_tick)} neurons read the next instruction, do it, and move on. Each of your moves runs {K} ticks of the game: move the head, eat, grow, put down new food, check for crashes.</li>
  <li><b>The game is a program:</b> {len(listing)} instructions in the console's own little language. The network knows nothing about Snake: give it another program and it plays another game.</li>
  <li><b>Try:</b> steer with the arrows. The snake's head is the bright square with the pale rim. Eat the red dots, don't hit the wall or your own tail. You have {TURNS} moves.</li>
</ul></div>
<form id="game" class="game">
{machine}
</form>
<details class="deep"><summary>How it works (for the curious)</summary>
<section>
  <h2>The console's instructions</h2>
  <p>Two 8-bit registers, A and B, a program counter, and 90 bytes of memory. The cartridge is {ROM} instructions, each a code and a number (n a number, m a memory cell, t an instruction). Numbers wrap around at 256, memory cells at 90.</p>
  <div class="scroll"><table class="isa">{"".join(ops_rows)}</table></div>
</section>
<section>
  <h2>The memory is the screen</h2>
  <p>A 9 × 10 grid: the 8 × 8 playfield and a wall around it (the left and right walls share one column). The snake only needs the wall to be "not empty", so wall cells also hold the game's variables. An empty cell is 0, the wall 1, food 2, the head 3, and a body cell holds the address of the next piece towards the head: moving the tail is one lookup. New food goes {STRIDE} cells further on, then further again while that cell is taken.</p>
  {memory_map()}
</section>
<section>
  <h2>The program</h2>
  <div class="scroll"><table class="listing"><tr><th class="n">#</th><th></th><th>instruction</th><th>what it does</th></tr>{listing_rows}</table></div>
</section>
<section>
  <h2>One tick, as ReLUs</h2>
  <div class="scroll"><table>
    <tr><th>Part</th><th>How</th></tr>
    <tr><td>fetch</td><td><code>[PC = i]</code> for every line, each gating that line's code and number: <code>relu(op_i − 31·(1 − [PC = i]) − 31·waiting)</code></td></tr>
    <tr><td>decode</td><td><code>[op = k]</code> for each code, from one shared row of units <code>relu(op − k + 1)</code></td></tr>
    <tr><td>address</td><td><code>m</code>, <code>B + n</code> or <code>B</code>, each mod 90 with one unit: <code>⌊x / 90⌋ = round(relu(2x − 89) / 180)</code></td></tr>
    <tr><td>memory</td><td>every cell read through <code>[address = j]</code>; the written cell becomes <code>cell − relu(cell − 255·h) + relu(data − 255·h)</code>, <code>h = 2 − [address = j] − write</code></td></tr>
    <tr><td>conditions</td><td>folded into the gates: <code>relu(x − 47·(1 − JXNZ) − 47·memory[B])</code> is "x if JXNZ and the cell is empty": with x the next line it falls through, and subtracted from t it jumps when the cell is not empty</td></tr>
  </table></div>
  <p>One tick is {fmt(per_tick)} units. The page writes {U} ticks out as one network of {fmt(net.n_units)} units and {fmt(net.n_weights)} weights, and runs it {steps} times per move: {fmt(net.n_units * steps)} units per click. As a strict layered network one tick is {one.depth} layers and {fmt(one.n_units)} units, most of them only passing values down a layer (grey below). In CSS every unit of a step sits on the same element and can read any other directly, so the page leaves those out.</p>
  {anatomy_html(one)}
  <p>A move whose buttons haven't been pressed yet is hidden, so the browser computes nothing inside it: a click computes just that move's {K} ticks. Browsers stop nesting HTML at 512 levels; {U} ticks per network keeps the {TURNS} moves at {TURNS * 2 * steps} levels.</p>
</section>
<section>
  <h2>Checked</h2>
  <p>A plain Python emulator runs the console, and a plain Python Snake checks the emulator: 5,000 games, every move. The networks match the emulator in numpy, every register and memory cell: one tick, every tick, on 3,000 random programs and 100 games; the {U}-tick network after every step, on 2,000 random programs and 300 full games. Every partial sum, measured on every state of those runs, stays below 2<sup>24</sup> by a factor of more than 100, so 32-bit CSS engines are exact too. The browser test plays 12 games by clicking the arrows in Chromium, Firefox and WebKit and compares the screen, score and game over after every move.</p>
</section>
</details>
"""
    turn_rules = """
.game, .turn, .cp, .step { display: contents; }
.turn > .step { display: none; }
.turn > input:checked ~ .step { display: contents; }
.turn > input:checked ~ .disp { display: none; }
"""
    styles = turn_rules + """
.disp { display: grid; grid-template-columns: auto 1fr; gap: 28px; align-items: start; }
@media (max-width: 640px) {
  .disp { grid-template-columns: 1fr; gap: 18px; }
  .app > .explain { order: 1; margin-top: 0; }                     /* on a phone the game comes first */
  .app > details.deep, .app > details.deep ~ * { order: 2; }
}
.screen { display: grid; grid-template-columns: repeat(8, 1fr); gap: 3px; padding: 10px; border-radius: 10px;
  width: min(380px, calc(100vw - 32px)); background: #1c1b19; box-shadow: inset 0 0 0 4px #3a3833, 0 0 0 1px var(--line); }
.screen i { aspect-ratio: 1; border-radius: 4px;
  --food: max(0, 1 - max(var(--v) - 2, 2 - var(--v)));
  --head: max(0, 1 - max(var(--v) - 3, 3 - var(--v)));
  --body: clamp(0, var(--v) - 3, 1);
  background: color-mix(in oklab, color-mix(in oklab, color-mix(in oklab, #2e2d29,
    #3f8a3a calc(var(--body) * 100%)), #a6e65f calc(var(--head) * 100%)), #e2412b calc(var(--food) * 100%));
  box-shadow: inset 0 0 0 calc(var(--head) * 3px) #efffd9;
  border-radius: calc(4px + var(--food) * 40% + var(--head) * 18%); transition: background-color .15s; }
.side { display: grid; gap: 16px; align-content: start; }
.score { display: flex; align-items: baseline; gap: 12px; }
.score span { color: var(--muted); font-size: 13px; }
.score b { font-size: 40px; line-height: 1; color: var(--hand); }
.score b::after { counter-reset: v var(--M0); content: counter(v); }
.status { display: grid; font-size: clamp(22px, 5vw, 28px); font-weight: 700; line-height: 1.1; }
.status span { grid-area: 1 / 1; }
.status .play { opacity: calc(2 - var(--M8)); }
.status .over { opacity: calc(var(--M8) - 1); color: #d93a2b; }
.pad { display: grid; grid-template-columns: repeat(3, 64px); column-gap: 6px;
  transform: scale(calc(2 - var(--M8))); transform-origin: left top; -webkit-tap-highlight-color: transparent;
  grid-template-rows: repeat(3, calc((2 - var(--M8)) * 56px)); row-gap: calc((2 - var(--M8)) * 6px); }
.pad:empty { display: none; }
.pad label { display: grid; place-items: center; border: 1px solid var(--line); border-radius: 10px; font-size: 24px;
  cursor: pointer; user-select: none; background: color-mix(in oklab, var(--bg), var(--fg) 4%); }
@media (hover: hover) { .pad label:hover { border-color: var(--hand); color: var(--hand); } }
.pad label:active { background: color-mix(in oklab, var(--bg), var(--hand) 20%); }
.pad .b1 { grid-area: 1 / 2; } .pad .b4 { grid-area: 2 / 1; } .pad .b2 { grid-area: 2 / 3; } .pad .b3 { grid-area: 3 / 2; }
.tools { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
.tools small, .ran { font-size: 12px; color: var(--muted); }
.tools small { opacity: calc(2 - var(--M8)); }
.ran b { color: var(--fg); font-weight: 600; }
.ran b::after { counter-reset: v var(--CY); content: counter(v); }
.isa td { padding: 4px 12px 4px 0; font-size: 12px; }
.isa td:nth-child(odd) { white-space: nowrap; }
.listing td, .listing th { padding: 3px 10px 3px 0; font-size: 12px; }
.listing td:nth-child(2) { color: var(--hand); }
.listing td:last-child { color: var(--muted); }
.memmap { display: grid; grid-template-columns: repeat(9, 1fr); gap: 2px; max-width: 420px; font-size: 10px; }
.memmap i { font-style: normal; aspect-ratio: 1; border-radius: 3px; background: var(--cell); position: relative;
  display: grid; place-items: center; color: var(--fg); overflow: hidden; }
.memmap i small { position: absolute; top: 1px; left: 3px; font-size: 8px; color: var(--muted); }
.memmap i.w { background: color-mix(in oklab, var(--cell), var(--fg) 30%); }
.memmap i.v { background: color-mix(in oklab, var(--cell), var(--hand) 45%); }
.explain a { color: var(--fg); }
"""
    kb = page("arcade", "Neural Game Console",
              "Snake in pure CSS on a tiny 8-bit computer whose every clock tick is one ReLU network.",
              css.render(), styles, body)
    print(f"arcade/index.html: {kb:.0f} KB, {len(css.names):,} properties, {net.n_units:,} units per step, "
          f"{steps} steps x {TURNS} turns, depth {TURNS * 2 * steps}")


if __name__ == "__main__":
    build()
