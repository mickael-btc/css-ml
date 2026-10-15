"""Mini OS page: a desktop with three windows and a dock whose window manager is one ReLU net per click.

    .venv/bin/python handwired/scripts/build_os.py

40 nested event slots, like the connect-four page: `<div class="slot">` copies the state it inherits
(--Po0: var(--o0) …) and holds that slot's 37 radios, one per control; `<div class="step">` runs the
event net on it. `.ui`, innermost, runs the screen net on the final state and draws the desktop. Every
control holds one <label> per slot, and only the next slot's labels are shown.
"""
import json

from common import NETS, SCRIPTS, Stylesheet, anatomy_html, fmt, page, var
from hw import Net

import mini_os as wm

PHOTOS = [  # (name, gradient)
    ("Dune", "linear-gradient(165deg, #f6d29b 0%, #e9a15f 45%, #b8643a 46%, #8c4a2f 100%)"),
    ("Lagoon", "radial-gradient(circle at 70% 30%, #e8fbff 0 9%, transparent 10%), linear-gradient(180deg, #7fd3e6 0%, #2a9bb8 55%, #11607a 100%)"),
    ("Aurora", "linear-gradient(200deg, transparent 40%, #0b1430 70%), linear-gradient(120deg, #0b1430 0%, #1f6f5c 35%, #6be3a4 50%, #3a2a7a 70%, #0b1430 100%)"),
    ("Ember", "radial-gradient(circle at 50% 110%, #ffd36b 0%, #ff7a2f 30%, #a3201d 60%, #2a0b0b 100%)"),
    ("Moss", "radial-gradient(ellipse at 30% 80%, #b6d36a 0%, transparent 45%), linear-gradient(180deg, #2f5d3a 0%, #5f8f3f 60%, #2b3d1f 100%)"),
    ("Dusk", "radial-gradient(circle at 50% 62%, #ffe2a8 0 12%, transparent 13%), linear-gradient(180deg, #34245e 0%, #b0517a 55%, #f39b6d 63%, #3b2d4f 64%, #1d1830 100%)"),
    ("Glacier", "linear-gradient(140deg, transparent 55%, #9ec7e0 56%, #e9f5fb 100%), linear-gradient(180deg, #d9eef8 0%, #9cc9e3 60%, #5f8fb5 100%)"),
    ("Poppy", "radial-gradient(circle at 35% 40%, #ff4b3e 0 14%, transparent 15%), radial-gradient(circle at 68% 62%, #ff6f3e 0 11%, transparent 12%), linear-gradient(180deg, #9fd0ff 0%, #c9e6a4 55%, #6e9b3d 100%)"),
    ("Night", "radial-gradient(circle at 75% 25%, #fff6c9 0 6%, transparent 7%), radial-gradient(1px 1px at 20% 30%, #fff 50%, transparent 51%), linear-gradient(180deg, #050b1f 0%, #142a5c 70%, #2b3f73 100%)"),
]
GLYPH = {"left": "←", "up": "↑", "down": "↓", "right": "→"}
TITLE = {"focus": "Bring to front", "close": "Close", "min": "Minimize", "max": "Maximize / restore",
         "left": "Move left", "up": "Move up", "down": "Move down", "right": "Move right"}


def u(v):
    return f"{v:g}cqw"


def geo_css(g, dot, glyph):
    """The rules that place the controls inside a window, for one layout of wm.LAYOUTS."""
    TB, BTN, BS, B0 = g["TB"], g["BTN"], g["BTN_STEP"], g["BTN0"]
    ARW, AS, AP = g["ARW"], g["ARW_STEP"], g["ARW_PAD"]
    GT, GP, GG = g["GRID_TOP"], g["GRID_PAD"], g["GRID_GAP"]
    bx = g["BACK_BOX"]
    box = f"left: {u(GP)}; right: {u(GP)}; top: {u(GT - TB)}; bottom: {u(GP)};"
    return "\n".join([
        f".win {{ grid-template-rows: {u(TB)} 1fr; }}",
        f".tb .title {{ left: {u(B0 + 2 * BS + BTN / 2 + 1)}; right: {u(AP + 4 * AS)}; }}",
        f".bclose, .bmin, .bmax {{ top: {u((TB - BTN) / 2)}; width: {u(BTN)}; height: {u(BTN)}; }}",
        " ".join(f".{c} {{ left: {u(B0 - BTN / 2 + i * BS)}; }}" for i, c in enumerate(("bclose", "bmin", "bmax"))),
        f".bclose i, .bmin i, .bmax i {{ width: {u(dot)}; height: {u(dot)}; }}",
        f".arw {{ top: {u((TB - ARW) / 2)}; width: {u(ARW)}; height: {u(ARW)}; font-size: {glyph}; }}",
        " ".join(f".arw.a{a[0].upper()} {{ right: {u(AP + (3 - i) * AS)}; }}" for i, a in enumerate(wm.ARROWS)),
        f".notes, .grid {{ {box} }}",
        f".grid {{ gap: {u(GG)}; }}",
        f".viewer {{ left: {u(GP)}; right: {u(GP)}; top: {u(bx[1] + bx[3] + 1.2 - TB)}; bottom: {u(GP)}; }}",
        f".back {{ left: {u(bx[0])}; top: {u(bx[1] - TB)}; width: {u(bx[2])}; height: {u(bx[3])}; }}",
    ])


def labels(ev):
    return "".join(f'<label for="e{t}_{ev}" class="ev s{t}"></label>' for t in range(wm.SLOTS))


def window_html(w):
    name = wm.WINS[w]
    ev = lambda a: wm.ev_window(w, a)  # noqa: E731
    btns = "".join(f'<span class="ctl b{a}" title="{TITLE[a]}"><i></i>{labels(ev(a))}</span>' for a in ("close", "min", "max"))
    arrows = "".join(f'<span class="ctl arw a{a[0].upper()}" title="{TITLE[a]}">'
                     f'<i style="opacity: calc(0.3 + 0.7 * var(--a{a[0].upper()}{w}))">{GLYPH[a]}</i>{labels(ev(a))}</span>'
                     for a in wm.ARROWS)
    bar = (f'<div class="tb"><span class="ctl focus" title="{TITLE["focus"]}">{labels(ev("focus"))}</span>'
           f'{btns}<b class="title">{name}</b>{arrows}</div>')
    if w == wm.NOTES:
        body = ('<textarea class="notes" form="notes" spellcheck="false" aria-label="Notes">'
                'Type anything here.\n\nThis text is not part of the network: the browser keeps it, so it stays '
                'while you move, minimize or maximize the window.</textarea>')
    elif w == wm.PHOTOS:
        thumbs = "".join(f'<span class="ctl thumb" title="{n}" style="background:{g}">{labels(wm.PHOTO0 + j)}</span>'
                         for j, (n, g) in enumerate(PHOTOS))
        big = "".join(f'<div class="big" style="background:{g}; opacity: var(--g{j + 1})"><span>{n}</span></div>'
                      for j, (n, g) in enumerate(PHOTOS))
        body = (f'<div class="grid">{thumbs}</div><div class="viewer">{big}</div>'
                f'<span class="ctl back" title="Back to all photos"><i>← All</i>{labels(wm.BACK)}</span>')
    else:
        body = MONITOR_BODY
    bodyf = f'<span class="ctl bodyf" title="{TITLE["focus"]}">{labels(ev("focus"))}</span>'
    return f'<div class="win" id="w{w}">{bar}<div class="body">{bodyf}{body}</div></div>'


MONITOR_BODY = ""  # filled in build(), it quotes the nets' sizes


def monitor_body(net, scr):
    rows = []
    for w, name in enumerate(wm.WINS):
        states = (f'<span style="opacity: var(--cl{w})">closed</span>'
                  f'<span style="opacity: var(--mn{w})">in the dock</span>'
                  f'<span style="opacity: var(--zm{w})">full screen</span>'
                  f'<span style="opacity: calc(var(--vis{w}) - var(--zm{w}))">open</span>')
        rows.append(f'<div class="mrow" style="order: calc(2 - var(--r{w}))"><b>{name}</b><span class="st">{states}</span>'
                    f'<span class="num" style="--a: var(--x{w}); --b: var(--y{w})"></span></div>')
    return (f'<div class="mon">'
            f'<div class="kv"><span>clicks used</span><b class="num1" style="--a: var(--E)"></b></div>'
            f'<div class="kv"><span>clicks left</span><b class="num1" style="--a: var(--evleft)"></b></div>'
            f'<div class="kv tinyhide"><span>photo shown</span><b class="num1 ph" style="--a: var(--ph)"></b></div>'
            f'<div class="mh"><span class="wideonly">windows, </span>top first <span>(x, y)</span></div><div class="stack">{"".join(rows)}</div>'
            f'<div class="mh wideonly">the network</div>'
            f'<div class="kv"><span>network</span><b><span class="wideonly">{net.depth} layers, </span>{fmt(net.n_units)} neurons</b></div>'
            f'<div class="kv wideonly"><span>weights</span><b>{fmt(net.n_weights)}</b></div>'
            f'<div class="kv wideonly"><span>run</span><b>{wm.SLOTS} times, once per click</b></div>'
            f'</div>')


def build():
    global MONITOR_BODY
    nets = json.loads((NETS / "os.json").read_text())
    net, scr = Net.from_json(nets["event"]), Net.from_json(nets["screen"])
    MONITOR_BODY = monitor_body(net, scr)
    css = Stylesheet()
    SL, ST, U = ".slot", ".step", ".ui"

    # one event: copy the state, read the slot's radio, run the net
    for k in wm.STATE:
        css.define(SL, f"P{k}", var(k))
    css.define(SL, "ev", "0")
    for k in range(1, wm.N_EV + 1):
        css.rule(f".slot:has(> .a{k}:checked)", f"--ev: {k};")
    css.net(ST, net, [f"P{k}" for k in wm.STATE] + ["ev"], "c", net.out_names)
    css.rule(".os", " ".join(f"--{k}: {v};" for k, v in wm.START.items()))
    # the final state, drawn
    css.net(U, scr, list(wm.STATE), "s", scr.out_names)

    turn_rules = [".os label.ev { display: none; }",
                  ".os:not(:has([name=t0]:checked)) label.s0 { display: block; }"]
    for t in range(1, wm.SLOTS):
        turn_rules.append(f".os:has([name=t{t - 1}]:checked):not(:has([name=t{t}]:checked)) label.s{t} {{ display: block; }}")

    opening = "\n".join(
        '<div class="slot">' + "".join(f'<input type="radio" name="t{t}" id="e{t}_{k}" class="a{k}" tabindex="-1">'
                                       for k in range(1, wm.N_EV + 1)) + '<div class="step">' for t in range(wm.SLOTS))
    closing = "</div></div>" * wm.SLOTS
    dock = "".join(f'<span class="ctl dk" id="dock{w}" title="{n}"><i class="ic ic{w}"></i><small>{n}</small>'
                   f'<em style="opacity: calc(1 - var(--cl{w})); --mn: var(--mn{w})"></em>{labels(wm.DOCK0 + w)}</span>'
                   for w, n in enumerate(wm.WINS))
    windows = "".join(window_html(w) for w in range(3))
    n_labels = (windows + dock).count("<label ")

    body = f"""
<header>
  <h1>A tiny desktop run by one network</h1>
  <p class="lead">Three windows and a dock. Every click goes through one neural network that decides what happens to every window, in CSS, with 0 bytes of JavaScript.</p>
</header>
<div class="explain"><ul>
  <li><b>The window manager:</b> which windows are open, minimized or full screen, where each one sits, and which is on top. That is 20 numbers. Each click hands them, plus the click, to one network of {fmt(net.n_units)} neurons, which returns the 20 new numbers.</li>
  <li><b>One click, one run:</b> the page chains that same network {wm.SLOTS} times, once per click, so you get {wm.SLOTS} clicks. Reset starts over. Buttons that would change nothing, like an arrow at the edge of the desktop or the window that is already in front, are switched off, so they don't use up a click.</li>
  <li><b>The motion is plain CSS:</b> the network decides where a window should be; the browser only animates it there.</li>
  <li><b>Notes is a real text box:</b> what you type is kept by the browser, not by the network, so it stays while windows move.</li>
  <li><b>Try:</b> click a window that is behind another to bring it forward, nudge it with the arrows, send it to the dock and back, open a photo, and watch the numbers in Monitor.</li>
</ul></div>
<form id="os" class="os">
{opening}
<div class="ui">
  <div class="topbar">
    <span class="left">clicks left <b class="num1" style="--a: var(--evleft)"></b></span>
    <span class="out">Out of clicks: press Reset.</span>
    <button type="reset" form="os">Reset</button>
  </div>
  <div class="screen">
    <div class="desk"><div class="wall"><span>All windows are away. Open one from the dock.</span></div>{windows}</div>
    <div class="dock">{dock}</div>
  </div>
</div>
{closing}
</form>
<form id="notes"></form>
<details class="deep"><summary>How it works (for the curious)</summary>
<section>
  <h2>Where this comes from</h2>
  <p>A reduced, CSS edition of Cheng Lou's <a href="https://chenglou.me/handwired/">HandwiredOS</a>, a desktop whose windows, dock and apps are run by one compiled program. CSS can't read where the pointer is, which key you pressed or what time it is, so this edition is driven by clicks only: every click is one event. You can't drag a window; the arrows move it one step on a coarse grid. Typing goes to a normal text box the network never sees.</p>
</section>
<section>
  <h2>One click, as ReLUs</h2>
  <div class="scroll"><table>
    <tr><th>Part</th><th>How</th></tr>
    <tr><td>which control</td><td>{wm.N_EV} controls, the click comes in as one number <code>ev</code>; <code>[ev = k]</code> for each k, three ReLUs each</td></tr>
    <tr><td>only on screen</td><td>a window's buttons work when it is open and not minimized: <code>relu([ev = k] + open − minimized − 1)</code></td></tr>
    <tr><td>switched off</td><td>the screen network also says which arrows can still move (or restore a full-screen window) and which window is in front; the page shrinks those buttons' labels to nothing, so a click there does nothing and costs nothing</td></tr>
    <tr><td>to the top</td><td>the window gets rank 2; every window above it moves down one: <code>rank − AND(top_w, [rank &gt; rank_w])</code></td></tr>
    <tr><td>to the bottom</td><td>close and minimize: rank 0, everyone below moves up one</td></tr>
    <tr><td>move</td><td><code>x + AND(right, [x &lt; max]) − AND(left, [x ≥ 1])</code>: clamped to the desktop</td></tr>
    <tr><td>maximize</td><td>a toggle, <code>mx + AND(max, 1 − mx) − AND(max, mx)</code>; a move or a close also restores it</td></tr>
    <tr><td>photos</td><td>a thumbnail opens its photo while the grid shows; back returns to the grid; closing Photos too</td></tr>
  </table></div>
  {anatomy_html(net).replace('<span><i style="background:var(--accent)"></i>learned</span>', '')}
  <p>A second, smaller network ({scr.depth} layers, {fmt(scr.n_units)} neurons) turns the final 20 numbers into what you see: each window's rectangle (full screen when maximized), whether it is on screen, which one is focused, which photo shows, the arrows that can still move. CSS then only places boxes at those numbers, with a transition.</p>
</section>
<section>
  <h2>The state</h2>
  <div class="scroll"><table>
    <tr><th>Per window</th><th>Values</th></tr>
    <tr><td>open, minimized, maximized</td><td>0 or 1</td></tr>
    <tr><td>x, y</td><td>grid cells: Notes 0…{wm.XMAX[0]} × 0…{wm.YMAX[0]}, Photos 0…{wm.XMAX[1]} × 0…{wm.YMAX[1]}, Monitor 0…{wm.XMAX[2]} × 0…{wm.YMAX[2]}</td></tr>
    <tr><td>rank</td><td>0, 1, 2: the stacking order, a permutation; 2 is on top</td></tr>
    <tr><td>and once</td><td>the photo shown (0 is the grid) and the clicks used</td></tr>
  </table></div>
  <p>Each of the {wm.SLOTS} slots is its own group of {wm.N_EV} radio buttons, and every control on screen holds {wm.SLOTS} labels, one per slot, of which only the next slot's is shown. A window's title bar and its inside are two labels for the same click. That's {fmt(n_labels)} labels in all.</p>
</section>
<footer>Checked in numpy against a plain Python window manager: every event in every combination of window flags, stacking order and photo (285,000 cases), every event at every grid position, and 4,000 random sequences of 60 events, every value after every event. The browser test clicks {len(wm.SCRIPTS)} scripted and 6 random 40-click sequences in Chromium, Firefox and WebKit, on a wide screen and at phone width, compares the whole state after every click, and checks that switched-off buttons and the keyboard change nothing. Built by <code>handwired/scripts/mini_os.py</code> and <code>build_os.py</code>.</footer>
</details>
"""
    # presentation: places boxes at the numbers the screen net gives
    win_rules = []
    for w in range(3):
        win_rules.append(
            f"#w{w} {{ left: calc(var(--L{w}) * 1cqw); top: calc(var(--T{w}) * 1cqw); width: calc(var(--W{w}) * 1cqw); "
            f"height: calc(var(--H{w}) * 1cqw); z-index: var(--r{w}); scale: var(--vis{w}); --top: var(--top{w}); "
            f"transform-origin: {28 + 22 * w}cqw 120%; }}")
    # which labels are switched off: the arrows that can't move, and the title bar, body and dock icon of
    # the window already in front. scale(0) takes a label out of hit testing; the control's span (with
    # pointer-events) then swallows the click, so nothing happens and no click is used.
    for w in range(3):
        win_rules.append(f"#w{w} .focus label.ev, #w{w} .bodyf label.ev, #dock{w} label.ev {{ scale: calc(1 - var(--top{w})); }}")
        win_rules.append(" ".join(f"#w{w} .arw.a{a} label.ev {{ scale: var(--a{a}{w}); }}" for a in "LUDR"))
    styles = f"""
.os, .slot, .step, .ui {{ display: contents; }}
.slot > input {{ display: none; }}  /* a radio that can't take focus: no arrow key can change a past click */
.topbar {{ display: grid; grid-template-columns: 1fr auto; align-items: center; gap: 12px; font-size: 13px; color: var(--muted); }}
.topbar .left, .topbar .out {{ grid-area: 1 / 1; transition: opacity .2s; }}
.topbar .left {{ opacity: calc(1 - var(--done)); }}
.topbar b {{ color: var(--fg); }}
.topbar .out {{ color: var(--accent); opacity: var(--done); }}
.num1::after {{ counter-reset: a var(--a); content: counter(a); }}
.screen {{ display: grid; gap: 0; border-radius: 12px; overflow: hidden; border: 1px solid var(--line); margin-top: -22px; }}
.desk {{ position: relative; container: desk / inline-size; aspect-ratio: {wm.DW} / {wm.DH}; overflow: hidden; isolation: isolate;
  background: radial-gradient(ellipse at 20% 0%, color-mix(in oklab, var(--hand), transparent 70%), transparent 60%),
    radial-gradient(ellipse at 100% 100%, color-mix(in oklab, var(--accent), transparent 78%), transparent 55%), var(--cell);
  -webkit-tap-highlight-color: transparent; font-size: max(9px, 2.1cqw); }}
.wall {{ position: absolute; inset: 0; display: grid; place-items: center; color: var(--muted); font-size: max(11px, 2.4cqw);
  opacity: calc(1 - var(--nvis)); transition: opacity .3s; pointer-events: none; }}
.win {{ position: absolute; display: grid; border-radius: 1.2cqw; overflow: hidden;
  background: var(--bg);
  box-shadow: 0 0 0 1px color-mix(in oklab, var(--line), var(--fg) calc(var(--top) * 25%)), 0 0.6cqw 2.4cqw rgb(0 0 0 / calc(0.12 + 0.18 * var(--top)));
  transition: left .35s cubic-bezier(.2,.8,.2,1), top .35s cubic-bezier(.2,.8,.2,1), width .35s cubic-bezier(.2,.8,.2,1),
    height .35s cubic-bezier(.2,.8,.2,1), scale .3s cubic-bezier(.3,.7,.2,1), box-shadow .3s; }}
{chr(10).join(win_rules)}
@media (prefers-reduced-motion: reduce) {{ .win, .grid, .back, .big, .wall {{ transition: none !important; }} }}
.win * {{ pointer-events: none; }}
.win .ctl, .win label.ev, .win textarea {{ pointer-events: auto; }}
.tb {{ position: relative; background: color-mix(in oklab, var(--cell), var(--hand) calc(var(--top) * 22%)); border-bottom: 1px solid var(--line);
  transition: background-color .3s; }}
.tb .title {{ position: absolute; top: 0; bottom: 0; display: grid; place-items: center; font-weight: 600; font-size: max(9px, 1.9cqw);
  white-space: nowrap; overflow: hidden; }}
.ctl {{ position: absolute; display: grid; place-items: center; }}
.ctl label.ev {{ position: absolute; inset: 0; cursor: pointer; }}
.ctl.focus, .ctl.bodyf {{ inset: 0; }}
.bclose i, .bmin i, .bmax i {{ border-radius: 50%; box-shadow: inset 0 0 0 1px rgb(0 0 0 / .15); }}
.bclose i {{ background: #ff5f57; }} .bmin i {{ background: #febc2e; }} .bmax i {{ background: #28c840; }}
.arw {{ border-radius: 0.8cqw; font-style: normal; }}
.arw i {{ font-style: normal; line-height: 1; }}
@media (hover: hover) {{ .ctl.arw:hover, .ctl.back:hover {{ background: color-mix(in oklab, transparent, var(--fg) 10%); }}
  .ctl.thumb:hover {{ outline: 0.4cqw solid var(--hand); }} .ctl.dk:hover {{ background: color-mix(in oklab, transparent, var(--fg) 7%); }} }}
.body {{ position: relative; }}
.notes {{ position: absolute; resize: none; border: 1px solid var(--line); border-radius: 0.8cqw; background: var(--bg); color: var(--fg);
  font: inherit; font-size: max(12px, 1.7cqw); line-height: 1.45; padding: 1.2cqw; outline: none; }}
.notes:focus {{ border-color: var(--hand); }}
.grid {{ position: absolute; display: grid; grid-template-columns: repeat(3, 1fr); grid-template-rows: repeat(3, 1fr);
  scale: var(--g0); opacity: var(--g0); transition: scale .25s, opacity .25s; }}
.thumb {{ position: relative; border-radius: 0.8cqw; }}
.viewer {{ position: absolute; }}
.big {{ position: absolute; inset: 0; border-radius: 1cqw; transition: opacity .3s; display: grid; align-items: end; }}
.big span {{ color: #fff; text-shadow: 0 1px 3px rgb(0 0 0 / .6); padding: 1cqw 1.4cqw; font-weight: 600; }}
.back {{ border-radius: 0.8cqw; border: 1px solid var(--line); scale: calc(1 - var(--g0)); transition: scale .2s; }}
.back i {{ font-style: normal; }}
.mon {{ position: absolute; inset: 1.4cqw 2cqw; display: grid; align-content: start; gap: 0.4cqw; font-size: max(8px, 1.55cqw); line-height: 1.35; overflow: hidden; }}
.mon .kv {{ display: flex; justify-content: space-between; gap: 2cqw; border-bottom: 1px solid var(--line); padding-bottom: 0.3cqw; }}
.mon .kv span, .mon .mh span {{ color: var(--muted); }}
.mon .kv b {{ font-weight: 600; color: var(--hand); }}
.mon .kv b span {{ color: inherit; }}
.mon .mh {{ margin-top: 0.8cqw; font-weight: 600; }}
.mon .stack {{ display: flex; flex-direction: column; gap: 0.3cqw; }}
.mon .mrow {{ display: grid; grid-template-columns: 1fr 1.2fr auto; gap: 1cqw; }}
.mon .st {{ display: grid; color: var(--muted); }}
.mon .st span {{ grid-area: 1 / 1; }}
.mon .num::after {{ counter-reset: a var(--a) b var(--b); content: counter(a) ", " counter(b); color: var(--hand); }}
.dock {{ display: flex; justify-content: center; gap: 6px; padding: 8px; background: color-mix(in oklab, var(--cell), var(--fg) 6%); border-top: 1px solid var(--line); }}
.dk {{ position: relative; display: grid; justify-items: center; gap: 2px; padding: 4px 10px 6px; border-radius: 8px; min-width: 64px; }}
.dk small {{ font-size: 11px; color: var(--muted); }}
.dk em {{ width: 5px; height: 5px; border-radius: 50%; background: color-mix(in oklab, var(--hand), var(--muted) calc(var(--mn) * 100%)); transition: opacity .3s; }}
.ic {{ width: 34px; height: 34px; border-radius: 9px; box-shadow: inset 0 0 0 1px rgb(0 0 0 / .12); }}
.ic0 {{ background: linear-gradient(180deg, #ffd65a 0 28%, #fff7d6 28%); }}
.ic1 {{ background: conic-gradient(from 0deg, #ff5f57, #febc2e, #28c840, #2f9bff, #a35bff, #ff5f57); }}
.ic2 {{ background: linear-gradient(180deg, #1c2733, #0d1218); position: relative; }}
.ic2::after {{ content: ""; position: absolute; inset: 9px 6px; background: linear-gradient(90deg, #28c840 0 22%, transparent 22% 39%, #28c840 39% 61%, transparent 61% 78%, #28c840 78%); clip-path: polygon(0 100%, 0 60%, 22% 60%, 22% 20%, 39% 20%, 39% 100%, 61% 100%, 61% 0, 78% 0, 78% 45%, 100% 45%, 100% 100%); }}
.dk label.ev {{ position: absolute; inset: 0; cursor: pointer; }}
.explain a, section a {{ color: var(--fg); }}
/* where the controls sit: the "wide" layout, and the "phone" one for a desktop under {wm.PHONE_PX}px wide
   (bigger buttons, a taller title bar, no title text); scripts/mini_os.py models both for the test */
{geo_css(wm.LAYOUTS["wide"], dot=2.4, glyph="max(9px, 2.3cqw)")}
@container desk (max-width: {wm.PHONE_PX - 0.02}px) {{
{geo_css(wm.LAYOUTS["phone"], dot=3.6, glyph="15px")}
  .tb .title, .mon .wideonly {{ display: none; }}
  .notes {{ font-size: 16px; padding: 1.6cqw; }}  /* 16px: iOS zooms into a smaller text box */
  .mon {{ font-size: 10px; line-height: 1.15; gap: 0.3cqw; inset: 1.2cqw 2cqw; }}
  .mon .kv {{ padding-bottom: 0; }}
  .mon .mh {{ margin-top: 0.4cqw; }}
  .mon .mrow {{ grid-template-columns: auto 1fr auto; white-space: nowrap; }}
  .big span {{ font-size: 10px; }}
}}
@container desk (max-width: 300px) {{ .mon .tinyhide {{ display: none; }} }}
"""
    kb = page("os", "Mini OS",
              "A desktop with windows, a dock and photos in pure CSS, whose window manager is one ReLU network per click.",
              css.render() + "\n" + "\n".join(turn_rules), styles, body)
    print(f"os/index.html: {kb:.0f} KB, {len(css.names):,} properties")


if __name__ == "__main__":
    build()
