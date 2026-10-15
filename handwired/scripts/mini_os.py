"""Mini OS: a desktop whose window manager, for one click, is one ReLU net.

    .venv/bin/python handwired/scripts/mini_os.py

A reduced, click-driven port of Cheng Lou's HandwiredOS (https://chenglou.me/handwired/). CSS can't
read pointer coordinates, keys or time, so every click is one event, and the page applies the event
net once per click (40 nested event slots, like the connect-four page).

The state, for each window w (0 Notes, 1 Photos, 2 Monitor):
    o{w}   open (not closed)          x{w}, y{w}   position on a coarse grid
    mn{w}  minimized to the dock      r{w}         rank in the stacking order, 2 = on top
    mx{w}  maximized
plus ph (the photo open in Photos, 0 = the grid) and E (events used so far).

The events (`ev`, 0 = no click yet):
    1 + 8w + a   window w's control a: focus (its title bar), close, minimize, maximize, up, down, left, right
    25 + w       the dock icon of window w
    27 + j       photo j (1..9) in the grid
    37           back to the grid

The net is total: it does what the plain `step` below does for every event in every state, including
clicks the page never offers (minimizing a closed window does nothing). A second, small net turns the
final state into what the screen shows: rectangles, visibility, which photo is shown.

Checked against `step` in numpy: every event in every combination of window flags, stacking order and
photo (with random positions), and thousands of random 60-event sequences. Writes handwired/nets/os.json
and the browser fixtures handwired/tests/os.json.
"""
import json
from itertools import permutations, product

import numpy as np

from common import NETS, ROOT
from hw import LIMIT, Prog, total

WINS = ["Notes", "Photos", "Monitor"]
NOTES, PHOTOS, MONITOR = 0, 1, 2
ACTS = ["focus", "close", "min", "max", "up", "down", "left", "right"]
N_PHOTOS = 9
SLOTS = 40
DOCK0 = 1 + 8 * len(WINS)          # 25
PHOTO0 = DOCK0 + len(WINS)         # 28 is photo 1
BACK = PHOTO0 + N_PHOTOS           # 37
N_EV = BACK                        # events are 0..37
STATE = [f"{k}{w}" for w in range(3) for k in ("o", "mn", "mx", "x", "y", "r")] + ["ph", "E"]

# ---- geometry, in desktop units: the desktop is 100 wide and 70 tall (1 unit = 1cqw on the page)
DW, DH = 100, 70
MARGIN, SX, SY = 1, 6, 5           # window left = MARGIN + SX * x, top = MARGIN + SY * y
SIZE = [(52, 40), (50, 46), (46, 42)]
XMAX = [(DW - 2 * MARGIN - w) // SX for w, _ in SIZE]
YMAX = [(DH - 2 * MARGIN - h) // SY for _, h in SIZE]
ARROWS = ["left", "up", "down", "right"]   # left to right
# Where the controls sit inside a window, in desktop units. Two layouts: "wide", and "phone" for a
# desktop narrower than PHONE_PX (a CSS container query), with a taller title bar, bigger buttons and
# no title text, so the buttons stay tappable. Window rectangles are the same in both.
PHONE_PX = 640
LAYOUTS = {
    "wide": dict(TB=5, BTN=3.4, BTN_STEP=3.6, BTN0=2.6,      # close / min / max boxes, first centre
                 ARW=3.6, ARW_STEP=3.9, ARW_PAD=1.2,         # the four arrows at the right
                 GRID_TOP=6.5, GRID_PAD=2, GRID_GAP=1.2, BACK_BOX=(2, 6.2, 10, 3.6)),  # x, y, w, h
    "phone": dict(TB=7, BTN=5.8, BTN_STEP=6, BTN0=3.9,
                  ARW=5.8, ARW_STEP=6, ARW_PAD=1,
                  GRID_TOP=8.5, GRID_PAD=2, GRID_GAP=1.5, BACK_BOX=(2, 8.2, 17, 6.4)),
}
TB = LAYOUTS["wide"]["TB"]

START = {"o0": 1, "mn0": 0, "mx0": 0, "x0": 0, "y0": 4, "r0": 2,
         "o1": 1, "mn1": 0, "mx1": 0, "x1": 5, "y1": 2, "r1": 1,
         "o2": 1, "mn2": 0, "mx2": 0, "x2": 8, "y2": 0, "r2": 0,
         "ph": 0, "E": 0}


def ev_window(w, act):
    return 1 + 8 * w + ACTS.index(act)


def describe(ev):
    if ev == 0:
        return "nothing"
    if ev < DOCK0:
        w, a = divmod(ev - 1, 8)
        return f"{WINS[w]}: {ACTS[a]}"
    if ev < PHOTO0:
        return f"dock: {WINS[ev - DOCK0]}"
    if ev < BACK:
        return f"photo {ev - PHOTO0 + 1}"
    return "back to the grid"


# ---------------------------------------------------------------- the reference window manager

def visible(s, w):
    return s[f"o{w}"] == 1 and s[f"mn{w}"] == 0


def to_top(s, w):
    for v in range(3):
        if s[f"r{v}"] > s[f"r{w}"]:
            s[f"r{v}"] -= 1
    s[f"r{w}"] = 2


def to_bottom(s, w):
    for v in range(3):
        if s[f"r{v}"] < s[f"r{w}"]:
            s[f"r{v}"] += 1
    s[f"r{w}"] = 0


def step(s, ev):
    """One click. Plain code: this is what the net must do."""
    s = dict(s)
    if ev == 0:
        return s
    s["E"] += 1
    if ev < DOCK0:
        w, a = divmod(ev - 1, 8)
        act = ACTS[a]
        if not visible(s, w):
            return s
        if act == "focus":
            to_top(s, w)
        elif act == "close":
            s[f"o{w}"], s[f"mx{w}"] = 0, 0
            if w == PHOTOS:
                s["ph"] = 0
            to_bottom(s, w)
        elif act == "min":
            s[f"mn{w}"] = 1
            to_bottom(s, w)
        elif act == "max":
            s[f"mx{w}"] = 1 - s[f"mx{w}"]
            to_top(s, w)
        else:  # a move: un-maximizes, steps one grid cell, stays on the desktop
            dx = {"left": -1, "right": 1}.get(act, 0)
            dy = {"up": -1, "down": 1}.get(act, 0)
            s[f"mx{w}"] = 0
            s[f"x{w}"] = min(max(s[f"x{w}"] + dx, 0), XMAX[w])
            s[f"y{w}"] = min(max(s[f"y{w}"] + dy, 0), YMAX[w])
            to_top(s, w)
    elif ev < PHOTO0:
        w = ev - DOCK0
        s[f"o{w}"], s[f"mn{w}"] = 1, 0
        to_top(s, w)
    elif ev < BACK:
        if visible(s, PHOTOS) and s["ph"] == 0:
            s["ph"] = ev - PHOTO0 + 1
            to_top(s, PHOTOS)
    elif ev == BACK:
        if visible(s, PHOTOS) and s["ph"] > 0:
            s["ph"] = 0
            to_top(s, PHOTOS)
    return s


def screen(s):
    """What the screen shows for a state, in the same plain style: the reference for the screen net."""
    out = {}
    for w in range(3):
        vis = int(visible(s, w))
        mx = s[f"mx{w}"]
        out[f"vis{w}"] = vis
        out[f"L{w}"] = 0 if mx else MARGIN + SX * s[f"x{w}"]
        out[f"T{w}"] = 0 if mx else MARGIN + SY * s[f"y{w}"]
        out[f"W{w}"] = DW if mx else SIZE[w][0]
        out[f"H{w}"] = DH if mx else SIZE[w][1]
        out[f"top{w}"] = int(vis and s[f"r{w}"] == 2)
        out[f"cl{w}"] = 1 - s[f"o{w}"]
        out[f"zm{w}"] = int(vis and mx == 1)
        # an arrow does something when the window can move that way, or when it is maximized (it restores)
        out[f"aL{w}"] = int(s[f"x{w}"] > 0 or mx == 1)
        out[f"aR{w}"] = int(s[f"x{w}"] < XMAX[w] or mx == 1)
        out[f"aU{w}"] = int(s[f"y{w}"] > 0 or mx == 1)
        out[f"aD{w}"] = int(s[f"y{w}"] < YMAX[w] or mx == 1)
    for k in range(N_PHOTOS + 1):
        out[f"g{k}"] = int(s["ph"] == k)
    out["evleft"] = SLOTS - s["E"]
    out["done"] = int(s["E"] >= SLOTS)
    out["nvis"] = sum(out[f"vis{w}"] for w in range(3))
    return out


# ---------------------------------------------------------------- the nets

def os_net():
    """Inputs: the state, then ev. Outputs: the next state."""
    p = Prog()
    st = {k: p.input(k) for k in STATE}
    ev = p.input("ev")
    hit = [None] + [p.eq(ev, k) for k in range(1, N_EV + 1)]  # one-hot of the click
    o = [st[f"o{w}"] for w in range(3)]
    mn = [st[f"mn{w}"] for w in range(3)]
    mx = [st[f"mx{w}"] for w in range(3)]
    x = [st[f"x{w}"] for w in range(3)]
    y = [st[f"y{w}"] for w in range(3)]
    r = [st[f"r{w}"] for w in range(3)]
    ph = st["ph"]

    vis = [p.AND(o[w], 1 - mn[w]) for w in range(3)]
    # a window's own controls only work while it is on screen
    act = [{a: p.AND(hit[ev_window(w, a)], vis[w]) for a in ACTS} for w in range(3)]
    dock = [hit[DOCK0 + w] for w in range(3)]
    grid = p.eq(ph, 0)
    pick = [p.AND(hit[PHOTO0 + j], vis[PHOTOS], grid) for j in range(N_PHOTOS)]
    back = p.AND(hit[BACK], vis[PHOTOS], 1 - grid)

    moves = [total(act[w][a] for a in ("up", "down", "left", "right")) for w in range(3)]
    top = [act[w]["focus"] + act[w]["max"] + moves[w] + dock[w] for w in range(3)]
    top[PHOTOS] = top[PHOTOS] + total(pick) + back
    bottom = [act[w]["close"] + act[w]["min"] for w in range(3)]

    out = {}
    for w in range(3):
        out[f"o{w}"] = o[w] - act[w]["close"] + p.AND(dock[w], 1 - o[w])
        out[f"mn{w}"] = mn[w] + act[w]["min"] - p.AND(dock[w], mn[w])
        out[f"mx{w}"] = mx[w] + p.AND(act[w]["max"], 1 - mx[w]) - p.AND(act[w]["max"] + act[w]["close"] + moves[w], mx[w])
        out[f"x{w}"] = x[w] + p.AND(act[w]["right"], 1 - p.ge(x[w], XMAX[w])) - p.AND(act[w]["left"], p.ge(x[w], 1))
        out[f"y{w}"] = y[w] + p.AND(act[w]["down"], 1 - p.ge(y[w], YMAX[w])) - p.AND(act[w]["up"], p.ge(y[w], 1))
    # the stacking order: to the top, everyone above moves down one; to the bottom, everyone below moves up
    above = {(v, w): p.ge(r[v] - r[w], 1) for v in range(3) for w in range(3) if v != w}  # r_v > r_w
    for v in range(3):
        shift = total(-p.AND(top[w], above[(v, w)]) + p.AND(bottom[w], above[(w, v)]) for w in range(3) if w != v)
        out[f"r{v}"] = r[v] + shift + p.gate(top[v], 2 - r[v], 2) - p.gate(bottom[v], r[v], 2)
    out["ph"] = ph + total((j + 1) * pick[j] for j in range(N_PHOTOS)) - p.gate(back + act[PHOTOS]["close"], ph, N_PHOTOS)
    out["E"] = st["E"] + p.ge(ev, 1)
    return p.compile({k: out[k] for k in STATE})


def screen_net():
    """Inputs: the state. Outputs: what `screen` returns."""
    p = Prog()
    st = {k: p.input(k) for k in STATE}
    out = {}
    for w in range(3):
        o, mn, mx, x, y, r = (st[f"{k}{w}"] for k in ("o", "mn", "mx", "x", "y", "r"))
        vis = p.AND(o, 1 - mn)
        out[f"vis{w}"] = vis
        out[f"L{w}"] = p.gate(1 - mx, MARGIN + SX * x, MARGIN + SX * XMAX[w])
        out[f"T{w}"] = p.gate(1 - mx, MARGIN + SY * y, MARGIN + SY * YMAX[w])
        out[f"W{w}"] = SIZE[w][0] + (DW - SIZE[w][0]) * mx
        out[f"H{w}"] = SIZE[w][1] + (DH - SIZE[w][1]) * mx
        out[f"top{w}"] = p.AND(vis, p.ge(r, 2))
        out[f"cl{w}"] = 1 - o
        out[f"zm{w}"] = p.AND(vis, mx)
        out[f"aL{w}"] = p.OR(p.ge(x, 1), mx)
        out[f"aR{w}"] = p.OR(1 - p.ge(x, XMAX[w]), mx)
        out[f"aU{w}"] = p.OR(p.ge(y, 1), mx)
        out[f"aD{w}"] = p.OR(1 - p.ge(y, YMAX[w]), mx)
    for k in range(N_PHOTOS + 1):
        out[f"g{k}"] = p.eq(st["ph"], k)
    out["evleft"] = SLOTS - st["E"]
    out["done"] = p.ge(st["E"], SLOTS)
    out["nvis"] = out["vis0"] + out["vis1"] + out["vis2"]
    return p.compile(out)


SCREEN = list(screen(START))


# ---------------------------------------------------------------- what can be clicked

# A click is (ev, where): `where` tells two labels of the same event apart ("bar" or "body" for focus,
# "" for the rest, "dock" for the dock). The page offers only controls that change something: an arrow
# that can't move (screen aX = 0), and the title bar, body and dock icon of the window already in
# front, are switched off.

def rect(s, w):
    sc = screen(s)
    return sc[f"L{w}"], sc[f"T{w}"], sc[f"W{w}"], sc[f"H{w}"]


def control_points(s, w, g):
    """{(ev, where): [(x, y), ...]} in desktop units, layout g: where the page puts each control of
    window w, a few candidate points each (the first one not under another window is used)."""
    L, T, W, H = rect(s, w)
    TBg = g["TB"]
    pts = {}
    # the title bar between the buttons and the arrows
    lo = g["BTN0"] + 2 * g["BTN_STEP"] + g["BTN"] / 2 + 0.6
    hi = W - g["ARW_PAD"] - 4 * g["ARW_STEP"] + (g["ARW_STEP"] - g["ARW"]) - 0.6
    bar = [(L + (lo + hi) / 2, T + TBg / 2)]
    if hi - lo >= 1:
        bar += [(L + lo + k * (hi - lo) / 8, T + TBg / 2) for k in range(9)]
    pts[(ev_window(w, "focus"), "bar")] = bar
    # the body, outside the text box / photo grid; anywhere in Monitor and on a big photo
    gp = g["GRID_PAD"]
    margin = [(L + W * k / 7, T + H - gp / 2) for k in range(1, 7)] + \
             [(L + gp / 2, T + TBg + (H - TBg) * k / 5) for k in range(1, 5)] + \
             [(L + W - gp / 2, T + TBg + (H - TBg) * k / 5) for k in range(1, 5)]
    if w == MONITOR or (w == PHOTOS and s["ph"] > 0):
        top = TBg + 1 if w == MONITOR else g["BACK_BOX"][1] + g["BACK_BOX"][3] + 2
        inner = [(L + W * i / 6, T + top + (H - top - 1) * k / 4) for k in range(1, 5) for i in range(1, 6)]
        body = [(L + W / 2, T + (top + H) / 2)] + inner + margin
    else:
        body = margin
    pts[(ev_window(w, "focus"), "body")] = body
    for i, a in enumerate(["close", "min", "max"]):
        pts[(ev_window(w, a), "")] = [(L + g["BTN0"] + i * g["BTN_STEP"], T + TBg / 2)]
    for i, a in enumerate(ARROWS):
        pts[(ev_window(w, a), "")] = [(L + W - g["ARW_PAD"] - g["ARW"] / 2 - (3 - i) * g["ARW_STEP"], T + TBg / 2)]
    if w == PHOTOS:
        if s["ph"] == 0:
            gw, gh = W - 2 * gp, H - g["GRID_TOP"] - gp
            cw, ch = (gw - 2 * g["GRID_GAP"]) / 3, (gh - 2 * g["GRID_GAP"]) / 3
            for j in range(N_PHOTOS):
                i, k = j % 3, j // 3
                pts[(PHOTO0 + j, "")] = [(L + gp + i * (cw + g["GRID_GAP"]) + cw / 2, T + g["GRID_TOP"] + k * (ch + g["GRID_GAP"]) + ch / 2)]
        else:
            bx, by, bw, bh = g["BACK_BOX"]
            pts[(BACK, "")] = [(L + bx + bw / 2, T + by + bh / 2)]
    return pts


def enabled(s, ev, where):
    """Is this control switched on? (It is when clicking it changes the state, E aside.)"""
    sc = screen(s)
    if ev == 0:
        return False
    if ev < DOCK0:
        w, a = divmod(ev - 1, 8)
        act = ACTS[a]
        if act == "focus":
            return not sc[f"top{w}"]
        if act in ARROWS:
            return bool(sc[f"a{act[0].upper()}{w}"])
        return True
    if ev < PHOTO0:
        return not sc[f"top{ev - DOCK0}"]
    return True


def covered(s, w, px, py, margin=1.0):
    """Is the point under a window stacked above window w (with a margin)?"""
    return any(L - margin <= px <= L + W + margin and T - margin <= py <= T + H + margin
               for L, T, W, H in (rect(s, v) for v in range(3) if v != w and visible(s, v) and s[f"r{v}"] > s[f"r{w}"]))


def clickable(s, g=None, margin=1.0):
    """{(ev, where): point or None} for every switched-on control a person can click now in layout g:
    on a visible window and not under a window stacked above it, or in the dock (point None)."""
    g = g or LAYOUTS["wide"]
    out = {}
    for w in range(3):
        if not visible(s, w):
            continue
        for (ev, where), pts in control_points(s, w, g).items():
            if not enabled(s, ev, where):
                continue
            for px, py in pts:
                if not covered(s, w, px, py, margin):
                    out[(ev, where)] = (round(px, 3), round(py, 3))
                    break
    for w in range(3):
        if enabled(s, DOCK0 + w, "dock"):
            out[(DOCK0 + w, "dock")] = None
    return out


def off_points(s, g, margin=1.0):
    """Points on screen where a switched-off control sits: a click there must change nothing."""
    out = []
    for w in range(3):
        if not visible(s, w):
            continue
        for (ev, where), pts in control_points(s, w, g).items():
            if ev >= DOCK0 or enabled(s, ev, where):
                continue
            for px, py in pts:
                if not covered(s, w, px, py, margin):
                    out.append((round(px, 3), round(py, 3)))
                    break
    return out


def top_window(s):
    sc = screen(s)
    return next((w for w in range(3) if sc[f"top{w}"]), None)


# ---------------------------------------------------------------- checks

def random_state(rng):
    s = {}
    flags = [(0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
    ranks = list(permutations(range(3)))[rng.integers(6)]
    for w in range(3):
        s[f"o{w}"], s[f"mn{w}"], s[f"mx{w}"] = flags[rng.integers(5)]
        s[f"x{w}"] = int(rng.choice([0, XMAX[w], rng.integers(XMAX[w] + 1)]))
        s[f"y{w}"] = int(rng.choice([0, YMAX[w], rng.integers(YMAX[w] + 1)]))
        s[f"r{w}"] = ranks[w]
    s["ph"] = int(rng.integers(N_PHOTOS + 1)) if s["o1"] else 0
    s["E"] = int(rng.integers(SLOTS))
    return s


def check_step(net, states, events):
    """Every (state, event) pair at once against `step`. Returns the max |partial sum|."""
    X = np.array([[s[k] for k in STATE] + [e] for s, e in zip(states, events)], dtype=np.int64)
    Y = net.forward(X)
    want = np.array([[step(s, e)[k] for k in STATE] for s, e in zip(states, events)], dtype=np.int64)
    bad = np.flatnonzero((Y != want).any(axis=1))
    assert not len(bad), (f"{describe(events[bad[0]])} on {states[bad[0]]}: got {dict(zip(STATE, Y[bad[0]]))}, "
                          f"want {step(states[bad[0]], events[bad[0]])}")
    return net.max_mass(X)


def check_screen(net, states):
    X = np.array([[s[k] for k in STATE] for s in states], dtype=np.int64)
    Y = net.forward(X)
    want = np.array([[screen(s)[k] for k in net.out_names] for s in states], dtype=np.int64)
    bad = np.flatnonzero((Y != want).any(axis=1))
    assert not len(bad), f"screen of {states[bad[0]]}: got {dict(zip(net.out_names, Y[bad[0]]))}"
    return net.max_mass(X)


def check(net, scr, seed=0, n_seq=4000, seq_len=60):
    rng = np.random.default_rng(seed)
    lines, mass = [], 0

    # 1. every event in every combination of flags (5 per window), stacking order (6) and photo (10),
    #    positions random but often at an edge
    flags = [(0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
    states, events = [], []
    for f in product(flags, repeat=3):
        for ranks in permutations(range(3)):
            for ph in range(N_PHOTOS + 1):
                s = random_state(rng)
                for w in range(3):
                    s[f"o{w}"], s[f"mn{w}"], s[f"mx{w}"] = f[w]
                    s[f"r{w}"] = ranks[w]
                s["ph"] = ph if s["o1"] else 0
                for e in range(N_EV + 1):
                    states.append(s)
                    events.append(e)
    mass = max(mass, check_step(net, states, events))
    mass = max(mass, check_screen(scr, states[::N_EV + 1]))
    lines.append(f"{len(states):,} (state, event) pairs: all {N_EV + 1} events x 125 flag combinations x 6 stacking orders x 10 photos")

    # 2. every event at every position of every window (the clamps), from random states
    states, events = [], []
    for w in range(3):
        for xx in range(XMAX[w] + 1):
            for yy in range(YMAX[w] + 1):
                for e in range(N_EV + 1):
                    s = random_state(rng)
                    s[f"x{w}"], s[f"y{w}"] = xx, yy
                    states.append(s)
                    events.append(e)
    mass = max(mass, check_step(net, states, events))
    mass = max(mass, check_screen(scr, states))
    lines.append(f"{len(states):,} pairs: every event at every grid position of every window")

    # 3. long random sequences from the start state, the net fed its own output, as on the page
    S = np.array([[START[k] for k in STATE]] * n_seq, dtype=np.int64)
    n_screen = 0
    ref = [dict(START) for _ in range(n_seq)]
    seen = np.zeros(N_EV + 1, dtype=np.int64)
    for t in range(seq_len):
        # mostly clickable events, some anything (including 0 and clicks on hidden controls)
        ev = []
        for i in range(n_seq):
            if rng.uniform() < 0.75:
                c = list(clickable(ref[i], LAYOUTS["wide" if i % 2 else "phone"]))
                ev.append(c[rng.integers(len(c))][0])
            else:
                ev.append(int(rng.integers(N_EV + 1)))
        ev = np.array(ev, dtype=np.int64)
        seen += np.bincount(ev, minlength=N_EV + 1)
        X = np.hstack([S, ev[:, None]])
        if t % 10 == 0:
            mass = max(mass, net.max_mass(X))
        S = net.forward(X)
        ref = [step(r, int(e)) for r, e in zip(ref, ev)]
        want = np.array([[r[k] for k in STATE] for r in ref], dtype=np.int64)
        bad = np.flatnonzero((S != want).any(axis=1))
        assert not len(bad), f"sequence {bad[0]}, event {t}: got {dict(zip(STATE, S[bad[0]]))}, want {ref[bad[0]]}"
        if t % 10 == 9:
            mass = max(mass, check_screen(scr, ref))
            n_screen += len(ref)
    assert seen.min() > 0
    lines.append(f"{n_seq:,} random sequences x {seq_len} events from the start state ({n_seq * seq_len:,} events, "
                 f"every event kind at least {seen.min():,} times): every state equals the reference after every event; "
                 f"the screen net equals `screen` on {n_screen:,} of those states")
    assert mass < LIMIT, mass
    lines.append(f"max |partial sum| {mass:,} (limit {LIMIT:,})")
    return lines


# ---------------------------------------------------------------- browser fixtures

SCRIPTS = {
    # a tour: photos, a window behind brought forward (title bar and body), moves, minimize and the dock,
    # maximize, close and reopen
    "tour": [
        ("Photos", "focus"), ("photo", 5), ("back", None), ("Monitor", "body"), ("Notes", "focus"),
        ("Notes", "up"), ("Notes", "min"), ("Monitor", "left"), ("Monitor", "left"), ("Monitor", "max"),
        ("Monitor", "max"), ("dock", "Notes"), ("Notes", "max"), ("Notes", "min"), ("Photos", "body"),
        ("Monitor", "close"), ("photo", 9), ("Photos", "max"), ("Photos", "min"), ("dock", "Monitor"),
        ("dock", "Photos"), ("back", None), ("photo", 1), ("Photos", "max"), ("Photos", "left"), ("Monitor", "body"), ("Photos", "body"),
        ("Photos", "close"), ("dock", "Notes"), ("Notes", "max"), ("Notes", "close"), ("dock", "Photos"),
        ("photo", 3), ("Photos", "right"), ("Photos", "up"), ("dock", "Notes"), ("Notes", "right"),
        ("Monitor", "focus"), ("Monitor", "min"), ("dock", "Monitor"),
    ],
    # windows pushed into the corners: the clamps, and lower windows clicked where they show
    "edges": [
        ("Notes", "up"), ("Notes", "up"), ("Notes", "up"), ("Notes", "up"), ("Notes", "max"), ("Notes", "left"),
        ("Monitor", "focus"), ("Monitor", "left"), ("Monitor", "down"), ("Monitor", "down"), ("Monitor", "down"),
        ("Monitor", "down"), ("Monitor", "down"), ("photo", 7), ("Photos", "min"), ("Notes", "body"),
        ("Notes", "down"), ("Notes", "down"), ("Notes", "down"), ("Notes", "down"), ("Notes", "down"),
        ("Notes", "right"), ("Notes", "right"), ("Notes", "right"), ("Notes", "right"), ("Notes", "right"),
        ("Notes", "right"), ("Notes", "up"), ("Monitor", "body"), ("dock", "Photos"), ("Photos", "up"),
        ("Photos", "up"), ("Photos", "left"), ("Photos", "left"), ("Photos", "left"), ("Photos", "left"),
        ("Photos", "left"), ("Photos", "max"), ("Photos", "down"), ("Photos", "close"),
    ],
}


def script_events(script):
    out = []
    for a, b in script:
        if a == "photo":
            out.append((PHOTO0 + b - 1, ""))
        elif a == "back":
            out.append((BACK, ""))
        elif a == "dock":
            out.append((DOCK0 + WINS.index(b), "dock"))
        elif b in ("focus", "body"):
            out.append((ev_window(WINS.index(a), "focus"), "bar" if b == "focus" else "body"))
        else:
            out.append((ev_window(WINS.index(a), b), ""))
    return out


def play(clicks):
    """The fixture of one sequence: per click, its point in both layouts, the state after it, and the
    switched-off points to probe before it. Checks every click is clickable in both layouts."""
    s, rows = dict(START), []
    for t, key in enumerate(clicks):
        at = {}
        for name, g in LAYOUTS.items():
            c = clickable(s, g)
            assert key in c, (f"click {t} ({describe(key[0])}, {key[1]}) can't be clicked ({name}) in {s}; "
                              f"clickable: {[(describe(e), wh) for e, wh in c]}")
            at[name] = c[key]
        off = {name: off_points(s, g)[:3] for name, g in LAYOUTS.items()}
        before = s
        s = step(s, key[0])
        rows.append({"ev": key[0], "where": key[1], "what": describe(key[0]) + (f" ({key[1]})" if key[1] in ("bar", "body") else ""),
                     "at": at["wide"], "atp": at["phone"], "off": off["wide"], "offp": off["phone"],
                     "offdock": top_window(before), "state": s})
    return rows


def fixtures(seed=1, n_random=6):
    rng = np.random.default_rng(seed)
    seqs = [{"name": name, "events": play(script_events(sc))} for name, sc in SCRIPTS.items()]
    for k in range(n_random):
        s, keys = dict(START), []
        for _ in range(SLOTS):
            both = [key for key, p in clickable(s, LAYOUTS["wide"]).items() if key in clickable(s, LAYOUTS["phone"])]
            # favour window controls so windows wander, keep dock clicks rarer
            weights = np.array([0.35 if key[1] == "dock" else 1.0 for key in both])
            key = both[rng.choice(len(both), p=weights / weights.sum())]
            keys.append(key)
            s = step(s, key[0])
        seqs.append({"name": f"random{k}", "events": play(keys)})
    return {"start": START, "state": STATE, "slots": SLOTS, "desk": [DW, DH], "phone_px": PHONE_PX, "sequences": seqs}


def main(quick=False):
    net, scr = os_net(), screen_net()
    print(f"event net: {net.depth} layers, {net.n_units:,} units, {net.n_weights:,} weights, {len(net.in_names)} inputs")
    print(f"screen net: {scr.depth} layers, {scr.n_units:,} units, {scr.n_weights:,} weights")
    for line in ([] if quick else check(net, scr)):
        print("  " + line)
    (NETS / "os.json").write_text(json.dumps({"event": net.to_json(), "screen": scr.to_json()}))
    fx = fixtures()
    for sq in fx["sequences"]:
        assert len(sq["events"]) <= SLOTS
        kinds = sorted({e["what"].split(":")[-1].strip() for e in sq["events"] if not e["what"].startswith("photo")})
        print(f"  fixture {sq['name']:8} {len(sq['events'])} clicks: {', '.join(kinds)}")
    (ROOT / "tests" / "os.json").write_text(json.dumps(fx))
    print("wrote nets/os.json and tests/os.json")


if __name__ == "__main__":
    import sys
    main(quick="--quick" in sys.argv)
