"""Step 6: a game console made of neurons. A bigger CPU, one clock cycle = one ReLU net, running Snake.

    .venv/bin/python handwired/scripts/cpu2.py

The machine: two 8-bit registers A and B, a program counter PC (0..47), a wait flag H, an
instruction counter CY, 90 bytes of RAM, and an input port (the button pressed this turn, 1..4).
The program (48 instructions, each an opcode and an argument) and the RAM's starting values are
inputs to the net, so the same net runs any program. RAM is also the screen: a 9 x 10 framebuffer
whose border is the wall around an 8 x 8 playfield. The wall cells double as storage for the game's
variables (they only need to be "not empty" for the snake to crash into them).

    0 WAIT     stop here for good           10 TBA      A = B
    1 SYNC t   wait for the next button,    11 ADD m    A = A + RAM[m]          (mod 256)
               then go on at t              12 SUB m    A = A - RAM[m]          (mod 256)
    2 LDI n    A = n                        13 ADDI n   A = A + n               (mod 256)
    3 LD m     A = RAM[m]                   14 WRAP n   if A >= n: A = A - n
    4 ST m     RAM[m] = A                   15 NEXT n   B = (B + n) mod 90
    5 LDX n    A = RAM[B + n]               16 JMP t    go to t
    6 STX n    RAM[B + n] = A               17 JZ t     go to t if A == 0
    7 SETX n   RAM[B] = n                   18 JNZ t    go to t if A != 0
    8 LDB m    B = RAM[m]                   19 JXNZ t   go to t if RAM[B] != 0
    9 TAB      B = A                        20 IN       A = the button
                                            21..31      do nothing

RAM addresses are taken mod 90, jump targets mod 48. One cycle, as ReLUs, in order:

  fetch     [PC >= i] thermometer, one unit per line ([PC == i] is a difference of three); then
            op = sum_i relu(op_i - 31 (1 - [PC == i]) - 31 H), arg likewise (0 when waiting)
  decode    op and arg as one unit each; [op >= k] thermometer; the ALU's carry bits;
            floor(x / 90) and floor(arg / 48), one unit each
  address   arg, B + arg or B, each mod 90, gated by the decoded opcode; [addr >= j] over 90 cells
  memory    read every cell gated by [addr == j]; write: cell - relu(cell - 255 h) + relu(D - 255 h),
            h = 2 - [addr == j] - write
  registers conditions folded into the gates: relu(t - 47 (1 - e_JXNZ) - 47 m) is "jump target if
            JXNZ and RAM[B] == 0"

A page turn runs K cycles: the turn resets H and CY and sets the button, then the CPU runs the game
loop until SYNC. The page's net is U cycles in one net (fewer nested elements: browsers stop nesting
HTML at 512 levels), in skip-connected form (`Dag`): a unit may read any earlier unit, so none of
the identity units a strictly layered net needs to carry values down a layer. Both forms are checked.

Checked against a plain Python emulator, every register and RAM cell at every cycle, on Snake
games and thousands of random programs; then thousands of random Snake games, each turn finishing
within K cycles. Writes handwired/nets/cpu2.json and handwired/tests/cpu2.json.
"""
import json
import time

import numpy as np

from common import NETS, ROOT
from hw import LIMIT, Prog, css_affine, total

ROM, RAM = 48, 90
OPB = 31                  # opcodes are 0..31
K = 36                    # cycles per turn
U = 12                    # cycles per compiled net on the page (K / U nested steps per turn)
TURNS = 60
OPS = ["WAIT", "SYNC", "LDI", "LD", "ST", "LDX", "STX", "SETX", "LDB", "TAB", "TBA",
       "ADD", "SUB", "ADDI", "WRAP", "NEXT", "JMP", "JZ", "JNZ", "JXNZ", "IN"]
OP = {name: k for k, name in enumerate(OPS)}
NO_ARG = {"WAIT", "TAB", "TBA", "IN"}
STATE = ["PC", "A", "B", "H", "CY"] + [f"M{j}" for j in range(RAM)]

# ---------------------------------------------------------------- the emulator


def step(s, ops, args, key):
    """One cycle on a dict state."""
    s = dict(s)
    if s["H"]:
        return s
    pc = s["PC"]
    op, arg = ops[pc], args[pc]
    A, B = s["A"], s["B"]
    name = OPS[op] if op < len(OPS) else "NOP"
    s["CY"] += 1
    nxt = (pc + 1) % ROM
    target = arg % ROM
    ad, ax, ab = arg % RAM, (B + arg) % RAM, B % RAM
    M = lambda a: s[f"M{a}"]
    if name == "WAIT":
        nxt = pc
    elif name == "SYNC":
        s["H"] = 1
        nxt = target
    elif name == "LDI":
        s["A"] = arg
    elif name == "LD":
        s["A"] = M(ad)
    elif name == "ST":
        s[f"M{ad}"] = A
    elif name == "LDX":
        s["A"] = M(ax)
    elif name == "STX":
        s[f"M{ax}"] = A
    elif name == "SETX":
        s[f"M{ab}"] = arg
    elif name == "LDB":
        s["B"] = M(ad)
    elif name == "TAB":
        s["B"] = A
    elif name == "TBA":
        s["A"] = B
    elif name == "ADD":
        s["A"] = (A + M(ad)) % 256
    elif name == "SUB":
        s["A"] = (A - M(ad)) % 256
    elif name == "ADDI":
        s["A"] = (A + arg) % 256
    elif name == "WRAP":
        s["A"] = A - arg if A >= arg else A
    elif name == "NEXT":
        s["B"] = ax
    elif name == "JMP":
        nxt = target
    elif name == "JZ":
        nxt = target if A == 0 else nxt
    elif name == "JNZ":
        nxt = target if A != 0 else nxt
    elif name == "JXNZ":
        nxt = target if M(ab) != 0 else nxt
    elif name == "IN":
        s["A"] = key
    s["PC"] = nxt
    return s


def new_turn(s):
    """What the page does at every turn: wake the CPU, zero the instruction counter."""
    return {**s, "H": 0, "CY": 0}


def run_turns(s, ops, args, keys, k=K):
    """States after every cycle, turn after turn."""
    states = [s]
    for key in keys:
        s = new_turn(s)
        for _ in range(k):
            s = step(s, ops, args, key)
            states.append(s)
    return states


# ---------------------------------------------------------------- the assembler

def assemble(src, consts):
    """Lines `label: OP arg ; comment`. Returns (ops, args, listing [(label, text, comment)])."""
    lines, labels = [], {}
    for raw in src.strip().splitlines():
        code, _, comment = raw.partition(";")
        label, _, code = code.rpartition(":") if ":" in code else ("", "", code)
        if label.strip():
            labels[label.strip()] = len(lines)
        if code.strip():
            lines.append((label.strip(), code.split(), comment.strip()))
        elif label.strip():
            raise ValueError(f"label without instruction: {raw}")
    ops, args, listing = [], [], []
    for label, (name, *rest), comment in lines:
        arg = 0
        if rest:
            word = rest[0]
            arg = labels[word] if word in labels else consts[word] if word in consts else int(word)
        assert (name in NO_ARG) == (not rest), name
        ops.append(OP[name])
        args.append(arg)
        listing.append((label, f"{name} {rest[0]}" if rest else name, comment))
    assert len(ops) <= ROM, len(ops)
    while len(ops) < ROM:
        ops.append(OP["WAIT"])
        args.append(0)
    return ops, args, listing


# ---------------------------------------------------------------- Snake

W = 9                     # framebuffer stride: 8 playfield columns + the shared wall column
SIDE = 8                  # the playfield is SIDE x SIDE, cells (r, c) for r, c in 1..8
SCORE, HEAD, TAIL, FOOD, NEW, STATUS = 0, 1, 2, 3, 4, 8
DELTA = 81                # RAM[81 + button] is the step for that button
EMPTY, WALL, FOODV, HEADV = 0, 1, 2, 3   # cell values; a body cell holds the address of the next one
BUTTONS = {1: "up", 2: "right", 3: "down", 4: "left"}
STRIDE = 23               # the food search steps through the screen 23 cells at a time (at most 2 walls in a row)
START_HEAD, START_FOOD = 4 * W + 4, 2 * W + 6

SNAKE = """
loop:  IN          ; A = the button
       TAB
       LDX DELTA   ; A = the step for that button
       ADD HEAD    ; A = the cell the head moves to
       ST NEW
       TAB         ; B = that cell
       JXNZ busy   ; not empty? look closer
move:  LDB HEAD
       STX 0       ; the old head points to the new one
       LDB TAIL
       LDX 0       ; A = the cell after the tail
       ST TAIL
       SETX 0      ; erase the old tail
head:  LD NEW
       TAB
       SETX 3      ; draw the head
       ST HEAD
       SYNC loop   ; wait for the next button
busy:  LDX 0       ; A = what is there
       ADDI 254    ; minus 2: food?
       JZ eat
       LD NEW
       SUB TAIL    ; the tail moves away, so it is fine
       JNZ dead
       LD NEW
       JMP move
eat:   LDB HEAD
       LD NEW
       STX 0       ; grow: the old head points to the new one, the tail stays
       LD SCORE
       ADDI 1
       ST SCORE
       LDB FOOD
probe: NEXT STRIDE ; the next place for food
       JXNZ probe  ; taken? try further
       TBA
       ST FOOD
       SETX 2      ; draw the food
       JMP head
dead:  LDI 2
       ST STATUS   ; game over
over:  SYNC over
"""
CONSTS = {"DELTA": DELTA, "HEAD": HEAD, "TAIL": TAIL, "FOOD": FOOD, "NEW": NEW, "SCORE": SCORE,
          "STATUS": STATUS, "STRIDE": STRIDE}


def cell(r, c):
    return r * W + c


def is_border(a):
    r, c = divmod(a, W)
    return r == 0 or r == SIDE + 1 or c == 0


def snake_ram():
    ram = [WALL if is_border(a) else EMPTY for a in range(RAM)]
    ram[SCORE], ram[STATUS] = 0, 1
    ram[HEAD] = ram[TAIL] = ram[NEW] = START_HEAD
    ram[FOOD] = START_FOOD
    for b, d in {1: -W, 2: 1, 3: W, 4: -1}.items():
        ram[DELTA + b] = d % 256
    ram[START_HEAD], ram[START_FOOD] = HEADV, FOODV
    return ram


def snake_cartridge():
    ops, args, listing = assemble(SNAKE, CONSTS)
    return ops, args, snake_ram(), listing


def start(ram, **regs):
    s = {k: 0 for k in STATE}
    s.update(regs)
    for j, v in enumerate(ram):
        s[f"M{j}"] = v
    return s


def screen(s):
    """The playfield as a SIDE x SIDE grid of cell kinds: 0 empty, 2 food, 3 head, 4 body."""
    out = []
    for r in range(1, SIDE + 1):
        row = []
        for c in range(1, SIDE + 1):
            v = s[f"M{cell(r, c)}"]
            row.append(4 if v >= 4 else v)
        out.append(row)
    return out


def play(keys, ops=None, args=None, ram=None):
    """Snake by the emulator: the state after each turn, and the cycles each turn used."""
    if ops is None:
        ops, args, ram, _ = snake_cartridge()
    s = start(ram)
    after, used = [], []
    for key in keys:
        s = new_turn(s)
        for c in range(K):
            s = step(s, ops, args, key)
        after.append(s)
        used.append(s["CY"] if s["H"] else None)   # None: SYNC not reached within K cycles
    return after, used


STEP = {1: -W, 2: 1, 3: W, 4: -1}
WALLS = frozenset(a for a in range(RAM) if is_border(a))


def reference_start():
    return {"body": (START_HEAD,), "food": START_FOOD, "score": 0, "over": False}


def reference_move(g, key):
    """Snake written plainly, without the CPU: the rules the program implements. body: tail first."""
    if g["over"]:
        return g
    body, food, score = list(g["body"]), g["food"], g["score"]
    new = body[-1] + STEP[key]
    if new == food:
        body.append(new)
        score += 1
        taken = set(body) | WALLS
        while food in taken:
            food = (food + STRIDE) % RAM
    elif new in WALLS or (new in body and new != body[0]):
        return {**g, "over": True}
    else:
        body = body[1:] + [new]
    return {"body": tuple(body), "food": food, "score": score, "over": False}


def reference_screen(g):
    grid = [[0] * SIDE for _ in range(SIDE)]
    for a in g["body"]:
        r, c = divmod(a, W)
        grid[r - 1][c - 1] = 4
    r, c = divmod(g["body"][-1], W)
    grid[r - 1][c - 1] = 3
    r, c = divmod(g["food"], W)
    grid[r - 1][c - 1] = 2
    return grid


def player(rng, style, turns=TURNS):
    """Button presses from a simple player. 'greedy' heads for the food, 'safe' wanders without
    crashing, 'wild' presses anything, 'late' plays greedy then wild."""
    g, keys = reference_start(), []
    crash_at = int(rng.integers(5, turns)) if style == "late" else None
    for t in range(turns):
        mode = "wild" if style == "wild" or (crash_at is not None and t >= crash_at) else style
        options = list(BUTTONS)
        safe = [k for k in options if not reference_move(g, k)["over"]]
        if mode == "wild" or not safe:
            k = int(rng.choice(options))
        elif mode == "greedy" and rng.uniform() < 0.85:
            hr, hc = divmod(g["body"][-1], W)
            fr, fc = divmod(g["food"], W)
            dist = lambda k: abs(hr + (k == 3) - (k == 1) - fr) + abs(hc + (k == 2) - (k == 4) - fc)
            best = min(dist(k) for k in safe)
            k = int(rng.choice([k for k in safe if dist(k) == best]))
        else:
            k = int(rng.choice(safe))
        keys.append(k)
        g = reference_move(g, k)
    return keys


STYLES = ["greedy", "greedy", "safe", "late", "wild"]


# ---------------------------------------------------------------- the net

def thermometer(p, v, n):
    """t_i = relu(v - i + 1) for i in 0..n+1; [v == i] = t_i - 2 t_{i+1} + t_{i+2} for integer 0 <= v <= n - 1."""
    t = [p.relu(v - i + 1) for i in range(n + 2)]
    return [t[i] - 2 * t[i + 1] + t[i + 2] for i in range(n)], [t[i] - t[i + 1] for i in range(n + 1)]


def floor_div(p, v, d):
    """floor(v / d) for integer v >= 0, one unit: round(relu(2v + 1 - d) / 2d); never an exact half."""
    return p.relu(2 * v + 1 - d, R=2 * d)


def cycle(p, s, ops, args, key):
    """One clock cycle on Vals. s: {name: Val}. Returns the next state as Vals."""
    PC, A, B, H = s["PC"], s["A"], s["B"], s["H"]
    Mem = [s[f"M{j}"] for j in range(RAM)]

    # 1-2 fetch (nothing, i.e. WAIT with argument 0, while waiting)
    at, _ = thermometer(p, PC, ROM)
    op = total(p.relu(ops[i] - OPB * (1 - at[i]) - OPB * H) for i in range(ROM))
    arg = total(p.relu(args[i] - 255 * (1 - at[i]) - 255 * H) for i in range(ROM))

    # 3 decode. The fetched op and arg are 48-term sums: each becomes one unit, read by everything after
    opm, argm = p.relu(op), p.relu(arg)
    is_, ge_ = thermometer(p, opm, len(OPS))
    E = dict(zip(OPS, is_))
    f_d = floor_div(p, argm, RAM)                # arg mod 90 = arg - 90 f_d
    f_x = floor_div(p, B + argm, RAM)            # (B + arg) mod 90
    f_t = floor_div(p, argm, ROM)                # arg mod 48
    c_addi = p.ge(A + argm, 256)
    c_wrap = p.ge(A - argm, 0)
    f_b = floor_div(p, B, RAM)                   # B mod 90 (layer 1)

    addr_d = argm - RAM * f_d
    addr_x = B + argm - RAM * f_x
    addr_b = B - RAM * f_b
    target = argm - ROM * f_t

    # 4 the address bus
    use_d = E["LD"] + E["ST"] + E["LDB"] + E["ADD"] + E["SUB"]
    use_x = E["LDX"] + E["STX"]
    use_b = E["SETX"] + E["JXNZ"]
    addr = p.gate(use_d, addr_d, RAM - 1) + p.gate(use_x, addr_x, RAM - 1) + p.gate(use_b, addr_b, RAM - 1)
    write = E["ST"] + E["STX"] + E["SETX"]
    data = p.relu(p.gate(E["ST"] + E["STX"], A, 255) + p.gate(E["SETX"], argm, 255))

    # the parts of the new A, B and PC that don't need the memory read
    A_set = [(E["LDI"], argm), (E["ADDI"], A + argm - 256 * c_addi), (E["WRAP"], A), (E["TBA"], B), (E["IN"], key)]
    keep_a = 1 - total(b for b, _ in A_set) - E["LD"] - E["LDX"] - E["ADD"] - E["SUB"]
    A_early = p.relu(total(p.gate(b, v, 255) for b, v in A_set) + p.gate(keep_a, A, 255)
                     - p.relu(argm - 255 * (2 - E["WRAP"] - c_wrap)))
    keep_b = 1 - E["TAB"] - E["NEXT"] - E["LDB"]
    B_early = p.relu(p.gate(E["TAB"], A, 255) + p.gate(E["NEXT"], addr_x, RAM - 1) + p.gate(keep_b, B, 255))
    T = ROM - 1
    pc_inc = PC + 1 - ROM * at[ROM - 1]

    def if_zero(bit, x, v):
        """v when bit and x == 0 (x a non-negative integer), else 0."""
        return p.relu(v - T * (1 - bit) - T * x)

    def if_nonzero(bit, x, v):
        return p.gate(bit, v, T) - if_zero(bit, x, v)

    advance = 1 - E["WAIT"] - E["SYNC"] - E["JMP"] - E["JZ"] - E["JNZ"] - E["JXNZ"]
    PC_early = p.relu(p.gate(E["WAIT"], PC, T) + p.gate(E["SYNC"] + E["JMP"], target, T) + p.gate(advance, pc_inc, T)
                      + if_zero(E["JZ"], A, target) + if_nonzero(E["JZ"], A, pc_inc)
                      + if_nonzero(E["JNZ"], A, target) + if_zero(E["JNZ"], A, pc_inc))

    # 5-6 memory: read every cell through the bus, write the one it names
    sel, _ = thermometer(p, addr, RAM)
    m = p.relu(total(p.relu(Mem[j] - 255 * (1 - sel[j])) for j in range(RAM)))   # the value read, one unit
    Mem2 = []
    for j in range(RAM):
        hit = 2 - sel[j] - write
        Mem2.append(p.relu(Mem[j] - p.relu(Mem[j] - 255 * hit) + p.relu(data - 255 * hit)))

    # 7 what needs the read: LD, LDX, ADD, SUB, LDB, JXNZ
    sa = A + m
    ad = E["ADD"]
    add = p.relu(sa - 510 * (1 - ad)) - 256 * (p.relu(sa - 255 - 510 * (1 - ad)) - p.relu(sa - 256 - 510 * (1 - ad)))
    sd = A + 256 - m               # 1..511; A - m mod 256 = sd - 256 [sd >= 256]
    su = E["SUB"]
    sub = p.relu(sd - 511 * (1 - su)) - 256 * (p.relu(sd - 255 - 511 * (1 - su)) - p.relu(sd - 256 - 511 * (1 - su)))
    A2 = A_early + p.gate(E["LD"] + E["LDX"], m, 255) + add + sub
    B2 = B_early + p.gate(E["LDB"], m, 255)
    PC2 = PC_early + if_zero(E["JXNZ"], m, pc_inc) + if_nonzero(E["JXNZ"], m, target)
    out = {"PC": p.relu(PC2), "A": p.relu(A2), "B": p.relu(B2), "H": p.relu(H + E["SYNC"]), "CY": p.relu(s["CY"] + 1 - H)}
    out.update({f"M{j}": Mem2[j] for j in range(RAM)})
    return out


def cpu2_prog(cycles=1):
    p = Prog()
    s = {k: p.input(k) for k in STATE}
    key = p.input("KEY")
    ops = [p.input(f"op{i}") for i in range(ROM)]
    args = [p.input(f"arg{i}") for i in range(ROM)]
    for _ in range(cycles):
        s = cycle(p, s, ops, args, key)
    return p, {k: s[k] for k in STATE}


def cpu2_net(cycles=1):
    """The strictly layered net (hw.compile): inputs the state, the button, then the program
    (48 ops, 48 args); outputs the state `cycles` later."""
    p, out = cpu2_prog(cycles)
    return p.compile(out)


def cpu2_dag(cycles=U):
    """The same units without the identity units that carry values down the layers: the form the page uses."""
    p, out = cpu2_prog(cycles)
    return Dag.from_prog(p, out)


class Dag:
    """A ReLU net whose units may read any earlier unit or input (skip connections), not only the
    layer above. In CSS every unit of one step sits on the same element, so a unit can read any
    other: the identity units a layered net needs to carry values down are not needed.

    nodes: list of (terms [(source, coef)], bias, R); a source is an input name or a node index.
    Each node is round(relu(bias + sum coef * source) / R). Outputs are affine forms over the same."""

    def __init__(self, in_names, nodes, out_names, outs):
        self.in_names, self.nodes, self.out_names, self.outs = in_names, nodes, out_names, outs
        self._plan()

    @staticmethod
    def from_prog(p, outputs):
        names = list(outputs)
        outs = [outputs[n] for n in names]
        live, stack = set(), [k for v in outs for k in v.terms]
        while stack:
            k = stack.pop()
            if k in live or k not in p.expr:
                continue
            live.add(k)
            stack.extend(p.expr[k].terms)
        order = sorted(live, key=lambda k: (p.depth[k], k))
        index = {k: n for n, k in enumerate(order)}
        src = lambda k: p.names[k] if k in p.names else index[k]
        nodes = [([(src(u), int(c)) for u, c in sorted(p.expr[k].terms.items())], p.expr[k].const, p.R[k]) for k in order]
        outs = [([(src(u), int(c)) for u, c in sorted(v.terms.items())], v.const) for v in outs]
        return Dag([p.names[i] for i in p.inputs], nodes, names, outs)

    def _plan(self):
        """Group nodes into levels (1 + the deepest source) for vectorised evaluation."""
        n_in = len(self.in_names)
        col = {name: i for i, name in enumerate(self.in_names)}
        c = lambda s: col[s] if isinstance(s, str) else n_in + s
        level = []
        for terms, _, _ in self.nodes:
            level.append(1 + max([level[s] if not isinstance(s, str) else 0 for s, _ in terms] + [0]))
        self.levels = []
        for d in range(1, max(level, default=0) + 1):
            ids = [n for n, l in enumerate(level) if l == d]
            src, dst, w = [], [], []
            for slot, n in enumerate(ids):
                for s, cf in self.nodes[n][0]:
                    src.append(c(s)); dst.append(slot); w.append(cf)
            self.levels.append((np.array(ids) + n_in, np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64),
                                np.array(w, dtype=np.int64), np.array([self.nodes[n][1] for n in ids], dtype=np.int64),
                                np.array([self.nodes[n][2] for n in ids], dtype=np.int64)))
        src, dst, w = [], [], []
        for slot, (terms, _) in enumerate(self.outs):
            for s, cf in terms:
                src.append(c(s)); dst.append(slot); w.append(cf)
        self.out_plan = (np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64), np.array(w, dtype=np.int64),
                         np.array([b for _, b in self.outs], dtype=np.int64))
        self.depth = len(self.levels)

    @staticmethod
    def _sums(V, src, dst, w, b, absolute=False):
        z = np.tile(np.abs(b) if absolute else b, (V.shape[0], 1))
        if len(w):
            prod = V[:, src] * w
            np.add.at(z.T, dst, np.abs(prod).T if absolute else prod.T)
        return z

    def forward(self, X, mass=False):
        X = np.atleast_2d(np.asarray(X, dtype=np.int64))
        V = np.zeros((X.shape[0], X.shape[1] + len(self.nodes)), dtype=np.int64)
        V[:, :X.shape[1]] = X
        m = 0
        for ids, src, dst, w, b, R in self.levels:
            z = self._sums(V, src, dst, w, b)
            if mass:
                m = max(m, int(self._sums(V, src, dst, w, b, True).max()))
            V[:, ids] = np.floor(np.maximum(z, 0) / R + 0.5).astype(np.int64)
        y = self._sums(V, *self.out_plan)
        if mass:
            return y, max(m, int(self._sums(V, *self.out_plan, True).max()))
        return y

    def max_mass(self, X):
        return self.forward(X, mass=True)[1]

    @property
    def n_units(self):
        return len(self.nodes)

    @property
    def n_weights(self):
        return sum(len(t) + (b != 0) for t, b, _ in self.nodes) + sum(len(t) + (b != 0) for t, b in self.outs)

    def to_json(self):
        return {"inputs": self.in_names, "outputs": self.out_names,
                "nodes": [[[[s, c] for s, c in t], b, r] for t, b, r in self.nodes],
                "outs": [[[[s, c] for s, c in t], b] for t, b in self.outs]}

    @staticmethod
    def from_json(d):
        return Dag(d["inputs"], [([(s, c) for s, c in t], b, r) for t, b, r in d["nodes"]], d["outputs"],
                   [([(s, c) for s, c in t], b) for t, b in d["outs"]])

    def css(self, inputs, prefix, outputs):
        """[(property, expression)]: units `max(0, calc(...))` (with `/ R`), outputs `calc(...)`.
        inputs: {input name: property name}."""
        name = lambda s: inputs[s] if isinstance(s, str) else f"{prefix}{s}"
        rules = []
        for n, (terms, b, R) in enumerate(self.nodes):
            body = css_affine(b, [(c, name(s)) for s, c in terms])
            rules.append((f"{prefix}{n}", f"max(0, calc({body}))" if R == 1 else f"max(0, calc(({body}) / {R}))"))
        for o, (terms, b) in zip(outputs, self.outs):
            rules.append((o, f"calc({css_affine(b, [(c, name(s)) for s, c in terms])})"))
        return rules


def net_inputs(S, key, prog):
    """Rows for net.forward: S (N, len(STATE)), key (N,), prog (N, 2 ROM)."""
    return np.hstack([S, np.asarray(key, dtype=np.int64)[:, None], prog])


# ---------------------------------------------------------------- checks

def snake_cases(n, seed):
    """(ops, args, start state, keys per turn) for Snake games by the players above."""
    rng = np.random.default_rng(seed)
    ops, args, ram, _ = snake_cartridge()
    return [(ops, args, start(ram), player(rng, STYLES[g % len(STYLES)])) for g in range(n)]


def random_cases(n, seed, turns=4):
    """Random programs from random states: mostly real opcodes, some WAITs and do-nothings."""
    rng = np.random.default_rng(seed)
    out = []
    for g in range(n):
        o = rng.integers(1, len(OPS), ROM)
        o = np.where(rng.uniform(size=ROM) < 0.08, rng.integers(0, OPB + 1, ROM), o)
        a = rng.integers(0, 256, ROM)
        a = np.where(rng.uniform(size=ROM) < 0.1, 255, a)                                  # boundary values
        r = rng.integers(0, 256, RAM)
        if rng.uniform() < 0.5:                                                            # small values: more aliasing
            r = rng.integers(0, 6, RAM)
        st = start(r.tolist(), PC=int(rng.integers(0, ROM)), A=int(rng.choice([0, 255, rng.integers(0, 256)])),
                   B=int(rng.choice([0, 255, 89, 90, rng.integers(0, 256)])), H=int(rng.uniform() < 0.1))
        out.append((o.tolist(), a.tolist(), st, rng.integers(0, 5, turns).tolist()))
    return out


def check_net(net, cyc, all_cases, turn_cycles):
    """The net (`cyc` cycles per application) against the emulator: every state value after every
    application, with the page's turn reset every turn_cycles cycles."""
    assert turn_cycles % cyc == 0
    col = [net.out_names.index(k) for k in STATE]
    emu = [run_turns(st, o, a, keys, turn_cycles) for o, a, st, keys in all_cases]
    n_turns = len(all_cases[0][3])
    assert all(len(c[3]) == n_turns for c in all_cases)
    prog = np.array([list(o) + list(a) for o, a, _, _ in all_cases], dtype=np.int64)
    S = np.array([[c[2][k] for k in STATE] for c in all_cases], dtype=np.int64)
    mass, compared = 0, 0
    iH, iCY = STATE.index("H"), STATE.index("CY")
    for t in range(n_turns):
        S[:, iH] = 0
        S[:, iCY] = 0
        key = np.array([c[3][t] for c in all_cases])
        for q in range(turn_cycles // cyc):
            X = net_inputs(S, key, prog)
            if isinstance(net, Dag):                       # every application: forward and mass in one pass
                Y, m = net.forward(X, mass=True)
            else:
                Y, m = net.forward(X), net.max_mass(X)
            mass = max(mass, int(m))
            S = Y[:, col]
            c = t * turn_cycles + (q + 1) * cyc
            want = np.array([[e[c][k] for k in STATE] for e in emu], dtype=np.int64)
            bad = np.flatnonzero((S != want).any(axis=1))
            if len(bad):
                i = bad[0]
                diff = {k: (int(S[i, n]), want[i, n]) for n, k in enumerate(STATE) if S[i, n] != want[i, n]}
                raise AssertionError(f"turn {t} cycle {c}: case {i} differs (net, emulator): {diff}")
            compared += len(all_cases) * len(STATE)
    assert mass < LIMIT, mass
    return mass, compared


def check_games(n_games, seed):
    """Snake games on the emulator, against the plain reference after every move; every turn
    finishes within K cycles."""
    rng = np.random.default_rng(seed)
    worst, scores, overs, spills, moves = 0, [], 0, 0, 0
    for n in range(n_games):
        keys = player(rng, STYLES[n % len(STYLES)])
        after, used = play(keys)
        g = reference_start()
        for t, (s, key) in enumerate(zip(after, keys)):
            g = reference_move(g, key)
            got = (screen(s), s[f"M{SCORE}"], s[f"M{STATUS}"] == 2)
            assert got == (reference_screen(g), g["score"], g["over"]), (n, t, keys[:t + 1])
            moves += 1
        spills += sum(u is None for u in used)
        worst = max([worst] + [u for u in used if u is not None])
        scores.append(after[-1][f"M{SCORE}"])
        overs += after[-1][f"M{STATUS}"] == 2
    assert spills == 0, spills
    return worst, scores, overs, moves


def fixtures(n, seed):
    """Games for the browser test: the screen, score and game over after every move."""
    rng = np.random.default_rng(seed)
    out = []
    for g in range(n):
        keys = player(rng, ["greedy", "late", "greedy", "safe", "greedy", "wild"][g % 6])
        after, _ = play(keys)
        moves = []
        for s in after:
            moves.append({"screen": "".join(str(v) for row in screen(s) for v in row), "score": s[f"M{SCORE}"],
                          "over": int(s[f"M{STATUS}"] == 2), "cy": s["CY"]})
            if s[f"M{STATUS}"] == 2:
                break
        out.append({"keys": keys[:len(moves)], "moves": moves})
    return out


def main():
    t0 = time.time()
    worst, scores, overs, moves = check_games(5000, 1)
    print(f"5,000 Snake games of {TURNS} moves ({moves:,} moves): the emulator plays exactly like the plain "
          f"Snake reference; slowest turn {worst} of {K} cycles; scores up to {max(scores)} "
          f"(mean {np.mean(scores):.1f}); {overs:,} end in a crash  [{time.time() - t0:.0f}s]")

    t0 = time.time()
    rc, sc = random_cases(3000, 0), snake_cases(100, 5)
    for label, one in (("layered", cpu2_net(1)), ("skip-connected", cpu2_dag(1))):
        mass, n = check_net(one, 1, rc, 12)
        mass2, n2 = check_net(one, 1, sc, K)
        print(f"one cycle, {label}: {one.depth} layers, {one.n_units:,} units, {one.n_weights:,} weights. "
              f"{len(rc):,} random programs x 4 turns x 12 cycles and {len(sc)} Snake games x {TURNS} turns x {K} cycles: "
              f"{n + n2:,} values equal the emulator, every cycle; max |partial sum| {max(mass, mass2):,}  [{time.time() - t0:.0f}s]")

    net = cpu2_dag(U)
    print(f"{U} cycles in one net (the page's): {net.depth} layers, {net.n_units:,} units, {net.n_weights:,} weights")
    t0 = time.time()
    rc = random_cases(2000, 2)
    mass, n = check_net(net, U, rc, K)
    sc = snake_cases(300, 6)
    mass2, n2 = check_net(net, U, sc, K)
    print(f"  {len(rc):,} random programs x 4 turns x {K} cycles and {len(sc)} Snake games x {TURNS} turns: "
          f"{n + n2:,} values equal the emulator after every {U} cycles; max |partial sum| {max(mass, mass2):,}  [{time.time() - t0:.0f}s]")

    (NETS / "cpu2.json").write_text(json.dumps(net.to_json()))
    fx = fixtures(12, 4)
    (ROOT / "tests" / "cpu2.json").write_text(json.dumps({"K": K, "U": U, "turns": TURNS, "games": fx}))
    print(f"wrote nets/cpu2.json and tests/cpu2.json ({len(fx)} games, {sum(len(g['moves']) for g in fx)} moves, "
          f"scores {[g['moves'][-1]['score'] for g in fx]})")


if __name__ == "__main__":
    main()
