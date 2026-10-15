"""Step 5: a computer made of neurons. One clock cycle of a tiny 8-bit CPU, compiled into one ReLU net.

    .venv/bin/python handwired/scripts/cpu.py

The machine: an accumulator A (0..255), a program counter PC (0..15), a halt flag, 8 bytes of RAM,
an output tape of 16 bytes, and a cycle counter. The program (the cartridge) is 16 instructions,
each an opcode and an argument, and the RAM's starting values. Both are inputs to the net, so the
same net runs any program.

    0 HLT        stop                       6 ADDI n   A = A + n (mod 256)
    1 LDI n      A = n                      7 JMP t    PC = t
    2 LD m       A = RAM[m]                 8 JZ t     PC = t if A == 0
    3 ST m       RAM[m] = A                 9 JNZ t    PC = t if A != 0
    4 ADD m      A = A + RAM[m] (mod 256)  10 OUT      write A to the tape
    5 SUB m      A = A - RAM[m] (mod 256)  other       do nothing

Addresses are taken mod 8 and jump targets mod 16. One cycle, as ReLUs:

  fetch     pc_i = [PC == i]; op = sum_i pc_i * op_i, arg likewise    (each product a gate)
  decode    run = 1 - halted; e_k = [op == k] AND run
  read      m = sum_j [addr == j] * RAM[j], addr = arg mod 8 = arg - 8 floor(arg / 8)
  ALU       A + m, A - m, A + arg, wrapped mod 256 with one comparison each
  write     A' = the result of whichever e_k is on (or A); RAM[j]' likewise; the tape at its next slot
  next PC   the target, PC + 1, or PC again (halted)

Checked against a plain Python emulator on the demo programs and on thousands of random programs,
every register at every cycle. Writes handwired/nets/cpu.json and handwired/tests/cpu.json.
"""
import json

import numpy as np

from common import NETS, ROOT
from hw import LIMIT, Prog, total

ROM, RAM, TAPE = 16, 8, 16
CYCLES = 128
OPS = ["HLT", "LDI", "LD", "ST", "ADD", "SUB", "ADDI", "JMP", "JZ", "JNZ", "OUT"]
OP = {name: k for k, name in enumerate(OPS)}
STATE = ["PC", "A", "H", "CY", "NT"] + [f"R{j}" for j in range(RAM)] + [f"T{k}" for k in range(TAPE)]


# ---------------------------------------------------------------- the emulator

def step(s, ops, args):
    """One cycle on a dict state. Returns (new state, op executed or None)."""
    s = dict(s)
    if s["H"]:
        return s, None
    op, arg = ops[s["PC"]], args[s["PC"]]
    addr, target = arg % RAM, arg % ROM
    nxt = (s["PC"] + 1) % ROM
    s["CY"] += 1
    if op == OP["HLT"]:
        s["H"] = 1
        return s, op
    if op == OP["LDI"]:
        s["A"] = arg
    elif op == OP["LD"]:
        s["A"] = s[f"R{addr}"]
    elif op == OP["ST"]:
        s[f"R{addr}"] = s["A"]
    elif op == OP["ADD"]:
        s["A"] = (s["A"] + s[f"R{addr}"]) % 256
    elif op == OP["SUB"]:
        s["A"] = (s["A"] - s[f"R{addr}"]) % 256
    elif op == OP["ADDI"]:
        s["A"] = (s["A"] + arg) % 256
    elif op == OP["OUT"]:
        if s["NT"] < TAPE:
            s[f"T{s['NT']}"] = s["A"]
        s["NT"] += 1
    if op == OP["JMP"] or (op == OP["JZ"] and s["A"] == 0) or (op == OP["JNZ"] and s["A"] != 0):
        nxt = target
    s["PC"] = nxt
    return s, op


def start(ram):
    s = {k: 0 for k in STATE}
    for j, v in enumerate(ram):
        s[f"R{j}"] = v
    return s


def run(ops, args, ram, cycles=CYCLES):
    states, executed = [start(ram)], []
    for _ in range(cycles):
        s, op = step(states[-1], ops, args)
        states.append(s)
        executed.append(op)
    return states, executed


# ---------------------------------------------------------------- the net

def cpu_net():
    """Inputs: the state, then the program (16 ops, 16 args). Outputs: the next state, plus what this
    cycle executed (one-hot e_k, the argument and the PC it ran at) for the trace."""
    p = Prog()
    st = {k: p.input(k) for k in STATE}
    ops = [p.input(f"op{i}") for i in range(ROM)]
    args = [p.input(f"arg{i}") for i in range(ROM)]
    PC, A, H = st["PC"], st["A"], st["H"]
    R = [st[f"R{j}"] for j in range(RAM)]
    T = [st[f"T{k}"] for k in range(TAPE)]

    # fetch
    at = [p.eq(PC, i) for i in range(ROM)]
    op = total(p.gate(at[i], ops[i], 15) for i in range(ROM))
    arg = total(p.gate(at[i], args[i], 255) for i in range(ROM))

    # decode
    run_ = 1 - H
    e = [p.AND(p.eq(op, k), run_) for k in range(len(OPS))]
    E = dict(zip(OPS, e))

    # arg mod 8 and mod 16, with one integer unit each: floor(u / 2^j) = round(relu(2u + 1 - 2^j) / 2^(j+1))
    addr = arg - 8 * p.relu(2 * arg + 1 - 8, R=16)
    target = arg - 16 * p.relu(2 * arg + 1 - 16, R=32)

    # read
    sel = [p.eq(addr, j) for j in range(RAM)]
    m = total(p.gate(sel[j], R[j], 255) for j in range(RAM))

    # ALU, wrapped mod 256
    s_add = A + m
    w_add = s_add - 256 * p.ge(s_add, 256)
    d = A - m
    w_sub = d + 256 - 256 * p.ge(d, 0)
    s_addi = A + arg
    w_addi = s_addi - 256 * p.ge(s_addi, 256)

    # write A
    writes = [(E["LDI"], arg), (E["LD"], m), (E["ADD"], w_add), (E["SUB"], w_sub), (E["ADDI"], w_addi)]
    keep = 1 - total(b for b, _ in writes)
    A2 = total(p.gate(b, v, 255) for b, v in writes) + p.gate(keep, A, 255)

    # write RAM
    R2 = []
    for j in range(RAM):
        store = p.AND(E["ST"], sel[j])
        R2.append(p.gate(store, A, 255) + p.gate(1 - store, R[j], 255))

    # the tape: the next free slot gets A
    NT = st["NT"]
    T2 = [T[k] + p.gate(p.AND(E["OUT"], p.eq(NT, k)), A, 255) for k in range(TAPE)]

    # next PC
    zero = p.eq(A, 0)
    jump = E["JMP"] + p.AND(E["JZ"], zero) + p.AND(E["JNZ"], 1 - zero)
    advance = run_ - E["HLT"] - jump
    pc_inc = PC + 1 - ROM * at[ROM - 1]
    PC2 = p.gate(jump, target, 15) + p.gate(advance, pc_inc, 15) + p.gate(1 - jump - advance, PC, 15)

    out = {"PC": PC2, "A": A2, "H": H + E["HLT"], "CY": st["CY"] + run_, "NT": NT + E["OUT"]}
    out.update({f"R{j}": R2[j] for j in range(RAM)})
    out.update({f"T{k}": T2[k] for k in range(TAPE)})
    out.update({f"e{k}": e[k] for k in range(len(OPS))})
    out["xarg"] = p.gate(run_, arg, 255)
    out["xpc"] = p.gate(run_, PC, 15)
    return p.compile(out)


def net_run(net, ops, args, ram, cycles=CYCLES):
    """The net applied cycle after cycle, as the page does. Returns the states and the trace."""
    col = {n: k for k, n in enumerate(net.out_names)}
    s = start(ram)
    states, trace = [s], []
    for _ in range(cycles):
        x = [s[k] for k in STATE] + list(ops) + list(args)
        y = net.forward(x)[0]
        s = {k: int(y[col[k]]) for k in STATE}
        states.append(s)
        hot = [k for k in range(len(OPS)) if y[col[f"e{k}"]]]
        trace.append(hot[0] if hot else None)
    return states, trace


# ---------------------------------------------------------------- the cartridges

def asm(lines):
    ops, args = [], []
    for line in lines:
        name, *rest = line.split()
        ops.append(OP[name]); args.append(int(rest[0]) if rest else 0)
    while len(ops) < ROM:
        ops.append(OP["HLT"]); args.append(0)
    return ops, args


# name: (title, listing with comments, RAM, inputs {ram cell: (label, values, default)})
PROGRAMS = {
    "count": ("Count to n", [
        ("LD 0", "A = counter"),
        ("ADDI 1", "add one"),
        ("ST 0", "save it"),
        ("OUT", "print it"),
        ("SUB 1", "A = counter − n"),
        ("JNZ 0", "not there yet? loop"),
        ("HLT", ""),
    ], [0, 8], {1: ("n", list(range(1, 16)), 8)}),
    "fibonacci": ("Fibonacci", [
        ("LD 0", "A = a"),
        ("OUT", "print a"),
        ("ADD 1", "A = a + b"),
        ("ST 2", "t = a + b"),
        ("LD 1", ""),
        ("ST 0", "a = b"),
        ("LD 2", ""),
        ("ST 1", "b = t"),
        ("JMP 0", "again"),
    ], [0, 1], {}),
    "multiply": ("Multiply x × y", [
        ("LD 2", "A = total"),
        ("ADD 0", "add x"),
        ("ST 2", "save total"),
        ("LD 1", "A = y"),
        ("SUB 3", "minus one"),
        ("ST 1", "save y"),
        ("JNZ 0", "y left? loop"),
        ("LD 2", ""),
        ("OUT", "print x × y"),
        ("HLT", ""),
    ], [6, 7, 0, 1], {0: ("x", list(range(0, 16)), 6), 1: ("y", list(range(1, 16)), 7)}),
    "sum": ("Sum 1 + 2 + … + n", [
        ("LD 1", "A = total"),
        ("ADD 0", "add n"),
        ("ST 1", "save total"),
        ("LD 0", "A = n"),
        ("SUB 2", "minus one"),
        ("ST 0", "save n"),
        ("JNZ 0", "n left? loop"),
        ("LD 1", ""),
        ("OUT", "print the sum"),
        ("HLT", ""),
    ], [10, 0, 1], {0: ("n", list(range(1, 16)), 10)}),
    "powers": ("Powers of two", [
        ("LD 0", "A = x"),
        ("OUT", "print x"),
        ("ADD 0", "A = x + x"),
        ("ST 0", "x = 2x"),
        ("JNZ 0", "until it overflows to 0"),
        ("HLT", ""),
    ], [1], {}),
}


# "Write your own": the editor's pickers offer the 11 opcodes, arguments 0..15 and starting memory values
# 0..15. Its default program counts down from 10 and prints 10, 9, ..., 1.
EDIT_ARGS, EDIT_MEM = 16, 16
CUSTOM = ([("LDI", 10), ("OUT", 0), ("SUB", 0), ("JNZ", 1), ("HLT", 0)], [1])


def custom_default():
    lines, ram = CUSTOM
    ops = [OP[o] for o, _ in lines] + [OP["HLT"]] * (ROM - len(lines))
    args = [a for _, a in lines] + [0] * (ROM - len(lines))
    return ops, args, list(ram) + [0] * (RAM - len(ram))


def cartridge(name, inputs=None):
    title, listing, ram, ins = PROGRAMS[name]
    ops, args = asm([l for l, _ in listing])
    ram = list(ram) + [0] * (RAM - len(ram))
    for cell, (_, _, default) in ins.items():
        ram[cell] = (inputs or {}).get(cell, default)
    return ops, args, ram


# ---------------------------------------------------------------- checks

def check(net, n_random=3000, seed=0):
    rng = np.random.default_rng(seed)
    cases = []
    for name, (_, _, _, ins) in PROGRAMS.items():
        cases.append(cartridge(name))
        for cell, (_, values, _) in ins.items():
            for v in values:
                cases.append(cartridge(name, {cell: v}))
    cases.append(custom_default())
    for _ in range(1000):  # programs the editor can express: opcodes 0..10, arguments and memory 0..15
        cases.append((rng.integers(0, len(OPS), ROM).tolist(), rng.integers(0, EDIT_ARGS, ROM).tolist(),
                      rng.integers(0, EDIT_MEM, RAM).tolist()))
    n_demo = len(cases)
    for _ in range(n_random):  # random programs: mostly real opcodes, some unknown ones
        ops = rng.integers(0, 13, ROM).tolist()
        ops = [o if rng.uniform() > 0.08 else int(rng.integers(0, 16)) for o in ops]
        if rng.uniform() < 0.7:  # fewer halts, so programs run longer
            ops = [o if o != 0 or rng.uniform() < 0.2 else int(rng.integers(1, 11)) for o in ops]
        cases.append((ops, rng.integers(0, 256, ROM).tolist(), rng.integers(0, 256, RAM).tolist()))

    # all cases at once, cycle by cycle, against the emulator
    col = {n: k for k, n in enumerate(net.out_names)}
    S = np.array([[start(ram)[k] for k in STATE] for _, _, ram in cases], dtype=np.int64)
    prog = np.array([list(o) + list(a) for o, a, _ in cases], dtype=np.int64)
    emu = [run(o, a, r) for o, a, r in cases]
    mass = 0
    for c in range(CYCLES):
        X = np.hstack([S, prog])
        if c % 16 == 0:
            mass = max(mass, net.max_mass(X))
        Y = net.forward(X)
        S = Y[:, [col[k] for k in STATE]]
        want = np.array([[emu[i][0][c + 1][k] for k in STATE] for i in range(len(cases))])
        bad = np.flatnonzero((S != want).any(axis=1))
        assert not len(bad), f"cycle {c}: case {bad[0]} differs: {dict(zip(STATE, S[bad[0]]))} vs {dict(zip(STATE, want[bad[0]]))}"
        ex = Y[:, [col[f"e{k}"] for k in range(len(OPS))]]
        for i in range(len(cases)):
            op = emu[i][1][c]
            assert ex[i].sum() == (op is not None and op < len(OPS)), (c, i)
    assert mass < LIMIT, mass
    halted = sum(1 for e in emu if e[0][-1]["H"])
    return (f"{len(cases):,} programs ({n_demo} demo and editor-range runs, {n_random:,} random; {halted:,} halt within {CYCLES} cycles) "
            f"x {CYCLES} cycles: every register equals the emulator, max |partial sum| {mass:,}")


def main():
    net = cpu_net()
    print(f"one cycle: {net.depth} layers, {net.n_units:,} units, {net.n_weights:,} weights, {len(net.in_names)} inputs")
    print(check(net))
    (NETS / "cpu.json").write_text(json.dumps(net.to_json()))
    fixtures = []
    for name, (_, _, _, ins) in PROGRAMS.items():
        choices = [{}] + [{cell: v} for cell, (_, values, _) in ins.items() for v in values[::4]]
        for inputs in choices:
            ops, args, ram = cartridge(name, inputs)
            states, executed = run(ops, args, ram)
            fixtures.append({"program": name, "inputs": {str(k): v for k, v in inputs.items()},
                             "final": states[-1], "trace": [s["A"] for s in states[1:]]})
    states, _ = run(*custom_default())
    fixtures.append({"program": "custom", "inputs": {}, "final": states[-1], "trace": [s["A"] for s in states[1:]]})
    f = states[-1]
    assert [f[f"T{k}"] for k in range(f["NT"])] == list(range(10, 0, -1)) and f["H"], f
    (ROOT / "tests" / "cpu.json").write_text(json.dumps(fixtures))
    for name in PROGRAMS:
        states, _ = run(*cartridge(name))
        f = states[-1]
        print(f"  {name:10} halted {bool(f['H'])} after {f['CY']} instructions, tape {[f[f'T{k}'] for k in range(min(f['NT'], TAPE))]}")
    print("wrote cpu.json and tests/cpu.json")


if __name__ == "__main__":
    main()
