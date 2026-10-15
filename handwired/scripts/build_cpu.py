"""Step 5 page: a CPU whose clock cycle is one ReLU network, run 128 times in pure CSS.

    .venv/bin/python handwired/scripts/build_cpu.py

128 nested cycles, like the connect-four page: `<div class="cyc">` copies the state it inherits
(--PA: var(--A) …) and `<div class="step">` runs the cycle net on it. Each step also holds one row of
the trace. The program (16 opcodes, 16 arguments) and the RAM's first values sit on .app, set by the
picked cartridge and its input chips.
"""
from common import Stylesheet, anatomy_html, fmt, load_net, page, var
from cpu import CYCLES, OPS, PROGRAMS, RAM, ROM, STATE, TAPE, asm

SHOW = {"HLT": "HLT", "LDI": "LDI", "LD": "LD", "ST": "ST", "ADD": "ADD", "SUB": "SUB", "ADDI": "ADDI",
        "JMP": "JMP", "JZ": "JZ", "JNZ": "JNZ", "OUT": "OUT"}
NO_ARG = {"HLT", "OUT"}


def build():
    net = load_net("cpu")
    css = Stylesheet()
    C, S, U = ".cyc", ".step", ".ui"

    # the cartridge: program and first RAM values on .app
    for i in range(ROM):
        css.register(f"op{i}")
        css.register(f"arg{i}")
    names = list(PROGRAMS)
    for n, name in enumerate(names):
        title, listing, ram, ins = PROGRAMS[name]
        ops, args = asm([l for l, _ in listing])
        decl = [f"--op{i}: {o}; --arg{i}: {a};" for i, (o, a) in enumerate(zip(ops, args))]
        ram = list(ram) + [0] * (RAM - len(ram))
        for j in range(RAM):
            decl.append(f"--R{j}: {var(f'in{name}_{j}') if j in ins else ram[j]};")
        css.rule(f".app:has(#prog-{name}:checked)", " ".join(decl))
        for j, (label, values, default) in ins.items():
            css.register(f"in{name}_{j}", initial=default)
            for v in values:
                css.rule(f".app:has(#i{name}_{j}_{v}:checked)", f"--in{name}_{j}: {v};")
                css.rule(f".app:has(#i{name}_{j}_{v}:checked) .chips label[for=i{name}_{j}_{v}]",
                         "border-color: var(--hand); color: var(--hand);")
        css.rule(f".app:has(#prog-{name}:checked) .progs label[for=prog-{name}]", "border-color: var(--hand); color: var(--hand);")
        css.rule(f".app:not(:has(#prog-{name}:checked)) .only-{name}", "display: none;")

    # one cycle: copy, then the net
    for k in STATE:
        css.define(C, f"P{k}", var(k))
    inputs = [f"P{k}" for k in STATE] + [f"op{i}" for i in range(ROM)] + [f"arg{i}" for i in range(ROM)]
    css.net(S, net, inputs, "c", net.out_names)

    # page
    progs = "".join(f'<input type="radio" name="prog" id="prog-{n}"{" checked" if k == 0 else ""}>' for k, n in enumerate(names))
    chips_radios, panels = [], []
    for name in names:
        title, listing, ram, ins = PROGRAMS[name]
        rows = "".join(f'<tr><td class="n">{i}</td><td><code>{l}</code></td><td>{c}</td></tr>' for i, (l, c) in enumerate(listing))
        chip_rows = ""
        for j, (label, values, default) in ins.items():
            chips_radios += [f'<input type="radio" name="i{name}_{j}" id="i{name}_{j}_{v}"{" checked" if v == default else ""}>' for v in values]
            chip_rows += (f'<div class="inrow"><span>{label} =</span><div class="chips">' +
                          "".join(f'<label for="i{name}_{j}_{v}">{v}</label>' for v in values) + "</div></div>")
        panels.append(f'<div class="only-{name}">{chip_rows}<table class="listing"><tr><th class="n">#</th><th>instruction</th><th>what it does</th></tr>{rows}</table></div>')
    prog_labels = "".join(f'<label for="prog-{n}" class="btn">{PROGRAMS[n][0]}</label>' for n in names)

    opspans = "".join(f'<span class="mn" style="opacity: var(--e{k})">{SHOW[o]}</span>' for k, o in enumerate(OPS))
    noarg = " + ".join(f"var(--e{OPS.index(o)})" for o in NO_ARG)
    row = (f'<div class="row" style="--ran: calc({" + ".join(f"var(--e{k})" for k in range(len(OPS)))}); --noarg: calc({noarg})">'
           f'<b class="cy"></b><span class="pc"></span><span class="ins">{opspans}</span><span class="arg"></span><span class="acc"></span></div>')
    opening = "\n".join(f'<div class="cyc"><div class="step">{row}' for _ in range(CYCLES))
    closing = "</div></div>" * CYCLES
    tape = "".join(f'<i style="--k: {k}; --v: var(--T{k})"></i>' for k in range(TAPE))
    ramcells = "".join(f'<i style="--v: var(--R{j})"><small>{j}</small></i>' for j in range(RAM))

    body = f"""
<header>
  <h1>A computer made of neurons</h1>
  <p class="lead">A tiny 8-bit computer where one tick of the clock is one neural network. It runs real programs, in CSS, with 0 bytes of JavaScript.</p>
</header>
<div class="explain"><ul>
  <li><b>The computer:</b> one number in its hand (A), 8 numbers of memory, a list of 16 instructions, and a printer. Like a 1970s kit computer.</li>
  <li><b>One tick is one network</b> of {fmt(net.n_units)} neurons: read the next instruction, do it (add, subtract, store, jump…), move on. The page runs that same network {CYCLES} times in a row.</li>
  <li><b>The program is data</b>, like the cartridge on the <a href="../interpreter/">Swap the brain</a> page: the network never changes, the program does.</li>
  <li><b>Try:</b> pick a program, change its inputs, and read what it printed. Below, every tick the computer made.</li>
</ul></div>
<form class="hidden-inputs">{progs}{"".join(chips_radios)}</form>
<div class="progs">{prog_labels}</div>
<div class="machine">
{opening}
<div class="ui">
  <div class="panel">
    <div class="left">{"".join(panels)}</div>
    <div class="right">
      <h2>Printed</h2>
      <div class="tape">{tape}</div>
      <div class="regs">
        <div><span>A</span><b style="--v: var(--A)"></b></div>
        <div><span>next instruction</span><b style="--v: var(--PC)"></b></div>
        <div><span>instructions run</span><b style="--v: var(--CY)"></b></div>
        <div class="state"><span class="on">halted</span><span class="off">still running after {CYCLES} ticks</span></div>
      </div>
      <h2>Memory</h2>
      <div class="ram">{ramcells}</div>
    </div>
  </div>
  <h2 class="tr">Every tick</h2>
</div>
{closing}
</div>
<details class="deep"><summary>How it works (for the curious)</summary>
<section>
  <h2>The instructions</h2>
  <div class="scroll"><table>
    <tr><th>Code</th><th>Does</th><th>Code</th><th>Does</th></tr>
    <tr><td><code>HLT</code></td><td>stop</td><td><code>ADDI n</code></td><td>A = A + n</td></tr>
    <tr><td><code>LDI n</code></td><td>A = n</td><td><code>JMP t</code></td><td>go to instruction t</td></tr>
    <tr><td><code>LD m</code></td><td>A = memory[m]</td><td><code>JZ t</code></td><td>go to t if A is 0</td></tr>
    <tr><td><code>ST m</code></td><td>memory[m] = A</td><td><code>JNZ t</code></td><td>go to t if A is not 0</td></tr>
    <tr><td><code>ADD m</code></td><td>A = A + memory[m]</td><td><code>OUT</code></td><td>print A</td></tr>
    <tr><td><code>SUB m</code></td><td>A = A − memory[m]</td><td></td><td>numbers wrap around at 256</td></tr>
  </table></div>
</section>
<section>
  <h2>One tick, as ReLUs</h2>
  <div class="scroll"><table>
    <tr><th>Part</th><th>How</th></tr>
    <tr><td>fetch</td><td><code>[PC = i]</code> for every line i, each gating that line's opcode and argument: <code>relu(op_i − 15·(1 − [PC = i]))</code></td></tr>
    <tr><td>decode</td><td><code>[op = k]</code> for each opcode, AND not halted</td></tr>
    <tr><td>address</td><td><code>arg mod 8 = arg − 8·round(relu(2·arg − 7) / 16)</code>, one integer unit</td></tr>
    <tr><td>ALU</td><td><code>A + m</code>, <code>A − m</code>, <code>A + n</code>, each wrapped with one comparison: <code>s − 256·[s ≥ 256]</code></td></tr>
    <tr><td>write</td><td>the new A is the result whose opcode is on, gated; the same for each memory cell and the printer's next slot</td></tr>
    <tr><td>next PC</td><td>the jump target, PC + 1, or PC again when halted</td></tr>
  </table></div>
  {anatomy_html(net)}
  <p>Checked in numpy against a plain Python emulator: the demo programs with every input, and 3,000 random programs, all 128 ticks, every register at every tick. The browser test runs the demo programs in Chromium, Firefox and WebKit and compares every tick's A and the final state.</p>
</section>
</details>
"""
    styles = f"""
.progs {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: -10px; }}
.machine {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); column-gap: 10px; align-items: start; }}
.cyc, .step {{ display: contents; }}
.ui {{ grid-column: 1 / -1; order: -1; display: grid; gap: 14px; margin-bottom: 8px; }}
.panel {{ display: grid; grid-template-columns: 1fr 1fr; gap: 32px; align-items: start; }}
@media (max-width: 720px) {{ .panel {{ grid-template-columns: 1fr; gap: 20px; }} }}
.ui h2 {{ font-size: 14px; margin: 0 0 8px; }}
.ui h2.tr {{ margin: 6px 0 0; border-top: 1px solid var(--line); padding-top: 16px; }}
.inrow {{ display: grid; grid-template-columns: 40px 1fr; gap: 8px; align-items: center; font-size: 13px; margin-bottom: 8px; }}
.inrow span {{ color: var(--muted); }}
.chips {{ display: flex; gap: 2px; flex-wrap: wrap; }}
.chips label {{ font-size: 11px; min-width: 24px; text-align: center; border: 1px solid var(--line); border-radius: 4px; padding: 1px 3px; cursor: pointer; }}
.listing td, .listing th {{ padding: 3px 10px 3px 0; font-size: 12px; }}
.listing td:last-child {{ color: var(--muted); }}
.tape {{ display: grid; grid-template-columns: repeat(8, 1fr); gap: 4px; margin-bottom: 16px; }}
.tape i {{ font-style: normal; text-align: center; border: 1px solid var(--line); border-radius: 4px; padding: 6px 0; font-size: 16px; font-weight: 700;
  color: var(--hand); opacity: clamp(0.15, var(--NT) - var(--k), 1); }}
.tape i::after, .regs b::after, .ram i::after, .row .acc::after {{ counter-reset: v var(--v); content: counter(v); }}
.regs {{ display: grid; gap: 4px; font-size: 13px; margin-bottom: 16px; }}
.regs div {{ display: flex; justify-content: space-between; border-bottom: 1px solid var(--line); padding: 3px 0; }}
.regs span {{ color: var(--muted); }}
.regs .state {{ display: grid; }}
.regs .state span {{ grid-area: 1 / 1; color: var(--fg); }}
.regs .state .on {{ opacity: var(--H); }}
.regs .state .off {{ opacity: calc(1 - var(--H)); }}
.ram {{ display: grid; grid-template-columns: repeat(8, 1fr); gap: 4px; }}
.ram i {{ font-style: normal; text-align: center; border: 1px solid var(--line); border-radius: 4px; padding: 4px 0; font-size: 14px; position: relative; }}
.ram small {{ position: absolute; top: -14px; left: 0; right: 0; font-size: 9px; color: var(--muted); }}
.row {{ display: grid; grid-template-columns: 30px 22px 40px 1fr 30px; gap: 4px; font-size: 11px; align-items: center;
  height: calc(var(--ran) * 19px); overflow: hidden; border-bottom: calc(var(--ran) * 1px) solid var(--line);
  background: color-mix(in oklab, transparent, var(--hand) calc(var(--e{OPS.index("OUT")}) * 14%)); }}
.row .cy {{ color: var(--muted); font-weight: 400; }}
.row .cy::after {{ counter-reset: v var(--CY); content: "#" counter(v); }}
.row .pc::after {{ counter-reset: v var(--xpc); content: counter(v); color: var(--muted); }}
.row .ins {{ display: grid; }}
.row .mn {{ grid-area: 1 / 1; }}
.row .arg {{ opacity: calc(1 - var(--noarg)); }}
.row .arg::after {{ counter-reset: v var(--xarg); content: counter(v); }}
.row .acc {{ text-align: right; color: var(--hand); --v: var(--A); }}
.explain a {{ color: var(--fg); }}
"""
    kb = page("cpu", "Neural CPU",
              "A tiny 8-bit computer in pure CSS whose clock cycle is one ReLU network, running real programs.",
              css.render(), styles, body)
    print(f"cpu/index.html: {kb:.0f} KB, {len(css.names):,} properties")


if __name__ == "__main__":
    build()
