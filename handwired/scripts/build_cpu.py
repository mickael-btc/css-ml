"""Step 5 page: a CPU whose clock cycle is one ReLU network, run 128 times in pure CSS.

    .venv/bin/python handwired/scripts/build_cpu.py

128 nested cycles, like the connect-four page: `<div class="cyc">` copies the state it inherits
(--PA: var(--A) …) and `<div class="step">` runs the cycle net on it. Each step also holds one row of
the trace. The program (16 opcodes, 16 arguments) and the RAM's first values sit on .app, set by the
picked cartridge and its input chips.
"""
from common import Stylesheet, anatomy_html, fmt, load_net, page, var
from cpu import CYCLES, EDIT_ARGS, EDIT_MEM, OPS, PROGRAMS, RAM, ROM, STATE, TAPE, asm, custom_default, net_run, run

SHOW = {"HLT": "HLT", "LDI": "LDI", "LD": "LD", "ST": "ST", "ADD": "ADD", "SUB": "SUB", "ADDI": "ADDI",
        "JMP": "JMP", "JZ": "JZ", "JNZ": "JNZ", "OUT": "OUT"}
NO_ARG = {"HLT", "OUT"}
# what each instruction does, in words, for the editor's live listing. <m> is the memory cell (argument
# mod 8), <t> the line, <n> the number.
SAYS = {"HLT": "stop", "LDI": "A = <n>", "LD": "A = memory[<m>]", "ST": "memory[<m>] = A",
        "ADD": "A = A + memory[<m>]", "SUB": "A = A − memory[<m>]", "ADDI": "A = A + <n>",
        "JMP": "go to line <t>", "JZ": "if A = 0, go to line <t>", "JNZ": "if A ≠ 0, go to line <t>", "OUT": "print A"}


def eq(a, k):
    """1 when the integer a equals k, else 0 (presentation styles only)."""
    return f"clamp(0, 1 - max({a} - {k}, {k} - {a}), 1)"


def editor(css):
    """The "Write your own" cartridge: radios for 16 opcodes, 16 arguments and 8 starting memory values.
    Each radio sets an integer on .app; picking the cartridge copies them into the program."""
    ops, args, ram = custom_default()
    for i in range(ROM):
        css.register(f"eop{i}", initial=ops[i], inherits=False)
        css.register(f"earg{i}", initial=args[i], inherits=False)
    for j in range(RAM):
        css.register(f"emem{j}", initial=ram[j])  # inherits: the editor shows it
    copy = [f"--op{i}: var(--eop{i}); --arg{i}: var(--earg{i});" for i in range(ROM)]
    copy += [f"--R{j}: var(--emem{j});" for j in range(RAM)]
    css.rule(".app:has(#prog-custom:checked)", " ".join(copy))
    for i in range(ROM):
        for k in range(len(OPS)):
            css.rule(f".app:has(#eop{i}_{k}:checked)", f"--eop{i}: {k};")
        for v in range(EDIT_ARGS):
            css.rule(f".app:has(#earg{i}_{v}:checked)", f"--earg{i}: {v};")
    for j in range(RAM):
        for v in range(EDIT_MEM):
            css.rule(f".app:has(#emem{j}_{v}:checked)", f"--emem{j}: {v};")

    def radio(name, value, default):
        return f'<input type="radio" name="{name}" id="{name}_{value}"{" checked" if value == default else ""}>'

    radios = []
    lines = []
    for i in range(ROM):
        radios += [radio(f"eop{i}", k, ops[i]) for k in range(len(OPS))]
        radios += [radio(f"earg{i}", v, args[i]) for v in range(EDIT_ARGS)]
        mns = "".join(f'<span style="--k: {k}">{o}</span>' for k, o in enumerate(OPS))
        says = "".join(f'<span style="--k: {k}">' + SAYS[o].replace("<m>", '<i class="m"></i>').replace("<t>", '<i class="t"></i>')
                       .replace("<n>", '<i class="t"></i>') + "</span>" for k, o in enumerate(OPS))
        op_labels = "".join(f'<label for="eop{i}_{k}" style="--k: {k}">{o}</label>' for k, o in enumerate(OPS))
        arg_labels = "".join(f'<label for="earg{i}_{v}" style="--k: {v}">{v}</label>' for v in range(EDIT_ARGS))
        lines.append(
            f'<details class="ln" style="--o: var(--op{i}); --a: var(--arg{i})"><summary><b>{i}</b>'
            f'<span class="stack mn">{mns}</span><span class="ag"></span><span class="stack says">{says}</span></summary>'
            f'<div class="pick"><span>do</span><div class="opts ops">{op_labels}</div>'
            f'<span>with</span><div class="opts args">{arg_labels}</div></div></details>')
    mem_rows = []
    for j in range(RAM):
        radios += [radio(f"emem{j}", v, ram[j]) for v in range(EDIT_MEM)]
        mem_rows.append(f'<span>[{j}]</span><div class="opts" style="--a: var(--emem{j})">' +
                        "".join(f'<label for="emem{j}_{v}" style="--k: {v}">{v}</label>' for v in range(EDIT_MEM)) + "</div>")
    start = "".join(f'<i style="--v: var(--emem{j})"></i>' for j in range(RAM))
    panel = f"""<div class="only-custom">
<p class="edhint">Tap a line to change it: pick what it does, then its number. Lines run from the top; a jump goes to any line.</p>
<div class="ed">{"".join(lines)}
<details class="ln mem"><summary><b>m</b><span class="memsum">starting memory {start}</span></summary>
<div class="pick">{"".join(mem_rows)}</div></details>
</div>
<p class="edhint">Numbers you can pick go from 0 to 15. Memory has only 8 cells, so cell 8 is cell 0 again, 9 is 1, and so on. Sums wrap around: 255 + 1 is 0, and 0 − 1 is 255.</p>
</div>"""
    return "".join(radios), panel


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

    css.rule(".app:has(#prog-custom:checked) .progs label[for=prog-custom]", "border-color: var(--hand); color: var(--hand);")
    css.rule(".app:not(:has(#prog-custom:checked)) .only-custom", "display: none;")
    ed_radios, ed_panel = editor(css)
    n_radios = ed_radios.count("<input")
    want, _ = run(*custom_default())
    got, _ = net_run(net, *custom_default())
    assert got == want and want[-1]["H"] and [want[-1][f"T{k}"] for k in range(10)] == list(range(10, 0, -1))

    # one cycle: copy, then the net
    for k in STATE:
        css.define(C, f"P{k}", var(k))
    inputs = [f"P{k}" for k in STATE] + [f"op{i}" for i in range(ROM)] + [f"arg{i}" for i in range(ROM)]
    css.net(S, net, inputs, "c", net.out_names)

    # page
    progs = "".join(f'<input type="radio" name="prog" id="prog-{n}"{" checked" if k == 0 else ""}>' for k, n in enumerate(names + ["custom"]))
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
    panels.append(ed_panel)
    prog_labels = "".join(f'<label for="prog-{n}" class="btn">{PROGRAMS[n][0]}</label>' for n in names)
    prog_labels += '<label for="prog-custom" class="btn">Write your own</label>'

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
  <li><b>Try:</b> pick a program and change its inputs, or press <i>Write your own</i> and change any of its 16 lines. Read what it printed; below, every tick the computer made.</li>
</ul></div>
<form class="hidden-inputs">{progs}{"".join(chips_radios)}{ed_radios}</form>
<div class="progs">{prog_labels}</div>
<div class="machine">
{opening}
<div class="ui">
  <div class="panel">
    <div class="out">
      <h2>Printed</h2>
      <div class="tape">{tape}</div>
    </div>
    <div class="left">{"".join(panels)}</div>
    <div class="right">
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
  <p>Checked in numpy against a plain Python emulator: the demo programs with every input, the editor's default, 1,000 random programs of the kind the editor can make and 3,000 random programs of any kind, all 128 ticks, every register at every tick. The browser tests run the demo programs in Chromium, Firefox and WebKit, and type programs into the editor with real clicks (random ones, and edge cases: a jump to itself, writing every memory cell, sums that wrap past 255, printing more than 16 times, stopping on line 0), comparing every tick's A and the final state with an emulator written in the test.</p>
</section>
<section>
  <h2>Writing your own</h2>
  <p>The editor is {n_radios} radio buttons: for each of the 16 lines, one per instruction (11) and one per number (16), and 16 for each of the 8 starting memory cells. Each picked radio sets one integer on the page (<code>.app:has(#eop3_7:checked) {{ --eop3: 7 }}</code>), and picking <i>Write your own</i> copies those integers into the program inputs of the same network every other cartridge uses. The network does not change at all: the program is just 32 more numbers it reads.</p>
  <p>Numbers stop at 15 to keep the page small (the network itself takes any argument up to 255). Memory addresses use the number mod 8 and jumps use it mod 16, so <code>ST 9</code> writes cell 1. There is no "do nothing" instruction, but <code>ADDI 0</code> is one. Each line's text (<code>LD 3</code>, "A = memory[3]") is drawn from the same integers: the numbers with CSS counters, and the instruction name and its sentence by showing one of 11 stacked words and hiding the rest.</p>
</section>
</details>
"""
    styles = f"""
.progs {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: -10px; }}
.machine {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); column-gap: 10px; align-items: start; }}
.cyc, .step {{ display: contents; }}
.ui {{ grid-column: 1 / -1; order: -1; display: grid; gap: 14px; margin-bottom: 8px; }}
.panel {{ display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: auto 1fr; grid-template-areas: "left out" "left right";
  column-gap: 32px; align-items: start; }}
.panel > .left {{ grid-area: left; }} .panel > .out {{ grid-area: out; }} .panel > .right {{ grid-area: right; }}
/* on a phone the printed tape comes first; while you write your own program it stays pinned at the top of the screen */
@media (max-width: 720px) {{
  .panel {{ grid-template-columns: minmax(0, 1fr); grid-template-rows: none; grid-template-areas: "out" "left" "right"; row-gap: 20px; }}
  .app:has(#prog-custom:checked) .panel > .out {{ position: sticky; top: 0; z-index: 2; background: var(--bg);
    padding: 8px 0 2px; border-bottom: 1px solid var(--line); }}
  .app:has(#prog-custom:checked) .panel > .out .tape {{ margin-bottom: 6px; }}
}}
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
/* the editor */
@property --edm {{ syntax: "<integer>"; inherits: true; initial-value: 0; }}
.edhint {{ font-size: 12px; color: var(--muted); margin: 0 0 10px; }}
.ed {{ display: grid; border-top: 1px solid var(--line); margin-bottom: 10px; }}
.ln {{ border-bottom: 1px solid var(--line); --noarg: calc({eq("var(--o)", 0)} + {eq("var(--o)", 10)});
  --edm: calc(var(--a) - 8 * clamp(0, var(--a) - 7, 1)); }}
.ln > summary {{ display: grid; grid-template-columns: 20px 38px 22px 1fr 14px; gap: 6px; align-items: center; font-size: 12px;
  padding: 4px 0; cursor: pointer; list-style: none; }}
.ln > summary::-webkit-details-marker {{ display: none; }}
.ln > summary::after {{ content: "+"; color: var(--muted); text-align: right; }}
.ln[open] > summary::after {{ content: "−"; }}
.ln > summary b {{ color: var(--muted); font-weight: 400; text-align: right; }}
.ln[open] > summary {{ color: var(--hand); }}
@media (hover: hover) {{ .ln > summary:hover {{ background: var(--cell); }} }}
.stack {{ display: grid; }}
.stack > span {{ grid-area: 1 / 1; opacity: {eq("var(--o)", "var(--k)")}; }}
.mn {{ font-weight: 700; }}
.ag {{ opacity: calc(1 - var(--noarg)); }}
.ag::after, .says .t::after {{ counter-reset: v var(--a); content: counter(v); }}
.says .m::after {{ counter-reset: v var(--edm); content: counter(v); }}
.says i {{ font-style: normal; }}
.says {{ color: var(--muted); min-width: 0; }}
.pick {{ display: grid; grid-template-columns: 32px minmax(0, 1fr); gap: 6px 8px; align-items: start; padding: 2px 0 12px; font-size: 12px; }}
.pick > span {{ color: var(--muted); padding-top: 2px; }}
.opts {{ display: flex; flex-wrap: wrap; gap: 3px; }}
.opts label {{ font-size: 11px; min-width: 26px; text-align: center; border: 1px solid var(--line); border-radius: 4px; padding: 1px 4px; cursor: pointer;
  --on: {eq("var(--a)", "var(--k)")};
  border-color: color-mix(in oklab, var(--line), var(--hand) calc(var(--on) * 100%));
  color: color-mix(in oklab, var(--fg), var(--hand) calc(var(--on) * 100%)); font-weight: calc(400 + 300 * var(--on)); }}
.opts.ops label {{ --on: {eq("var(--o)", "var(--k)")}; }}
.ln .args {{ opacity: calc(1 - 0.6 * var(--noarg)); }}
.mem > summary {{ grid-template-columns: 20px 1fr 14px; }}
.memsum {{ color: var(--muted); }}
.memsum i {{ font-style: normal; color: var(--fg); margin-left: 8px; }}
.memsum i::after {{ counter-reset: v var(--v); content: counter(v); }}
/* bigger tap targets on a phone or a touch screen */
@media (max-width: 720px), (pointer: coarse) {{
  .opts {{ gap: 5px; }}
  .opts label {{ min-width: 32px; min-height: 32px; padding: 5px 6px; font-size: 12px; display: inline-flex; align-items: center; justify-content: center; box-sizing: border-box; }}
  .ln > summary {{ padding: 9px 0; }}
}}
"""
    kb = page("cpu", "Neural CPU",
              "A tiny 8-bit computer in pure CSS whose clock cycle is one ReLU network, running real programs.",
              css.render(), styles, body)
    print(f"cpu/index.html: {kb:.0f} KB, {len(css.names):,} properties")


if __name__ == "__main__":
    build()
