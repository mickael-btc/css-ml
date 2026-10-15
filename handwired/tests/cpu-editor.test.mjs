// Types programs into the Neural CPU's "Write your own" editor with real clicks, in Chromium, Firefox and
// WebKit, and checks every tick's A and the final registers, memory and tape against an emulator written
// here (a line-for-line port of step() in handwired/scripts/cpu.py). The page under test has no JavaScript.
//   node handwired/tests/cpu-editor.test.mjs
import { chromium, firefox, webkit } from 'playwright';
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const url = pathToFileURL(`${root}cpu/index.html`).href;
let failed = false;

// ---------------------------------------------------------------- the emulator (port of cpu.py)

const ROM = 16, RAM = 8, TAPE = 16, CYCLES = 128;
const OPS = ['HLT', 'LDI', 'LD', 'ST', 'ADD', 'SUB', 'ADDI', 'JMP', 'JZ', 'JNZ', 'OUT'];
const OP = Object.fromEntries(OPS.map((o, k) => [o, k]));
const STATE = ['PC', 'A', 'H', 'CY', 'NT', ...[...Array(RAM).keys()].map(j => `R${j}`), ...[...Array(TAPE).keys()].map(k => `T${k}`)];
const mod = (a, n) => ((a % n) + n) % n;

function step(s0, ops, args) {
  const s = { ...s0 };
  if (s.H) return s;
  const op = ops[s.PC], arg = args[s.PC];
  const addr = arg % RAM, target = arg % ROM;
  let nxt = (s.PC + 1) % ROM;
  s.CY += 1;
  if (op === OP.HLT) { s.H = 1; return s; }
  if (op === OP.LDI) s.A = arg;
  else if (op === OP.LD) s.A = s[`R${addr}`];
  else if (op === OP.ST) s[`R${addr}`] = s.A;
  else if (op === OP.ADD) s.A = mod(s.A + s[`R${addr}`], 256);
  else if (op === OP.SUB) s.A = mod(s.A - s[`R${addr}`], 256);
  else if (op === OP.ADDI) s.A = mod(s.A + arg, 256);
  else if (op === OP.OUT) { if (s.NT < TAPE) s[`T${s.NT}`] = s.A; s.NT += 1; }
  if (op === OP.JMP || (op === OP.JZ && s.A === 0) || (op === OP.JNZ && s.A !== 0)) nxt = target;
  s.PC = nxt;
  return s;
}

function run(ops, args, ram) {
  let s = Object.fromEntries(STATE.map(k => [k, 0]));
  ram.forEach((v, j) => { s[`R${j}`] = v; });
  const trace = [];
  for (let c = 0; c < CYCLES; c++) { s = step(s, ops, args); trace.push(s.A); }
  return { trace, final: s };
}

// ---------------------------------------------------------------- the programs

// lines as [mnemonic, argument]; missing lines are HLT 0, missing memory 0
function prog(name, lines, ram = []) {
  const ops = [], args = [];
  for (let i = 0; i < ROM; i++) { const [o, a] = lines[i] || ['HLT', 0]; ops.push(OP[o]); args.push(a); }
  return { name, ops, args, ram: [...Array(RAM).keys()].map(j => ram[j] || 0) };
}

const programs = [
  prog('default countdown (as loaded)', [['LDI', 10], ['OUT', 0], ['SUB', 0], ['JNZ', 1], ['HLT', 0]], [1]),
  prog('jump to itself', [['LDI', 7], ['JMP', 1], ['OUT', 0]], [3, 3]),
  // fills all 16 lines with no HLT: PC runs off line 15 back to line 0; ST 9.. use the address mod 8
  prog('write every memory cell', [...Array(8).keys()].flatMap(k => [['LDI', 15 - k], ['ST', k % 2 ? k + 8 : k]]), [9, 9, 9, 9, 9, 9, 9, 9]),
  prog('overflow past 255 and below 0', [['LD', 0], ['ADD', 0], ['ST', 0], ['OUT', 0], ['JNZ', 0],
    ['LDI', 0], ['SUB', 1], ['OUT', 0], ['ADDI', 15], ['OUT', 0], ['ADD', 2], ['OUT', 0], ['ADDI', 15], ['ADDI', 15], ['ADD', 2], ['OUT', 0]], [15, 1, 15]),
  prog('printing more than 16 times', [['LDI', 1], ['OUT', 9], ['ADDI', 1], ['JMP', 1]]),
  prog('halt on line 0', [['HLT', 13], ['OUT', 5], ['LDI', 9]], [15, 14, 13, 12, 11, 10, 9, 8]),
  // no NOP instruction: ADDI 0, a JZ that doesn't jump, a JNZ that doesn't, a JMP to the next line
  prog('do-nothing lines', [['LDI', 4], ['ADDI', 0], ['JZ', 0], ['JMP', 4], ['JNZ', 15], ['OUT', 12], ['SUB', 0], ['JNZ', 15],
    ['JZ', 10], ['OUT', 0], ['OUT', 0], ['HLT', 0], ['OUT', 0], ['OUT', 0], ['OUT', 0], ['ADDI', 1]], [4]),
];
let seed = 20261015;
const rnd = n => (seed = (seed * 1103515245 + 12345) % 2147483648) % n;
for (let r = 0; r < 5; r++) {
  const lines = [...Array(ROM).keys()].map(() => {
    let o = rnd(OPS.length);
    if (o === 0 && rnd(4)) o = 1 + rnd(OPS.length - 1);  // fewer halts, longer runs
    return [OPS[o], rnd(16)];
  });
  programs.push(prog(`random #${r + 1}`, lines, [...Array(RAM).keys()].map(() => rnd(16))));
}

// ---------------------------------------------------------------- static check (same rule as browser.test.mjs)

function pureNetCss() {
  const html = readFileSync(`${root}cpu/index.html`, 'utf8');
  const bad = [];
  if (html.toLowerCase().includes('<script') || /\son[a-z]+=/i.test(html) || html.includes('javascript:')) bad.push('JavaScript');
  const css = html.split('/* ===== shared page styles')[0];
  for (const m of css.matchAll(/--[\w-]+:\s*([^;]+);/g)) {
    const e = m[1].trim();
    if (/^-?\d+$/.test(e)) continue;
    const body = e.replace(/^max\(0, calc\((.*)\)\)$/, '$1').replace(/^calc\((.*)\)$/, '$1');
    if (/clamp|min\(|max\(|exp\(|round|\bmod\b/.test(body)) { bad.push(e.slice(0, 80)); continue; }
    if (/var\([^)]*\)\s*\*\s*var\(|\)\s*\*\s*\(/.test(body)) bad.push(e.slice(0, 80));
  }
  return bad;
}

// ---------------------------------------------------------------- the browser

async function test(browser) {
  const page = await browser.newPage({ viewport: { width: 1000, height: 1000 } });
  await page.goto(url);
  const errors = [];
  const read = () => page.evaluate(STATE => {
    const trace = [...document.querySelectorAll('.step')].map(e => +getComputedStyle(e).getPropertyValue('--A'));
    const s = getComputedStyle(document.querySelector('.ui'));
    const app = getComputedStyle(document.querySelector('.app'));
    const g = (st, n) => +st.getPropertyValue(n);
    return { trace, final: Object.fromEntries(STATE.map(k => [k, g(s, `--${k}`)])),
      ops: [...Array(16).keys()].map(i => g(app, `--op${i}`)), args: [...Array(16).keys()].map(i => g(app, `--arg${i}`)) };
  }, STATE);

  await page.click('label[for=prog-custom]');
  for (const sum of await page.locator('.ed .ln > summary').all()) await sum.click();  // open every line and the memory
  // what the page holds now: the radios' defaults
  const cur = await page.evaluate(() => {
    const val = n => +[...document.querySelectorAll(`input[name=${n}]`)].find(i => i.checked).id.split('_')[1];
    return { ops: [...Array(16).keys()].map(i => val(`eop${i}`)), args: [...Array(16).keys()].map(i => val(`earg${i}`)),
      ram: [...Array(8).keys()].map(j => val(`emem${j}`)) };
  });
  let clicks = 0, slowest = 0;
  const pick = async (id) => {
    const t0 = Date.now();
    await page.click(`label[for=${id}]`);
    await page.evaluate(() => getComputedStyle(document.querySelectorAll('.step')[127]).getPropertyValue('--A'));
    slowest = Math.max(slowest, Date.now() - t0); clicks++;
  };
  let same = 0;
  for (const p of programs) {
    for (let i = 0; i < ROM; i++) {
      if (cur.ops[i] !== p.ops[i]) { await pick(`eop${i}_${p.ops[i]}`); cur.ops[i] = p.ops[i]; }
      if (cur.args[i] !== p.args[i]) { await pick(`earg${i}_${p.args[i]}`); cur.args[i] = p.args[i]; }
    }
    for (let j = 0; j < RAM; j++) if (cur.ram[j] !== p.ram[j]) { await pick(`emem${j}_${p.ram[j]}`); cur.ram[j] = p.ram[j]; }
    const want = run(p.ops, p.args, p.ram);
    const got = await read();
    let err = '';
    if (got.ops.join() !== p.ops.join() || got.args.join() !== p.args.join()) err = `program on the page ${got.ops}/${got.args}`;
    const badFinal = STATE.filter(k => got.final[k] !== want.final[k]);
    const badTick = got.trace.findIndex((a, c) => a !== want.trace[c]);
    if (!err && badFinal.length) err = `final ${badFinal.map(k => `${k}=${got.final[k]} want ${want.final[k]}`).join(', ')}`;
    if (!err && badTick >= 0) err = `tick ${badTick + 1}: A=${got.trace[badTick]} want ${want.trace[badTick]}`;
    // the live listing reads like code: line number, mnemonic, argument (the stacked mnemonics are all in the text,
    // so check that the visible one is the right one by its opacity)
    if (!err) {
      const shown = await page.evaluate(() => [...document.querySelectorAll('.ed .ln:not(.mem) .mn')].map(m =>
        [...m.children].find(c => +getComputedStyle(c).opacity === 1)?.textContent));
      const expect = p.ops.map(o => OPS[o]);
      if (shown.join() !== expect.join()) err = `listing shows ${shown}, want ${expect}`;
    }
    if (err) errors.push(`${p.name}: ${err}`); else same++;
    if (p === programs[0] || p.name === 'printing more than 16 times')
      console.log(`   ${p.name}: tape ${STATE.filter(k => k[0] === 'T').slice(0, Math.min(want.final.NT, TAPE)).map(k => want.final[k])}, NT ${want.final.NT}, halted ${want.final.H}`);
  }
  // the editor keeps its program while another cartridge plays
  await page.click('label[for=prog-fibonacci]');
  const fib = await read();
  await page.click('label[for=prog-custom]');
  const back = await read();
  const last = programs[programs.length - 1], wantBack = run(last.ops, last.args, last.ram);
  const keeps = back.trace.join() === wantBack.trace.join() && fib.final.T6 === 8;
  if (!keeps) errors.push('switching to Fibonacci and back changed the editor program');
  // one recompute measured inside the page: flip a radio, force style on the last tick
  const inPage = await page.evaluate(() => {
    const r = [];
    for (const v of [3, 9, 3, 9, 3]) { const t = performance.now(); document.getElementById('earg1_' + v).checked = true;
      getComputedStyle(document.querySelectorAll('.step')[127]).getPropertyValue('--A'); document.body.offsetHeight; r.push(performance.now() - t); }
    return Math.round(r.sort((a, b) => a - b)[2]);
  });
  await page.close();
  // phone width, editor open
  const phone = await browser.newPage({ viewport: { width: 390, height: 844 } });
  await phone.goto(url);
  await phone.click('label[for=prog-custom]');
  for (const sum of await phone.locator('.ed .ln > summary').all()) await sum.click();
  const sw = await phone.evaluate(() => document.documentElement.scrollWidth);
  await phone.close();
  if (sw > 390) errors.push(`phone: scrollWidth ${sw} > 390`);
  const ok = same === programs.length && keeps && sw <= 390;
  return { ok, errors, msg: `programs ${same}/${programs.length} typed in with ${clicks} clicks (every tick's A, final registers, RAM, tape, listing), ` +
    `keeps program ${keeps ? 'ok' : 'FAIL'}, phone scrollWidth ${sw}, slowest click+recompute ${slowest} ms, in-page recompute ~${inPage} ms` };
}

const bad = pureNetCss();
console.log(`cpu: only inputs, sums and ReLU units in the CSS: ${bad.length ? 'FAIL' : 'ok'}`);
for (const b of bad.slice(0, 5)) console.log('   ' + b);
failed ||= bad.length > 0;
for (const [name, engine] of Object.entries({ chromium, firefox, webkit })) {
  const browser = await engine.launch();
  const t0 = Date.now();
  let r;
  try { r = await test(browser); } catch (e) { r = { ok: false, msg: 'crashed', errors: [String(e)] }; }
  failed ||= !r.ok;
  console.log(`${name.padEnd(8)} cpu editor  ${r.msg}  ${((Date.now() - t0) / 1000).toFixed(1)}s  ${r.ok ? 'ok' : 'FAIL'}`);
  for (const e of r.errors.slice(0, 8)) console.log('   ' + e);
  await browser.close();
}
process.exit(failed ? 1 : 0);
