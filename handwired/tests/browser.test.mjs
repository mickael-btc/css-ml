// Plays and probes every handwired page in Chromium, Firefox and WebKit, and checks each one equals
// the Python reference exactly. The pages under test contain no JavaScript.
//   node handwired/tests/browser.test.mjs              all pages
//   node handwired/tests/browser.test.mjs tower        one page
import { chromium, firefox, webkit } from 'playwright';
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const repo = fileURLToPath(new URL('../..', import.meta.url));
const only = process.argv[2];
const json = p => JSON.parse(readFileSync(p, 'utf8'));
const url = folder => pathToFileURL(`${root}${folder}/index.html`).href;
let failed = false;

// ---------------------------------------------------------------- static checks

// in the step 1 pages every computed value is a radio input, a sum, or a ReLU unit
function pureNetCss(folder) {
  const html = readFileSync(`${root}${folder}/index.html`, 'utf8');
  const bad = [];
  if (html.toLowerCase().includes('<script') || /\son[a-z]+=/i.test(html) || html.includes('javascript:')) bad.push('JavaScript');
  const css = html.split('/* ===== shared page styles')[0];
  for (const m of css.matchAll(/--[\w-]+:\s*([^;]+);/g)) {
    const e = m[1].trim();
    if (/^-?\d+$/.test(e)) continue;                                   // a constant (radio inputs set 1)
    const body = e.replace(/^max\(0, calc\((.*)\)\)$/, '$1').replace(/^calc\((.*)\)$/, '$1');
    if (/clamp|min\(|max\(|exp\(|round|\bmod\b/.test(body)) { bad.push(e.slice(0, 80)); continue; }
    if (/var\([^)]*\)\s*\*\s*var\(|\)\s*\*\s*\(/.test(body)) bad.push(e.slice(0, 80));  // value times value
  }
  return bad;
}

// ---------------------------------------------------------------- pages

const pages = {};

pages['tic-tac-toe'] = async (browser, name) => {
  const games = json(`${repo}tic-tac-toe/tests/fixtures.json`);
  const page = await browser.newPage({ viewport: { width: 900, height: 900 } });
  await page.goto(url('tic-tac-toe'));
  const board = () => page.evaluate(() => {
    const s = getComputedStyle(document.querySelector('.app'));
    const get = n => +s.getPropertyValue(n).trim();
    return {
      x: [...Array(9).keys()].filter(i => get(`--x${i}`)),
      o: [...Array(9).keys()].filter(i => get(`--o${i}`)),
      result: get('--owin') ? 'O' : get('--xwin') ? 'X' : get('--draw') ? 'draw' : 'on',
    };
  });
  const boxes = await Promise.all([...Array(9).keys()].map(async i => {
    const b = await page.locator('#c' + i).boundingBox(); return [b.x + b.width / 2, b.y + b.height / 2];
  }));
  let same = 0; const errors = [];
  for (const g of games) {
    await page.click('button[type=reset]');
    let err = '';
    for (let n = 0; n < g.x.length && !err; n++) {
      await page.mouse.click(...boxes[g.x[n]]);
      const b = await board();
      const wantO = g.o.slice(0, Math.min(n + 1, g.o.length)).sort((a, c) => a - c).join();
      if (b.o.join() !== wantO) err = `after X ${g.x.slice(0, n + 1)}: o ${b.o}, want ${wantO}`;
    }
    const b = await board();
    if (!err && b.result !== g.result) err = `result ${b.result}, want ${g.result}`;
    if (!err) { for (let i = 0; i < 9; i++) await page.mouse.click(...boxes[i]); const a = await board();
      if (a.x.join() !== b.x.join()) err = 'board not locked'; }
    if (err) errors.push(`x ${g.x}: ${err}`); else same++;
  }
  await page.close();
  return { ok: same === games.length, msg: `games ${same}/${games.length}`, errors };
};

pages['connect-four'] = async browser => {
  const games = json(`${repo}connect-four/tests/fixtures.json`);
  const page = await browser.newPage({ viewport: { width: 1000, height: 1000 } });
  await page.goto(url('connect-four'));
  const read = () => page.evaluate(() => {
    const s = getComputedStyle(document.querySelector('.ui'));
    const get = n => +s.getPropertyValue(n).trim();
    return { you: [...Array(42).keys()].map(i => get(`--Y${i}`)).join(''), me: [...Array(42).keys()].map(i => get(`--M${i}`)).join(''),
      result: get('--mewin') ? 'me' : get('--youwin') ? 'you' : get('--full') ? 'draw' : 'on' };
  });
  const boxes = await Promise.all([...Array(7).keys()].map(async c => {
    const b = await page.locator('#col' + c).boundingBox(); return [b.x + b.width / 2, b.y + b.height / 2];
  }));
  let same = 0; const errors = [];
  for (const g of games) {
    await page.click('button[type=reset]');
    const you = Array(42).fill(0), me = Array(42).fill(0), h = Array(7).fill(0);
    let err = '';
    for (let n = 0; n < g.you.length && !err; n++) {
      await page.mouse.click(...boxes[g.you[n]]);
      you[h[g.you[n]]++ * 7 + g.you[n]] = 1;
      if (n < g.me.length) me[h[g.me[n]]++ * 7 + g.me[n]] = 1;
      const b = await read();
      if (b.you !== you.join('') || b.me !== me.join('')) err = `board differs after your move ${n}`;
    }
    const b = await read();
    if (!err && b.result !== g.result) err = `result ${b.result}, want ${g.result}`;
    if (!err) { for (let c = 0; c < 7; c++) await page.mouse.click(...boxes[c]); const a = await read();
      if (a.you !== b.you) err = 'board not locked'; }
    if (err) errors.push(`you ${g.you}: ${err}`); else same++;
  }
  const t0 = Date.now(); await page.click('button[type=reset]'); await page.mouse.click(...boxes[3]); await read();
  const ms = Date.now() - t0;
  await page.close();
  return { ok: same === games.length, msg: `games ${same}/${games.length}, one move ~${ms} ms`, errors };
};

pages['interpreter'] = async browser => {
  const fixtures = json(`${root}tests/interpreter.json`);
  // the trained model must also replay the original demo's games move for move, draws included
  fixtures.learned = [...fixtures.learned, ...json(`${repo}tic-tac-toe/tests/fixtures.json`)];
  const page = await browser.newPage({ viewport: { width: 1000, height: 1000 } });
  await page.goto(url('interpreter'));
  const read = () => page.evaluate(() => {
    const s = getComputedStyle(document.querySelector('.ui'));
    const get = n => +s.getPropertyValue(n).trim();
    return { x: [...Array(9).keys()].filter(i => get(`--X${i}`)), o: [...Array(9).keys()].filter(i => get(`--O${i}`)),
      result: get('--owin') ? 'O' : get('--xwin') ? 'X' : get('--draw') ? 'draw' : 'on' };
  });
  const boxes = await Promise.all([...Array(9).keys()].map(async i => {
    const b = await page.locator('#c' + i).boundingBox(); return [b.x + b.width / 2, b.y + b.height / 2];
  }));
  let same = 0, total = 0; const errors = []; let ms = 0;
  for (const [model, games] of Object.entries(fixtures)) {
    await page.click(`label[for=model-${model}]`);
    for (const g of games) {
      total++;
      await page.click('button[type=reset]');
      let err = '';
      for (let n = 0; n < g.x.length && !err; n++) {
        const t0 = Date.now();
        await page.mouse.click(...boxes[g.x[n]]);
        const b = await read();
        ms = Math.max(ms, Date.now() - t0);
        const want = g.o.slice(0, Math.min(n + 1, g.o.length)).sort((a, c) => a - c).join();
        if (b.o.join() !== want) err = `after X ${g.x.slice(0, n + 1)}: o ${b.o}, want ${want}`;
      }
      const b = await read();
      if (!err && b.result !== g.result) err = `result ${b.result}, want ${g.result}`;
      if (err) errors.push(`${model} x ${g.x}: ${err}`); else same++;
    }
  }
  await page.close();
  return { ok: same === total, msg: `games ${same}/${total} over ${Object.keys(fixtures).length} models, slowest move ${ms} ms`, errors };
};

pages['tower'] = async browser => {
  const truth = json(`${root}tests/tower.json`);
  const page = await browser.newPage({ viewport: { width: 1000, height: 1000 } });
  await page.goto(url('tower'));
  const read = () => page.evaluate(() => {
    const s = getComputedStyle(document.querySelector('.app'));
    return [0, 1, 2, 3].map(a => [+s.getPropertyValue(`--y1_${a}`), +s.getPropertyValue(`--y2_${a}`)]);
  });
  const X = [[0, 0], [0, 1], [1, 0], [1, 1]];
  let same = 0, total = 0; const errors = [];
  const compare = (name, want, got) => {
    total++;
    const ok = got.every(([a, b], k) => a === want[k] && b === want[k]);
    if (ok) same++; else errors.push(`${name}: got ${JSON.stringify(got)}, want ${want}`);
  };
  const names = Object.keys(truth);
  for (let k = 0; k < names.length; k++) {
    await page.click(`label[for=p${k}]`);
    compare(names[k], truth[names[k]], await read());
  }
  // custom N's: click random numbers, compute N here, compare both levels
  await page.click('label[for=pcustom]');
  let seed = 7; const rnd = n => (seed = (seed * 1103515245 + 12345) % 2147483648) % n;
  for (let trial = 0; trial < 8; trial++) {
    const v = [];
    for (let s = 0; s < 9; s++) {
      const weight = ![4, 5, 8].includes(s);
      const val = weight ? rnd(8) - 4 : rnd(9) - 4;
      v.push(val);
      await page.click(`label[for=c${s}_${val + 10}]`);
    }
    // slots: w11 w21 w12 w22 b1 b2 v1 v2 c
    const relu = z => Math.max(0, z);
    const want = X.map(([a, b]) => v[6] * relu(v[0] * a + v[1] * b + v[4]) + v[7] * relu(v[2] * a + v[3] * b + v[5]) + v[8]);
    compare(`custom ${v}`, want, await read());
  }
  await page.close();
  return { ok: same === total, msg: `nets ${same}/${total} (both levels, all 4 inputs)`, errors };
};

pages['unfrozen'] = async browser => {
  const fx = json(`${root}tests/unfrozen.json`);
  const page = await browser.newPage({ viewport: { width: 1000, height: 1000 } });
  await page.goto(url('unfrozen'));
  const H = 4;
  const read = () => page.evaluate(([E, H]) => {
    const s = getComputedStyle(document.querySelector('.app'));
    const g = n => +s.getPropertyValue(n).trim();
    const w = [], err = [];
    for (let e = 0; e <= E; e++) {
      const o = {};
      for (const k of 'abcv') o[k] = [...Array(H).keys()].map(i => g(`--${k}${e}_${i}`));
      o.d = g(`--d${e}`);
      w.push(o); err.push(g(`--err${e}`));
    }
    return { w, err };
  }, [fx.epochs, H]);
  // the test sets radios directly; the page itself has no script
  const setUp = (set) => page.evaluate(({ t, start }) => {
    document.getElementById(`start-${start}`).checked = true;
    t.forEach((v, p) => { document.getElementById((v > 0 ? 'r' : v < 0 ? 'b' : 'n') + p).checked = true; });
  }, set);
  let same = 0; const errors = []; let ms = 0;
  for (const set of fx.sets) {
    await setUp(set);
    const t0 = Date.now(); const got = await read(); ms = Math.max(ms, Date.now() - t0);
    const key = w => JSON.stringify(['a', 'b', 'c', 'v', 'd'].map(k => w[k]));
    const bad = got.w.findIndex((w, e) => key(w) !== key(set.w[e]));
    const badErr = JSON.stringify(got.err) !== JSON.stringify(set.errors);
    if (bad < 0 && !badErr) same++;
    else errors.push(`${set.name} from ${set.start}: ` + (bad >= 0 ? `epoch ${bad} weights ${JSON.stringify(got.w[bad])} want ${JSON.stringify(set.w[bad])}` : `errors ${got.err} want ${set.errors}`));
  }
  // and by hand: clear, paint two points red with real clicks, one blue, remove one
  await page.click('button[type=reset]');
  const clickPt = async p => { const b = await page.locator('#pt' + p).boundingBox(); await page.mouse.click(b.x + b.width / 2, b.y + b.height / 2); };
  await clickPt(27); await clickPt(28); await page.click('label[for=br-b]'); await clickPt(0); await clickPt(0); await clickPt(63);
  const labels = await page.evaluate(() => { const s = getComputedStyle(document.querySelector('.app'));
    return [27, 28, 0, 63].map(p => s.getPropertyValue(`--t${p}`).trim()).join(); });
  const clicksOk = labels === '1,1,0,-1';
  if (!clicksOk) errors.push(`clicked labels ${labels}, want 1,1,0,-1`);
  await page.close();
  return { ok: same === fx.sets.length && clicksOk, msg: `sets ${same}/${fx.sets.length} (every weight, every epoch), clicks ${clicksOk ? 'ok' : 'FAIL'}, slowest recompute ${ms} ms`, errors };
};

// ---------------------------------------------------------------- run

const selected = Object.keys(pages).filter(p => !only || p === only);
for (const folder of selected) {
  if (['tic-tac-toe', 'connect-four', 'interpreter'].includes(folder)) {
    const bad = pureNetCss(folder);
    console.log(`${folder}: only inputs, sums and ReLU units in the CSS: ${bad.length ? 'FAIL' : 'ok'}`);
    for (const b of bad.slice(0, 5)) console.log('   ' + b);
    failed ||= bad.length > 0;
  }
}
for (const [name, engine] of Object.entries({ chromium, firefox, webkit })) {
  const browser = await engine.launch();
  for (const folder of selected) {
    const t0 = Date.now();
    const r = await pages[folder](browser, name);
    failed ||= !r.ok;
    console.log(`${name.padEnd(8)} ${folder.padEnd(13)} ${r.msg}  ${((Date.now() - t0) / 1000).toFixed(1)}s  ${r.ok ? 'ok' : 'FAIL'}`);
    for (const e of (r.errors || []).slice(0, 5)) console.log('   ' + e);
  }
  await browser.close();
}
process.exit(failed ? 1 : 0);
