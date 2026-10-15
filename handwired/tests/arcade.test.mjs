// Plays Snake on the arcade page by clicking its arrows, in Chromium, Firefox and WebKit, and checks the
// screen, score and game over after every move equal the Python emulator. The page has no JavaScript.
//   node handwired/tests/arcade.test.mjs              all three engines
//   node handwired/tests/arcade.test.mjs webkit       one engine
import { chromium, firefox, webkit } from 'playwright';
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const only = process.argv[2];
const fx = JSON.parse(readFileSync(`${root}tests/cpu2.json`, 'utf8'));
const url = pathToFileURL(`${root}arcade/index.html`).href;
let failed = false;

// ---------------------------------------------------------------- static checks

// every computed value in the network part is a radio input, a constant, a copy, a sum, or a ReLU unit
function pureNetCss() {
  const html = readFileSync(`${root}arcade/index.html`, 'utf8');
  const bad = [];
  if (html.toLowerCase().includes('<script') || /\son[a-z]+=/i.test(html) || html.includes('javascript:')) bad.push('JavaScript');
  const css = html.split('/* ===== shared page styles')[0];
  let n = 0;
  for (const m of css.matchAll(/--[\w-]+:\s*([^;]+);/g)) {
    n++;
    const e = m[1].trim();
    if (/^-?\d+$/.test(e)) continue;
    const body = e.replace(/^max\(0, calc\((.*)\)\)$/, '$1').replace(/^calc\((.*)\)$/, '$1');
    if (/clamp|min\(|max\(|exp\(|round|\bmod\b/.test(body)) { bad.push(e.slice(0, 80)); continue; }
    if (/var\([^)]*\)\s*\*\s*var\(|\)\s*\*\s*\(/.test(body)) bad.push(e.slice(0, 80));
  }
  return { bad, n, kb: Math.round(html.length / 1024) };
}

// ---------------------------------------------------------------- the game

const KINDS = 8;
async function read(page, t) {
  return page.evaluate(([t, side]) => {
    const el = document.getElementById('s' + t);
    const s = getComputedStyle(el);
    const g = n => +s.getPropertyValue(n).trim();
    let screen = '';
    for (let r = 1; r <= side; r++) for (let c = 1; c <= side; c++) { const v = g(`--M${r * 9 + c}`); screen += v >= 4 ? 4 : v; }
    const shown = [...document.querySelectorAll('.disp')].filter(d => d.checkVisibility()).map(d => d.id);
    return { screen, score: g('--M0'), over: g('--M8') - 1, cy: g('--CY'), shown };
  }, [t, KINDS]);
}

async function play(browser, name) {
  const page = await browser.newPage({ viewport: { width: 1000, height: 1000 } });
  const t0 = Date.now();
  await page.goto(url);
  const load = Date.now() - t0;
  let same = 0, clicks = 0; const errors = [], ms = [];
  for (const [n, g] of fx.games.entries()) {
    if (n) await page.click(`#s${(await page.evaluate(() => [...document.querySelectorAll('.disp')].find(d => d.checkVisibility()).id)).slice(1)} button[type=reset]`);
    const start = await read(page, 0);
    let err = start.shown.join() === 's0' ? '' : `after New game the screens shown are ${start.shown}`;
    for (let t = 0; t < g.keys.length && !err; t++) {
      const c0 = Date.now();
      await page.click(`#s${t} label[for=m${t}_${g.keys[t]}]`);
      const got = await read(page, t + 1);
      ms.push(Date.now() - c0); clicks++;
      const want = g.moves[t];
      if (got.shown.join() !== `s${t + 1}`) err = `move ${t}: screens shown ${got.shown}, want s${t + 1}`;
      else if (got.screen !== want.screen || got.score !== want.score || got.over !== want.over || got.cy !== want.cy)
        err = `move ${t} (${g.keys.slice(0, t + 1)}): got ${JSON.stringify({ ...got, shown: undefined })}, want ${JSON.stringify(want)}`;
    }
    // after a crash the arrows are gone: the pad is scaled to nothing
    if (!err && g.moves.at(-1).over) {
      const t = g.keys.length;
      const box = await page.locator(`#s${t} .pad label`).first().boundingBox();
      if (box && box.width > 1) err = `game over but the arrows are still ${box.width}px wide`;
      const rest = await page.evaluate(t => {
        const d = document.getElementById('s' + t);
        return { padH: d.querySelector('.pad').getBoundingClientRect().height, left: +getComputedStyle(d.querySelector('.tools small')).opacity };
      }, t);
      if (!err && (rest.padH > 1 || rest.left !== 0)) err = `game over but the arrow pad is ${rest.padH}px tall, moves-left opacity ${rest.left}`;
    }
    if (err) errors.push(`game ${n}: ${err}`); else same++;
  }
  ms.sort((a, b) => a - b);
  // a full game's last move: the deepest turn
  const g = fx.games.find(g => g.keys.length === fx.turns);
  let last = null;
  if (g) {
    await page.click(`#s${fx.games.at(-1).keys.length} button[type=reset]`);
    for (let t = 0; t < g.keys.length - 1; t++) await page.click(`#s${t} label[for=m${t}_${g.keys[t]}]`);
    const t = g.keys.length - 1, c0 = Date.now();
    await page.click(`#s${t} label[for=m${t}_${g.keys[t]}]`);
    const got = await read(page, t + 1);
    last = Date.now() - c0;
    if (got.screen !== g.moves[t].screen) errors.push('last move of a full game differs');
  }
  await page.close();
  return { ok: same === fx.games.length && !errors.length, errors,
    msg: `games ${same}/${fx.games.length} (${clicks} moves, screen + score + game over + instruction count after each), ` +
         `load ${load} ms, click median ${ms[ms.length >> 1]} ms, max ${ms.at(-1)} ms, move ${fx.turns} ${last} ms` };
}

async function phone(browser) {
  const page = await browser.newPage({ viewport: { width: 390, height: 844 }, isMobile: browser.browserType().name() !== 'firefox' });
  await page.goto(url);
  const w0 = await page.evaluate(() => document.documentElement.scrollWidth);
  const scr = await page.evaluate(() => document.querySelector('#s0 .screen').getBoundingClientRect().bottom);
  await page.click('#s0 label[for=m0_2]');
  await page.click('details.deep summary');
  const w1 = await page.evaluate(() => document.documentElement.scrollWidth);
  await page.close();
  return { ok: w0 <= 390 && w1 <= 390 && scr <= 844,
    msg: `phone 390px: page width ${w0}px, ${w1}px with the details open; screen bottom at load ${Math.round(scr)}px of 844` };
}

// ---------------------------------------------------------------- run

const st = pureNetCss();
console.log(`arcade: ${st.kb} KB, ${st.n.toLocaleString()} declarations in the network CSS, only inputs, sums and ReLU units: ${st.bad.length ? 'FAIL' : 'ok'}`);
for (const b of st.bad.slice(0, 5)) console.log('   ' + b);
failed ||= st.bad.length > 0;
for (const [name, engine] of Object.entries({ chromium, firefox, webkit })) {
  if (only && only !== name) continue;
  const browser = await engine.launch();
  for (const test of [play, phone]) {
    const t0 = Date.now();
    const r = await test(browser, name);
    failed ||= !r.ok;
    console.log(`${name.padEnd(8)} ${r.msg}  ${((Date.now() - t0) / 1000).toFixed(1)}s  ${r.ok ? 'ok' : 'FAIL'}`);
    for (const e of (r.errors || []).slice(0, 5)) console.log('   ' + e);
  }
  await browser.close();
}
process.exit(failed ? 1 : 0);
