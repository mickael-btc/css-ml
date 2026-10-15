// Clicks through the Mini OS page in Chromium, Firefox and WebKit and checks the whole window-manager
// state after every click against the Python reference (handwired/tests/os.json, from scripts/mini_os.py).
// The page under test contains no JavaScript.
//   node handwired/tests/os.test.mjs
import { chromium, firefox, webkit } from 'playwright';
import { readFileSync, statSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const fx = JSON.parse(readFileSync(`${root}tests/os.json`, 'utf8'));
const file = `${root}os/index.html`;
const url = pathToFileURL(file).href;
const STATE = fx.state;
let failed = false;

// ---------------------------------------------------------------- static checks

function pureNetCss() {
  const html = readFileSync(file, 'utf8');
  const bad = [];
  if (html.toLowerCase().includes('<script') || /\son[a-z]+=/i.test(html) || html.includes('javascript:')) bad.push('JavaScript');
  const css = html.split('/* ===== shared page styles')[0];
  for (const m of css.matchAll(/--[\w-]+:\s*([^;]+);/g)) {
    const e = m[1].trim();
    if (/^-?\d+$/.test(e)) continue;                                   // a constant (radios and the start state)
    const body = e.replace(/^max\(0, calc\((.*)\)\)$/, '$1').replace(/^calc\((.*)\)$/, '$1');
    if (/clamp|min\(|max\(|exp\(|round|\bmod\b/.test(body)) { bad.push(e.slice(0, 80)); continue; }
    if (/var\([^)]*\)\s*\*\s*var\(|\)\s*\*\s*\(/.test(body)) bad.push(e.slice(0, 80));  // value times value
  }
  return bad;
}
{
  const bad = pureNetCss();
  console.log(`os: only inputs, copies, sums and ReLU units in the network CSS: ${bad.length ? 'FAIL' : 'ok'}`);
  for (const b of bad.slice(0, 5)) console.log('   ' + b);
  const kb = statSync(file).size / 1024;
  console.log(`os: page ${kb.toFixed(0)} KB ${kb < 3072 ? 'ok' : 'FAIL'}`);
  failed ||= bad.length > 0 || kb >= 3072;
}

// ---------------------------------------------------------------- helpers

const readState = page => page.evaluate(STATE => {
  const s = getComputedStyle(document.querySelector('.ui'));
  return Object.fromEntries(STATE.map(k => [k, +s.getPropertyValue(`--${k}`).trim()]));
}, STATE);

const diff = (got, want) => STATE.filter(k => got[k] !== want[k]).map(k => `${k}=${got[k]} want ${want[k]}`).join(', ');

// labels on screen that belong to a slot other than `t` (there must be none)
const strayLabels = (page, t) => page.evaluate(t => [...document.querySelectorAll('label.ev')]
  .filter(l => !l.classList.contains(`s${t}`) && getComputedStyle(l).display !== 'none').length, t);

// the desk's position and unit; under fx.phone_px wide the page uses the "phone" control layout
async function desk(page) {
  const b = await page.locator('.desk').boundingBox();
  return { x: b.x, y: b.y, u: b.width / fx.desk[0], phone: b.width < fx.phone_px };
}

const hitAt = (page, x, y) => page.evaluate(([x, y]) => {
  const e = document.elementFromPoint(x, y);
  return e ? (e.htmlFor || e.tagName + '.' + e.className) : 'nothing';
}, [x, y]);
const checkedIds = page => page.evaluate(() => [...document.querySelectorAll('.slot > input:checked')].map(i => i.id).join(','));

// clicks where a switched-off control sits (an arrow that can't move, the front window's title bar or
// inside, the front window's dock icon): no label may be under the point, and nothing may change
async function probeOff(page, d, r, before, where) {
  const errors = [];
  let n = 0;
  for (const [px, py] of (d.phone ? r.offp : r.off)) {
    const x = d.x + px * d.u, y = d.y + py * d.u;
    const hit = await hitAt(page, x, y);
    if (hit.startsWith('e')) { errors.push(`${where}: a label (${hit}) is under a switched-off control at ${px},${py}`); continue; }
    await page.mouse.click(x, y); n++;
    const got = await readState(page);
    if (diff(got, before)) errors.push(`${where}: a click on a switched-off control at ${px},${py} changed ${diff(got, before)}`);
  }
  if (r.offdock !== null) {
    const b = await page.locator(`#dock${r.offdock}`).boundingBox();
    const hit = await hitAt(page, b.x + b.width / 2, b.y + b.height / 2);
    if (hit.startsWith('e')) errors.push(`${where}: the dock icon of the front window still has a label (${hit})`);
    await page.mouse.click(b.x + b.width / 2, b.y + b.height / 2); n++;
    const got = await readState(page);
    if (diff(got, before)) errors.push(`${where}: the dock icon of the front window changed ${diff(got, before)}`);
  }
  return { errors, n };
}

// one click of event row r at slot t: at the point the Python model says, after checking that the
// label for this slot and event is what's under it
async function click(page, d, t, r) {
  const id = `e${t}_${r.ev}`;
  const at = d.phone ? r.atp : r.at;
  if (!at) { const t0 = performance.now(); await page.locator(`label[for=${id}]`).click({ timeout: 3000 }); return { t0 }; }
  const x = d.x + at[0] * d.u, y = d.y + at[1] * d.u;
  const hit = await hitAt(page, x, y);
  if (hit !== id) return { err: `under the click for ${r.what}: ${hit}, want label ${id}` };
  const t0 = performance.now();
  await page.mouse.click(x, y);
  return { t0 };
}

// plays one sequence from Reset; every `offEvery` clicks it first probes the switched-off controls.
// Returns {errors, ms[], probes}
async function play(page, seq, { wait = 0, labelsEvery = 10, offEvery = 1 } = {}) {
  const errors = [], ms = [];
  let probes = 0;
  await page.click('button[type=reset]');
  const s0 = await readState(page);
  if (diff(s0, fx.start)) errors.push(`after Reset: ${diff(s0, fx.start)}`);
  const d = await desk(page);
  let before = fx.start;
  for (let t = 0; t < seq.events.length && !errors.length; t++) {
    const r = seq.events[t];
    if (wait) await page.waitForTimeout(wait);
    if (offEvery && t % offEvery === 0) {
      const o = await probeOff(page, d, r, before, `${seq.name} before click ${t}`);
      errors.push(...o.errors); probes += o.n;
      if (o.errors.length) break;
    }
    const { err, t0 } = await click(page, d, t, r);
    if (err) { errors.push(`${seq.name} click ${t}: ${err}`); break; }
    const got = await readState(page);
    ms.push(performance.now() - t0);
    const bad = diff(got, r.state);
    if (bad) errors.push(`${seq.name} click ${t} (${r.what}): ${bad}`);
    before = r.state;
    if (t % labelsEvery === labelsEvery - 1 || t === seq.events.length - 1) {
      const stray = await strayLabels(page, t + 1);
      if (stray) errors.push(`${seq.name} after click ${t}: ${stray} labels of other slots on screen`);
    }
  }
  return { errors, ms, probes };
}

// ---------------------------------------------------------------- run

const median = a => { const b = [...a].sort((x, y) => x - y); return b.length ? b[Math.floor(b.length / 2)] : 0; };

for (const [name, engine] of Object.entries({ chromium, firefox, webkit })) {
  const browser = await engine.launch();
  const t0 = Date.now();
  const errors = [], ms = [];
  let seqOk = 0;

  // every fixture sequence, by real clicks, transitions off so each click lands on a still layout
  const ctx = await browser.newContext({ viewport: { width: 1000, height: 1500 }, reducedMotion: 'reduce' });
  const page = await ctx.newPage();
  await page.goto(url);

  // the text box: click into it, type, and check that it is not an event
  const typed = 'Hello from the test. 123';
  const area = page.locator('textarea.notes');
  await area.click();
  await page.keyboard.press(process.platform === 'darwin' ? 'Meta+A' : 'Control+A');
  await page.keyboard.type(typed);
  const afterTyping = await readState(page);
  let textOk = (await area.inputValue()) === typed && !diff(afterTyping, fx.start);
  if (!textOk) errors.push(`typing: value ${JSON.stringify(await area.inputValue())}, state ${diff(afterTyping, fx.start)}`);

  // the keyboard: after a click, arrow keys and Space must not change any click (no radio can take focus)
  let keysOk = true;
  {
    await page.click('button[type=reset]');
    const r0 = fx.sequences[0].events[0];
    const { err } = await click(page, await desk(page), 0, r0);
    const ids = await checkedIds(page);
    const active = await page.evaluate(() => document.activeElement.tagName);
    for (const k of ['ArrowRight', 'ArrowDown', 'ArrowDown', 'Space']) await page.keyboard.press(k);
    const got = await readState(page);
    // Space scrolls the page (smoothly in Chromium): let it finish, then go back to the top
    await page.waitForTimeout(600);
    await page.evaluate(() => scrollTo({ top: 0, behavior: 'instant' }));
    if (err || active === 'INPUT' || diff(got, r0.state) || (await checkedIds(page)) !== ids) {
      keysOk = false;
      errors.push(`keyboard: ${err || ''} focus on ${active}, ${diff(got, r0.state)}, checked ${ids} -> ${await checkedIds(page)}`);
    }
  }

  let probes = 0;
  for (const seq of fx.sequences) {
    const r = await play(page, seq);
    ms.push(...r.ms); probes += r.probes;
    if (r.errors.length) errors.push(...r.errors); else seqOk++;
    const v = await area.inputValue();
    if (v !== typed) { textOk = false; errors.push(`text box after ${seq.name}: ${JSON.stringify(v)}`); }
  }
  await page.close(); await ctx.close();

  // the same tour at phone width: the layout scales, so the same click points must land
  const phone = await browser.newContext({ viewport: { width: 390, height: 900 }, reducedMotion: 'reduce' });
  const pp = await phone.newPage();
  await pp.goto(url);
  const sw = await pp.evaluate(() => document.documentElement.scrollWidth);
  const phoneFit = sw <= 390;
  if (!phoneFit) errors.push(`phone: page is ${sw}px wide`);
  await pp.locator('.screen').scrollIntoViewIfNeeded();
  const pd = await desk(pp);
  if (!pd.phone) errors.push(`phone: the desk is ${pd.u * fx.desk[0]}px, not the phone layout`);
  const y0 = await pp.evaluate(() => scrollY);
  const pr = await play(pp, fx.sequences[0]);
  const pr2 = await play(pp, fx.sequences[fx.sequences.length - 1]);
  const y1 = await pp.evaluate(() => scrollY);
  if (y1 !== y0) errors.push(`phone: the page scrolled from ${y0} to ${y1} while clicking`);
  const phoneOk = !pr.errors.length && !pr2.errors.length && y1 === y0;
  if (pr.errors.length) errors.push(...pr.errors.map(e => 'phone ' + e));
  if (pr2.errors.length) errors.push(...pr2.errors.map(e => 'phone ' + e));
  // the phone controls are big enough to tap
  const small = await pp.evaluate(() => [...document.querySelectorAll('#w0 .ctl:not(.focus):not(.bodyf), #w1 .ctl.thumb, .dk')]
    .map(e => e.getBoundingClientRect()).filter(b => b.width > 0 && Math.min(b.width, b.height) < 19).length);
  if (small) errors.push(`phone: ${small} controls under 19px`);
  const monFits = await pp.evaluate(() => {
    const m = document.querySelector('.mon');
    const rows = [...m.children].filter(c => getComputedStyle(c).display !== 'none');
    return Math.max(...rows.map(c => c.getBoundingClientRect().bottom)) <= m.getBoundingClientRect().bottom;
  });
  if (!monFits) errors.push('phone: Monitor content is cut off');
  await pp.close(); await phone.close();

  // with the animations on: wait for them between clicks
  const anim = await browser.newContext({ viewport: { width: 1000, height: 1500 } });
  const ap = await anim.newPage();
  await ap.goto(url);
  const ar = await play(ap, { ...fx.sequences[2], events: fx.sequences[2].events.slice(0, 12) }, { wait: 500, labelsEvery: 4, offEvery: 0 });
  if (ar.errors.length) errors.push(...ar.errors.map(e => 'animated ' + e));
  await ap.close(); await anim.close();

  const clicks = fx.sequences.reduce((n, s) => n + s.events.length, 0);
  const ok = !errors.length;
  failed ||= !ok;
  console.log(`${name.padEnd(8)} os  sequences ${seqOk}/${fx.sequences.length} (${clicks} clicks, every state value after every click), ` +
    `${probes} switched-off clicks ignored, keyboard ${keysOk ? 'ignored' : 'FAIL'}, text box ${textOk ? 'kept' : 'LOST'}, ` +
    `phone ${sw}px + 2 sequences ${phoneOk ? 'ok' : 'FAIL'}, animated ${ar.errors.length ? 'FAIL' : 'ok'}; ` +
    `click to new state median ${median(ms).toFixed(0)} ms, max ${Math.max(...ms).toFixed(0)} ms  ${((Date.now() - t0) / 1000).toFixed(1)}s  ${ok ? 'ok' : 'FAIL'}`);
  for (const e of errors.slice(0, 8)) console.log('   ' + e);
  await browser.close();
}
process.exit(failed ? 1 : 0);
