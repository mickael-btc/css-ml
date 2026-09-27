// Plays full games in real browsers and checks every network move and result equal the Python integer model.
// Playwright clicks the columns; the page under test contains no JavaScript.
import { chromium, firefox, webkit } from 'playwright';
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const url = pathToFileURL(root + 'index.html').href;
const games = JSON.parse(readFileSync(root + 'tests/fixtures.json', 'utf8'));
const html = readFileSync(root + 'index.html', 'utf8').toLowerCase();
let failed = html.includes('<script') || /\son[a-z]+=/.test(html) || html.includes('javascript:');
console.log(`no JavaScript in index.html: ${failed ? 'FAIL' : 'ok'}`);

const COLS = 7, CELLS = 42;

// the page's final state lives on .ui, inside the deepest of the 21 nested turns
const state = p => p.evaluate(CELLS => {
  const s = getComputedStyle(document.querySelector('.ui'));
  const get = n => +s.getPropertyValue(n).trim();
  const cells = k => [...Array(CELLS).keys()].map(i => get(`--${k}${i}`)).join('');
  return {
    you: cells('H'), me: cells('A'),
    result: get('--awin') ? 'me' : get('--hwin') ? 'you' : get('--draw') ? 'draw' : 'on',
  };
}, CELLS);

// the same board, dropped in Python order: cell i = row * 7 + col, row 0 at the bottom
function expected(game, n) {
  const you = Array(CELLS).fill(0), me = Array(CELLS).fill(0), h = Array(COLS).fill(0);
  for (let k = 0; k <= n; k++) {
    you[h[game.you[k]]++ * COLS + game.you[k]] = 1;
    if (k < game.me.length) me[h[game.me[k]]++ * COLS + game.me[k]] = 1;
  }
  return { you: you.join(''), me: me.join('') };
}

// replays a fixture; `press` clicks or taps column c. Returns the first mismatch, or ''.
async function play(page, game, press) {
  for (let n = 0; n < game.you.length; n++) {
    await press(game.you[n]);
    const got = await state(page), want = expected(game, n);
    if (got.you !== want.you) return `after your move ${n + 1}: your pieces differ`;
    if (got.me !== want.me) return `after your move ${n + 1}: the network played differently`;
  }
  const end = await state(page);
  if (end.result !== game.result) return `result ${end.result}, want ${game.result}`;
  // the board is locked: pressing every column changes nothing
  for (let c = 0; c < COLS; c++) await press(c);
  const after = await state(page);
  if (after.you !== end.you || after.me !== end.me) return 'board not locked after game over';
  return '';
}

for (const [name, engine] of Object.entries({ chromium, firefox, webkit })) {
  const browser = await engine.launch();
  const page = await browser.newPage({ viewport: { width: 900, height: 1000 } });
  await page.goto(url);
  await page.evaluate(() => new Promise(requestAnimationFrame));  // Firefox can be read before its first style pass
  const fresh = await state(page);
  const centre = async c => { const b = await page.locator('#col' + c).boundingBox(); return [b.x + b.width / 2, b.y + b.height / 2]; };
  const cols = await Promise.all([...Array(COLS).keys()].map(centre));

  let same = 0, moves = 0, errors = [];
  const t0 = Date.now();
  for (const game of games) {
    await page.click('button[type=reset]');
    const err = await play(page, game, c => page.mouse.click(...cols[c]));
    if (err) errors.push(`you ${game.you}: ${err}`); else same++;
    moves += game.you.length;
  }
  const ms = (Date.now() - t0) / moves;
  await page.click('button[type=reset]');
  const cleared = await state(page);
  const clearOk = !cleared.you.includes('1') && !cleared.me.includes('1') && cleared.result === 'on';

  // the time the style engine itself takes for one move, measured inside the page
  await page.click('button[type=reset]');
  const recalc = await page.evaluate(() => {
    const t = performance.now();
    document.querySelector('#m0_3').checked = true;
    getComputedStyle(document.querySelector('.ui')).getPropertyValue('--acol');
    return performance.now() - t;
  });
  await page.click('button[type=reset]');

  // one game by touch
  const touch = await browser.newPage({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: name !== 'firefox' });
  await touch.goto(url);
  await touch.evaluate(() => new Promise(requestAnimationFrame));
  const touchErr = await play(touch, games[0], c => touch.locator('#col' + c).tap());

  const empty = !fresh.you.includes('1') && !fresh.me.includes('1') && fresh.result === 'on';
  const ok = empty && same === games.length && clearOk && !touchErr;
  failed ||= !ok;
  console.log(`${name.padEnd(8)} games match python ${same}/${games.length}  new game clears ${clearOk ? 'yes' : 'no'}  ` +
    `touch ${touchErr || 'ok'}  first move recalc ${recalc.toFixed(0)} ms  ~${ms.toFixed(0)} ms/move with Playwright  ${ok ? 'ok' : 'FAIL'}`);
  for (const e of errors.slice(0, 5)) console.log('   ' + e);
  await browser.close();
}
process.exit(failed ? 1 : 0);
