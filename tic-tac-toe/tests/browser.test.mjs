// Plays full games in real browsers and checks every O move and result equal the Python integer model.
// Playwright clicks the squares; the page under test contains no JavaScript.
import { chromium, firefox, webkit } from 'playwright';
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const url = pathToFileURL(root + 'index.html').href;
const games = JSON.parse(readFileSync(root + 'tests/fixtures.json', 'utf8'));
const html = readFileSync(root + 'index.html', 'utf8').toLowerCase();
let failed = html.includes('<script') || /\son[a-z]+=/.test(html) || html.includes('javascript:');
console.log(`no JavaScript in index.html: ${failed ? 'FAIL' : 'ok'}`);

const board = p => p.evaluate(() => {
  const s = getComputedStyle(document.querySelector('.app'));
  const get = n => +s.getPropertyValue(n).trim();
  return {
    x: [...Array(9).keys()].filter(i => get(`--x${i}`)),
    o: [...Array(9).keys()].filter(i => get(`--o${i}`)),
    result: get('--owin') ? 'O' : get('--xwin') ? 'X' : get('--draw') ? 'draw' : 'on',
  };
});

// replays a fixture; `press` clicks or taps square i. Returns a description of the first mismatch, or ''.
async function play(page, game, press) {
  const o = [];
  for (let n = 0; n < game.x.length; n++) {
    await press(game.x[n]);
    if (n < game.o.length) o.push(game.o[n]);
    const b = await board(page);
    const want = [...game.x.slice(0, n + 1)].sort((a, c) => a - c).join();
    if (b.x.join() !== want) return `after X ${game.x.slice(0, n + 1)}: x ${b.x}, want ${want}`;
    if (b.o.join() !== [...o].sort((a, c) => a - c).join()) return `after X ${game.x.slice(0, n + 1)}: o ${b.o}, want ${o}`;
  }
  const b = await board(page);
  if (b.result !== game.result) return `result ${b.result}, want ${game.result}`;
  // the board is locked: clicking every square changes nothing
  for (let i = 0; i < 9; i++) await press(i);
  const after = await board(page);
  if (after.x.join() !== b.x.join() || after.o.join() !== b.o.join()) return 'board not locked after game over';
  return '';
}

for (const [name, engine] of Object.entries({ chromium, firefox, webkit })) {
  const browser = await engine.launch();
  const page = await browser.newPage({ viewport: { width: 900, height: 900 } });
  await page.goto(url);
  const centre = async i => { const b = await page.locator('#c' + i).boundingBox(); return [b.x + b.width / 2, b.y + b.height / 2]; };
  const boxes = await Promise.all([...Array(9).keys()].map(centre));

  let same = 0, errors = [];
  for (const game of games) {
    await page.click('button[type=reset]');
    const err = await play(page, game, i => page.mouse.click(...boxes[i]));
    if (err) errors.push(`x ${game.x}: ${err}`); else same++;
  }
  await page.click('button[type=reset]');
  const cleared = await board(page);
  const clearOk = cleared.x.length === 0 && cleared.o.length === 0 && cleared.result === 'on';

  // one game by touch
  const touch = await browser.newPage({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: name !== 'firefox' });
  await touch.goto(url);
  const touchErr = await play(touch, games[0], i => touch.locator('#c' + i).tap());

  const ok = same === games.length && clearOk && !touchErr;
  failed ||= !ok;
  console.log(`${name.padEnd(8)} games match python ${same}/${games.length}  new game clears ${clearOk ? 'yes' : 'no'}  ` +
    `touch ${touchErr || 'ok'}  ${ok ? 'ok' : 'FAIL'}`);
  for (const e of errors.slice(0, 5)) console.log('   ' + e);
  await browser.close();
}
process.exit(failed ? 1 : 0);
