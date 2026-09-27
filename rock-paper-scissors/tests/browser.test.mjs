// Plays every fixture in real browsers and checks each CSS move and the final score equal the Python model.
// Playwright clicks the labels; the page under test contains no JavaScript.
import { chromium, firefox, webkit } from 'playwright';
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const url = pathToFileURL(root + 'index.html').href;
const fixtures = JSON.parse(readFileSync(root + 'tests/fixtures.json', 'utf8'));
const html = readFileSync(root + 'index.html', 'utf8').toLowerCase();
let failed = html.includes('<script') || /\son[a-z]+=/.test(html) || html.includes('javascript:');
console.log(`no JavaScript in index.html: ${failed ? 'FAIL' : 'ok'}`);

const read = (p, names) => p.evaluate(ns => {
  const s = getComputedStyle(document.querySelector('.app'));
  return ns.map(n => +s.getPropertyValue(n).trim());
}, names);

// the CSS's move in round r, read before the player's move in round r is clicked
const cssMove = async (p, r) => {
  const [a0, a1, a2] = await read(p, [`--a${r}_0`, `--a${r}_1`, `--a${r}_2`]);
  return a0 + a1 + a2 === 1 ? a1 + 2 * a2 : -1;
};

async function play(page, fx, tap) {
  await (tap ? page.locator('button[type=reset]').tap() : page.click('button[type=reset]'));
  let same = 0;
  for (let r = 0; r < fx.moves.length; r++) {
    same += (await cssMove(page, r)) === fx.ai[r];
    const label = page.locator(`label[for="r${r}m${fx.moves[r]}"]`);  // only visible in its own round
    await (tap ? label.tap() : label.click());
  }
  const [you, draws, cpu, played] = await read(page, ['--you', '--draws', '--cpu', '--played']);
  const ok = same === fx.moves.length && played === fx.moves.length &&
    [you, draws, cpu].join() === fx.tally.join();
  return { ok, same, tally: [you, draws, cpu] };
}

for (const [name, engine] of Object.entries({ chromium, firefox, webkit })) {
  const browser = await engine.launch();
  const page = await browser.newPage({ viewport: { width: 900, height: 900 } });
  await page.goto(url);
  const start = Date.now();
  let passed = 0;
  for (const fx of fixtures) {
    const res = await play(page, fx, false);
    passed += res.ok;
    if (!res.ok) console.log(`  ${name} "${fx.name}": ${res.same}/${fx.moves.length} moves match, ` +
      `score ${res.tally} vs python ${fx.tally}`);
  }
  const ms = Math.round((Date.now() - start) / (fixtures.length * fixtures[0].moves.length));

  // a finished game hides every move
  const visible = await page.locator('.move label:visible').count();

  // touch screen: the same game by taps
  const touch = await browser.newPage({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: name !== 'firefox' });
  await touch.goto(url);
  const t = await play(touch, fixtures[0], true);

  const ok = passed === fixtures.length && visible === 0 && t.ok;
  failed ||= !ok;
  console.log(`${name.padEnd(8)} games matching python ${passed}/${fixtures.length}  ` +
    `touch game ${t.ok ? 'ok' : 'FAIL'}  moves left after game over ${visible}  ~${ms} ms/round  ${ok ? 'ok' : 'FAIL'}`);
  await browser.close();
}
process.exit(failed ? 1 : 0);
