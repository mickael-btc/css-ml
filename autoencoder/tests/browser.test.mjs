// Draws hand-made digits and picks map cells in real browsers; every encoder unit, the map cell,
// every decoder unit and every pixel must equal the Python integer model. The page contains no JavaScript.
import { chromium, firefox, webkit } from 'playwright';
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const url = pathToFileURL(root + 'index.html').href;
const { drawn, cells } = JSON.parse(readFileSync(root + 'tests/fixtures.json', 'utf8'));
const model = JSON.parse(readFileSync(root + 'model.json', 'utf8'));
const html = readFileSync(root + 'index.html', 'utf8').toLowerCase();
let failed = html.includes('<script') || /\son[a-z]+=/.test(html) || html.includes('javascript:');
console.log(`no JavaScript in index.html: ${failed ? 'FAIL' : 'ok'}`);

const names = (layers, prefix, last) => layers.flatMap((layer, l) => layer.b.map((_, k) =>
  l === layers.length - 1 ? last(k) : `--${prefix}${l}_${k}`));
const encNames = names(model.encoder, 'e', k => ['--ex', '--ey'][k]);
const decNames = names(model.decoder, 'h', k => `--y${k}`);

const read = (page, ns) => page.evaluate(ns => {
  const style = getComputedStyle(document.querySelector('.app'));
  return ns.map(n => +style.getPropertyValue(n));
}, ns);
const same = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);

// the decoder shows (col, row) and draws exactly the Python output for it
async function decodes(page, col, row, expected) {
  const [c, r, ...dec] = await read(page, ['--col', '--row', ...decNames]);
  return c === col && r === row && same(dec, expected);
}

// the encoder turned the drawing into exactly the Python code and cell
async function encodes(page, fixture) {
  const [dcol, drow, ...enc] = await read(page, ['--dcol', '--drow', ...encNames]);
  return dcol === fixture.col && drow === fixture.row && same(enc, fixture.encoder);
}

const frame = page => page.evaluate(() => new Promise(requestAnimationFrame));

for (const [name, engine] of Object.entries({ chromium, firefox, webkit })) {
  const browser = await engine.launch();
  const page = await browser.newPage({ viewport: { width: 1200, height: 1000 } });
  await page.goto(url);
  await frame(page);  // Firefox can be read before its first style pass
  const [ink0, show0] = await read(page, ['--ink', '--show']);
  let ok = ink0 === 0 && show0 === 0;

  const boxes = [];
  for (let i = 0; i < 64; i++) {
    const b = await page.locator('#c' + i).boundingBox();
    boxes.push([b.x + b.width / 2, b.y + b.height / 2]);
  }
  const draw = async X => {
    await page.click('.clear'); await page.mouse.move(1, 1);
    for (let i = 0; i < 64; i++) if (X[i]) await page.mouse.move(...boxes[i]);
    await page.mouse.move(1, 1);
  };

  // drawings, in hover mode (works in every engine): encoder, dot cell and decoder
  await page.click('.pen');
  let drawnOk = 0;
  for (const f of drawn) {
    await draw(f.X);
    drawnOk += (await encodes(page, f)) && (await decodes(page, f.col, f.row, f.decoder));
  }

  // map: a click pins a cell over the drawing, hover previews, "back to my drawing" releases the pin
  const f = drawn.at(-1);  // still on the grid from the loop above
  let pinned = 0;
  for (const [i, cell] of Object.entries(cells)) {
    await page.click(`label[for=m${i}]`); await page.mouse.move(1, 1);
    pinned += await decodes(page, cell.col, cell.row, cell.decoder);
  }
  const stillEncoded = await encodes(page, f);  // the dot keeps showing the drawing's own cell
  const [hi, hover] = Object.entries(cells)[2];
  await page.hover(`label[for=m${hi}]`);
  const previewed = await decodes(page, hover.col, hover.row, hover.decoder);
  const [li, last] = Object.entries(cells).at(-1);
  await page.mouse.move(1, 1);
  const backToPin = await decodes(page, last.col, last.row, last.decoder);
  await page.click('label[for=mine]'); await page.mouse.move(1, 1);
  const released = await decodes(page, f.col, f.row, f.decoder);

  // Clear empties the drawing and releases a pin
  await page.click(`label[for=m${li}]`);
  await page.click('.clear'); await page.mouse.move(1, 1);
  const [ink, pin, show] = await read(page, ['--ink', '--pinned', '--show']);
  const cleared = ink === 0 && pin === 0 && show === 0;

  ok &&= drawnOk === drawn.length && pinned === Object.keys(cells).length && stillEncoded && previewed &&
    backToPin && released && cleared;

  // recalculation time for one painted pixel: toggle a cell, read the last pixel, median of 15
  const ms = await page.evaluate(() => {
    const times = [];
    const box = document.getElementById('k27');
    for (let n = 0; n < 15; n++) {
      const t = performance.now();
      box.checked = !box.checked;
      getComputedStyle(document.querySelector('.app')).getPropertyValue('--y255');
      times.push(performance.now() - t);
    }
    return times.sort((a, b) => a - b)[7];
  });

  // touch screen: taps paint cells and pick map points
  const touch = await browser.newPage({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: name !== 'firefox' });
  await touch.goto(url);
  await frame(touch);
  const t = drawn[1];
  for (let i = 0; i < 64; i++) if (t.X[i]) await touch.locator('#c' + i).tap();
  const tapDrawn = (await encodes(touch, t)) && (await decodes(touch, t.col, t.row, t.decoder));
  const [ti, tcell] = Object.entries(cells)[3];
  await touch.locator(`label[for=m${ti}]`).tap();
  const tapPinned = await decodes(touch, tcell.col, tcell.row, tcell.decoder);
  await touch.locator('label[for=mine]').tap();
  const tapBack = await decodes(touch, t.col, t.row, t.decoder);
  const touchOk = tapDrawn && tapPinned && tapBack;
  ok &&= touchOk;

  failed ||= !ok;
  console.log(`${name.padEnd(8)} empty ${ink0 === 0 && show0 === 0 ? 'yes' : 'no'}  drawn ${drawnOk}/${drawn.length}  ` +
    `pinned ${pinned}/${Object.keys(cells).length}  dot stays ${stillEncoded ? 'yes' : 'no'}  ` +
    `hover ${previewed ? 'yes' : 'no'}  back to pin ${backToPin ? 'yes' : 'no'}  released ${released ? 'yes' : 'no'}  ` +
    `clear ${cleared ? 'yes' : 'no'}  touch ${touchOk ? 'ok' : 'FAIL'}  ~${ms.toFixed(0)} ms/pixel  ${ok ? 'ok' : 'FAIL'}`);
  await browser.close();
}
process.exit(failed ? 1 : 0);
