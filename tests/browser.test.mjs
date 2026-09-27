// Draws every fixture in real browsers and checks the CSS prediction equals the Python integer model.
// Playwright drives the mouse; the page under test contains no JavaScript.
import { chromium, firefox, webkit } from 'playwright';
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const url = pathToFileURL(root + 'index.html').href;
const { X, pred, label } = JSON.parse(readFileSync(root + 'tests/fixtures.json', 'utf8'));
const html = readFileSync(root + 'index.html', 'utf8').toLowerCase();
let failed = html.includes('<script') || /\son[a-z]+=/.test(html) || html.includes('javascript:');
console.log(`no JavaScript in index.html: ${failed ? 'FAIL' : 'ok'}`);

const read = (p, name) => p.evaluate(n => getComputedStyle(document.querySelector('.app')).getPropertyValue(n).trim(), name);

for (const [name, engine] of Object.entries({ chromium, firefox, webkit })) {
  const browser = await engine.launch();
  const page = await browser.newPage({ viewport: { width: 900, height: 900 } });
  await page.goto(url);
  const boxes = [];
  for (let i = 0; i < 64; i++) {
    const b = await page.locator('#c' + i).boundingBox();
    boxes.push([b.x + b.width / 2, b.y + b.height / 2]);
  }

  // press-and-drag across a row (WebKit freezes :hover during a drag, so it only gets the first cell)
  await page.mouse.move(...boxes[24]); await page.mouse.down();
  await page.mouse.move(boxes[31][0], boxes[31][1], { steps: 20 }); await page.mouse.up();
  await page.mouse.move(1, 1);
  const dragInk = +(await read(page, '--ink'));
  await page.click('.clear'); await page.mouse.move(1, 1);
  const clearInk = +(await read(page, '--ink'));

  // fixtures drawn in hover mode (works in every engine): one jump per inked cell
  await page.click('.pen');  // the checkbox itself is visually hidden
  let same = 0, correct = 0;
  for (let n = 0; n < X.length; n++) {
    await page.click('.clear'); await page.mouse.move(1, 1);
    for (let i = 0; i < 64; i++) if (X[n][i]) await page.mouse.move(...boxes[i]);
    await page.mouse.move(1, 1);
    const p = +(await read(page, '--pred'));
    same += p === pred[n]; correct += p === label[n];
  }
  const ok = same === X.length && clearInk === 0 && dragInk >= (name === 'webkit' ? 1 : 8);
  failed ||= !ok;
  console.log(`${name.padEnd(8)} matches python ${same}/${X.length}  correct ${correct}/${X.length}  ` +
    `drag ink ${dragInk}  after clear ${clearInk}  ${ok ? 'ok' : 'FAIL'}`);

  // touch screen: :hover never follows a finger, so each tap toggles that cell's checkbox
  const touch = await browser.newPage({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: name !== 'firefox' });
  await touch.goto(url);
  const taps = [19, 27, 35, 43];
  for (const i of taps) await touch.locator('#c' + i).tap();
  const tapInk = +(await read(touch, '--ink'));
  await touch.locator('#c' + taps[0]).tap();  // tapping again erases
  const retapInk = +(await read(touch, '--ink'));
  await touch.locator('.clear').tap();
  const touchClear = +(await read(touch, '--ink'));
  const touchOk = tapInk === taps.length && retapInk === taps.length - 1 && touchClear === 0;
  failed ||= !touchOk;
  console.log(`${name.padEnd(8)} touch: tap ink ${tapInk}  after re-tap ${retapInk}  after clear ${touchClear}  ${touchOk ? 'ok' : 'FAIL'}`);
  await browser.close();
}
process.exit(failed ? 1 : 0);
