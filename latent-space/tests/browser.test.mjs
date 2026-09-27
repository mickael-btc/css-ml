// Moves over and taps the latent map in real browsers and checks every generated pixel and hidden
// unit equals the Python integer decoder. Playwright drives the pointer; the page has no JavaScript.
import { chromium, firefox, webkit } from 'playwright';
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const url = pathToFileURL(root + 'index.html').href;
const { hidden, pixels } = JSON.parse(readFileSync(root + 'tests/fixtures.json', 'utf8'));
const { anchors } = JSON.parse(readFileSync(root + 'model.json', 'utf8'));
const html = readFileSync(root + 'index.html', 'utf8').toLowerCase();
let failed = html.includes('<script') || /\son[a-z]+=/.test(html) || html.includes('javascript:');
console.log(`no JavaScript in index.html: ${failed ? 'FAIL' : 'ok'}`);

const names = [...hidden.map((layer, l) => layer[0].map((_, k) => `--h${l}_${k}`)).flat(), ...pixels[0].map((_, k) => `--y${k}`)];
const expected = cell => [...hidden.map(layer => layer[cell]).flat(), ...pixels[cell]];

// true when every hidden unit and pixel on the page equals the Python decoder at `cell`
async function matches(page, cell) {
  const got = await page.evaluate(ns => {
    const style = getComputedStyle(document.querySelector('.app'));
    return ns.map(n => +style.getPropertyValue(n));
  }, names);
  return got.every((v, i) => v === expected(cell)[i]);
}

const clicked = [0, 19, 57, 150, 222, 333, 380, 399];
const hovered = [5, 88, 171, 209, 264, 318];
const tapped = [44, 210, 377];

for (const [name, engine] of Object.entries({ chromium, firefox, webkit })) {
  const browser = await engine.launch();

  // mouse: the default cell, clicks that stay after the pointer leaves, hovers that preview
  const page = await browser.newPage({ viewport: { width: 1000, height: 900 } });
  await page.goto(url);
  await page.evaluate(() => new Promise(requestAnimationFrame));  // Firefox can be read before its first style pass
  const initial = await matches(page, anchors['8']);
  let ok = initial;
  let pinned = 0, previewed = 0;
  for (const cell of clicked) {
    await page.click(`label[for=m${cell}]`);
    await page.mouse.move(1, 1);
    pinned += await matches(page, cell);
  }
  for (const cell of hovered) {
    await page.hover(`label[for=m${cell}]`);
    previewed += await matches(page, cell);
  }
  await page.mouse.move(1, 1);
  const back = await matches(page, clicked.at(-1));  // leaving the map returns to the pinned cell
  ok &&= pinned === clicked.length && previewed === hovered.length && back;

  // touch screen: taps pin a cell
  const touch = await browser.newPage({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: name !== 'firefox' });
  await touch.goto(url);
  let taps = 0;
  for (const cell of tapped) {
    await touch.locator(`label[for=m${cell}]`).tap();
    taps += await matches(touch, cell);
  }
  ok &&= taps === tapped.length;

  failed ||= !ok;
  console.log(`${name.padEnd(8)} default ${initial ? 'yes' : 'no'}  clicked ${pinned}/${clicked.length}  hovered ${previewed}/${hovered.length}  ` +
    `back to pinned ${back ? 'yes' : 'no'}  tapped ${taps}/${tapped.length}  ${ok ? 'ok' : 'FAIL'}`);
  await browser.close();
}
process.exit(failed ? 1 : 0);
