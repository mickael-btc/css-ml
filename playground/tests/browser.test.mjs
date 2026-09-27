// Places every fixture's points by clicking, in real browsers, and checks the training the CSS ran
// (every weight of every epoch, the error counts, the background plane) equals scripts/reference.py.
// The page under test contains no JavaScript.
import { chromium, firefox, webkit } from 'playwright';
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const url = pathToFileURL(root + 'index.html').href;
const { epochs, features, sets } = JSON.parse(readFileSync(root + 'tests/fixtures.json', 'utf8'));
const byName = Object.fromEntries(sets.map(s => [s.name, s]));
const html = readFileSync(root + 'index.html', 'utf8').toLowerCase();
let failed = html.includes('<script') || /\son[a-z]+=/.test(html) || html.includes('javascript:');
console.log(`no JavaScript in index.html: ${failed ? 'FAIL' : 'ok'}`);

const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const plane = w => features.map(f => f.reduce((s, x, k) => s + x * w[k], 0));

// everything the CSS computed: weights per epoch, errors per epoch, point count, and each plane cell's score
async function state(page) {
  return page.evaluate(E => {
    const cs = getComputedStyle(document.querySelector('.app'));
    const get = n => +cs.getPropertyValue('--' + n);
    const w = [], errors = [];
    for (let e = 0; e <= E; e++) {
      w.push([0, 1, 2, 3, 4, 5].map(k => get(`w${e}_${k}`)));
      errors.push(get(`err${e}`));
    }
    const plane = [...document.querySelectorAll('.plane i')].map(i => +getComputedStyle(i).getPropertyValue('--q'));
    return { w, errors, n: get('n'), ep: get('ep'), acc: get('acc'), plane };
  }, epochs);
}

function matches(got, set, shownEpoch = epochs) {
  const n = set.t.filter(v => v).length;
  const acc = Math.round(100 * (n - set.errors[shownEpoch]) / Math.max(n, 1));
  return same(got.w, set.w) && same(got.errors, set.errors) && got.n === n && got.ep === shownEpoch &&
    got.acc === acc && same(got.plane, plane(set.w[shownEpoch]));
}

// click (or tap) every point of a set: red ones with the red brush, then blue ones with the blue brush
async function place(page, t, act) {
  for (const [brush, sign, id] of [['br-r', 1, 'r'], ['br-b', -1, 'b']]) {
    const points = t.flatMap((v, p) => v === sign ? [p] : []);
    if (!points.length) continue;
    await act(page.locator(`label[for=${brush}]`));
    for (const p of points) await act(page.locator(`label[for=${id}${p}]`));
  }
}

const click = l => l.click();
const tap = l => l.tap();

for (const [name, engine] of Object.entries({ chromium, firefox, webkit })) {
  const browser = await engine.launch();
  const page = await browser.newPage({ viewport: { width: 1000, height: 1000 } });
  await page.goto(url);
  await page.evaluate(() => new Promise(requestAnimationFrame));  // Firefox can be read before its first style pass

  let placed = 0, cleared = 0, total = 0;
  for (const set of sets.filter(s => s.name !== 'few-edited')) {
    await place(page, set.t, click);
    await page.mouse.move(1, 1);
    placed += matches(await state(page), set);
    await page.click('.clear');
    cleared += matches(await state(page), byName.empty);
    total++;
  }

  // clicking a point again with its own brush removes it; the other brush repaints it
  await place(page, byName.few.t, click);
  await page.click('label[for=br-r]');
  await page.click('label[for=n22]');
  await page.click('label[for=br-b]');
  await page.click('label[for=b23]');
  await page.mouse.move(1, 1);
  const edited = matches(await state(page), byName['few-edited']);

  // picking an epoch in the chart shows that epoch's weights on the plane; hovering previews one
  await page.click('.clear');
  await place(page, byName.circle.t, click);
  await page.click('label.bar2');
  await page.mouse.move(1, 1);
  const picked = matches(await state(page), byName.circle, 2);
  await page.hover('label.bar1');
  const hovered = matches(await state(page), byName.circle, 1);
  await page.mouse.move(1, 1);
  const back = matches(await state(page), byName.circle, 2);

  // touch screen: taps place points
  const touch = await browser.newPage({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: name !== 'firefox' });
  await touch.goto(url);
  await touch.evaluate(() => new Promise(requestAnimationFrame));
  await place(touch, byName.xor.t, tap);
  const tapped = matches(await state(touch), byName.xor);

  // time one click: from checking a point to the last epoch's error count being recomputed
  const ms = await page.evaluate(() => {
    const app = document.querySelector('.app'), cell = document.querySelector('.plane i'), times = [];
    for (const p of [5, 50, 95]) for (const id of ['r', 'n']) {  // place a point, then remove it
      const t0 = performance.now();
      document.getElementById(id + p).checked = true;
      getComputedStyle(app).getPropertyValue('--err24');
      getComputedStyle(cell).getPropertyValue('--q');
      times.push(performance.now() - t0);
    }
    return times.sort((a, b) => a - b)[times.length >> 1];
  });

  const ok = placed === total && cleared === total && edited && picked && hovered && back && tapped;
  failed ||= !ok;
  console.log(`${name.padEnd(8)} sets match python ${placed}/${total}  clear ${cleared}/${total}  ` +
    `remove+repaint ${edited ? 'yes' : 'no'}  pick epoch ${picked ? 'yes' : 'no'}  hover epoch ${hovered && back ? 'yes' : 'no'}  ` +
    `touch ${tapped ? 'ok' : 'no'}  ~${ms.toFixed(0)} ms/click  ${ok ? 'ok' : 'FAIL'}`);
  await browser.close();
}
process.exit(failed ? 1 : 0);
