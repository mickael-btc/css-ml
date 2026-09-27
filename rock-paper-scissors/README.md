# Rock, paper, scissors against CSS

Play 30 rounds against a model that learns your habits while you play and picks what beats your next move. It runs entirely in CSS: **zero bytes of JavaScript.**

Open `index.html`, or play it at `/rock-paper-scissors` on the live site.

## How it works

Each round is a group of three radio buttons. The big Rock / Paper / Scissors buttons are `<label>`s, and `:has()` shows only the labels of the round being played. They work by click or by tap. **New game** is a `<button type="reset">`.

Everything else is a registered `<integer>` custom property (6,067 of them), recomputed by the browser whenever a radio changes:

| Stage | CSS |
|---|---|
| **Your move** | `.app:has(#r7m1:checked) { --h7_1: 1 }` |
| **Online learning** | Count tables grow by one row per round: `--TL8_0_2: calc(var(--TL7_0_2) + var(--cL7_0) * var(--h7_2))`, which reads "after rock, you played scissors" |
| **Five predictors** | Each one looks up your next move in its own context: your favourite move, your last move, your last two moves, your last move and whether you won, and my last move |
| **15 experts** | Each predictor, plus two "second-guess" variants rotated by +1 and +2 for players who try to outthink the counter |
| **Scores** | Each expert gets +1 when the move it would have played wins and −1 when it loses, summed over the last 6 rounds |
| **Decision** | The best-scoring expert decides (`clamp(0, key - max(keys) + 1, 1)` picks it), and the CSS plays what beats its guess |

Round *r* only reads rounds before *r*, so the CSS never sees the move it is answering. All ties break deterministically, so the browser matches the Python model in `scripts/model.py` exactly. The page shows the guess behind the round you just played, never the next one.

## How well it plays

Win rates over 400 games of 30 rounds each against scripted players (`scripts/evaluate.py`):

| Player | CSS wins | Draws | Player wins |
|---|---|---|---|
| Cycles R→P→S (10% noise) | 76% | 13% | 11% |
| Reverse cycle (10% noise) | 77% | 12% | 11% |
| Plays what beats my last move (10% noise) | 73% | 10% | 18% |
| Win-stay, lose-shift (10% noise) | 65% | 14% | 21% |
| Never repeats a move | 39% | 31% | 30% |
| Rock 50% of the time | 39% | 31% | 30% |
| Uniform random | 33% | 33% | 34% |

Against random play nobody can do better than a third. Against anything with a pattern, the model finds it within a few rounds.

## Rebuild

From the repository root, with the virtualenv and `npm install` done as in the main README:

```sh
python3 rock-paper-scissors/scripts/evaluate.py --fixtures   # win rates, writes tests/fixtures.json
python3 rock-paper-scissors/scripts/build.py                 # writes index.html
node rock-paper-scissors/tests/browser.test.mjs              # plays every fixture in Chromium, Firefox and WebKit
```

| File | Purpose |
|---|---|
| `index.html` | The whole game |
| `scripts/model.py` | The model in plain Python, the reference the CSS must match |
| `scripts/build.py` | Compiles the model into CSS |
| `scripts/page.html` | Page template: layout and styles |
| `scripts/evaluate.py` | Scripted players, win rates, test fixtures |
| `tests/` | Browser test and fixtures |
