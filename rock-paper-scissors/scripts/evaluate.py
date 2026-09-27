"""How often the model beats scripted human-like players, and fixtures for the browser test.

    python3 scripts/evaluate.py            # win rates
    python3 scripts/evaluate.py --fixtures # also writes tests/fixtures.json
"""
import json
import sys
from pathlib import Path

import numpy as np

from model import ROUNDS, outcome, run

ROOT = Path(__file__).resolve().parent.parent


def noisy(policy, noise):
    """Follow the policy, but play a random move with probability `noise`."""
    def play(hs, ais, rng):
        return int(rng.integers(3)) if rng.random() < noise else policy(hs, ais, rng)
    return play


def cycle(hs, ais, rng):
    return (hs[-1] + 1) % 3 if hs else int(rng.integers(3))


def reverse_cycle(hs, ais, rng):
    return (hs[-1] + 2) % 3 if hs else int(rng.integers(3))


def beat_last(hs, ais, rng):
    """Play what beats the CSS's previous move."""
    return (ais[-1] + 1) % 3 if ais else 0


def win_stay_lose_shift(hs, ais, rng):
    """The classic human pattern: repeat after a win, switch to what beats the opponent after a loss."""
    if not hs:
        return 0
    result = outcome(hs[-1], ais[-1])
    if result == 0:
        return hs[-1]
    if result == 2:
        return (ais[-1] + 1) % 3
    return int(rng.integers(3))


def rock_lover(hs, ais, rng):
    return int(rng.choice(3, p=[0.5, 0.25, 0.25]))


def avoid_repeat(hs, ais, rng):
    """Never the same move twice in a row: a very common human bias."""
    return int((hs[-1] + rng.integers(1, 3)) % 3) if hs else int(rng.integers(3))


def uniform(hs, ais, rng):
    return int(rng.integers(3))


STRATEGIES = {
    "cycle R→P→S": noisy(cycle, 0.1),
    "reverse cycle": noisy(reverse_cycle, 0.1),
    "beat its last move": noisy(beat_last, 0.1),
    "win-stay, lose-shift": noisy(win_stay_lose_shift, 0.1),
    "never repeat": avoid_repeat,
    "rock 50%": rock_lover,
    "uniform random": uniform,
}


def play_game(strategy, rng):
    """Replays a strategy against the model; the model only ever sees past rounds."""
    from model import Model
    m = Model()
    hs, ais = [], []
    for _ in range(ROUNDS):
        h = strategy(hs, ais, rng)
        ais.append(m.play(h))
        hs.append(h)
    return hs, ais


def main():
    rng = np.random.default_rng(0)
    games = 400
    print(f"{ROUNDS} rounds per game, {games} games per strategy")
    print(f"{'strategy':<24} {'CSS wins':>9} {'draws':>7} {'you win':>8}")
    for name, strategy in STRATEGIES.items():
        tally = np.zeros(3)
        for _ in range(games):
            hs, ais = play_game(strategy, rng)
            for h, a in zip(hs, ais):
                tally[outcome(h, a)] += 1
        you, draw, css = tally / tally.sum()
        print(f"{name:<24} {css:>9.0%} {draw:>7.0%} {you:>8.0%}")

    if "--fixtures" in sys.argv:
        rng = np.random.default_rng(1)
        fixtures = []
        for name, strategy in STRATEGIES.items():
            hs, _ = play_game(strategy, rng)
            ais, tally = run(hs)
            fixtures.append({"name": name, "moves": hs, "ai": ais, "tally": tally})
        (ROOT / "tests" / "fixtures.json").write_text(json.dumps(fixtures))
        print(f"wrote {len(fixtures)} fixtures")


if __name__ == "__main__":
    main()
