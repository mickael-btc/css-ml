"""Step 1: game rules as weights. Each game's whole player is one ReLU net: the trained policy, plus
the legality mask, the tactics, the argmax and the win check, all hand-compiled into layers.

    .venv/bin/python handwired/scripts/rules.py

Writes handwired/nets/{ttt_player,ttt_judge,c4_player,c4_judge}.json and checks each one against the
existing Python players in tic-tac-toe/scripts/train.py and connect-four/scripts/game.py:

  tic-tac-toe   every reachable position (exhaustive)
  connect four  the browser-test games plus random self-play positions (the game tree is too big)
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
REPO = ROOT.parent
NETS = ROOT / "nets"
sys.path.insert(0, str(SCRIPTS))

from hw import LIMIT, Net, Prog, total  # noqa: E402


def load(name, path):
    """Import a sibling demo's script under its own name (both games have a train.py)."""
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    sys.path.pop(0)
    return mod


ttt = load("ttt_train", REPO / "tic-tac-toe" / "scripts" / "train.py")
c4 = load("c4_game", REPO / "connect-four" / "scripts" / "game.py")


def learned_logits(p, layers, inputs):
    """The trained MLP, its logits each split into relu(g) and relu(-g) so later reads are cheap."""
    g = p.dense(layers, inputs)
    with p.tagging("learned"):
        return [p.split(v) for v in g]


def any_line(p, cells, lines, need):
    """OR over lines of [all `need` cells of the line set]."""
    return p.OR(*[p.relu(total(cells[i] for i in l) - (need - 1)) for l in lines])


# ---------------------------------------------------------------- tic-tac-toe

def ttt_player(layers):
    """18 board inputs (X then O, the board O sees) + go (you just moved) -> 9 one-hot O moves."""
    p = Prog()
    x = [p.input(f"x{i}") for i in range(9)]
    o = [p.input(f"o{i}") for i in range(9)]
    go = p.input("go")
    g = learned_logits(p, layers, x + o)                         # learned: 9 logits
    base = [ttt.TIE_SCALE * g[c] + (8 - c) for c in range(9)]    # rules: the best free cell, as
    w = p.pick(base, [[1 - x[c] - o[c] for c in range(9)]])      # 16 * logit + tie-break - 2^20 * occupied
    xwon = any_line(p, x, ttt.LINES, 3)
    full = p.relu(total(x) + total(o) - 8)
    move = [p.AND(w[c], go, 1 - xwon, 1 - full) for c in range(9)]   # only if you moved and the game is on
    return p.compile({f"m{c}": move[c] for c in range(9)})


def ttt_judge():
    """18 board inputs -> who won, full board, and whether each cell is on a winning line."""
    p = Prog()
    x = [p.input(f"x{i}") for i in range(9)]
    o = [p.input(f"o{i}") for i in range(9)]
    xl = [p.relu(x[a] + x[b] + x[c] - 2) for a, b, c in ttt.LINES]
    ol = [p.relu(o[a] + o[b] + o[c] - 2) for a, b, c in ttt.LINES]
    xwin, owin = p.OR(*xl), p.OR(*ol)
    full = p.relu(total(x) + total(o) - 8)
    over = p.OR(xwin, owin, full)
    out = {"xwin": xwin, "owin": owin, "full": full, "over": over, "draw": p.AND(full, 1 - xwin, 1 - owin)}
    for i in range(9):
        out[f"free{i}"] = p.relu(1 - x[i] - o[i] - over)        # an empty cell, while the game is on
    for i in range(9):
        out[f"win{i}"] = p.OR(*[xl[n] for n, l in enumerate(ttt.LINES) if i in l],
                              *[ol[n] for n, l in enumerate(ttt.LINES) if i in l])
    return p.compile(out)


def ttt_states():
    """Every reachable board, with whoever is to move."""
    seen, stack = {}, [(0,) * 9]
    while stack:
        b = stack.pop()
        if b in seen:
            continue
        n = sum(1 for v in b if v)
        seen[b] = "X" if n % 2 == 0 else "O"
        if ttt.winner(b) or all(b):
            continue
        for i in range(9):
            if not b[i]:
                stack.append(ttt.play(b, i, seen[b]))
    return seen


def check_ttt(player, judge, layers):
    states = ttt_states()
    boards = list(states)
    X = np.array([ttt.encode(b) for b in boards])

    # player, on every board right after an X move: exactly int_move, or nothing once X has won.
    # With go = 0 (you haven't moved yet) it never moves.
    after_x = [n for n, b in enumerate(boards) if states[b] == "O"]
    on = np.ones((len(after_x), 1), dtype=np.int64)
    moves = player.forward(np.hstack([X[after_x], on]))
    want = np.zeros_like(moves)
    for row, n in enumerate(after_x):
        b = boards[n]
        if not ttt.winner(b) and not all(b):
            want[row, ttt.int_move(layers, b)] = 1
    o_turn = int(want.sum())
    assert (moves == want).all(), "tic-tac-toe player differs from int_move"
    assert not player.forward(np.hstack([X, np.zeros((len(X), 1), dtype=np.int64)])).any(), "moved without go"

    # judge: winner, full, draw on every reachable board
    J = judge.forward(X)
    col = {n: k for k, n in enumerate(judge.out_names)}
    for n, b in enumerate(boards):
        w = ttt.winner(b)
        assert J[n, col["xwin"]] == (w == "X") and J[n, col["owin"]] == (w == "O"), b
        assert J[n, col["full"]] == all(b) and J[n, col["over"]] == bool(w or all(b)), b
        assert J[n, col["draw"]] == (all(b) and not w), b
        for i in range(9):
            assert J[n, col[f"free{i}"]] == (not b[i] and not w and not all(b)), (b, i)
        for i in range(9):
            on_line = any(i in l and b[l[0]] and b[l[0]] == b[l[1]] == b[l[2]] for l in ttt.LINES)
            assert J[n, col[f"win{i}"]] == on_line, (b, i)

    mass = max(player.max_mass(np.hstack([X[after_x], on])), judge.max_mass(X))
    assert mass < LIMIT, mass
    return f"{len(boards)} reachable boards ({o_turn} with O to move), max |partial sum| {mass:,}"


# ---------------------------------------------------------------- connect four

def c4_choose(p, layers, you, me, go):
    """The network's move as 7 one-hot columns, exactly int_move, for boards given as Vals."""
    occ = [you[i] + me[i] for i in range(c4.CELLS)]
    g = learned_logits(p, layers, you + me)
    land = landing(p, occ)

    def completes_at(plane, c, rr):
        """OR over rows: [the piece lands on (row, c)] and [cell (row + rr, c) finishes a line of plane]."""
        terms = []
        for row in range(c4.ROWS):
            target = row + rr
            if target >= c4.ROWS:
                continue
            i = c4.cell(target, c)
            for l in c4.THROUGH[i]:
                others = total(plane[j] for j in l if j != i)
                terms.append(p.relu(land[(row, c)] + others - 3))   # 1 iff lands here and 3 of 3
        return p.OR(*terms) if terms else 0

    # score = 8 * logit + tie-break + 2^21 win + 2^20 block - 2^19 danger - 2^22 full, picked as tiers:
    # a legal column; a winning one if any; else a blocking one if any; a safe one if any; best logit
    cols = range(c4.COLS)
    legal = [1 - occ[c4.cell(c4.ROWS - 1, c)] for c in cols]
    win = [completes_at(me, c, 0) for c in cols]
    block = [completes_at(you, c, 0) for c in cols]
    safe = [1 - completes_at(you, c, 1) for c in cols]
    base = [c4.TIE_SCALE * g[c] + c4.TIE[c] for c in cols]
    w = p.pick(base, [legal, win, block, safe])
    youwon = any_line(p, you, c4.LINES, 4)
    full = p.relu(total(occ) - (c4.CELLS - 1))
    return [p.AND(w[c], go, 1 - youwon, 1 - full) for c in cols]


def landing(p, occ):
    """{(r, c): 1 if the next piece in column c lands on row r}: the cell below is filled, this one empty."""
    return {(r, c): p.relu((1 if r == 0 else occ[c4.cell(r - 1, c)]) - occ[c4.cell(r, c)])
            for r in range(c4.ROWS) for c in range(c4.COLS)}


def c4_player(layers):
    """84 board inputs (yours then the network's) + go -> 7 one-hot columns, exactly int_move."""
    p = Prog()
    you = [p.input(f"y{i}") for i in range(c4.CELLS)]
    me = [p.input(f"m{i}") for i in range(c4.CELLS)]
    go = p.input("go")
    move = c4_choose(p, layers, you, me, go)
    return p.compile({f"m{c}": move[c] for c in range(c4.COLS)})


def c4_turn(layers):
    """One whole turn: the board before it (84) and the column you picked (7, one-hot or all 0) ->
    the board after your piece falls, the network answers, and its piece falls (84)."""
    p = Prog()
    you = [p.input(f"y{i}") for i in range(c4.CELLS)]
    me = [p.input(f"m{i}") for i in range(c4.CELLS)]
    pick = [p.input(f"h{c}") for c in range(c4.COLS)]
    land = landing(p, [you[i] + me[i] for i in range(c4.CELLS)])
    you2 = [p.relu(you[i] + p.AND(pick[i % c4.COLS], land[divmod(i, c4.COLS)])) for i in range(c4.CELLS)]  # one unit per cell
    go = total(pick)
    move = c4_choose(p, layers, you2, me, go)
    land2 = landing(p, [you2[i] + me[i] for i in range(c4.CELLS)])
    me2 = [me[i] + p.AND(move[i % c4.COLS], land2[divmod(i, c4.COLS)]) for i in range(c4.CELLS)]
    out = {f"Y{i}": you2[i] for i in range(c4.CELLS)}
    out.update({f"M{i}": me2[i] for i in range(c4.CELLS)})
    out.update({f"a{c}": move[c] for c in range(c4.COLS)})   # the column it played, for display
    return p.compile(out)


def check_c4_turn(turn, layers, n_games=400, seed=1):
    """Random first players against the turn net, applied turn after turn, vs. Board + int_move."""
    rng = np.random.default_rng(seed)
    turns, mass, seen = 0, 0, []
    for _ in range(n_games):
        b = c4.Board()
        state = np.zeros(2 * c4.CELLS, dtype=np.int64)
        greedy = rng.uniform() < 0.5
        while not (b.won("you") or b.won("me") or b.full()):
            legal = b.legal()
            col = int(rng.choice(legal))
            if greedy:  # win, else block, else random: reaches the tactics more often
                for who in ("you", "me"):
                    for c in legal:
                        t = b.copy(); t.play(c, who)
                        if t.won(who):
                            col = c
            pick = np.zeros(c4.COLS, dtype=np.int64); pick[col] = 1
            seen.append(np.r_[state, pick])
            out = turn.forward(seen[-1])[0]
            b.play(col, "you")
            if not (b.won("you") or b.full()):
                b.play(c4.int_move(layers, b), "me")
            state = out[:2 * c4.CELLS]
            assert (state == np.array(b.you + b.me)).all(), "turn net differs from Board + int_move"
            turns += 1
        # a finished game stays put, and an unplayed turn changes nothing
        assert (turn.forward(np.r_[state, np.zeros(c4.COLS, dtype=np.int64)])[0][:2 * c4.CELLS] == state).all()
    mass = turn.max_mass(np.array(seen))
    assert mass < LIMIT, mass
    return f"{n_games} games, {turns:,} turns, max |partial sum| {mass:,}"


def c4_judge():
    p = Prog()
    you = [p.input(f"y{i}") for i in range(c4.CELLS)]
    me = [p.input(f"m{i}") for i in range(c4.CELLS)]
    yl = [p.relu(total(you[i] for i in l) - 3) for l in c4.LINES]
    ml = [p.relu(total(me[i] for i in l) - 3) for l in c4.LINES]
    youwin, mewin = p.OR(*yl), p.OR(*ml)
    full = p.relu(total(you) + total(me) - (c4.CELLS - 1))
    over = p.OR(youwin, mewin, full)
    out = {"youwin": youwin, "mewin": mewin, "full": full, "over": over}
    for c in range(c4.COLS):
        out[f"free{c}"] = p.relu(1 - you[c4.cell(c4.ROWS - 1, c)] - me[c4.cell(c4.ROWS - 1, c)] - over)
    for i in range(c4.CELLS):
        out[f"win{i}"] = p.OR(*[yl[n] for n, l in enumerate(c4.LINES) if i in l],
                              *[ml[n] for n, l in enumerate(c4.LINES) if i in l])
    return p.compile(out)


def c4_positions(layers, n_games=3000, seed=0):
    """Positions with the network to move: the fixture games, then games against a random or greedy first player."""
    rng = np.random.default_rng(seed)
    out = []
    fixtures = json.loads((REPO / "connect-four" / "tests" / "fixtures.json").read_text())
    for g in fixtures:
        b = c4.Board()
        for n, col in enumerate(g["you"]):
            b.play(col, "you")
            if b.won("you") or b.full():
                break
            out.append(b.copy())
            if n < len(g["me"]):
                b.play(g["me"][n], "me")
                if b.won("me"):
                    break
    for k in range(n_games):
        b = c4.Board()
        eps = rng.uniform(0, 1)  # how often each side plays at random
        while True:
            legal = b.legal()
            col = int(rng.choice(legal))
            b.play(col, "you")
            if b.won("you") or b.full():
                out.append(b.copy())      # a finished board: the player must stay still
                break
            out.append(b.copy())
            mv = int(rng.choice(b.legal())) if rng.uniform() < eps else c4.int_move(layers, b)
            b.play(mv, "me")
            if b.won("me") or b.full():
                break
    return out


def check_c4(player, judge, layers):
    boards = c4_positions(layers)
    X = np.array([c4.encode(b) for b in boards])
    on = np.ones((len(X), 1), dtype=np.int64)
    moves = player.forward(np.hstack([X, on]))
    J = judge.forward(X)
    col = {n: k for k, n in enumerate(judge.out_names)}
    played = 0
    for n, b in enumerate(boards):
        want = np.zeros(c4.COLS, dtype=np.int64)
        if not b.won("you") and not b.full():
            want[c4.int_move(layers, b)] = 1
            played += 1
        assert (moves[n] == want).all(), f"connect-four player differs on board {n}"
        assert J[n, col["youwin"]] == b.won("you") and J[n, col["mewin"]] == b.won("me"), n
        assert J[n, col["full"]] == b.full(), n
        over = b.won("you") or b.won("me") or b.full()
        for c in range(c4.COLS):
            assert J[n, col[f"free{c}"]] == (b.height[c] < c4.ROWS and not over), (n, c)
    assert not player.forward(np.hstack([X, 0 * on])).any()
    mass = max(player.max_mass(np.hstack([X, on])), judge.max_mass(X))
    assert mass < LIMIT, mass
    return f"{len(boards):,} boards ({played:,} with the network to move), max |partial sum| {mass:,}"


# ---------------------------------------------------------------- main

def describe(name, net):
    tags = net.count_by_tag()
    return (f"{name}: {net.depth} layers, widths {net.widths()[:4]}…, {net.n_units:,} units, "
            f"{net.n_weights:,} weights ({', '.join(f'{k} {v:,}' for k, v in sorted(tags.items()))})")


def main():
    NETS.mkdir(exist_ok=True)
    ttt_layers = json.loads((REPO / "tic-tac-toe" / "model.json").read_text())["layers"]
    c4_layers = json.loads((REPO / "connect-four" / "model.json").read_text())["layers"]

    nets = {"ttt_player": ttt_player(ttt_layers), "ttt_judge": ttt_judge(),
            "c4_player": c4_player(c4_layers), "c4_judge": c4_judge(), "c4_turn": c4_turn(c4_layers)}
    for name, net in nets.items():
        print(describe(name, net))
    print("tic-tac-toe:", check_ttt(nets["ttt_player"], nets["ttt_judge"], ttt_layers))
    print("connect four:", check_c4(nets["c4_player"], nets["c4_judge"], c4_layers))
    print("connect four turn net:", check_c4_turn(nets["c4_turn"], c4_layers))
    for name, net in nets.items():
        (NETS / f"{name}.json").write_text(json.dumps(net.to_json()))
    print(f"wrote {', '.join(nets)} to {NETS.relative_to(REPO)}")


if __name__ == "__main__":
    main()
