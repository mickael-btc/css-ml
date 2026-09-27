"""Connect 4 rules, an alpha-beta search, and the network's move exactly as the CSS computes it.

Cells are numbered i = row * 7 + col with row 0 at the bottom. Bitboards use the usual layout,
bit col * 7 + row, with an empty sentinel row on top of each column.
"""
import numpy as np

ROWS, COLS = 6, 7
CELLS = ROWS * COLS
CENTRE_FIRST = [3, 2, 4, 1, 5, 0, 6]
TIE = {c: 6 - CENTRE_FIRST.index(c) for c in range(COLS)}  # centre wins ties: 3→6, 2→5, 4→4, …

# score = TIE_SCALE * logit + tie-break + tactics - full column, every legal score distinct
TIE_SCALE = 8
WIN = 1 << 21      # this move wins now
BLOCK = 1 << 20    # this move stops your immediate win
DANGER = 1 << 19   # this move lets you win on top of it
MASK = 1 << 22     # full column


def cell(r, c):
    return r * COLS + c


def lines():
    """The 69 lines of four, as tuples of cell numbers."""
    out = []
    for r in range(ROWS):
        for c in range(COLS):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                cells = [(r + k * dr, c + k * dc) for k in range(4)]
                if all(0 <= rr < ROWS and 0 <= cc < COLS for rr, cc in cells):
                    out.append(tuple(cell(rr, cc) for rr, cc in cells))
    return out


LINES = lines()
assert len(LINES) == 69
THROUGH = [[l for l in LINES if i in l] for i in range(CELLS)]


# ---------------------------------------------------------------- plain boards (for the net and the CSS model)

class Board:
    """Two planes of 42 cells: `me` is the network (second player), `you` the human (first)."""

    def __init__(self):
        self.you = [0] * CELLS
        self.me = [0] * CELLS
        self.height = [0] * COLS

    def copy(self):
        b = Board()
        b.you, b.me, b.height = self.you[:], self.me[:], self.height[:]
        return b

    def play(self, c, who):
        r = self.height[c]
        assert r < ROWS, "full column"
        (self.you if who == "you" else self.me)[cell(r, c)] = 1
        self.height[c] += 1
        return cell(r, c)

    def legal(self):
        return [c for c in range(COLS) if self.height[c] < ROWS]

    def won(self, who):
        plane = self.you if who == "you" else self.me
        return any(all(plane[i] for i in l) for l in LINES)

    def full(self):
        return all(h == ROWS for h in self.height)


def completes(plane, i):
    """1 if putting a piece on cell i would finish a line of `plane` (a list of 42)."""
    return int(any(sum(plane[j] for j in l if j != i) == 3 for l in THROUGH[i]))


def tactics(b):
    """Per column: (wins now, blocks your win, lets you win on top). Matches the CSS."""
    win, block, danger = [], [], []
    for c in range(COLS):
        r = b.height[c]
        if r >= ROWS:
            win.append(0); block.append(0); danger.append(0)
            continue
        i = cell(r, c)
        win.append(completes(b.me, i))
        block.append(completes(b.you, i))
        danger.append(completes(b.you, cell(r + 1, c)) if r + 1 < ROWS else 0)
    return win, block, danger


def encode(b):
    """84 inputs: your 42 cells, then the network's 42."""
    return b.you + b.me


def css_round(x):
    """A registered <integer> custom property rounds to nearest, halves toward +infinity."""
    return np.floor(x + 0.5).astype(np.int64)


def int_forward(layers, X):
    """The network exactly as the CSS computes it: logits."""
    a = np.asarray(X, dtype=np.int64)
    for layer in layers:
        z = a @ np.array(layer["W"], dtype=np.int64) + np.array(layer["b"], dtype=np.int64)
        a = css_round(np.maximum(z, 0) / layer["R"]) if "R" in layer else z
    return a


def int_scores(layers, b, use_tactics=True):
    g = int_forward(layers, [encode(b)])[0]
    win, block, danger = tactics(b) if use_tactics else ([0] * COLS,) * 3
    return [int(TIE_SCALE * g[c] + TIE[c] + WIN * win[c] + BLOCK * block[c] - DANGER * danger[c]
                - MASK * (b.height[c] >= ROWS)) for c in range(COLS)]


def int_move(layers, b, use_tactics=True):
    s = int_scores(layers, b, use_tactics)
    return int(np.argmax(s))


# ---------------------------------------------------------------- bitboards and search

H1 = ROWS + 1
BOTTOM = sum(1 << (c * H1) for c in range(COLS))
TOP = sum(1 << (c * H1 + ROWS - 1) for c in range(COLS))
WINDOW_MASKS = [sum(1 << ((i % COLS) * H1 + i // COLS) for i in l) for l in LINES]


def bb_won(b):
    for s in (1, H1, H1 - 1, H1 + 1):
        m = b & (b >> s)
        if m & (m >> (2 * s)):
            return True
    return False


def bb_from(board, who_to_move):
    """(pieces of the player to move, all pieces) from a Board."""
    me = you = 0
    for i in range(CELLS):
        bit = 1 << ((i % COLS) * H1 + i // COLS)
        if board.me[i]:
            me |= bit
        if board.you[i]:
            you |= bit
    cur = me if who_to_move == "me" else you
    return cur, me | you


def can_play(mask, c):
    return not mask & (1 << (c * H1 + ROWS - 1))


def play_bit(mask, c):
    return (mask + (1 << (c * H1))) & (((1 << ROWS) - 1) << (c * H1))


def evaluate(cur, mask):
    """Heuristic for the player to move: open windows, weighted by how full they are."""
    opp = mask ^ cur
    score = 0
    for w in WINDOW_MASKS:
        a, b = cur & w, opp & w
        if a and not b:
            score += WEIGHT[bin(a).count("1")]
        elif b and not a:
            score -= WEIGHT[bin(b).count("1")]
    centre = ((1 << ROWS) - 1) << (3 * H1)
    return score + 2 * (bin(cur & centre).count("1") - bin(opp & centre).count("1"))


WIN_SCORE = 10000
WEIGHT = (0, 1, 4, 16, 0)  # by pieces in an open window


def negamax(cur, mask, depth, alpha, beta, ply=0):
    moves = [c for c in CENTRE_FIRST if can_play(mask, c)]
    if not moves:
        return 0
    for c in moves:  # win right now
        if bb_won(cur | play_bit(mask, c)):
            return WIN_SCORE - ply
    if depth == 0:
        return evaluate(cur, mask)
    best = -WIN_SCORE * 2
    for c in moves:
        bit = play_bit(mask, c)
        v = -negamax(cur ^ mask, mask | bit, depth - 1, -beta, -alpha, ply + 1)
        if v > best:
            best = v
        if v > alpha:
            alpha = v
        if alpha >= beta:
            break
    return best


def move_values(board, who, depth):
    """Search value of every legal column for `who` ("me" or "you")."""
    cur, mask = bb_from(board, who)
    out = {}
    for c in range(COLS):
        if not can_play(mask, c):
            continue
        bit = play_bit(mask, c)
        if bb_won(cur | bit):
            out[c] = WIN_SCORE
        else:
            out[c] = -negamax(cur ^ mask, mask | bit, depth - 1, -WIN_SCORE * 2, WIN_SCORE * 2, 1)
    return out
