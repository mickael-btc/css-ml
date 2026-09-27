"""The integer model the CSS computes, written plainly in Python.

Moves: 0 rock, 1 paper, 2 scissors. (m + 1) % 3 beats m.

Five base predictors guess the player's next move from frequency counts in a context:
  freq    all past moves
  last    the player's previous move
  last2   the player's previous two moves
  outcome the player's previous move and whether they won, drew or lost
  reply   the CSS's previous move (players often pick what beats it)

Each base predictor also has two "second-guess" variants rotated by +1 and +2, for players
who anticipate the counter. That makes 15 experts. Every expert gets +1 when the move it
would have played wins, -1 when it loses, summed over the last WINDOW rounds.
The best-scoring expert decides; the CSS plays what beats its prediction.
All ties break deterministically, so this is bit-exact with the CSS.
"""
ROUNDS = 30
WINDOW = 6
TIE = [2, 1, 0]  # on equal counts prefer rock, then paper

# (name, context size, function(history, r) -> context index or None)
BASES = ["freq", "last", "last2", "outcome", "reply"]
CONTEXTS = {"freq": 1, "last": 3, "last2": 9, "outcome": 9, "reply": 3}
ROTATIONS = 3
# preference among experts with equal scores: higher wins
PRIORITY = {
    ("last", 0): 14, ("outcome", 0): 13, ("reply", 0): 12, ("last2", 0): 11, ("freq", 0): 10,
    ("last", 1): 9, ("outcome", 1): 8, ("reply", 1): 7, ("last2", 1): 6, ("freq", 1): 5,
    ("last", 2): 4, ("outcome", 2): 3, ("reply", 2): 2, ("last2", 2): 1, ("freq", 2): 0,
}
EXPERTS = [(b, rot) for rot in range(ROTATIONS) for b in BASES]


def outcome(h, a):
    """0 player wins, 1 draw, 2 CSS wins."""
    if h == a:
        return 1
    return 0 if h == (a + 1) % 3 else 2


def context(base, hs, ais, r):
    """Context index at round r (uses rounds < r only), or None when it does not exist yet."""
    if base == "freq":
        return 0
    if base == "last":
        return hs[r - 1] if r >= 1 else None
    if base == "last2":
        return hs[r - 2] * 3 + hs[r - 1] if r >= 2 else None
    if base == "outcome":
        return hs[r - 1] * 3 + outcome(hs[r - 1], ais[r - 1]) if r >= 1 else None
    if base == "reply":
        return ais[r - 1] if r >= 1 else None


class Model:
    def __init__(self):
        self.hs, self.ais = [], []
        self.results = {e: [] for e in EXPERTS}  # per round, +1 / 0 / -1

    def counts(self, base, r):
        """Counts of each next move seen in the current context, before round r."""
        ctx = context(base, self.hs, self.ais, r)
        out = [0, 0, 0]
        if ctx is None:
            return out
        for t in range(r):
            if context(base, self.hs, self.ais, t) == ctx:
                out[self.hs[t]] += 1
        return out

    def base_prediction(self, base, r):
        freq = [0, 0, 0]
        for t in range(r):
            freq[self.hs[t]] += 1
        cnt = self.counts(base, r)
        keys = [cnt[k] * 128 + freq[k] * 3 + TIE[k] for k in range(3)]
        return max(range(3), key=lambda k: keys[k])

    def scores(self, r):
        return {e: sum(self.results[e][max(0, r - WINDOW):r]) for e in EXPERTS}

    def decide(self):
        """Everything the CSS shows for the round about to be played."""
        r = len(self.hs)
        base = {b: self.base_prediction(b, r) for b in BASES}
        preds = {(b, rot): (base[b] + rot) % 3 for b, rot in EXPERTS}
        scores = self.scores(r)
        best = max(EXPERTS, key=lambda e: (scores[e] + 64) * 16 + PRIORITY[e])
        return {"pred": preds[best], "ai": (preds[best] + 1) % 3, "best": best,
                "preds": preds, "scores": scores}

    def play(self, h):
        d = self.decide()
        for e, q in d["preds"].items():
            self.results[e].append(1 if h == q else -1 if h == (q + 2) % 3 else 0)
        self.hs.append(h)
        self.ais.append(d["ai"])
        return d["ai"]


def run(moves):
    """AI moves and final score (player wins, draws, CSS wins) for a sequence of player moves."""
    m = Model()
    ais = [m.play(h) for h in moves]
    tally = [0, 0, 0]
    for h, a in zip(moves, ais):
        tally[outcome(h, a)] += 1
    return ais, tally
