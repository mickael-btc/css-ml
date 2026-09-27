"""Compile the model in model.py into a single self-contained index.html with zero JavaScript.

    python3 scripts/build.py

Every value is a registered <integer> custom property on `.app`. Each round r has one radio
group; --hR_K is 1 when the player picked move K in round R. From that the browser computes,
for every round, what the model predicts and which move the CSS plays:

  tables       --T{B}R_C_K      how often move K followed context C, before round R
  counts       --n{B}R_K        the row of that table for round R's context
  base guess   --b{B}R_K        one-hot argmax of the counts (ties broken by frequency, then rock)
  experts      15 = 5 bases x 3 rotations; --x{E}_R is +1/0/-1 for how the expert's move did
  scores       --S{E}_R         sum of the last WINDOW results
  decision     --w{E}_R         one-hot best expert; --pR_K its guess; --aR_K the CSS's move

Round R only reads rounds < R, so the CSS never looks at the move it is answering.
"""
import sys
from pathlib import Path
from string import Template

from model import BASES, CONTEXTS, EXPERTS, PRIORITY, ROUNDS, TIE, WINDOW

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.append(str(ROOT.parent / "scripts"))  # shared demo nav

from nav import nav_css, nav_html  # noqa: E402
MOVES = ["Rock", "Paper", "Scissors"]
GLYPHS = ["✊", "✋", "✌️"]
NAMES = {
    "freq": ("Favourite", "your most played move"),
    "last": ("After X", "what you play after your last move"),
    "last2": ("After X, Y", "what follows your last two moves"),
    "outcome": ("After a win / loss", "your last move and how it went"),
    "reply": ("Answer to me", "what you play after my last move"),
}
CODE = {"freq": "F", "last": "L", "last2": "M", "outcome": "O", "reply": "A"}


def var(name):
    return f"var(--{name})"


def total(terms):
    terms = list(terms)
    return "calc(" + " + ".join(terms) + ")" if terms else "0"


def onehot_max(css, name, keys):
    """keys are distinct integers: 1 for the largest, 0 for the others. --name holds the max."""
    css.define(name, f"max({', '.join(keys)})")
    return [f"clamp(0, {key} - var(--{name}) + 1, 1)" for key in keys]


class Stylesheet:
    def __init__(self):
        self.registrations, self.rules, self.values = [], [], []
        self.names = set()

    def register(self, name, syntax="<integer>", initial=0):
        assert name not in self.names, f"--{name} defined twice"
        self.names.add(name)
        self.registrations.append(f'@property --{name}{{syntax:"{syntax}";inherits:true;initial-value:{initial}}}')

    def define(self, name, expression, syntax="<integer>", initial=0):
        self.register(name, syntax, initial)
        self.values.append(f"--{name}:{expression};")

    def rule(self, selector, body):
        self.rules.append(f"{selector}{{{body}}}")

    def render(self):
        return "\n".join([*self.registrations, *self.rules, ".app{\n" + "\n".join(self.values) + "\n}"])


def h(r, k):
    return var(f"h{r}_{k}")


def expert_id(e):
    base, rot = e
    return f"{CODE[base]}{rot}"


# ---------------------------------------------------------------- the model, round by round

def add_inputs(css):
    for r in range(ROUNDS):
        for k in range(3):
            css.register(f"h{r}_{k}")
            css.rule(f".app:has(#r{r}m{k}:checked)", f"--h{r}_{k}:1")
        css.define(f"d{r}", total(h(r, k) for k in range(3)))  # 1 once round r is played


def context_terms(base, r):
    """One-hot context indicators at round r (all 0 while the context does not exist yet)."""
    if base == "freq":
        return ["1"]
    if base == "last":
        return [h(r - 1, c) if r >= 1 else "0" for c in range(3)]
    if base == "last2":
        return [f"{h(r - 2, a)} * {h(r - 1, b)}" if r >= 2 else "0" for a in range(3) for b in range(3)]
    if base == "outcome":
        return [f"{h(r - 1, a)} * {var(f'o{r - 1}_{o}')}" if r >= 1 else "0" for a in range(3) for o in range(3)]
    if base == "reply":
        return [var(f"a{r - 1}_{c}") if r >= 1 else "0" for c in range(3)]


def add_round(css, r):
    """Everything the model knows before round r, and the move it plays in round r."""
    playable = r < ROUNDS

    for base in BASES:
        ctx = context_terms(base, r)
        for c in range(CONTEXTS[base]):
            if base != "freq" and r:
                css.define(f"c{CODE[base]}{r}_{c}", f"calc({ctx[c]})")
            for k in range(3):
                if r == 0:
                    css.define(f"T{CODE[base]}{r}_{c}_{k}", "0")
                    continue
                seen = "1" if base == "freq" else var(f"c{CODE[base]}{r - 1}_{c}") if r - 1 else "0"
                step = h(r - 1, k) if seen == "1" else f"{seen} * {h(r - 1, k)}" if seen != "0" else None
                prev = var(f"T{CODE[base]}{r - 1}_{c}_{k}")
                css.define(f"T{CODE[base]}{r}_{c}_{k}", f"calc({prev} + {step})" if step else prev)

        for k in range(3):
            if base == "freq":
                count = var(f"TF{r}_0_{k}")
            elif r == 0:
                count = "0"
            else:
                count = total(f"{var(f'c{CODE[base]}{r}_{c}')} * {var(f'T{CODE[base]}{r}_{c}_{k}')}" for c in range(CONTEXTS[base]))
            css.define(f"n{CODE[base]}{r}_{k}", count)

        keys = [f"calc({var(f'n{CODE[base]}{r}_{k}')} * 128 + {var(f'TF{r}_0_{k}')} * 3 + {TIE[k]})" for k in range(3)]
        for k, pick in enumerate(onehot_max(css, f"B{CODE[base]}{r}", keys)):
            css.define(f"b{CODE[base]}{r}_{k}", pick)

    def guess(e, k):
        """Expert e's one-hot guess of the player's move in round r."""
        base, rot = e
        return var(f"b{CODE[base]}{r}_{(k - rot) % 3}")

    for e in EXPERTS:
        name = expert_id(e)
        if playable:
            # +1 if the player played the guess (the counter wins), -1 if they played guess + 2
            css.define(f"x{name}_{r}", total(f"{guess(e, k)} * ({h(r, k)} - {h(r, (k + 2) % 3)})" for k in range(3)))
        css.define(f"S{name}_{r}", total(var(f"x{name}_{t}") for t in range(max(0, r - WINDOW), r)))

    keys = [f"calc(({var(f'S{expert_id(e)}_{r}')} + 64) * 16 + {PRIORITY[e]})" for e in EXPERTS]
    for e, pick in zip(EXPERTS, onehot_max(css, f"W{r}", keys)):
        css.define(f"w{expert_id(e)}_{r}", pick)

    for k in range(3):
        css.define(f"p{r}_{k}", total(f"{var(f'w{expert_id(e)}_{r}')} * {guess(e, k)}" for e in EXPERTS))
    for k in range(3):
        css.define(f"a{r}_{k}", var(f"p{r}_{(k + 2) % 3}"))  # play what beats the guess

    if playable:
        # outcome of round r: 0 player wins, 1 draw, 2 CSS wins
        css.define(f"o{r}_0", total(f"{h(r, k)} * {var(f'a{r}_{(k + 2) % 3}')}" for k in range(3)))
        css.define(f"o{r}_1", total(f"{h(r, k)} * {var(f'a{r}_{k}')}" for k in range(3)))
        css.define(f"o{r}_2", total(f"{h(r, k)} * {var(f'a{r}_{(k + 1) % 3}')}" for k in range(3)))


def add_display(css):
    """Collapse the per-round values into what the page shows."""
    R = range(ROUNDS)
    css.define("played", total(var(f"d{r}") for r in R))
    css.define("live", "min(1, var(--played))")
    css.define("over", f"var(--d{ROUNDS - 1})")
    for o, name in enumerate(["you", "draws", "cpu"]):
        css.define(name, total(var(f"o{r}_{o}") for r in R))

    # last played round
    for r in R:
        nxt = var(f"d{r + 1}") if r + 1 < ROUNDS else "0"
        css.define(f"L{r}", f"calc({var(f'd{r}')} - {nxt})")
    css.define("lh", total(f"{var(f'L{r}')} * {h(r, k)} * {k}" for r in R for k in (1, 2)))
    css.define("la", total(f"{var(f'L{r}')} * {var(f'a{r}_{k}')} * {k}" for r in R for k in (1, 2)))
    css.define("lo", total(f"{var(f'L{r}')} * {var(f'o{r}_{o}')} * {o}" for r in R for o in (1, 2)))

    # the round about to be played (ROUNDS once the game is over): only its scores are shown,
    # its guess would give the CSS's move away
    for r in range(ROUNDS + 1):
        before = var(f"d{r - 1}") if r else "1"
        after = var(f"d{r}") if r < ROUNDS else "0"
        css.define(f"C{r}", f"calc({before} - {after})")

    def at(weight, prop):
        return total(f"{var(f'{weight}{r}')} * {var(prop.format(r=r))}" for r in R)

    # the guess behind the last round, and which expert made it
    css.define("lp", total(f"{k} * {at('L', 'p{r}_' + str(k))}" for k in (1, 2)))
    for e in EXPERTS:
        name = expert_id(e)
        base, rot = e
        css.define(f"NS{name}", total(f"{var(f'C{r}')} * {var(f'S{name}_{r}')}" for r in range(ROUNDS + 1)))
        css.define(f"Nw{name}", at("L", f"w{name}_{{r}}"))
        css.define(f"Ng{name}", total(f"{k} * {at('L', f'b{CODE[base]}{{r}}_{(k - rot) % 3}')}" for k in (1, 2)))

    # what the deciding expert had seen in that situation, rotated like its guess
    for k in range(3):
        css.define(f"Nv{k}", total(
            f"{var(f'L{r}')} * {var(f'w{CODE[base]}{rot}_{r}')} * {var(f'n{CODE[base]}{r}_{(k - rot) % 3}')}"
            for r in R for base, rot in EXPERTS))
    css.define("Nvmax", f"max(1, {', '.join(var(f'Nv{k}') for k in range(3))})")

    # the "after X" table, counted over every round played so far
    for c in range(3):
        for k in range(3):
            css.define(f"M{c}_{k}", var(f"TL{ROUNDS}_{c}_{k}"))
    css.define("Mmax", f"max(1, {', '.join(var(f'M{c}_{k}') for c in range(3) for k in range(3))})")


def add_visuals(css):
    for r in range(ROUNDS):
        css.rule(f".hist i:nth-child({r + 1})",
                 f"--d:var(--d{r});--hm:calc(var(--h{r}_1) + 2 * var(--h{r}_2));"
                 f"--am:calc(var(--a{r}_1) + 2 * var(--a{r}_2));--oc:calc(var(--o{r}_1) + 2 * var(--o{r}_2))")
    for e in EXPERTS:
        name = expert_id(e)
        css.rule(f".x-{name}", f"--s:var(--NS{name});--w:var(--Nw{name});--g:var(--Ng{name})")
    for c in range(3):
        for k in range(3):
            css.rule(f".mat i:nth-of-type({c * 3 + k + 1})", f"--n:var(--M{c}_{k})")
    for k in range(3):
        css.rule(f".vote:nth-child({k + 1})", f"--n:var(--Nv{k});--on:clamp(0, 1 - (var(--lp) - {k}) * (var(--lp) - {k}), 1)")

    # only the current round's three labels are shown
    for r in range(ROUNDS):
        done_before = f":has(input[name=r{r - 1}]:checked)" if r else ""
        css.rule(f".app{done_before}:not(:has(input[name=r{r}]:checked)) .l{r}", "display:grid")


# ---------------------------------------------------------------- page

def moves_html():
    radios = "\n".join(
        f'      <input type="radio" name="r{r}" id="r{r}m{k}" tabindex="-1">'
        for r in range(ROUNDS) for k in range(3)
    )
    buttons = []
    for k, move in enumerate(MOVES):
        labels = "".join(f'<label class="l{r}" for="r{r}m{k}">{move}</label>' for r in range(ROUNDS))
        buttons.append(f'      <div class="move m{k}">{labels}</div>')
    return radios + "\n" + "\n".join(buttons)


def experts_html():
    rows = []
    for base in BASES:
        title, detail = NAMES[base]
        cells = "".join(f'<span class="expert x-{CODE[base]}{rot}"></span>' for rot in range(3))
        rows.append(f'      <div class="erow"><span class="ename">{title}<small>{detail}</small></span>{cells}</div>')
    return "\n".join(rows)


def main():
    css = Stylesheet()
    add_inputs(css)
    for r in range(ROUNDS + 1):
        add_round(css, r)
    add_display(css)
    add_visuals(css)

    html = Template((SCRIPTS / "page.html").read_text()).substitute(
        nav_css=nav_css(),
        nav=nav_html("rock-paper-scissors"),
        network_css=css.render(),
        rounds=ROUNDS,
        last=ROUNDS - 1,
        window=WINDOW,
        moves=moves_html(),
        history="<i></i>" * ROUNDS,
        experts=experts_html(),
        n_props=f"{len(css.registrations):,}",
    )
    assert "<script" not in html.lower() and "javascript:" not in html.lower()
    (ROOT / "index.html").write_text(html)
    print(f"index.html: {len(html) / 1024:.0f} KB, {len(css.registrations):,} properties")


if __name__ == "__main__":
    main()
