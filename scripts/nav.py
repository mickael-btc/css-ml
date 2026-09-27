"""The list of demos, shared by every page's build script so the footer nav stays in sync.

    from nav import nav_css, nav_html
    template.substitute(..., nav_css=nav_css(), nav=nav_html("tic-tac-toe"))

Links are relative: the digit demo lives at the site root, every other demo in its own folder.
"""
GITHUB = "https://github.com/mickael-btc/css-ml"

DEMOS = [
    ("", "Digit recognizer"),
    ("latent-space", "Latent space"),
    ("autoencoder", "Autoencoder"),
    ("playground", "Live training"),
    ("tic-tac-toe", "Tic-tac-toe"),
    ("connect-four", "Connect four"),
    ("rock-paper-scissors", "Rock paper scissors"),
]


def nav_html(current):
    """current is the demo's folder name, or "" for the digit demo at the root."""
    prefix = "" if current == "" else "../"
    items = []
    for folder, title in DEMOS:
        if folder == current:
            items.append(f'<a aria-current="page">{title}</a>')
        else:
            items.append(f'<a href="{prefix + folder or "./"}">{title}</a>')
    items.append(f'<a href="{GITHUB}">GitHub</a>')
    return '<nav class="demos" aria-label="CSS ML demos"><span>More in pure CSS</span>' + "".join(items) + "</nav>"


def nav_css():
    return """
.demos { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; border-top: 1px solid var(--line); padding-top: 18px; font-size: 12px; }
.demos span { color: var(--muted); margin-right: 4px; }
.demos a { color: var(--fg); text-decoration: none; border: 1px solid var(--line); border-radius: 999px; padding: 4px 12px; }
.demos a[aria-current] { color: var(--accent); border-color: var(--accent); }
@media (hover: hover) { .demos a[href]:hover { border-color: var(--fg); } }
.demos a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.demos a:last-child { margin-left: auto; }
@media (max-width: 660px) { .demos a:last-child { margin-left: 0; } }
""".strip()
