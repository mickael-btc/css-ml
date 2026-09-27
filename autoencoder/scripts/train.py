"""Train an autoencoder whose two halves both run in CSS, and export them as integers.

    .venv/bin/python autoencoder/scripts/train.py      # ~2 min on a laptop

Pipeline
  1. MNIST (from ../data/, shared with the other demos)
  2. encoder input: each digit rasterised to a binary 8x8 drawing at a random size and position,
     then bounding-box normalised exactly like the CSS does it (norm.py, copied from the digit demo)
  3. decoder target: the same digit as a clean 16x16 greyscale image
  4. VAE: encoder 64 -> 64 -> 32 -> (mu, log var) in 2D, decoder 2 -> 32 -> 64 -> 256,
     trained with Adam on binary cross-entropy + KL divergence, plus a small linear digit
     classifier reading mu (training only, never in the page): an 8x8 drawing is so coarse that
     a plain 2D VAE mixes 4/9 and 3/5/8, and this nudge keeps each digit in its own region
  5. both halves quantised to int8 weights and integer activations; the encoder's 2 outputs are
     snapped to a MAP x MAP grid of cells, and the decoder draws the centre of the chosen cell

Writes model.json and tests/fixtures.json.
"""
import glob
import gzip
import json
import re
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from norm import GRID, normalize

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT.parent / "data"
MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/{}.gz"

FONT_DIRS = ["/System/Library/Fonts/**", "/usr/share/fonts/**", "C:/Windows/Fonts"]
NOT_DIGIT_FONTS = re.compile(
    r"Symbol|Webdings|Wingding|Zapf|Emoji|Braille|LastResort|Keyboard|Dingbat|Ornaments|NISC|"
    r"STIX|Noto Sans (?!Mono)|Kokonor|Farisi|Mishafi|Diwan|Sana|Ayuthaya|Chalkduster",
    re.I,
)

SIDE = 16                  # decoded image is SIDE x SIDE
ENCODER = [64, 32]
DECODER = [32, 64]
EPOCHS = 30
BATCH = 128
LEARNING_RATE = 1e-3
DRAWINGS = 3               # random 8x8 drawings of every MNIST digit
FONT_DRAWINGS = 60         # fonts give far fewer digits, so draw each one more often
SEPARATE = 100.0            # weight of the auxiliary digit classifier that keeps digits apart on the map

MAP = 20                   # the map is MAP x MAP cells
Z_UNITS = 100              # the encoder's integer output is roughly round(z * Z_UNITS)
LEVELS = 100               # pixel intensity 0..LEVELS

rng = np.random.default_rng(0)


# ---------------------------------------------------------------- data

def load_mnist(name):
    path = DATA / f"{name}.gz"
    if not path.exists():
        DATA.mkdir(exist_ok=True)
        urllib.request.urlretrieve(MNIST_URL.format(name), path)

    raw = gzip.open(path).read()
    if "images" in name:
        return np.frombuffer(raw, np.uint8, offset=16).reshape(-1, 28, 28) / 255.0

    return np.frombuffer(raw, np.uint8, offset=8)


def render_digits(font):
    """0-9 in one font, as 28x28 grayscale images like MNIST (copied from the digit demo)."""
    glyphs = []
    for digit in range(10):
        image = Image.new("L", (28, 28))
        ImageDraw.Draw(image).text((14, 14), str(digit), fill=255, font=font, anchor="mm")
        glyphs.append(np.asarray(image) / 255.0)

    return glyphs


def font_digits():
    """Digits from every installed font: clean strokes, close to how people click on a grid."""
    paths = sorted(p for d in FONT_DIRS for p in glob.glob(f"{d}/*.tt[fc]", recursive=True))
    images, labels = [], []

    for path in paths:
        if NOT_DIGIT_FONTS.search(path):
            continue
        try:
            font = ImageFont.truetype(path, 24)
        except OSError:
            continue

        glyphs = render_digits(font)
        has_real_digits = min(g.sum() for g in glyphs) >= 5 and not np.allclose(glyphs[1], glyphs[7])
        if has_real_digits:
            images += glyphs
            labels += range(10)

    return np.array(images), np.array(labels, np.uint8)


def rasterize(image, size, dx, dy, threshold):
    """Crop to the ink, scale the long side to `size` cells, place on the 8x8 grid, binarise.

    Copied from the digit demo (scripts/train.py) so the drawings look the same.
    """
    ys, xs = np.nonzero(image > 0.2)
    if len(ys) == 0:
        return np.zeros((GRID, GRID))

    crop = image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    scale = size / max(h, w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))

    # area-average downsampling via 8x supersampling
    yi = ((np.arange(new_h * 8) + 0.5) / (new_h * 8) * h).astype(int)
    xi = ((np.arange(new_w * 8) + 0.5) / (new_w * 8) * w).astype(int)
    small = crop[yi][:, xi].reshape(new_h, 8, new_w, 8).mean(axis=(1, 3))

    top = min(max((GRID - new_h) // 2 + dy, 0), GRID - new_h)
    left = min(max((GRID - new_w) // 2 + dx, 0), GRID - new_w)
    out = np.zeros((GRID, GRID))
    out[top:top + new_h, left:left + new_w] = small > threshold

    return out


def drawings(images):
    """One random 8x8 drawing per image (random size, position, stroke), normalised like the CSS."""
    X = np.zeros((len(images), GRID * GRID), np.uint8)
    for i, image in enumerate(images):
        size = rng.integers(5, 9)
        dx, dy = rng.integers(-4, 5, size=2)
        threshold = rng.uniform(0.2, 0.6)
        X[i] = rasterize(image, size, dx, dy, threshold).ravel()

    return normalize(X)


def ink_box(image):
    """A square box around the ink, centred, at least 20x20 like MNIST's digit box."""
    ys, xs = np.nonzero(image > 0.1)
    side = max(20, ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
    top = (ys.min() + ys.max() + 1 - side) // 2
    left = (xs.min() + xs.max() + 1 - side) // 2
    padded = np.pad(image, side)
    return padded[top + side:top + 2 * side, left + side:left + 2 * side]


def shrink(images, crop=lambda image: image[4:24, 4:24]):
    """MNIST digits sit in a centred 20x20 box: crop it and resize to SIDE x SIDE."""
    out = np.empty((len(images), SIDE * SIDE), np.float32)
    for i, image in enumerate(images):
        small = Image.fromarray((crop(image) * 255).astype(np.uint8)).resize((SIDE, SIDE), Image.LANCZOS)
        out[i] = np.asarray(small, np.float32).ravel() / 255.0

    return np.clip(out, 0, 1)


# ---------------------------------------------------------------- numpy VAE

def dense_init(dims):
    weights = [rng.normal(0, np.sqrt(2 / n_in), (n_in, n_out)).astype(np.float32)
               for n_in, n_out in zip(dims, dims[1:])]
    biases = [np.zeros(n_out, np.float32) for n_out in dims[1:]]
    return weights, biases


def mlp_forward(weights, biases, x):
    """ReLU hidden layers, linear last layer. Returns every layer's input and pre-activation."""
    inputs, pre = [], []
    for i, (W, b) in enumerate(zip(weights, biases)):
        inputs.append(x)
        z = x @ W + b
        pre.append(z)
        x = z if i == len(weights) - 1 else np.maximum(z, 0)

    return x, inputs, pre


def mlp_backward(weights, inputs, pre, grad):
    """Gradients for weights and biases, and the gradient with respect to the MLP's input."""
    n = len(weights)
    grad_W, grad_b = [None] * n, [None] * n
    for i in reversed(range(n)):
        grad_W[i] = inputs[i].T @ grad
        grad_b[i] = grad.sum(axis=0)
        grad = grad @ weights[i].T
        if i > 0:
            grad = grad * (pre[i - 1] > 0)

    return grad_W, grad_b, grad


class Adam:
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m[:] = self.beta1 * m + (1 - self.beta1) * g
            v[:] = self.beta2 * v + (1 - self.beta2) * g * g

            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def train(X, T, Y):
    """X: 8x8 drawings the encoder reads; T: the 16x16 images the decoder must draw back; Y: digits."""
    enc_W, enc_b = dense_init([GRID * GRID, *ENCODER, 4])  # 4 = (mu, log var) for 2 dims
    dec_W, dec_b = dense_init([2, *DECODER, SIDE * SIDE])
    head_W, head_b = dense_init([2, 10])
    params = enc_W + enc_b + dec_W + dec_b + head_W + head_b
    optimizer = Adam(params, lr=LEARNING_RATE)
    X = X.astype(np.float32)

    for epoch in range(EPOCHS):
        if epoch == int(EPOCHS * 0.7):
            optimizer.lr = LEARNING_RATE / 3

        order = rng.permutation(len(X))
        total_rec = total_kl = total_ce = 0.0
        for start in range(0, len(X), BATCH):
            batch = order[start:start + BATCH]
            x, t, y = X[batch], T[batch], Y[batch]
            n = len(x)

            stats, enc_in, enc_pre = mlp_forward(enc_W, enc_b, x)
            mu, logvar = stats[:, :2], np.clip(stats[:, 2:], -8, 8)
            eps = rng.standard_normal(mu.shape).astype(np.float32)
            std = np.exp(0.5 * logvar)
            z = mu + std * eps

            logits, dec_in, dec_pre = mlp_forward(dec_W, dec_b, z)
            p = sigmoid(logits)

            rec = -(t * np.log(p + 1e-7) + (1 - t) * np.log(1 - p + 1e-7)).sum() / n
            kl = -0.5 * (1 + logvar - mu ** 2 - np.exp(logvar)).sum() / n
            total_rec += rec * n
            total_kl += kl * n

            g_dec_W, g_dec_b, g_z = mlp_backward(dec_W, dec_in, dec_pre, (p - t) / n)

            probs = softmax(mu @ head_W[0] + head_b[0])
            total_ce += -np.log(probs[np.arange(n), y] + 1e-7).sum()
            g_logits = probs.copy()
            g_logits[np.arange(n), y] -= 1
            g_logits *= SEPARATE / n
            g_head_W, g_head_b = mu.T @ g_logits, g_logits.sum(axis=0)

            g_mu = g_z + mu / n + g_logits @ head_W[0].T
            g_logvar = g_z * eps * 0.5 * std + 0.5 * (np.exp(logvar) - 1) / n
            g_enc_W, g_enc_b, _ = mlp_backward(enc_W, enc_in, enc_pre, np.concatenate([g_mu, g_logvar], 1))

            optimizer.step(g_enc_W + g_enc_b + g_dec_W + g_dec_b + [g_head_W] + [g_head_b])

        print(f"epoch {epoch + 1}/{EPOCHS}  reconstruction {total_rec / len(X):.1f}  kl {total_kl / len(X):.2f}  "
              f"digit loss {total_ce / len(X):.2f}",
              flush=True)

    # the encoder keeps only mu: in the browser it gives one point, not a distribution
    enc_W[-1], enc_b[-1] = enc_W[-1][:, :2], enc_b[-1][:2]
    return (enc_W, enc_b), (dec_W, dec_b)


# ---------------------------------------------------------------- integers

def css_round(x):
    """A registered <integer> custom property rounds to nearest, halves toward +infinity."""
    return np.floor(x + 0.5).astype(np.int64)


def odd(n):
    n = max(1, int(round(n)))
    return n + 1 - n % 2


def quantize_layer(W, b, input_scale):
    """Weights to int8; the bias is scaled to match the integer inputs it is added to."""
    w_scale = 127 / np.abs(W).max()
    Wq = np.round(W * w_scale).astype(int)
    bq = np.round(b * input_scale * w_scale).astype(int)

    return Wq, bq, w_scale


def quantize_hidden(W, b, a, scale):
    """One integer ReLU layer, divided by an odd R to stay around 0..255.

    Odd R means an integer / R never lands exactly on k + 0.5, so CSS and numpy can't round differently.
    """
    Wq, bq, w_scale = quantize_layer(W, b, scale)
    z = a @ Wq + bq
    R = odd(np.percentile(np.maximum(z, 0), 99.9) / 255)

    return {"W": Wq.tolist(), "b": bq.tolist(), "R": R}, css_round(np.maximum(z, 0) / R), scale * w_scale / R


def quantize_encoder(weights, biases, calibration):
    """Binary 8x8 in, two integers out: z = (bias + sum) / R, R odd, about Z_UNITS per latent unit.

    Returns the layers and the exact number of integer units per latent unit.
    """
    layers = []
    scale = 1.0
    a = calibration.astype(np.int64)
    for W, b in zip(weights[:-1], biases[:-1]):
        layer, a, scale = quantize_hidden(W, b, a, scale)
        layers.append(layer)

    Wq, bq, w_scale = quantize_layer(weights[-1], biases[-1], scale)
    R = odd(scale * w_scale / Z_UNITS)
    layers.append({"W": Wq.tolist(), "b": bq.tolist(), "R": R, "linear": True})

    return layers, scale * w_scale / R


def quantize_decoder(weights, biases, calibration, units):
    """Integer latent in (at `units` per latent unit); the output folds in a hard sigmoid 0..LEVELS."""
    layers = []
    scale = units
    a = calibration.astype(np.int64)
    for W, b in zip(weights[:-1], biases[:-1]):
        layer, a, scale = quantize_hidden(W, b, a, scale)
        layers.append(layer)

    # hard sigmoid: clamp(0, (logit + 2) / 4, 1), so shift the bias by 2 logits before quantizing
    Wq, bq, w_scale = quantize_layer(weights[-1], biases[-1] + 2, scale)
    layers.append({"W": Wq.tolist(), "b": bq.tolist(), "R": odd(4 * scale * w_scale / LEVELS), "top": LEVELS})

    return layers


def int_forward(layers, X):
    """A network exactly as the CSS computes it; returns every layer's integer output."""
    a = X.astype(np.int64)
    outputs = []
    for layer in layers:
        z = a @ np.array(layer["W"]) + np.array(layer["b"])
        if "top" in layer:
            a = np.clip(css_round(z / layer["R"]), 0, layer["top"])
        elif layer.get("linear"):
            a = css_round(z / layer["R"])
        else:
            a = css_round(np.maximum(z, 0) / layer["R"])
        outputs.append(a)

    return outputs


# ---------------------------------------------------------------- the map

def cell_of(z, grid):
    """Map column and row of integer codes: floor((z - origin) / width), clamped; row 0 is the top.

    CSS has no floor(), so it computes round((z - origin - (W - 1) / 2) / W): with an odd width W
    that is exactly floor, and never a .5 tie.
    """
    W, origin = grid["width"], grid["origin"]
    col = np.clip(css_round((z[:, 0] - origin - (W - 1) // 2) / W), 0, MAP - 1)
    row = MAP - 1 - np.clip(css_round((z[:, 1] - origin - (W - 1) // 2) / W), 0, MAP - 1)

    return col, row


def centres(grid):
    """Integer latent coordinates of every column and row centre."""
    W, origin = grid["width"], grid["origin"]
    xs = [origin + c * W + (W - 1) // 2 for c in range(MAP)]
    return xs, xs[::-1]


def digit_map(codes, labels, grid):
    """Dominant digit of each cell (nearest training codes) and how crowded the cell is."""
    xs, ys = centres(grid)
    cell_centres = np.array([(x, y) for y in ys for x in xs], np.float32)

    classes, density = [], []
    for c in cell_centres:
        d = ((codes - c) ** 2).sum(1)
        near = np.argpartition(d, 50)[:50]
        classes.append(int(np.bincount(labels[near], minlength=10).argmax()))
        density.append(int((d < (grid["width"] / 2) ** 2).sum()))

    anchors = {}
    for digit in range(10):
        median = np.median(codes[labels == digit], axis=0)
        anchors[digit] = int(((cell_centres - median) ** 2).sum(1).argmin())

    return classes, density, anchors


def peak_intermediate(layers, X):
    outputs = int_forward(layers, X)
    inputs = [X.astype(np.int64), *outputs[:-1]]
    return max(int(np.abs(a @ np.array(l["W"])).max() + np.abs(l["b"]).max()) for a, l in zip(inputs, layers))


# ---------------------------------------------------------------- main

def handdrawn_fixtures():
    """The digit demo's hand-drawn 8x8 test drawings (never trained on), raw as drawn."""
    fixtures = json.loads((ROOT.parent / "tests" / "fixtures.json").read_text())
    return np.array(fixtures["X"], np.int64), np.array(fixtures["label"])


def main():
    images = load_mnist("train-images-idx3-ubyte")
    labels = load_mnist("train-labels-idx1-ubyte")
    font_images, font_labels = font_digits()
    print(f"font digits: {len(font_images)}")

    X = np.concatenate([drawings(images) for _ in range(DRAWINGS)] +
                       [drawings(font_images) for _ in range(FONT_DRAWINGS)])
    T = np.concatenate([shrink(images)] * DRAWINGS + [shrink(font_images, ink_box)] * FONT_DRAWINGS)
    Y = np.concatenate([labels] * DRAWINGS + [font_labels] * FONT_DRAWINGS)
    print(f"training pairs: {X.shape} -> {T.shape}", flush=True)

    encoder, decoder = train(X, T, Y)

    enc_layers, units = quantize_encoder(*encoder, X[:20000])
    codes = int_forward(enc_layers, X)[-1]
    mnist = slice(0, len(images) * DRAWINGS)
    reach = int(np.percentile(np.abs(codes), 99))
    width = odd(2 * reach / MAP)
    grid = {"width": width, "origin": -(MAP // 2) * width}
    print(f"{units:.1f} integer units per latent unit, cell width {width}, map ±{MAP // 2 * width}")

    xs, ys = centres(grid)
    cells = np.array([(x, y) for y in ys for x in xs])
    dec_layers = quantize_decoder(*decoder, cells, units)

    peak = max(peak_intermediate(enc_layers, X[:20000]), peak_intermediate(dec_layers, cells))
    print(f"peak |intermediate| {peak}")
    assert peak < 2 ** 24

    classes, density, anchors = digit_map(codes, Y, grid)

    # does the encoder put a drawing in the part of the map where its digit lives?
    col, row = cell_of(codes[mnist], grid)
    on_map = np.array(classes)[row * MAP + col] == Y[mnist]
    print(f"MNIST drawings landing on their own digit's colour: {on_map.mean():.1%}")

    hand_X, hand_Y = handdrawn_fixtures()
    hand_codes = int_forward(enc_layers, normalize(hand_X))[-1]
    col, row = cell_of(hand_codes, grid)
    hand_on_map = np.array(classes)[row * MAP + col] == hand_Y
    print(f"hand-drawn fixtures landing on their own digit's colour: {hand_on_map.sum()}/{len(hand_Y)} "
          f"({hand_on_map.mean():.1%})")

    (ROOT / "model.json").write_text(json.dumps({
        "encoder": enc_layers, "decoder": dec_layers, "map": MAP, "grid": grid, "units": float(units),
        "levels": int(LEVELS), "xs": xs, "ys": ys, "classes": classes, "density": density, "anchors": anchors,
        "on_map": {"mnist": float(on_map.mean()), "handdrawn": [int(hand_on_map.sum()), len(hand_Y)]},
    }))


if __name__ == "__main__":
    main()
