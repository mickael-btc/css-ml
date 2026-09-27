"""Train the digit network and export integer weights for the CSS build.

    python3 scripts/train.py        # ~5 min on a laptop

Pipeline
  1. MNIST (downloaded to data/) + digits rendered from the system's fonts
  2. rasterised to binary 8x8 at random sizes (5-8 cells) and positions
  3. bounding-box normalised exactly like the CSS does it (norm.py)
  4. MLP 64 -> 64 -> 32 -> 10 with ReLU, trained with Adam on cross-entropy
  5. quantised to int8 weights and integer activations, so every CSS value is an integer

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

from handdrawn import dataset as handdrawn_dataset
from norm import GRID, normalize

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/{}.gz"

HIDDEN = [64, 32]
EPOCHS = 30
BATCH = 128
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-4
MNIST_REPS = 3    # augmented copies of MNIST
FONT_REPS = 60    # augmented copies of the font digits (there are far fewer of them)

FONT_DIRS = ["/System/Library/Fonts/**", "/usr/share/fonts/**", "C:/Windows/Fonts"]
NOT_DIGIT_FONTS = re.compile(
    r"Symbol|Webdings|Wingding|Zapf|Emoji|Braille|LastResort|Keyboard|Dingbat|Ornaments|NISC|"
    r"STIX|Noto Sans (?!Mono)|Kokonor|Farisi|Mishafi|Diwan|Sana|Ayuthaya|Chalkduster",
    re.I,
)

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
    """0-9 in one font, as 28x28 grayscale images like MNIST."""
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
    """Crop to the ink, scale the long side to `size` cells, place on the 8x8 grid, binarise."""
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


def random_drawings(images, copies):
    """Each image drawn `copies` times at a random size, position and stroke thickness."""
    batches = []
    for _ in range(copies):
        X = np.zeros((len(images), GRID * GRID), np.uint8)
        for i, image in enumerate(images):
            size = rng.integers(5, 9)
            dx, dy = rng.integers(-4, 5, size=2)
            threshold = rng.uniform(0.2, 0.6)
            X[i] = rasterize(image, size, dx, dy, threshold).ravel()
        batches.append(X)

    return np.concatenate(batches)


# ---------------------------------------------------------------- model

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def forward(weights, biases, x):
    """Pre-activations and activations of every layer; the last layer stays linear (logits)."""
    activations, pre_activations = [x], []

    for i, (W, b) in enumerate(zip(weights, biases)):
        z = activations[-1] @ W + b
        pre_activations.append(z)

        is_output = i == len(weights) - 1
        activations.append(z if is_output else np.maximum(z, 0))

    return pre_activations, activations


def backward(weights, pre_activations, activations, labels):
    """Gradients of mean cross-entropy (+ L2 weight decay), weights first then biases."""
    grad = softmax(activations[-1])
    grad[np.arange(len(labels)), labels] -= 1
    grad /= len(labels)

    n = len(weights)
    grad_W, grad_b = [None] * n, [None] * n

    for i in reversed(range(n)):
        grad_W[i] = activations[i].T @ grad + WEIGHT_DECAY * weights[i]
        grad_b[i] = grad.sum(axis=0)

        if i > 0:
            grad = (grad @ weights[i].T) * (pre_activations[i - 1] > 0)

    return grad_W + grad_b


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


def train(X, Y):
    dims = [GRID * GRID, *HIDDEN, 10]
    weights = [rng.normal(0, np.sqrt(2 / n_in), (n_in, n_out)).astype(np.float32)
               for n_in, n_out in zip(dims, dims[1:])]
    biases = [np.zeros(n_out, np.float32) for n_out in dims[1:]]
    optimizer = Adam(weights + biases, lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        if epoch == int(EPOCHS * 0.7):
            optimizer.lr = LEARNING_RATE / 3

        order = rng.permutation(len(X))
        for start in range(0, len(X), BATCH):
            batch = order[start:start + BATCH]
            pre, acts = forward(weights, biases, X[batch].astype(np.float32))
            optimizer.step(backward(weights, pre, acts, Y[batch]))

        print(f"epoch {epoch + 1}/{EPOCHS}", flush=True)

    return weights, biases


# ---------------------------------------------------------------- integer model

def css_round(x):
    """A registered <integer> custom property rounds to nearest, halves toward +infinity."""
    return np.floor(x + 0.5).astype(np.int64)


def quantize_layer(W, b, input_scale):
    """Weights to int8; the bias is scaled to match the integer inputs it is added to."""
    w_scale = 127 / np.abs(W).max()
    Wq = np.round(W * w_scale).astype(int)
    bq = np.round(b * input_scale * w_scale).astype(int)

    return Wq, bq, w_scale


def quantize(weights, biases, calibration):
    """int8 weights; each hidden layer is divided by an odd integer R to keep it around 0..255.

    Odd R means x / R never lands exactly on k + 0.5, so CSS and numpy can't round differently.
    Every intermediate sum stays far below 2^24, exact even in engines doing float32 CSS math.
    Returns the layers and the factor that turns output integers back into logits.
    """
    layers = []
    scale = 1.0  # integer activation = scale * float activation
    a = calibration.astype(np.int64)

    for W, b in zip(weights[:-1], biases[:-1]):
        Wq, bq, w_scale = quantize_layer(W, b, scale)
        z = a @ Wq + bq

        R = max(1, int(np.percentile(np.maximum(z, 0), 99.9) / 255))
        R += 1 - R % 2  # make it odd
        layers.append({"W": Wq.tolist(), "b": bq.tolist(), "R": R})

        a = css_round(np.maximum(z, 0) / R)
        scale *= w_scale / R

    Wq, bq, w_scale = quantize_layer(weights[-1], biases[-1], scale)
    layers.append({"W": Wq.tolist(), "b": bq.tolist()})

    return layers, float(scale * w_scale)


def int_forward(layers, X):
    """The network exactly as the CSS computes it."""
    a = X.astype(np.int64)
    for layer in layers:
        z = a @ np.array(layer["W"]) + np.array(layer["b"])
        a = css_round(np.maximum(z, 0) / layer["R"]) if "R" in layer else z

    return a


# ---------------------------------------------------------------- main

def main():
    train_images = load_mnist("train-images-idx3-ubyte")
    train_labels = load_mnist("train-labels-idx1-ubyte")
    test_images = load_mnist("t10k-images-idx3-ubyte")
    test_labels = load_mnist("t10k-labels-idx1-ubyte")
    font_images, font_labels = font_digits()
    print(f"font digits: {len(font_images)}")

    X = normalize(np.concatenate([
        random_drawings(train_images, MNIST_REPS),
        random_drawings(font_images, FONT_REPS),
    ]))
    Y = np.concatenate([train_labels] * MNIST_REPS + [font_labels] * FONT_REPS)

    X_test_raw = random_drawings(test_images, 1)
    X_hand_raw, y_hand = handdrawn_dataset()

    weights, biases = train(X, Y)
    layers, scale = quantize(weights, biases, X[:20000])

    test_acc = (int_forward(layers, normalize(X_test_raw)).argmax(1) == test_labels).mean()
    hand_acc = (int_forward(layers, normalize(X_hand_raw)).argmax(1) == y_hand).mean()
    peak = max(np.abs(int_forward(layers[:i + 1], normalize(X_test_raw))).max() for i in range(len(layers)))
    print(f"integer model: MNIST (random size/position) {test_acc:.3f}   "
          f"hand-drawn {hand_acc:.3f}   peak |value| {peak}")
    assert peak < 2 ** 24

    (ROOT / "model.json").write_text(json.dumps({"layers": layers, "scale": scale}))

    # raw (un-normalised) drawings + the integer model's answers: the browser must match exactly
    fixtures = np.concatenate([X_hand_raw, X_test_raw[:60]])
    (ROOT / "tests" / "fixtures.json").write_text(json.dumps({
        "X": fixtures.tolist(),
        "pred": int_forward(layers, normalize(fixtures)).argmax(1).tolist(),
        "label": np.concatenate([y_hand, test_labels[:60]]).tolist(),
    }))


if __name__ == "__main__":
    main()
