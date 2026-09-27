"""Train a 2D variational autoencoder on MNIST and export the decoder as integers for the CSS build.

    .venv/bin/python latent-space/scripts/train.py      # ~3 min on a laptop

Pipeline
  1. MNIST (from ../data/, shared with the digit demo), cropped to the 20x20 digit box, resized to 16x16
  2. VAE: encoder 256 -> 256 -> 128 -> (mu, log var) in 2D, decoder 2 -> 32 -> 64 -> 256
     trained with Adam on binary cross-entropy + KL divergence
  3. the decoder is quantised to int8 weights and integer activations; its sigmoid output becomes
     a hard sigmoid clamp(0, (logit + 2) / 4, 1) scaled to 0..LEVELS, so every CSS value is an integer
  4. the training set is encoded to find which digit lives where on the map

Writes model.json and tests/fixtures.json.
"""
import gzip
import json
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT.parent / "data"
MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/{}.gz"

SIDE = 16                  # generated image is SIDE x SIDE
ENCODER = [256, 128]
DECODER = [32, 64]
EPOCHS = 40
BATCH = 128
LEARNING_RATE = 1e-3

MAP = 20                   # the map is MAP x MAP cells
Z_SCALE = 100              # latent coordinates reach CSS as integers: round(z * Z_SCALE)
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
        return np.frombuffer(raw, np.uint8, offset=16).reshape(-1, 28, 28)

    return np.frombuffer(raw, np.uint8, offset=8)


def shrink(images):
    """MNIST digits sit in a centred 20x20 box: crop it and resize to SIDE x SIDE."""
    out = np.empty((len(images), SIDE * SIDE), np.float32)
    for i, image in enumerate(images):
        small = Image.fromarray(image[4:24, 4:24]).resize((SIDE, SIDE), Image.LANCZOS)
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


def train(X):
    enc_W, enc_b = dense_init([SIDE * SIDE, *ENCODER, 4])  # 4 = (mu, log var) for 2 dims
    dec_W, dec_b = dense_init([2, *DECODER, SIDE * SIDE])
    params = enc_W + enc_b + dec_W + dec_b
    optimizer = Adam(params, lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        if epoch == int(EPOCHS * 0.7):
            optimizer.lr = LEARNING_RATE / 3

        order = rng.permutation(len(X))
        total_rec = total_kl = 0.0
        for start in range(0, len(X), BATCH):
            x = X[order[start:start + BATCH]]
            n = len(x)

            stats, enc_in, enc_pre = mlp_forward(enc_W, enc_b, x)
            mu, logvar = stats[:, :2], np.clip(stats[:, 2:], -8, 8)
            eps = rng.standard_normal(mu.shape).astype(np.float32)
            std = np.exp(0.5 * logvar)
            z = mu + std * eps

            logits, dec_in, dec_pre = mlp_forward(dec_W, dec_b, z)
            p = sigmoid(logits)

            # per-image sums, averaged over the batch
            rec = -(x * np.log(p + 1e-7) + (1 - x) * np.log(1 - p + 1e-7)).sum() / n
            kl = -0.5 * (1 + logvar - mu ** 2 - np.exp(logvar)).sum() / n
            total_rec += rec * n
            total_kl += kl * n

            g_dec_W, g_dec_b, g_z = mlp_backward(dec_W, dec_in, dec_pre, (p - x) / n)

            g_mu = g_z + mu / n
            g_logvar = g_z * eps * 0.5 * std + 0.5 * (np.exp(logvar) - 1) / n
            g_enc_W, g_enc_b, _ = mlp_backward(enc_W, enc_in, enc_pre, np.concatenate([g_mu, g_logvar], 1))

            optimizer.step(g_enc_W + g_enc_b + g_dec_W + g_dec_b)

        print(f"epoch {epoch + 1}/{EPOCHS}  reconstruction {total_rec / len(X):.1f}  kl {total_kl / len(X):.2f}",
              flush=True)

    return (enc_W, enc_b), (dec_W, dec_b)


def encode(encoder, X):
    stats, _, _ = mlp_forward(*encoder, X)
    return stats[:, :2]


# ---------------------------------------------------------------- integer decoder

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


def quantize(weights, biases, calibration):
    """int8 weights, integer activations; hidden layers divided by an odd R to stay around 0..255.

    Odd R means x / R never lands exactly on k + 0.5, so CSS and numpy can't round differently.
    The output layer folds in the hard sigmoid: pixel = clamp(0, (bias + sum) / R, LEVELS).
    """
    layers = []
    scale = float(Z_SCALE)  # integer activation = scale * float activation
    a = calibration.astype(np.int64)

    for W, b in zip(weights[:-1], biases[:-1]):
        Wq, bq, w_scale = quantize_layer(W, b, scale)
        z = a @ Wq + bq

        R = odd(np.percentile(np.maximum(z, 0), 99.9) / 255)
        layers.append({"W": Wq.tolist(), "b": bq.tolist(), "R": R})

        a = css_round(np.maximum(z, 0) / R)
        scale *= w_scale / R

    # hard sigmoid: (logit + 2) / 4, so shift the bias by 2 logits before quantizing
    Wq, bq, w_scale = quantize_layer(weights[-1], biases[-1] + 2, scale)
    layers.append({"W": Wq.tolist(), "b": bq.tolist(), "R": odd(4 * scale * w_scale / LEVELS), "top": LEVELS})

    return layers


def int_forward(layers, Z):
    """The decoder exactly as the CSS computes it; returns every layer's integer output."""
    a = Z.astype(np.int64)
    outputs = []
    for layer in layers:
        z = a @ np.array(layer["W"]) + np.array(layer["b"])
        if "top" in layer:
            a = np.clip(css_round(z / layer["R"]), 0, layer["top"])
        else:
            a = css_round(np.maximum(z, 0) / layer["R"])
        outputs.append(a)

    return outputs


# ---------------------------------------------------------------- the map

def map_centres(extent):
    """Integer latent coordinates of every cell centre; x grows rightwards, y upwards (row 0 is the top)."""
    steps = [(-extent + (i + 0.5) * 2 * extent / MAP) for i in range(MAP)]
    xs = [int(round(s * Z_SCALE)) for s in steps]
    ys = xs[::-1]
    return xs, ys


def digit_map(codes, labels, extent):
    """Dominant digit of each cell (nearest training codes) and how crowded the cell is."""
    xs, ys = map_centres(extent)
    centres = np.array([(x, y) for y in ys for x in xs], np.float32) / Z_SCALE

    classes, density = [], []
    for c in centres:
        d = ((codes - c) ** 2).sum(1)
        near = np.argpartition(d, 50)[:50]
        classes.append(int(np.bincount(labels[near], minlength=10).argmax()))
        density.append(int((d < (extent / MAP) ** 2).sum()))

    # one label per digit, at the cell closest to that digit's median code
    anchors = {}
    for digit in range(10):
        median = np.median(codes[labels == digit], axis=0)
        anchors[digit] = int(((centres - median) ** 2).sum(1).argmin())

    return classes, density, anchors


# ---------------------------------------------------------------- main

def main():
    X = shrink(load_mnist("train-images-idx3-ubyte"))
    Y = load_mnist("train-labels-idx1-ubyte")
    print(f"training images: {X.shape}")

    encoder, (dec_W, dec_b) = train(X)
    codes = encode(encoder, X)
    extent = float(np.ceil(np.percentile(np.abs(codes), 99) * 10) / 10)
    print(f"latent extent ±{extent}")

    xs, ys = map_centres(extent)
    grid = np.array([(x, y) for y in ys for x in xs])
    layers = quantize(dec_W, dec_b, grid)

    outputs = int_forward(layers, grid)
    inputs = [grid, *outputs[:-1]]
    peak = max(int(np.abs(a @ np.array(l["W"])).max() + np.abs(l["b"]).max()) for a, l in zip(inputs, layers))
    print(f"peak |intermediate| {peak}")
    assert peak < 2 ** 24

    # how close is the integer decoder to the float one (on 0..1 pixels)?
    _, _, pre = mlp_forward(dec_W, dec_b, grid / Z_SCALE)
    float_pixels = np.clip((pre[-1] + 2) / 4, 0, 1)
    print(f"integer vs float decoder: mean |diff| {np.abs(outputs[-1] / LEVELS - float_pixels).mean():.4f}")

    classes, density, anchors = digit_map(codes, Y, extent)
    (ROOT / "model.json").write_text(json.dumps({
        "layers": layers, "map": MAP, "extent": extent, "z_scale": Z_SCALE, "levels": LEVELS,
        "xs": xs, "ys": ys, "classes": classes, "density": density, "anchors": anchors,
    }))

    # every cell's integer outputs: the browser must match exactly
    (ROOT / "tests" / "fixtures.json").write_text(json.dumps({
        "hidden": [o.tolist() for o in outputs[:-1]],
        "pixels": outputs[-1].tolist(),
    }))


if __name__ == "__main__":
    main()
