import gzip
import os

import numpy as np


def to_signed(word):
    value = int(word, 2)
    return value - 256 if word[0] == "1" else value


def read_layer(out_dir, count, layer):
    rows = []
    for i in range(count):
        with open(os.path.join(out_dir, "%d_layer%d.txt" % (i, layer))) as f:
            rows.append([to_signed(line.strip()) for line in f if line.strip()])
    return np.array(rows)


def read_stats(out_dir):
    stats = {}
    with open(os.path.join(out_dir, "stats.txt")) as f:
        for line in f:
            k, n, abs_err, _, max_exact, _, _ = [int(v) for v in line.split()]
            stats[k] = {"n": n, "abs_err": abs_err, "max_exact": max_exact}
    return stats


def nmed(out_dir, exact_dir, layer):
    stats, ref = read_stats(out_dir), read_stats(exact_dir)
    return stats[layer]["abs_err"] / stats[layer]["n"] / ref[layer]["max_exact"]


def nmed_all_inputs(out_dir):
    with open(os.path.join(out_dir, "nmed_all.txt")) as f:
        count, abs_err, sum_err, max_err, err_count = [int(v) for v in f.read().split()]
    return {"nmed_all_inputs": abs_err / count / 16384.0, "med": abs_err / count,
            "mean_error": sum_err / count, "max_error": max_err, "error_rate": err_count / count}


def accuracy(out_dir, count, label_file):
    with gzip.open(label_file, "rb") as f:
        labels = np.frombuffer(f.read(), dtype=np.uint8)[8:]
    pred = read_layer(out_dir, count, 2).argmax(axis=1)
    return 100.0 * np.mean(pred == labels[:count])


def images(out_dir, count, zero_point, scale):
    return np.clip((read_layer(out_dir, count, 2) - zero_point) * scale, 0, 1)


def psnr(a, b):
    return 10 * np.log10(1.0 / np.maximum(np.mean((a - b) ** 2, axis=1), 1e-10))


def window(size=11, sigma=1.5):
    x = np.arange(size) - size // 2
    g = np.exp(-x ** 2 / (2.0 * sigma ** 2))
    g = g / g.sum()
    return np.outer(g, g)


def blur(img, win):
    k = win.shape[0]
    h = img.shape[1] - k + 1
    out = np.zeros((img.shape[0], h, h))
    for i in range(k):
        for j in range(k):
            out += win[i, j] * img[:, i:i + h, j:j + h]
    return out


def ssim(a, b):
    x, y = a.reshape(-1, 28, 28), b.reshape(-1, 28, 28)
    win = window()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mx, my = blur(x, win), blur(y, win)
    sxx, syy, sxy = blur(x * x, win) - mx * mx, blur(y * y, win) - my * my, blur(x * y, win) - mx * my
    m = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))
    return m.mean(axis=(1, 2))
