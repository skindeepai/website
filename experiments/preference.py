"""CPU reference for the browser's bounded latent preference model."""
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -700, 700)))

def fit(x, y, epochs=260, l2=0.02):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.zeros(x.shape[1]); b = 0.0
    if not len(y): return w, b
    for _ in range(epochs):
        error = sigmoid(x @ w + b) - y
        w -= 0.5 * (x.T @ error / len(y) + l2 * w)
        b -= 0.5 * error.mean()
    return w, b

def optimum(w, scale=1):
    return np.sign(w) * np.clip(scale, 0, 1)

def minimum_change(w, b, reference, probability):
    w = np.asarray(w, dtype=float); reference = np.asarray(reference, dtype=float)
    if not 0 < probability < 1: raise ValueError('Probability must be inside (0,1)')
    if np.any(np.abs(reference) > 1) or not np.isfinite(reference).all(): raise ValueError('Reference outside box')
    target = np.log(probability / (1 - probability))
    if w @ reference + b >= target: return reference.copy(), 'already-met'
    if np.abs(w).sum() + b < target:
        return np.where(w == 0, reference, np.sign(w)), 'unreachable'
    direction = w / np.max(np.abs(w))
    def at(t): return np.clip(reference + t * direction, -1, 1)
    lo, hi = 0.0, 1.0
    while w @ at(hi) + b < target and hi < np.finfo(float).max / 2: hi *= 2
    for _ in range(90):
        mid = (lo + hi) / 2
        if w @ at(mid) + b >= target: hi = mid
        else: lo = mid
    return at(hi), 'reached'

if __name__ == '__main__':
    rng = np.random.default_rng(17)
    x = rng.uniform(-1, 1, (50, 8)); y = (x[:, 0] - x[:, 1] > 0).astype(float)
    w, b = fit(x, y)
    z, status = minimum_change(w, b, np.zeros(8), 0.8)
    print({'ratings': len(y), 'training_fit': float(((x @ w + b > 0) == y).mean()), 'edit_status': status, 'achieved_score': float(sigmoid(w @ z + b))})
