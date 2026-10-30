# Copyright (c) 2026 Siyu Cai. SPDX-License-Identifier: MIT
from fractions import Fraction as F
from functools import lru_cache
from math import isqrt
import numpy as np
from .hardness_fixed_threshold import squares
from .intervals import Interval, ci_certificate, scatter_bounds

@lru_cache(None)
def parameters(threshold):
    tau = F(threshold)
    if not 0 < tau < 1:
        raise ValueError('Need a fixed rational threshold in (0,1)')
    p, q = (tau.numerator, tau.denominator)
    c_mid = tau + tau * tau * (1 - tau) / 2
    r_squared = (q * q * (1 + c_mid) / 2 - p * p) / 2
    exponent = 0
    while True:
        scale = 2 ** exponent
        scaled = r_squared * scale * scale
        r = F(isqrt(scaled.numerator // scaled.denominator), scale)
        c = 2 * (p * p + 2 * r * r) / F(q * q) - 1
        if tau < c < tau + tau * tau * (1 - tau):
            break
        exponent += 1
    epsilon = F(1, 2)
    while True:
        expected = {(0, 0): F(1), (1, 1): F(1), (2, 2): F(1), (0, 1): c, (0, 2): tau, (1, 2): tau}
        box = {}
        for (i, j), center in expected.items():
            box[i, j] = box[j, i] = Interval(center - epsilon, center + epsilon)
        comparisons = [ci_certificate(box, *test, threshold=tau) for test in ((0, 1, (2,)), (0, 2, (1,)), (1, 2, (0,)))]
        marginal, _ = ci_certificate(box, 0, 1, threshold=tau)
        if marginal == 'dependent' and all((decision == 'independent' and margin.hi < 0 for decision, margin in comparisons)):
            break
        epsilon /= 2
    return {'tau': tau, 'p': p, 'q': q, 'r': r, 'c': c, 'epsilon': epsilon}

def construct(weights, target, threshold=F(1, 10)):
    weights = tuple((int(a) for a in weights))
    target = int(target)
    S, m = (sum(weights), len(weights))
    if not weights or min(weights) <= 0 or (not 0 < target < S):
        raise ValueError('Need positive weights and 0 < target < their sum')
    par = parameters(F(threshold))
    p, q, r, c0, epsilon = (par[key] for key in ('p', 'q', 'r', 'c', 'epsilon'))
    h = max(4, q.bit_length())
    attempts = 0
    while True:
        attempts += 1
        blocks = 8 * h + 16
        n = m + 4 * blocks
        A = p * n + 1
        H = 2 ** h + 1
        M = (r * H).numerator // (r * H).denominator
        a = 2 * (H * H + blocks - 1)
        E = 16 * A * A * sum((w * w for w in weights)) + 8 * A * A * target * target + 4 * p * p * target * target + 4 * A * A * S * S
        square_sum = (q * q - p * p) * (H * H + blocks - 1) - 2 * (M * M + (blocks - 1) * A * A) - E // 2
        if M < A or square_sum < 0:
            h += 1
            continue
        fillers = squares(square_sum)
        assert len(fillers) <= blocks - 2
        z = np.zeros(n, dtype=object)
        W = np.zeros(n, dtype=object)
        V = np.zeros(n, dtype=object)
        V[:m] = [4 * A * w for w in weights]
        for j in range(blocks):
            at = m + 4 * j
            height = H if j == 1 else 1
            amplitude = M if j == 1 else A
            z[at:at + 4] = [-height, height, 0, 0]
            W[at:at + 4] = [amplitude, amplitude, -amplitude, -amplitude]
        V[m:m + 4] = [-2 * A * target - p * target - A * S, 2 * A * target - p * target - A * S, p * target - A * S, p * target - A * S]
        for j, v in enumerate(fillers, start=2):
            at = m + 4 * j
            V[at + 2], V[at + 3] = (v, -v)
        U = p * z + W
        X, Y, Z = (U + V, U - V, 1 + z)
        assert sum(z) == sum(W) == sum(V) == 0
        assert z @ z == a and z @ W == 0 and (U @ V == 0)
        assert X @ X == Y @ Y == q * q * a
        assert z @ V == 4 * A * target
        loz = Z - np.asarray(W <= 0, dtype=object)
        assert all((int(v) % 2 == 0 for v in loz))
        lower = np.column_stack((X, Y, loz))
        upper = lower.copy()
        upper[:, 2] += 1
        intervals = scatter_bounds(lower, upper)
        expected = {(0, 0): F(1), (1, 1): F(1), (2, 2): F(1), (0, 1): c0, (0, 2): par['tau'], (1, 2): par['tau']}
        scaling = [q, q, 1]
        inside = True
        for (i, j), center in expected.items():
            normalizer = 4 * n * a * scaling[i] * scaling[j]
            interval = intervals[i, j]
            if not (F(interval.lo, normalizer) >= center - epsilon and F(interval.hi, normalizer) <= center + epsilon):
                inside = False
                break
        if not inside:
            h += 1
            continue
        assert all((ci_certificate(intervals, *test, threshold=par['tau'])[0] == 'independent' for test in ((0, 1, (2,)), (0, 2, (1,)), (1, 2, (0,)))))
        assert ci_certificate(intervals, 0, 1, threshold=par['tau'])[0] == 'dependent'
        magnitude = max((abs(int(v)) for v in np.concatenate((lower.ravel(), upper.ravel()))))
        bits = magnitude.bit_length() + 2
        offset = 2 ** (bits - 1)
        lower += offset
        upper += offset
        nominal = np.column_stack((X, Y, Z)) + offset
        assert lower.min() >= 0 and upper.max() < 2 ** bits
        info = {'n': n, 'd': 3, 'm': m, 'bits': bits, 'depths': [bits, bits, bits - 1], 'h': h, 'blocks': blocks, 'A': A, 'H': H, 'M': M, 'a': a, 'square_sum': square_sum, 'square_terms': len(fillers), 'threshold': str(par['tau']), 'r': str(r), 'nominal_c': str(c0), 'epsilon': str(epsilon), 'offset': offset, 'attempts': attempts, 'weights': list(weights), 'target': target}
        return (lower, upper, nominal, W, info)
