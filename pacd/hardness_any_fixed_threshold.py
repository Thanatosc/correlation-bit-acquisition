# Copyright (c) 2026 Siyu Cai. SPDX-License-Identifier: MIT
from fractions import Fraction
from functools import lru_cache
from math import lcm
import numpy as np
from .hardness_fixed_threshold import squares
from .intervals import Interval, ci_certificate

def ceil_fraction(value):
    value = Fraction(value)
    return -(-value.numerator // value.denominator)

@lru_cache(None)
def parameters(threshold):
    tau = Fraction(threshold)
    if not 0 < tau < 1:
        raise ValueError('A fixed rational threshold in (0,1) is required')
    c = tau + tau * tau * (1 - tau) / 2
    q = (1 + tau) / 2
    epsilon = Fraction(1, 2)
    halvings = 0
    while True:
        values = {(0, 0): (2, 2), (1, 1): (2, 2), (2, 2): (2, 2 + epsilon), (0, 1): (2 * c, 2 * c), (0, 2): (2 * tau - 2 * epsilon, 2 * tau + 2 * epsilon), (1, 2): (2 * tau - 2 * epsilon, 2 * tau + 2 * epsilon)}
        denominator = lcm(*(Fraction(v).denominator for ends in values.values() for v in ends))
        scatter = {}
        for (x, y), (lo, hi) in values.items():
            scatter[x, y] = scatter[y, x] = Interval(int(lo * denominator), int(hi * denominator))
        records = [ci_certificate(scatter, *test, tau) for test in ((0, 1, (2,)), (0, 2, (1,)), (1, 2, (0,)))]
        if all((decision == 'independent' and margin.hi < 0 for decision, margin in records)):
            break
        epsilon /= 2
        halvings += 1
    dbase = lcm(2 * c.denominator, q.denominator)
    D = dbase * ceil_fraction(Fraction(4, 1) / (1 - c))
    h = tau / q
    C = h.denominator * ceil_fraction(max(4 / epsilon, 4 / (1 - h * h), Fraction(4)))
    return (tau, c, q, epsilon, D, C, halvings)

def construct(weights, target, threshold=Fraction(1, 10)):
    tau, c, q, epsilon, D, C, halvings = parameters(Fraction(threshold))
    total, m = (sum(weights), len(weights))
    if not weights or min(weights) < 1 or (not 0 < target < total):
        raise ValueError('Positive weights and 0<target<sum(weights) required')
    L = D * (total + 1)
    M = C * L * (m + 1)
    u, z = (q * L, tau * M / q)
    amounts = [L * L * (1 - c) / 2 - 2 * (sum((a * a for a in weights)) + total * total) - 1, L * L * (1 + c) / 2 - u * u, M * M - z * z - target * target]
    assert u.denominator == z.denominator == 1
    assert all((v.denominator == 1 and v >= 0 for v in amounts))
    rows = [[2 * a, -2 * a, 0] for a in weights]
    rows.extend([[-2 * total, 2 * total, 0], [1, -1, -target], [-1, 1, target]])
    for v in squares(int(amounts[0])):
        rows.extend([[v, -v, 0], [-v, v, 0]])
    rows.extend([[int(u), int(u), int(z)], [-int(u), -int(u), -int(z)]])
    for v in squares(int(amounts[1])):
        rows.extend([[v, v, 0], [-v, -v, 0]])
    for v in squares(int(amounts[2])):
        rows.extend([[0, 0, v], [0, 0, -v]])
    centered = np.array(rows, dtype=object)
    assert all(centered.sum(axis=0) == 0)
    gram = centered.T @ centered
    cross = 2 * tau * L * M
    expected = np.array([[2 * L * L, 2 * c * L * L, cross - 2 * target], [2 * c * L * L, 2 * L * L, cross + 2 * target], [cross - 2 * target, cross + 2 * target, 2 * M * M]], dtype=object)
    assert np.array_equal(gram, expected)
    bits = max((abs(int(v)) for v in centered.flat)).bit_length() + 2
    offset = 2 ** (bits - 1)
    lower = centered + offset
    upper = lower.copy()
    upper[:m, 2] += 1
    assert all((int(v) % 2 == 0 for v in lower[:m, 2]))
    assert lower.min() >= 0 and upper.max() < 2 ** bits
    return (lower, upper, {'n': len(rows), 'm': m, 'L': L, 'M': M, 'bits': bits, 'offset': offset, 'threshold': str(tau), 'c': str(c), 'q': str(q), 'epsilon': str(epsilon), 'D': D, 'C': C, 'epsilon_halvings': halvings})
