# Copyright (c) 2026 Siyu Cai. SPDX-License-Identifier: MIT
from fractions import Fraction
import numpy as np
from .intervals import Interval

def base(m):
    if not isinstance(m, int) or m < 1:
        raise ValueError('m must be a positive integer')
    n = 1 << (max(64, 3 * m + 7) - 1).bit_length()
    shared = (1, 2, 4, 8, 16, 32)
    private = [c for c in range(1, n) if c not in shared][:3 * m]
    h = {c: np.array([1 - 2 * ((r & c).bit_count() % 2) for r in range(n)], dtype=object) for c in (*shared, *private)}
    u, v = (h[1] + h[2] + h[4], h[8] + h[16])
    columns = [9 * (u + v), 9 * (u - v)]
    for i in range(m):
        a, b, c = private[3 * i:3 * i + 3]
        columns.append(5 * u + 2 * h[32] + 18 * h[a] + h[b] + h[c])
    return np.column_stack(columns)

def gram_entry(i, j):
    return 405 if i == j else 81 if {i, j} == {0, 1} else 135 if min(i, j) < 2 else 79

def instance(labels, bits):
    labels = tuple(labels)
    if not labels or any((x not in (0, 1) for x in labels)) or (not isinstance(bits, int)) or (bits < 8):
        raise ValueError('Nonempty binary labels and B >= 8 required')
    a = base(len(labels))
    s = 1 << bits - 7
    data = 64 * s + s * a
    for i, label in enumerate(labels, 2):
        data[0 if label else 7, i] += 1
    return data

def closed_scatter(labels, bits):
    labels = tuple(labels)
    a = base(len(labels))
    n, d = a.shape
    s = 1 << bits - 7
    rows = [None, None, *(0 if label else 7 for label in labels)]
    result = {}
    for i in range(d):
        for j in range(d):
            value = n * n * s * s * gram_entry(i, j)
            if rows[i] is not None:
                value += n * s * int(a[rows[i], j])
            if rows[j] is not None:
                value += n * s * int(a[rows[j], i])
            if rows[i] is not None and rows[j] is not None:
                value += n * int(rows[i] == rows[j]) - 1
            result[i, j] = value
    return result

def envelope(d, depth):
    delta = Fraction(128, 1 << depth)
    error = 180 * delta + delta * delta
    return {(i, j): Interval(1620 - 180 * delta, 1620 + error) if i == j else Interval(4 * gram_entry(i, j) - error, 4 * gram_entry(i, j) + error) for i in range(d) for j in range(d)}

def target_polynomials(n, x, y, z):
    return ([145800, 810 * z - 270 * x, 405 * (n - 1) - x * x], [145800, 810 * z - 270 * y, 405 * (n - 1) - y * y], [14580, 162 * z - 135 * (x + y), 81 * (n - 1) - x * y])

def masks(m):
    values = [(0,) * m]
    values += [tuple((int(i == j) for i in range(m))) for j in range(m)]
    values += [(1,) * m, tuple((i % 2 for i in range(m))), tuple((1 - i % 2 for i in range(m)))]
    return list(dict.fromkeys(values))
