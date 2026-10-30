# Copyright (c) 2026 Siyu Cai. SPDX-License-Identifier: MIT
from dataclasses import dataclass
from fractions import Fraction
from math import isqrt
import numpy as np

@dataclass(frozen=True)
class Interval:
    lo: int
    hi: int

    def __post_init__(self):
        if self.lo > self.hi:
            raise ValueError('Reversed interval')

    @property
    def width(self):
        return self.hi - self.lo

    def __add__(self, other):
        return Interval(self.lo + other.lo, self.hi + other.hi)

    def __sub__(self, other):
        return Interval(self.lo - other.hi, self.hi - other.lo)

    def __mul__(self, other):
        values = (self.lo * other.lo, self.lo * other.hi, self.hi * other.lo, self.hi * other.hi)
        return Interval(min(values), max(values))

    def scale(self, k):
        return self * Interval(k, k)

    def square(self):
        return Interval(0 if self.lo <= 0 <= self.hi else min(self.lo ** 2, self.hi ** 2), max(self.lo ** 2, self.hi ** 2))

def ceil_sqrt(value):
    root = isqrt(value)
    return root + (root * root != value)

def scatter_bounds(lower, upper):
    lo, hi = (np.asarray(lower, dtype=object), np.asarray(upper, dtype=object))
    if lo.shape != hi.shape or lo.ndim != 2 or lo.shape[0] < 2 or np.any(lo > hi):
        raise ValueError('Need ordered n-by-d bounds, n >= 2')
    n, d = lo.shape
    mid, rad = (lo + hi, hi - lo)
    sums = mid.sum(axis=0)
    centered = n * mid - sums
    gram = n * (mid.T @ mid) - np.outer(sums, sums)
    linear = np.abs(centered).T @ rad
    rad_norm2 = (rad * rad).sum(axis=0)
    rad_sums = rad.sum(axis=0)
    rad_products = rad.T @ rad
    result = {}
    for a in range(d):
        for b in range(a, d):
            c = int(gram[a, b])
            error = int(linear[a, b] + linear[b, a])
            if a == b:
                interval = Interval(max(0, c - error), c + error + n * int(rad_norm2[a]))
            else:
                quadratic = min(n * int(rad_products[a, b]) + int(rad_sums[a] * rad_sums[b]), n * ceil_sqrt(int(rad_norm2[a] * rad_norm2[b])))
                interval = Interval(c - error - quadratic, c + error + quadratic)
            result[a, b] = result[b, a] = interval
    return result

def ci_certificate(scatter, x, y, condition=(), threshold=Fraction(1, 10)):
    threshold = Fraction(threshold)
    if not 0 <= threshold < 1:
        raise ValueError('threshold must be in [0, 1)')
    if len(condition) > 1 or len({x, y, *condition}) != 2 + len(condition):
        raise ValueError('Only distinct variables with |condition| <= 1 are supported')
    involved = (x, y, *condition)
    exact = all((scatter[a, b].width == 0 for a in involved for b in involved))
    a, b, c = (scatter[x, x], scatter[y, y], scatter[x, y])
    if condition:
        z = condition[0]
        zz, xz, yz = (scatter[z, z], scatter[x, z], scatter[y, z])
        if zz.lo <= 0:
            return ('singular' if exact else 'unknown', None)
        numerator = c * zz - xz * yz
        vx, vy = (a * zz - xz.square(), b * zz - yz.square())
    else:
        numerator, vx, vy = (c, a, b)
    if vx.lo <= 0 or vy.lo <= 0:
        return ('singular' if exact else 'unknown', None)
    p, q = (threshold.numerator, threshold.denominator)
    margin = numerator.square().scale(q * q) - (vx * vy).scale(p * p)
    if margin.hi <= 0:
        return ('independent', margin)
    if margin.lo > 0:
        return ('dependent', margin)
    return ('unknown', margin)

def exact_scatter(data):
    rows = [[int(v) for v in row] for row in data]
    n, d = (len(rows), len(rows[0]))
    sums = [sum((row[j] for row in rows)) for j in range(d)]
    return {(a, b): n * sum((row[a] * row[b] for row in rows)) - sums[a] * sums[b] for a in range(d) for b in range(d)}

def exact_ci(scatter, x, y, condition=(), threshold=Fraction(1, 10)):
    vx, vy, cov = (Fraction(scatter[a, b]) for a, b in ((x, x), (y, y), (x, y)))
    if condition:
        z = condition[0]
        zz = scatter[z, z]
        if zz <= 0:
            return 'singular'
        vx -= Fraction(scatter[x, z] ** 2, zz)
        vy -= Fraction(scatter[y, z] ** 2, zz)
        cov -= Fraction(scatter[x, z] * scatter[y, z], zz)
    if vx <= 0 or vy <= 0:
        return 'singular'
    return 'independent' if cov * cov <= Fraction(threshold) ** 2 * vx * vy else 'dependent'
