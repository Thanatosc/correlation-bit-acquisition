# Copyright (c) 2026 Siyu Cai. SPDX-License-Identifier: MIT
from fractions import Fraction
import numpy as np
from .intervals import ci_certificate, scatter_bounds

def residual_certificate(lower, upper, x, y, condition=(), threshold=Fraction(1, 10)):
    if len(condition) != 1:
        return ci_certificate(scatter_bounds(lower, upper), x, y, condition, threshold)
    z = condition[0]
    lo = np.asarray(lower[:, (x, y, z)], dtype=object)
    hi = np.asarray(upper[:, (x, y, z)], dtype=object)
    mid = lo + hi
    n = len(mid)
    sums = mid.sum(axis=0)
    center = n * (mid.T @ mid) - np.outer(sums, sums)
    zz = int(center[2, 2])
    if zz <= 0:
        return ('unknown', None)
    transformed_lo, transformed_hi = (lo.copy(), hi.copy())
    for j in (0, 1):
        coef = int(center[j, 2])
        if coef >= 0:
            transformed_lo[:, j] = zz * lo[:, j] - coef * hi[:, 2]
            transformed_hi[:, j] = zz * hi[:, j] - coef * lo[:, 2]
        else:
            transformed_lo[:, j] = zz * lo[:, j] - coef * lo[:, 2]
            transformed_hi[:, j] = zz * hi[:, j] - coef * hi[:, 2]
    return ci_certificate(scatter_bounds(transformed_lo, transformed_hi), 0, 1, (2,), threshold)

class BoundEvaluator:

    def __init__(self, reader, threshold=Fraction(1, 10), kind='interval'):
        if kind not in ('interval', 'residual'):
            raise ValueError(kind)
        self.reader, self.threshold, self.kind = (reader, Fraction(threshold), kind)
        self.version = -1
        self.cache = {}
        self.calls = 0
        self.residual_calls = 0

    def evaluate(self, test):
        if self.version != self.reader.version:
            self.scatter = scatter_bounds(self.reader.lower, self.reader.upper)
            self.version = self.reader.version
            self.cache.clear()
        if test not in self.cache:
            self.calls += 1
            decision, margin = ci_certificate(self.scatter, *test, threshold=self.threshold)
            source = 'interval'
            if decision == 'unknown' and test[2] and (self.kind == 'residual'):
                self.residual_calls += 1
                decision, margin = residual_certificate(self.reader.lower, self.reader.upper, *test, self.threshold)
                source = 'residual'
            self.cache[test] = (decision, margin, source)
        return self.cache[test]
