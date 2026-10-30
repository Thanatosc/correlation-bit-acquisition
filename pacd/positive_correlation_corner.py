# Copyright (c) 2026 Siyu Cai. SPDX-License-Identifier: MIT
from fractions import Fraction
import numpy as np
from .intervals import Interval, ci_certificate, exact_ci, exact_scatter, scatter_bounds

def positive_corner_certificate(lower, upper, threshold=Fraction(1, 10)):
    lo, hi = (np.asarray(lower, dtype=object), np.asarray(upper, dtype=object))
    tau = Fraction(threshold)
    if lo.ndim != 2 or lo.shape != hi.shape or lo.shape[0] < 2 or (lo.shape[1] != 2) or np.any(lo > hi):
        raise ValueError('Ordered n-by-2 bounds with n >= 2 required')
    if not 0 < tau < 1:
        raise ValueError('Threshold must be strictly between zero and one')
    if any((int(v) != v for v in lo.flat)) or any((int(v) != v for v in hi.flat)):
        raise ValueError('Integer bounds required')
    scatter = scatter_bounds(lo, hi)
    record = {'decision': 'unknown', 'threshold': str(tau), 'reason': None, 'derivative_bounds': [], 'minimum_corner': None, 'corner_decision': None}
    if scatter[0, 1].lo <= 0 or scatter[0, 0].lo <= 0 or scatter[1, 1].lo <= 0:
        record['reason'] = 'positive covariance and variances not certified'
        return record
    n = len(lo)
    centered = {}
    lo_sums, hi_sums = (lo.sum(axis=0), hi.sum(axis=0))
    for c in range(2):
        for row in range(n):
            centered[row, c] = Interval(int((n - 1) * lo[row, c] - hi_sums[c] + hi[row, c]), int((n - 1) * hi[row, c] - lo_sums[c] + lo[row, c]))
    corner = lo.copy()
    for c in range(2):
        for row in range(n):
            derivative = centered[row, 1 - c] * scatter[c, c] - centered[row, c] * scatter[0, 1]
            sign = 1 if derivative.lo > 0 else -1 if derivative.hi < 0 else 0
            record['derivative_bounds'].append({'row': row, 'column': c, 'bounds': [derivative.lo, derivative.hi], 'sign': sign})
            if sign == 0:
                record['reason'] = 'a coordinate derivative is not strictly signed'
                return record
            if sign < 0:
                corner[row, c] = hi[row, c]
    record['minimum_corner'] = [[int(v) for v in row] for row in corner]
    record['corner_decision'] = exact_ci(exact_scatter(corner), 0, 1, threshold=tau)
    if record['corner_decision'] == 'dependent':
        record['decision'] = 'dependent'
        record['reason'] = 'positive correlation has a certified minimum corner above threshold'
    else:
        record['reason'] = 'minimum corner is not strictly above threshold'
    return record
