# Copyright (c) 2026 Siyu Cai. SPDX-License-Identifier: MIT
from fractions import Fraction as F
from functools import lru_cache
from itertools import combinations
import numpy as np
from .intervals import Interval
from .whole_column_or import structural_tests as one_block_tests
from .whole_column_or_linear_kappa import base as one_block_base, gram_entry as one_block_gram

def layout(m, groups, remainder):
    if any((not isinstance(v, int) for v in (m, groups, remainder))) or m < 1 or groups < 1 or (not 0 <= remainder < m + 2):
        raise ValueError('m>=1, groups>=1, 0<=remainder<m+2 required')
    n0 = 1 << (max(64, 3 * m + 7) - 1).bit_length()
    outer = 1 << (groups + int(remainder > 0) - 1).bit_length()
    return (n0, outer, groups * (m + 2) + remainder)

@lru_cache(None)
def base(m, groups, remainder):
    n0, outer, d = layout(m, groups, remainder)
    a = one_block_base(m)
    columns = []
    for block in range(groups):
        signs = np.array([1 - 2 * ((t & block).bit_count() % 2) for t in range(outer)], dtype=object)
        columns.extend((np.kron(signs, a[:, c]) for c in range(m + 2)))
    if remainder:
        signs = np.array([1 - 2 * ((t & groups).bit_count() % 2) for t in range(outer)], dtype=object)
        for c in range(remainder):
            h = lambda k: np.array([1 - 2 * ((u & k).bit_count() % 2) for u in range(n0)], dtype=object)
            local = 20 * h(3 * c + 1) + 2 * h(3 * c + 2) + h(3 * c + 3)
            columns.append(np.kron(signs, local))
    answer = np.column_stack(columns)
    assert answer.shape == (n0 * outer, d)
    answer.flags.writeable = False
    return answer

def block_of(c, m, groups):
    return c // (m + 2) if c < groups * (m + 2) else None

def gram_entry(i, j, m, groups):
    if i == j:
        return 405
    block = block_of(i, m, groups)
    if block is not None and block == block_of(j, m, groups):
        return one_block_gram(i % (m + 2), j % (m + 2))
    return 0

def instance(labels, bits, remainder=0):
    labels = tuple((tuple(group) for group in labels))
    if not labels or not labels[0] or any((len(group) != len(labels[0]) or any((x not in (0, 1) for x in group)) for group in labels)):
        raise ValueError('Nonempty equally sized binary label groups required')
    if not isinstance(bits, int) or bits < 14:
        raise ValueError('B>=14 required')
    m, groups = (len(labels[0]), len(labels))
    s = 1 << bits - 7
    data = 64 * s + s * base(m, groups, remainder)
    for block, group in enumerate(labels):
        for i, label in enumerate(group):
            data[0 if label else 7, block * (m + 2) + i + 2] += 1
    return data

def structural_tests(m, groups, remainder):
    _, _, d = layout(m, groups, remainder)
    tests = {}
    for block in range(groups):
        start = block * (m + 2)
        for (x, y, cond), decision in one_block_tests(m).items():
            tests[x + start, y + start, tuple((z + start for z in cond))] = decision
    for x, y in combinations(range(d), 2):
        bx, by = (block_of(x, m, groups), block_of(y, m, groups))
        if bx is None or bx != by:
            tests[x, y, ()] = 'independent'
    return tests

def target_tests(labels):
    m = len(labels[0])
    tests = {}
    for block, group in enumerate(labels):
        start = block * (m + 2)
        if any(group):
            tests[start, start + 1, (start + 2 + group.index(1),)] = 'independent'
        else:
            for i in range(m):
                tests[start, start + 1, (start + 2 + i,)] = 'dependent'
    return tests

def expected_edges(labels):
    m = len(labels[0])
    edges = []
    for block, group in enumerate(labels):
        start = block * (m + 2)
        edges.extend(((start + x, start + 2 + i) for x in (0, 1) for i in range(m)))
        if not any(group):
            edges.append((start, start + 1))
    return sorted(edges)

def certificate_depths(labels, bits, remainder):
    m = len(labels[0])
    depths = []
    for group in labels:
        if any(group):
            target = group.index(1) + 2
            depths.extend((bits if c in (0, 1, target) else 13 for c in range(m + 2)))
        else:
            depths.extend([bits] * (m + 2))
    return depths + [13] * remainder

def envelope(m, groups, remainder, depth=13):
    _, _, d = layout(m, groups, remainder)
    delta = F(128, 2 ** depth)
    error = 180 * delta + delta * delta
    return {(i, j): Interval(1620 - 180 * delta, 1620 + error) if i == j else Interval(4 * gram_entry(i, j, m, groups) - error, 4 * gram_entry(i, j, m, groups) + error) for i in range(d) for j in range(d)}
