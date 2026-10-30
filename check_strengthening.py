# Copyright (c) 2026 Siyu Cai. SPDX-License-Identifier: MIT
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import sympy as sp
HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / 'pacd').is_dir() else HERE.parent
sys.path.insert(0, str(ROOT))
from pacd.intervals import Interval, ci_certificate, exact_ci, exact_scatter, scatter_bounds
from pacd.positive_correlation_corner import positive_corner_certificate
from pacd.skeleton import reference_skeleton
from pacd.whole_column_or_direct_sum import base, instance, expected_edges, structural_tests, gram_entry

def symbolic_checks():
    x, y, z, n, t, L, r, v = sp.symbols('x y z n t L r v')
    A = 145800 + (810 * z - 270 * x) * t + (405 * (n - 1) - x * x) * t * t
    C = 145800 + (810 * z - 270 * y) * t + (405 * (n - 1) - y * y) * t * t
    N = 14580 + (162 * z - 135 * (x + y)) * t + (81 * (n - 1) - x * y) * t * t
    f = 100 * N * N - A * C
    equations = {'spectral_slack': (r - 1) ** 2 - (r * r + 1 - v - r * r / v) - (v - r) ** 2 / v, 'switch_zero_constant': f.subs(t, 0), 'switch_linear_coefficient': sp.diff(f, t).subs(t, 0) - 118098000 * (2 * z - 3 * (x + y)), 'gap_comparison_polynomial': 100 * (8 * L ** 2 + 14 * L - 16) ** 2 - (80 * L ** 2 + 306 * L + 1985) * (80 * L ** 2 - 4 * L + 1) - (-1760 * L ** 3 - 163656 * L ** 2 - 37166 * L + 23615), 'gap_actual_threshold_margin': 10 * (2 * L ** 2 + F(7, 2) * L - F(1, 8)) - (20 * L ** 2 - L + F(1, 4)) - (36 * L - F(3, 2))}
    results = {name: sp.factor(value) == 0 for name, value in equations.items()}
    assert all(results.values())
    assert F(1944, 79) > F(49, 10) ** 2
    assert (30 + F(1, 128)) ** 2 / (F(49, 10) - F(1, 128)) ** 2 < 38
    assert F(70884061625, 64) < 10392624000
    assert F(180, 64) + F(1, 4096) < 3
    assert 100 * 3 ** 2 - 1617 ** 2 < 0
    eta = F(1, 32)
    assert 44 * eta + (16 + 4 * eta) * (7 * eta + eta ** 2) < 5 < 12
    return results

def uniform_margins():
    box = {}
    for i, j in product(range(4), repeat=2):
        center = 1620 if i == j else 324 if {i, j} == {0, 1} else 316 if i >= 2 and j >= 2 else 540
        box[i, j] = Interval(center - 3, center + 3)
    cases = [((0, 1, ()), 'dependent', 7669971), ((0, 2, ()), 'dependent', 26202771), ((2, 3, ()), 'dependent', 7162771), ((0, 2, (1,)), 'dependent', 41778717995520), ((0, 2, (3,)), 'dependent', 42368846572800), ((2, 3, (0,)), 'independent', -120689683200)]
    result = []
    for test, wanted, endpoint in cases:
        decision, margin = ci_certificate(box, *test)
        assert decision == wanted
        assert (margin.lo if wanted == 'dependent' else margin.hi) == endpoint
        result.append({'test': test, 'decision': decision, 'lo': margin.lo, 'hi': margin.hi})
    return result

def prefixes(data, bits, depths):
    step = np.array([1 << bits - k for k in depths], dtype=object)
    lower = data // step * step
    return (lower, lower + step - 1)

def finite_low_width():
    results = []
    for m, groups, remainder in ((1, 1, 0), (2, 2, 3), (4, 3, 5), (8, 2, 0)):
        original = base(m, groups, remainder)
        n, d = original.shape
        gram = original.T @ original
        assert all((int(v) == 0 for v in original.sum(axis=0)))
        assert all((gram[i, j] == n * gram_entry(i, j, m, groups) for i, j in product(range(d), repeat=2)))
        for bits in (14, 16, 32, 64):
            for pattern in ('zeros', 'singletons', 'ones'):
                labels = tuple((tuple((int(pattern == 'ones' or (pattern == 'singletons' and i == j % m)) for i in range(m))) for j in range(groups)))
                data = instance(labels, bits, remainder)
                assert all((0 <= v < 2 ** bits for v in data.flat))
                scatter = exact_scatter(data)
                box = scatter_bounds(*prefixes(data, bits, [13] * d))
                tests = structural_tests(m, groups, remainder)
                for test, wanted in tests.items():
                    assert ci_certificate(box, *test)[0] == wanted
                for j, group in enumerate(labels):
                    offset = j * (m + 2)
                    for i, label in enumerate(group):
                        assert exact_ci(scatter, offset, offset + 1, (offset + 2 + i,)) == ('independent' if label else 'dependent')
                assert reference_skeleton(data)[0] == expected_edges(labels)
                other = instance(tuple((tuple((1 - v for v in row)) for row in labels)), bits, remainder)
                assert np.array_equal(data // 2, other // 2)
                if pattern != 'zeros':
                    depths = []
                    for group in labels:
                        positive = group.index(1) + 2
                        depths += [bits if i in (0, 1, positive) else 13 for i in range(m + 2)]
                    depths += [13] * remainder
                    assert sum(depths) == groups * (3 * bits + 13 * (m - 1)) + 13 * remainder
                    assert sum(depths) <= groups * (3 * bits + 39 * m)
                results.append({'m': m, 'groups': groups, 'remainder': remainder, 'bits': bits, 'pattern': pattern, 'n': n, 'd': d, 'structural_tests': len(tests)})
    return results

def finite_gap():
    signs = np.array(list(product((-1, 1), repeat=4)), dtype=object)
    U = 3 * signs[:, 0] + signs[:, 1] + signs[:, 2]
    V = 3 * signs[:, 3]
    x, y = (U + V, U - V)
    sx, sy = (np.sign(10 * y - x), np.sign(10 * x - y))
    assert all((int(v) == 0 for v in (sum(x), sum(y), sum(sx), sum(sy))))
    assert sum(x * x) == sum(y * y) == 320 and sum(x * y) == 32
    assert sum(abs(x)) == sum(abs(y)) == 56
    assert sum(x * sx) == sum(y * sy) == -16
    assert sum(x * sy) == sum(y * sx) == 56 and sum(sx * sy) == -8
    assert min(abs(2 * (10 * y - x))) == min(abs(2 * (10 * x - y))) == 12
    results = []
    for bits in (12, 16, 20, 28, 32, 64, 128, 256):
        L = 1 << bits - 5
        data = np.column_stack((16 * L + L * x - (sx < 0), 16 * L + L * y - (sy < 0)))
        scatter = exact_scatter(data)
        A, C = (F(scatter[0, 0], 256), F(scatter[0, 1], 256))
        assert A == F(scatter[1, 1], 256) == 20 * L * L - L + F(1, 4)
        assert C == 2 * L * L + F(7, 2) * L - F(1, 8)
        assert 10 * C > A > C > 0 and (A + C) / (A - C) < F(3, 2)
        corner = positive_corner_certificate(*prefixes(data, bits, [10, 10]))
        assert corner['decision'] == 'dependent' and corner['minimum_corner'] == data.tolist()
        for depths in ((bits - 5, bits), (bits, bits - 5)):
            box = scatter_bounds(*prefixes(data, bits, depths))
            uncertain = depths.index(bits - 5)
            exact = 1 - uncertain
            assert box[0, 1].lo == 256 * (8 * L * L + 14 * L - 16)
            assert box[uncertain, uncertain].hi == 256 * (80 * L * L + 306 * L + 1985)
            assert box[exact, exact].hi == 256 * (80 * L * L - 4 * L + 1)
            assert ci_certificate(box, 0, 1)[0] == 'unknown'
        results.append({'bits': bits, 'kappa': str((A + C) / (A - C)), 'semantic_upper': 20, 'H_lower': 2 * bits - 8, 'visible_corner_certificate': 'dependent'})
    return results

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    symbolic = symbolic_checks()
    margins = uniform_margins()
    low = finite_low_width()
    gap = finite_gap()
    names = ['pacd/intervals.py', 'pacd/positive_correlation_corner.py', 'pacd/skeleton.py', 'pacd/whole_column_or_direct_sum.py', 'pacd/whole_column_or_linear_kappa.py', 'pacd/whole_column_or.py']
    report = {'status': 'passed', 'symbolic_equalities': symbolic, 'uniform_box_margins': margins, 'low_width_cases': low, 'gap_cases': gap, 'counts': {'symbolic_equalities': len(symbolic), 'margin_types': len(margins), 'low_width_cases': len(low), 'gap_cases': len(gap)}, 'sources': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}, 'checker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'scope': 'Symbolic identities, interval margins, integer matrices, and corner certificates.'}
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text, encoding='utf-8')
    print(json.dumps({'status': 'passed', 'counts': report['counts']}, indent=2))
if __name__ == '__main__':
    main()
