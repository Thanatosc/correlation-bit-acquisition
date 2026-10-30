# Copyright (c) 2026 Siyu Cai. SPDX-License-Identifier: MIT
import argparse
from fractions import Fraction as F
from itertools import combinations, product
from math import inf
from pathlib import Path
import hashlib
import io
import json
import sys
import traceback
import unittest
import numpy as np
import sympy as sp
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
from pacd.codec import PrefixReader, encode
from pacd.column_monotone import discover_column_balanced
from pacd.edge_formula import edge_formula, value
from pacd.intervals import Interval, ci_certificate, exact_ci, exact_scatter
from pacd.skeleton import reference_skeleton
from support import MemoryReader
SOURCES = ['scripts/check_semantic_margin.py', 'pacd/codec.py', 'pacd/column_monotone.py', 'pacd/ancestor_acquisition.py', 'pacd/edge_formula.py', 'pacd/intervals.py', 'pacd/skeleton.py', 'tests/support.py', 'tests/test_column_monotone.py']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def tests(d):
    return [(x, y, z) for x, y in combinations(range(d), 2) for z in [(), *((z,) for z in range(d) if z not in (x, y))]]

def squared_rho(scatter, test):
    x, y, cond = test
    a, b, c = (F(scatter[i, j]) for i, j in ((x, x), (y, y), (x, y)))
    if cond:
        z = cond[0]
        zz = scatter[z, z]
        a, b, c = (a * zz - scatter[x, z] ** 2, b * zz - scatter[y, z] ** 2, c * zz - scatter[x, z] * scatter[y, z])
    assert a > 0 and b > 0
    return c * c / (a * b)

def ceil_log2(x):
    x = F(x)
    assert x >= 1
    k = 0
    while 2 ** k < x:
        k += 1
    return k

def factor(bits, K, beta, tau):
    D = F(2 ** 14) / (tau * tau * (1 - tau))
    g = 0
    while 4 ** g < K:
        g += 1
    return min(bits, g + ceil_log2(216 * D * K * K / beta) + 2)

def algebra_checks():
    e, s, u, v = sp.symbols('e s u v', nonnegative=True)
    assert sp.expand(6 * e - (4 * e + 2 * e ** 2) - 2 * e * (1 - e)) == 0
    charging = sp.expand((s + 2) * (1 + u) * (2 + v) - (2 + u) * (2 + v + s))
    assert charging == sp.expand(u * (2 + v) + s * (u + v + u * v))
    assert all((c >= 0 for c in sp.Poly(charging, s, u, v).coeffs()))
    return {'product_slack': '2 e (1-e), nonnegative for 0<=e<=1', 'charging_slack': str(charging), 'conditional_error_constant': 108, 'sufficient_width_denominator': 216}

def rational_interval_checks():
    count = 0
    matrices = [([[F(1) if i == j else r for j in range(3)] for i in range(3)], (1 + 2 * abs(r)) / (1 - 2 * abs(r))) for r in (F(-1, 3), F(-1, 5), F(0), F(1, 5), F(1, 3))]
    matrices.append(([[F(1), F(1, 10), F(1, 2)], [F(1, 10), F(1), F(1, 5)], [F(1, 2), F(1, 5), F(1)]], F(17, 3)))
    for R, K in matrices:
        actual = {(i, j): R[i][j] for i in range(3) for j in range(3)}
        for tau in (F(1, 10), F(1, 2), F(9, 10)):
            selected = [t for t in tests(3) if squared_rho(actual, t) != tau * tau]
            beta = min((abs(squared_rho(actual, t) - tau * tau) for t in selected))
            eta = beta / (216 * K * K)
            eps = 3 * eta
            box = {key: Interval(val - eps, val + eps) for key, val in actual.items()}
            for test in selected:
                got, _ = ci_certificate(box, *test, threshold=tau)
                assert got == exact_ci(actual, *test, threshold=tau)
                count += 1
    return {'certified_atom_checks': count, 'arithmetic': 'exact rational intervals at the sufficient eta boundary', 'correlation_matrices': len(matrices), 'thresholds': ['1/10', '1/2', '9/10']}

def family(kind, bits):
    if kind == 'orthogonal':
        W = np.array([[1 - 2 * ((r & c).bit_count() % 2) for c in (1, 2, 3)] for r in range(4)], dtype=object)
        return (2 ** (bits - 1) + 2 ** (bits - 3) * W, F(1), tests(3))
    W = np.array([[1 - 2 * ((r & c).bit_count() % 2) for c in range(1, 7)] for r in range(8)], dtype=object)
    coefficients = np.array([[10, 0, 0, 0, 0, 0], [1, 7, 7, 1, 0, 0], [5, 1, 1, 1, 6, 6]], dtype=object).T
    return (2 ** (bits - 1) + 2 ** (bits - 7) * (W @ coefficients), F(17, 3), [t for t in tests(3) if t != (0, 1, ())])

def scaled_checks(out):
    records = []
    tau = F(1, 10)
    for kind in ('orthogonal', 'unused_equality'):
        for bits in (12, 20, 32, 64, 128):
            data, K, selected = family(kind, bits)
            assert np.all(data >= 0) and np.all(data < 2 ** bits)
            scatter = exact_scatter(data)
            beta = min((abs(squared_rho(scatter, t) - tau * tau) for t in selected))
            assert beta == F(1, 100)
            states = {t: exact_ci(scatter, *t, threshold=tau) != 'independent' for t in selected}
            predicted = {e: value(edge_formula(3, *e), states) for e in combinations(range(3), 2)}
            assert None not in predicted.values()
            if kind == 'unused_equality':
                assert squared_rho(scatter, (0, 1, ())) == tau * tau
                assert [e for e, present in predicted.items() if present] == [(0, 2), (1, 2)]
            if bits <= 32:
                path = out / f'{kind}_B{bits}.bp'
                encode(path, np.asarray(data, dtype=np.int64), bits, len(data))
                with PrefixReader(path) as reader:
                    result = discover_column_balanced(reader, timeout_s=inf)
                transport = 'actual codec'
            else:
                reader = MemoryReader(data, bits, len(data))
                result = discover_column_balanced(reader, timeout_s=inf)
                transport = 'test-only exact prefix transport; no physical codec or byte claim'
            assert result['status'] == 'certified'
            assert result['edges'] == reference_skeleton(data, tau)[0]
            assert result['actions'] == [(i % 3, i // 3 + 1) for i in range(len(result['actions']))]
            bound = factor(bits, K, beta, tau)
            assert len(result['actions']) <= 3 * bound
            assert max(result['depths']) < bits
            if kind == 'unused_equality':
                assert not any((tuple(item['test']) == (0, 1, ()) for item in result['trace']))
            record = {'family': kind, 'bits': bits, 'transport': transport, 'queries': len(result['actions']), 'depths': result['depths'], 'edges': result['edges'], 'beta': str(beta), 'K_upper': str(K), 'ratio_upper': bound, 'comparison_denominator': 'Q_sem >= d = 3'}
            if bits <= 32:
                record['wire_bytes'] = result['wire_bytes']
                record['encoded_sha256'] = sha(path)
            trace = {k: v for k, v in result.items() if k not in ('seconds', 'cpu_seconds')}
            (out / f'{kind}_B{bits}_trace.json').write_text(json.dumps(trace, indent=2), encoding='utf-8')
            records.append(record)
    return records

def exact_semantic_comparison():
    columns = np.array(list(product(range(8), repeat=3)), dtype=np.int64)
    sums = columns.sum(axis=1)
    variance = 3 * (columns * columns).sum(axis=1) - sums * sums
    covariance = 3 * (columns @ columns.T) - np.outer(sums, sums)
    vv = np.outer(variance, variance)
    retained = (vv == 0) | (4 * covariance * covariance > vv)
    optimum = np.full(retained.shape, 6, dtype=np.int8)
    for qx, qy in product(range(4), repeat=2):
        codes = []
        for q in (qx, qy):
            base = 2 ** q
            coarsened = columns >> 3 - q
            codes.append(coarsened @ np.array([base ** 2, base, 1]))
        keys = codes[0][:, None] * 2 ** (3 * qy) + codes[1][None, :]
        group_min = np.ones(2 ** (3 * (qx + qy)), dtype=np.int8)
        group_max = np.zeros_like(group_min)
        np.minimum.at(group_min, keys.ravel(), retained.ravel())
        np.maximum.at(group_max, keys.ravel(), retained.ravel())
        constant = group_min[keys] == group_max[keys]
        optimum[constant] = np.minimum(optimum[constant], qx + qy)
    rng = np.random.default_rng(20261030)
    chosen = []
    while len(chosen) < 32:
        x, y = map(int, rng.integers(0, len(columns), size=2))
        if vv[x, y] <= covariance[x, y] ** 2 or 4 * covariance[x, y] ** 2 == vv[x, y]:
            continue
        data = np.column_stack((columns[x], columns[y]))
        result = discover_column_balanced(MemoryReader(data, 3, 3), threshold=F(1, 2), timeout_s=inf)
        assert result['status'] == 'certified'
        assert bool(result['edges']) == bool(retained[x, y])
        qsem = int(optimum[x, y])
        assert qsem >= 2 and len(result['actions']) <= 3 * qsem
        chosen.append({'column_indices': [x, y], 'Q_sem': qsem, 'Q_uniform': len(result['actions'])})
    return {'universe_matrices': int(retained.size), 'depth_vectors': 16, 'threshold': '1/2', 'n': 3, 'd': 2, 'B': 3, 'cases': chosen, 'scope': 'Exact semantic optima from all completions; at B=3 the proved ratio uses the B cap.'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/semantic_margin/verified')
    out = parser.parse_args().output_dir
    out.mkdir(parents=True, exist_ok=False)
    try:
        log = io.StringIO()
        suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_column_monotone.py')
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
        (out / 'unit_tests.log').write_text(log.getvalue(), encoding='utf-8')
        assert result.wasSuccessful()
        report = {'status': 'passed', 'unit_tests_run': result.testsRun, 'algebra': algebra_checks(), 'rational_intervals': rational_interval_checks(), 'scaled_acquisition': scaled_checks(out), 'exact_semantic_comparison': exact_semantic_comparison(), 'sources': {name: sha(ROOT / name) for name in SOURCES}, 'scope': 'Exact interval comparisons, encoded acquisition, and exhaustive small-matrix optima.'}
        (out / 'checks.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({'status': 'passed', 'unit_tests': result.testsRun, 'interval_checks': report['rational_intervals']['certified_atom_checks'], 'scaled_acquisition': report['scaled_acquisition'], 'exact_semantic_cases': len(report['exact_semantic_comparison']['cases'])}, indent=2))
    except Exception:
        sources = {}
        for name in SOURCES:
            path = ROOT / name
            digest = sha(path)
            snapshot = ROOT / 'artifacts/source_snapshots' / digest / path.name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            if not snapshot.exists():
                snapshot.write_bytes(path.read_bytes())
            sources[name] = {'sha256': digest, 'snapshot': snapshot.relative_to(ROOT).as_posix()}
        (out / 'failure.json').write_text(json.dumps({'traceback': traceback.format_exc(), 'sources': sources}, indent=2), encoding='utf-8')
        raise
if __name__ == '__main__':
    main()
