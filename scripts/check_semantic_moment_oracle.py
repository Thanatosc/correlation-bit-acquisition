# Copyright (c) 2026 Siyu Cai. SPDX-License-Identifier: MIT
import argparse
from fractions import Fraction as F
from itertools import product
from math import comb
from pathlib import Path
import hashlib
import io
import json
import sys
import unittest
import numpy as np
import sympy as sp
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
from pacd.codec import PrefixReader, encode
from pacd.hardness_any_fixed_threshold import construct
from pacd.positive_correlation_corner import positive_corner_certificate
from pacd.semantic_acquisition import discover_semantic
from pacd.semantic_moments import UNLIMITED, certify_graph, moment_edge
from test_semantic_acquisition import gap_data

def corner_polynomials():
    L, t = sp.symbols('L t', real=True)
    w = L / 4 - 1
    variance = (sp.Rational(1209, 16) * L ** 2 + 14 * L, sp.Rational(661, 8) * L ** 2 - sp.Rational(29, 2) * L + 1)
    covariance = (sp.Rational(253, 32) * L ** 2 + sp.Rational(29, 2) * L - 1, sp.Rational(481, 32) * L ** 2 - sp.Rational(29, 2) * L + 1)

    def centered(a, sg):
        middle = 16 * L * (a + sp.Rational(sg, 8))
        return (middle - 15 * w, middle + 15 * w)

    def multiply(a, b):
        signs = [1 if end.subs(L, 128) > 0 else -1 for end in a]
        for end, sg in zip(a, signs):
            assert all((v >= 0 for v in sp.Poly(sg * end.subs(L, t + 128), t).all_coeffs()))
        return (a[0] * b[0 if signs[0] > 0 else 1], a[1] * b[1 if signs[1] > 0 else 0])
    polynomials = []
    for signs in product((-1, 1), repeat=4):
        u = 3 * signs[0] + signs[1] + signs[2]
        v = 3 * signs[3]
        x, y = (u + v, u - v)
        sx = 1 if 10 * y - x > 0 else -1
        sy = 1 if 10 * x - y > 0 else -1
        for xx, yy, ssx, ssy in ((x, y, sx, sy), (y, x, sy, sx)):
            p = multiply(centered(yy, ssy), variance)
            q = multiply(centered(xx, ssx), covariance)
            margin = sp.expand(p[0] - q[1] if ssx > 0 else q[0] - p[1])
            shifted = sp.Poly(margin.subs(L, t + 128), t)
            assert all((v >= 0 for v in shifted.all_coeffs())) and shifted.eval(0) > 0
            polynomials.append(margin)
    coefficients = sorted((tuple((int(v) for v in sp.Poly(256 * p, L).all_coeffs())) for p in set(polynomials)))
    expected = [(61042, 703992, -24576, 0), (70486, 952776, -36160, 3840), (625678, 630824, 94528, -3840), (1179918, 626728, 102720, -3840), (1723670, 692680, -3392, 3840), (2296918, 940488, -11584, 3840)]
    assert coefficients == expected
    for bits in (12, 16, 20, 28, 32, 64, 128, 256):
        data = gap_data(bits)
        step = 2 ** (bits - 7)
        lo = data // step * step
        hi = lo + step - 1
        certificate = positive_corner_certificate(lo, hi)
        assert certificate['decision'] == 'dependent' and certificate['minimum_corner'] == data.tolist()
        actual = sorted({F(row['bounds'][0] if row['sign'] > 0 else -row['bounds'][1], 256) for row in certificate['derivative_bounds']})
        predicted = sorted({int(p.subs(L, 2 ** (bits - 5))) for p in polynomials})
        assert actual == predicted
    return {'signed_derivatives': 32, 'distinct_polynomials': coefficients, 'positivity_domain': 'L >= 128, by nonnegative coefficients in L-128', 'checked_bit_widths': [12, 16, 20, 28, 32, 64, 128, 256], 'certificate_planes': 14}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/semantic_moment_oracle/verified')
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    log = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_semantic_*.py')
    tests = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    (out / 'unit_tests.log').write_text(log.getvalue(), encoding='utf-8')
    assert tests.wasSuccessful()
    polynomial = corner_polynomials()
    lo = np.zeros((32, 2), dtype=object)
    binary = moment_edge(lo, lo + 1, limits=UNLIMITED)
    assert binary['stats']['terminal_states'] == comb(35, 3) == 6545
    cell_lo, cell_hi, info = construct([2] * 32, 1, F(1, 10))
    cell = certify_graph(cell_lo, cell_hi, limits=UNLIMITED)
    assert cell['status'] == 'certified' and cell['edges'] == []
    cells = [r['stats'] for r in cell['decisions'] if r['method'] == 'moment_dp']
    assert cells and cells[0]['terminal_states'] == 33
    gaps = []
    for bits in (12, 16, 20, 28, 32):
        path = out / f'gap_B{bits}.bp'
        encode(path, np.asarray(gap_data(bits), dtype=np.int64), bits, 16)
        with PrefixReader(path) as reader:
            result = discover_semantic(reader)
        assert result['status'] == 'certified' and result['edges'] == [(0, 1)]
        assert result['queries'] <= 14
        (out / f'gap_B{bits}_trace.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        gaps.append({'bits': bits, 'queries': result['queries'], 'H_query_lower_bound': 2 * bits - 8, 'wire_bytes': result['wire_bytes'], 'resource_limited_checks': result['resource_limited_checks'], 'semantic_competitive_bound': result['semantic_competitive_bound'], 'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    source_names = ['pacd/semantic_moments.py', 'pacd/semantic_acquisition.py', 'pacd/positive_correlation_corner.py', 'pacd/column_monotone.py', 'pacd/calibrated_witness.py', 'pacd/intervals.py', 'pacd/skeleton.py', 'pacd/codec.py', 'pacd/hardness_any_fixed_threshold.py', 'pacd/hardness_fixed_threshold.py', 'tests/test_semantic_moments.py', 'tests/test_semantic_acquisition.py', 'scripts/check_semantic_moment_oracle.py']
    report = {'status': 'passed', 'unit_tests_run': tests.testsRun, 'corner_proof': polynomial, 'binary_whole_column': {'status': binary['status'], 'stats': binary['stats']}, 'cellwise_joint_control': {'status': cell['status'], 'n': info['n'], 'raw_completions': 2 ** 32, 'moment_calls': cells, 'scope': 'Cellwise prefix control, not the whole-column access model.'}, 'gap_acquisition': gaps, 'sources': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_names}, 'scope': 'Constructed exact arithmetic and actual codec checks; no ordinary-data speedup or general polynomial-time semantic stopping theorem.'}
    (out / 'checks.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
if __name__ == '__main__':
    main()
