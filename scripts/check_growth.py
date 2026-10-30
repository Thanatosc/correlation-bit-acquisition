# Copyright (c) 2026 Siyu Cai. SPDX-License-Identifier: MIT
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json
from check_lower import parameters, integer_base, scales, locations, realize, certify, scatter, graph, expected, gram
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    out = parser.parse_args().output_dir
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for record in json.loads((ROOT / 'data/inputs/higher_order_parameters.json').read_text()):
        k, tau = (record['k'], F(record['tau']))
        if k < 2 or tau not in (F(1, 10), F(1, k + 1)):
            continue
        p = parameters(tau, k)
        old_rows, M = integer_base(record, p, k + 2)
        q0, q, s, B = scales(record, p, M)
        m = k + 5
        rows, M_new = integer_base(record, p, m)
        assert M_new == M
        where = locations(rows, p)
        s, B = (s * 2 ** 32, B + 32)
        for label in ((0,) * m, (1,) + (0,) * (m - 1)):
            data = realize(rows, where, label, q0, s)
            assert graph(scatter(data), k, tau) == expected(p, label)
            for r, row in enumerate(data):
                for i, v in enumerate(row):
                    assert 0 <= v < 2 ** B
                    assert v // 2 == (2 ** q0 + rows[r][i]) * s // 2
                    assert v // 2 ** (B - q) * 2 ** (B - q) == (2 ** q0 + rows[r][i]) * s
        certificate = certify(data, p, B, q)
        L = record['L']
        direct = []
        for t in range(4):
            for u, row in enumerate(rows):
                sign = lambda j: 1 - 2 * ((t & j).bit_count() % 2)
                direct.append(row + [sign(1) * v for v in row] + [sign(2) * L * (1 - 2 * (u & 1))])
        T, G, n = (scatter(direct), gram(p, m), len(direct))
        for i in range(len(T)):
            for j in range(len(T)):
                g = F(1) if i == j else G[i % (m + 2)][j % (m + 2)] if i // (m + 2) == j // (m + 2) and i // (m + 2) < 2 else F(0)
                assert T[i][j] == n * n * L * L * g
        results.append({'k': k, 'tau': str(tau), 'old_m': k + 2, 'm': m, 'B': B, 'added_bit_planes': 32, 'unchanged_coarse_depth': q, 'n': len(rows), 'whole_graph_certificate': certificate, 'outer_walsh_gram_exact': True})
    report = {'status': 'passed', 'cases': results, 'matrices': 2 * len(results), 'scope': 'Fixed coefficient records, coarse depths, and exact direct-sum Gram validation.', 'sources': {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in (('check_growth.py', Path(__file__)), ('check_lower.py', HERE / 'check_lower.py'), ('higher_order_parameters.json', ROOT / 'data/inputs/higher_order_parameters.json'))}}
    (out / 'growth_checks.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'passed', 'cases': len(results), 'matrices': 2 * len(results)}))
if __name__ == '__main__':
    main()
