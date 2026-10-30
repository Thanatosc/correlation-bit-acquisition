# Copyright (c) 2026 Siyu Cai. SPDX-License-Identifier: MIT
from fractions import Fraction as F
from itertools import combinations, product
from pathlib import Path
import argparse
import hashlib
import json
import sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
from check_extension import I, boxes, det, graph, inverse, partial_atom, retain, retain_schur, scatter, sets, sub

def parameters(tau, k):
    a = (7 + tau) / 8
    u, v = (a * a, 1 - a * a)
    c = -tau + (1 + tau) * u
    gamma = min(tau * tau / 4, (1 - tau) / 8)
    if tau < F(1, 2):
        gamma = min(gamma, (1 - 2 * tau) / (4 * (1 - tau)))
    if tau < F(1, k + 1):
        delta = (1 - (k + 1) * tau) / (1 - k * tau)
        gamma = min(gamma, (1 - tau) * delta / 8)
    b = 1 - gamma * v
    spectral_delta = (1 + c) * b - 2 * u
    lam = min(1 - c, 1 - b, spectral_delta / 3)
    assert spectral_delta > 3 * (1 - tau) * v / 4 > 0 and lam > 0
    return dict(tau=tau, k=k, a=a, u=u, v=v, c=c, gamma=gamma, b=b, lam=lam)

def gram(p, m):
    return [[F(1) if i == j else p['c'] if {i, j} == {0, 1} else p['a'] if min(i, j) < 2 else p['b'] for j in range(m + 2)] for i in range(m + 2)]

def schur_correlation(T, x, y, S):
    inv = inverse(sub(T, S))
    assert inv is not None

    def residual(i, j):
        return T[i][j] - sum((T[i][u] * inv[a][b] * T[v][j] for a, u in enumerate(S) for b, v in enumerate(S)))
    A, B, C = (residual(x, x), residual(y, y), residual(x, y))
    assert A == B and A > 0
    return C / A

def obligations(p):
    k = p['k']
    tests = {(0, 1, ()): True, (0, 2, ()): True, (2, 3, ()): True, (0, 2, (3,)): False}
    for s in range(2, k + 1):
        tests[0, 1, tuple(range(2, 2 + s))] = True
    if p['tau'] >= F(1, k + 1):
        tests[2, 3, tuple(range(4, 4 + k))] = False
    else:
        for h in range(min(2, k) + 1):
            for r in range(k - h + 1):
                S = tuple(range(h)) + tuple(range(4, 4 + r))
                tests[2, 3, S] = True
    return tests

def formula_checks():
    count = 0
    records = []
    for k in range(1, 9):
        thresholds = sorted({F(1, 100), F(1, 10), F(9, 10 * (k + 1)), F(1, k + 1), F(11, 10 * (k + 1)), F(1, 2), F(99, 100)})
        for tau in thresholds:
            p = parameters(tau, k)
            T = gram(p, k + 2)
            for s in range(1, k + 1):
                t = p['u'] * s / (1 - p['b'] + p['b'] * s)
                expected = (p['c'] - t) / (1 - t)
                rho = schur_correlation(T, 0, 1, tuple(range(2, s + 2)))
                assert rho == expected
                assert rho == -tau if s == 1 else rho < -tau
                count += 1
            for h in range(min(2, k) + 1):
                theta = (p['b'], 1 - p['gamma'], 1 - p['gamma'] * (1 + p['c']) / (1 - tau))[h]
                for r in range(k - h + 1):
                    S = tuple(range(h)) + tuple(range(4, 4 + r))
                    rho = schur_correlation(T, 2, 3, S)
                    assert rho == theta / (1 + r * theta)
                    if tau < F(1, k + 1):
                        assert rho > tau
                    count += 1
            for (x, y, S), wanted in obligations(p).items():
                assert retain_schur(T, x, y, S, tau) is wanted
                count += 1
            records.append({'k': k, 'tau': str(tau)})
    return {'parameter_cases': records, 'exact_formula_and_decision_checks': count}

def integer_base(record, p, m):
    common, private, L = (record['common'], record['private'], record['L'])
    dot = lambda a, b: sum((x * y for x, y in zip(a, b)))
    H = [[1, p['c'], p['a']], [p['c'], 1, p['a']], [p['a'], p['a'], p['b']]]
    assert all((dot(common[i], common[j]) == L * L * H[i][j] for i in range(3) for j in range(3)))
    assert dot(private, private) == L * L * (1 - p['b'])
    M = max(L, *(sum(map(abs, row)) + (sum(map(abs, private)) if i == 2 else 0) for i, row in enumerate(common)))
    assert M * M <= 16 * L * L
    n = 1 << (max(32, 13 + 4 * m) - 1).bit_length()
    rows = []
    for r in range(n):
        w = [1 - 2 * ((r & j).bit_count() % 2) for j in range(1, 13 + 4 * m)]
        shared = [dot(w[:12], v) for v in common]
        rows.append(shared[:2] + [shared[2] + dot(w[12 + 4 * i:16 + 4 * i], private) for i in range(m)])
    T = scatter(rows)
    G = gram(p, m)
    assert all((sum((row[i] for row in rows)) == 0 for i in range(m + 2)))
    assert all((T[i][j] == n * n * L * L * G[i][j] for i in range(m + 2) for j in range(m + 2)))
    return (rows, M)

def scales(record, p, M):
    L, k = (record['L'], p['k'])
    q0 = M.bit_length()
    G = gram(p, k + 2)
    depth = q0 + 1
    while True:
        eta = F(2 ** (q0 + 1), 2 ** depth * L)
        error = 2 * eta + eta * eta
        a = [[I(v - error, v + error) for v in row] for row in G]
        good = all((partial_atom(a, x, y, S, p['tau']) is wanted for (x, y, S), wanted in obligations(p).items()))
        cross = [[I(1 - error, 1 + error), I(-error, error)], [I(-error, error), I(1 - error, 1 + error)]]
        good = good and partial_atom(cross, 0, 1, (), p['tau']) is False
        if good:
            break
        depth += 8
        assert depth < q0 + 257
    depth += 8
    tau, v, a = (p['tau'], p['v'], p['a'])
    alpha = 113 * a.denominator / (8 * tau * v * a * (1 + tau))
    bounds = [2, int(alpha) + 1, 2 ** (depth - q0)]
    for value in (36 / (tau * v * L), 12 / (p['lam'] * L)):
        bounds.append(-(-value.numerator // value.denominator))
    scale = 1 << (max(bounds) - 1).bit_length()
    B = q0 + 1 + scale.bit_length() - 1
    assert scale > alpha and scale * L >= 36 / (tau * v)
    assert scale * L >= 12 / p['lam'] and B >= depth + 1
    return (q0, depth, scale, B)

def locations(rows, p):
    out = []
    for i in range(2, len(rows[0])):
        slopes = [row[0] + row[1] - 2 * p['a'] * row[i] for row in rows]
        assert min(slopes) < 0 < max(slopes)
        out.append((slopes.index(max(slopes)), slopes.index(min(slopes))))
    return out

def realize(rows, where, labels, q0, scale):
    data = [[(2 ** q0 + v) * scale for v in row] for row in rows]
    for i, label in enumerate(labels):
        data[where[i][label]][2 + i] += 1
    return data

def expected(p, labels, groups=1, remainder=0):
    m = len(labels) // groups
    out = []
    for g in range(groups):
        start = g * (m + 2)
        if not any(labels[g * m:(g + 1) * m]):
            out.append((start, start + 1))
        if p['tau'] < F(1, p['k'] + 1):
            out.extend(combinations(range(start + 2, start + m + 2), 2))
    return tuple(sorted(out))

def certify(data, p, B, q, groups=1, remainder=0):
    d, m = (len(data[0]), (len(data[0]) - remainder) // groups - 2)
    depths = [B if i < groups * (m + 2) and i % (m + 2) in (0, 1, 2) else q for i in range(d)]
    box, low, widths = boxes(data, B, depths)
    count = 0
    for x, y in combinations(range(d), 2):
        g = x // (m + 2)
        same = x < groups * (m + 2) and y < groups * (m + 2) and (g == y // (m + 2))
        assert partial_atom(box, x, y, (), p['tau']) is same
        count += 1
        if not same:
            continue
        start = g * (m + 2)
        a, b = (x - start, y - start)
        if (a, b) == (0, 1):
            S = (start + 2,)
            assert partial_atom(box, x, y, S, p['tau']) is False
            count += 1
        elif a < 2:
            z = next((z for z in range(start + 2, start + m + 2) if z != y))
            assert partial_atom(box, x, y, (z,), p['tau']) is False
            count += 1
        elif p['tau'] >= F(1, p['k'] + 1):
            S = tuple((z for z in range(start + 2, start + m + 2) if z not in (x, y)))[:p['k']]
            assert partial_atom(box, x, y, S, p['tau']) is False
            count += 1
        else:
            local = [z for z in range(start, start + m + 2) if z not in (x, y)]
            for size in range(1, p['k'] + 1):
                for S in combinations(local, size):
                    assert partial_atom(box, x, y, S, p['tau']) is True
                    count += 1
    assert sum(depths) == groups * (3 * B + q * (m - 1)) + q * remainder
    return {'certified_atoms': count, 'cost': sum(depths), 'depths': depths, 'all_completions_covered_by_intervals': True}

def integer_checks():
    records = json.loads((ROOT / 'data/inputs/higher_order_parameters.json').read_text())
    reports = []
    det_schur_comparisons = 0
    for record in records:
        p = parameters(F(record['tau']), record['k'])
        k, m = (p['k'], p['k'] + 2)
        rows, M = integer_base(record, p, m)
        q0, q, scale, B = scales(record, p, M)
        where = locations(rows, p)
        labels = list(product((0, 1), repeat=m)) if k <= 2 else sorted({(0,) * m, (1,) * m, tuple((i % 2 for i in range(m))), *(tuple((int(i == j) for i in range(m))) for j in range(m))})
        base_data = [[(2 ** q0 + v) * scale for v in row] for row in rows]
        for label in labels:
            data = realize(rows, where, label, q0, scale)
            assert all((0 <= v < 2 ** B and v // 2 == base_data[r][i] // 2 for r, row in enumerate(data) for i, v in enumerate(row)))
            T = scatter(data)
            assert all((det(sub(T, range(j))) > 0 for j in range(1, m + 3)))
            actual = graph(T, k, p['tau'])
            assert actual == expected(p, label), (p, label, actual)
            for S in sets(m + 2, 0, 1, k):
                assert retain(T, 0, 1, S, p['tau']) == retain_schur(T, 0, 1, S, p['tau'])
                det_schur_comparisons += 1
        one = (1,) + (0,) * (m - 1)
        data = realize(rows, where, one, q0, scale)
        certificate = certify(data, p, B, q)
        L = record['L']
        direct = []
        for t in range(4):
            for u, row in enumerate(rows):
                sign = lambda j: 1 - 2 * ((t & j).bit_count() % 2)
                direct.append(row + [sign(1) * v for v in row] + [sign(2) * L * (1 - 2 * (u & 1))])
        direct_data = [[(2 ** q0 + v) * scale for v in row] for row in direct]
        for g in range(2):
            for i, label in enumerate(one):
                direct_data[where[i][label]][g * (m + 2) + 2 + i] += 1
        direct_cert = certify(direct_data, p, B, q, groups=2, remainder=1)
        assert graph(scatter(direct_data), k, p['tau']) == expected(p, one + one, 2, 1)
        assert len(direct) >= 2 * m and scale * L >= 12 / p['lam']
        reports.append({'k': k, 'tau': str(p['tau']), 'm': m, 'n': len(rows), 'B': B, 'coarse_depth': q, 'lambda0': str(p['lam']), 'condition_bound_Cm': str(6 * m / p['lam']), 'labelings_checked': len(labels), 'all_labels_enumerated': k <= 2, 'whole_block_certificate': certificate, 'direct_sum_certificate': direct_cert})
        print(json.dumps({'case': len(reports), 'k': k, 'tau': str(p['tau']), 'labelings': len(labels), 'B': B}), flush=True)
    return (reports, det_schur_comparisons)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    out = parser.parse_args().output_dir
    out.mkdir(parents=True, exist_ok=True)
    formula = formula_checks()
    cases, comparisons = integer_checks()
    report = {'status': 'passed', 'formula_checks': formula, 'integer_cases': cases, 'integer_labelings': sum((r['labelings_checked'] for r in cases)), 'integer_target_det_schur_comparisons': comparisons, 'whole_graph_interval_certificates': 2 * len(cases), 'certified_interval_atoms': sum((r['whole_block_certificate']['certified_atoms'] + r['direct_sum_certificate']['certified_atoms'] for r in cases)), 'scope': 'Rational arithmetic and sound determinant interval certificates; mathematical prefix transport.', 'sources': {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in (('check_lower.py', Path(__file__)), ('parameters.json', ROOT / 'data/inputs/higher_order_parameters.json'), ('check_extension.py', HERE / 'check_extension.py'))}}
    (out / 'lower_checks.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('status', 'integer_labelings', 'integer_target_det_schur_comparisons', 'whole_graph_interval_certificates', 'certified_interval_atoms')}, indent=2))
if __name__ == '__main__':
    main()
