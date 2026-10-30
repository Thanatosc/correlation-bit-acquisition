# Copyright (c) 2026 Siyu Cai. SPDX-License-Identifier: MIT
from dataclasses import dataclass
from fractions import Fraction as F
from itertools import combinations, permutations, product
from math import factorial
from pathlib import Path
import argparse
import hashlib
import json
import random
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

def det(matrix):
    n = len(matrix)
    if not n:
        return 1
    a = [list(row) for row in matrix]
    denominator = 1
    sign = 1
    for j in range(n - 1):
        pivot = next((i for i in range(j, n) if a[i][j]), None)
        if pivot is None:
            return 0
        if pivot != j:
            a[pivot], a[j] = (a[j], a[pivot])
            sign = -sign
        p = a[j][j]
        for i in range(j + 1, n):
            for l in range(j + 1, n):
                v = a[i][l] * p - a[i][j] * a[j][l]
                assert v % denominator == 0
                a[i][l] = v // denominator
            a[i][j] = 0
        denominator = p
    return sign * a[-1][-1]

def inverse(matrix):
    n = len(matrix)
    a = [[F(v) for v in row] + [F(i == j) for j in range(n)] for i, row in enumerate(matrix)]
    for j in range(n):
        pivot = next((i for i in range(j, n) if a[i][j]), None)
        if pivot is None:
            return None
        a[j], a[pivot] = (a[pivot], a[j])
        p = a[j][j]
        a[j] = [v / p for v in a[j]]
        for i in range(n):
            if i != j:
                t = a[i][j]
                a[i] = [v - t * w for v, w in zip(a[i], a[j])]
    return [row[n:] for row in a]

def scatter(data):
    n, d = (len(data), len(data[0]))
    sums = [sum((row[i] for row in data)) for i in range(d)]
    return [[n * sum((row[i] * row[j] for row in data)) - sums[i] * sums[j] for j in range(d)] for i in range(d)]

def sub(matrix, rows, cols=None):
    if cols is None:
        cols = rows
    return [[matrix[i][j] for j in cols] for i in rows]

def sets(d, x, y, k):
    remaining = [i for i in range(d) if i not in (x, y)]
    return [S for size in range(1, min(k, len(remaining)) + 1) for S in combinations(remaining, size)]

def retain(T, x, y, S, tau):
    xs, ys = ((x, *S), (y, *S))
    conditioning = det(sub(T, S))
    A, B = (det(sub(T, xs)), det(sub(T, ys)))
    if conditioning <= 0 or A <= 0 or B <= 0:
        return True
    C = det(sub(T, xs, ys))
    return C * C > tau * tau * A * B

def retain_schur(T, x, y, S, tau):
    if not S:
        A, B, C = (F(T[x][x]), F(T[y][y]), F(T[x][y]))
    else:
        inv = inverse(sub(T, S))
        if inv is None:
            return True

        def residual(i, j):
            return F(T[i][j]) - sum((T[i][u] * inv[a][b] * T[v][j] for a, u in enumerate(S) for b, v in enumerate(S)))
        A, B, C = (residual(x, x), residual(y, y), residual(x, y))
    return A <= 0 or B <= 0 or C * C > tau * tau * A * B

def graph(T, k, tau, method=retain):
    d = len(T)
    marginal = {(x, y): method(T, x, y, (), tau) for x, y in combinations(range(d), 2)}

    def neighbor(x, y):
        return marginal[tuple(sorted((x, y)))]
    return tuple(((x, y) for x, y in marginal if marginal[x, y] and all((method(T, x, y, S, tau) for S in sets(d, x, y, k) if all((neighbor(x, z) or neighbor(y, z) for z in S))))))

@dataclass(frozen=True)
class I:
    lo: object
    hi: object

    def __add__(self, other):
        return I(self.lo + other.lo, self.hi + other.hi)

    def __sub__(self, other):
        return I(self.lo - other.hi, self.hi - other.lo)

    def __mul__(self, other):
        p = [a * b for a in (self.lo, self.hi) for b in (other.lo, other.hi)]
        return I(min(p), max(p))

    def square(self):
        return I(0 if self.lo <= 0 <= self.hi else min(self.lo ** 2, self.hi ** 2), max(self.lo ** 2, self.hi ** 2))

def interval_det(a):
    n = len(a)
    total = I(0, 0)
    for perm in permutations(range(n)):
        term = I(1, 1)
        for i, j in enumerate(perm):
            term = term * a[i][j]
        odd = sum((perm[i] > perm[j] for i in range(n) for j in range(i + 1, n))) % 2
        total = total - term if odd else total + term
    return total

def boxes(data, B, depths):
    n, d = (len(data), len(data[0]))
    widths = [2 ** (B - q) - 1 for q in depths]
    low = [[v // (w + 1) * (w + 1) for v, w in zip(row, widths)] for row in data]
    mid = [[2 * v + w for v, w in zip(row, widths)] for row in low]
    G = scatter(mid)
    sums = [sum((row[i] for row in mid)) for i in range(d)]
    L = [sum((abs(n * row[i] - sums[i]) for row in mid)) for i in range(d)]
    intervals = []
    for i in range(d):
        row = []
        for j in range(d):
            if i == j:
                row.append(I(max(0, G[i][i] - 2 * widths[i] * L[i]), G[i][i] + 2 * widths[i] * L[i] + n * n * widths[i] ** 2))
            else:
                r = widths[j] * L[i] + widths[i] * L[j] + n * n * widths[i] * widths[j]
                row.append(I(G[i][j] - r, G[i][j] + r))
        intervals.append(row)
    return (intervals, low, widths)

def partial_atom(a, x, y, S, tau):
    cond = interval_det(sub(a, S))
    A = interval_det(sub(a, (x, *S)))
    B = interval_det(sub(a, (y, *S)))
    if cond.hi == 0 or A.hi == 0 or B.hi == 0:
        return True
    if min(cond.lo, A.lo, B.lo) <= 0:
        return None
    C = interval_det(sub(a, (x, *S), (y, *S)))
    test = C.square() - I(tau * tau, tau * tau) * (A * B)
    if test.hi <= 0:
        return False
    if test.lo > 0:
        return True
    return None

def land(values):
    values = list(values)
    return False if False in values else None if None in values else True

def lor(values):
    values = list(values)
    return True if True in values else None if None in values else False

def partial_graph(a, k, tau):
    d = len(a)
    marginal = {(x, y): partial_atom(a, x, y, (), tau) for x, y in combinations(range(d), 2)}
    result = {}
    for x, y in marginal:
        terms = [marginal[x, y]]
        if terms[0] is not False:
            for S in sets(d, x, y, k):
                eligible = land((lor((marginal[tuple(sorted((x, z)))], marginal[tuple(sorted((y, z)))])) for z in S))
                if eligible is False:
                    continue
                decision = partial_atom(a, x, y, S, tau)
                terms.append(lor((decision, None if eligible is None else not eligible)))
        result[x, y] = land(terms)
    return result

def walsh_fixture(k, B, ill_conditioned=False):
    d = k + 2
    n = 2 ** (d + 1 - 1).bit_length()
    w = [[(-1) ** ((r & j).bit_count() % 2) for j in range(1, d + 1)] for r in range(n)]
    centered = [[sum(row[:k]) + row[k], sum(row[:k]) + row[k + 1], *row[:k]] for row in w]
    if ill_conditioned:
        for row in centered:
            old = row[2:]
            row[2:] = [old[j] + (100 * old[j - 1] if j else 0) for j in range(k)]
    if B is None:
        return centered
    scale = 2 ** max(0, B - (k + 3).bit_length() - 2)
    data = [[2 ** (B - 1) + scale * v for v in row] for row in centered]
    assert all((0 <= v < 2 ** B for row in data for v in row))
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    out = parser.parse_args().output_dir
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261030)
    independent = 0
    for d in range(2, 7):
        for rep in range(7):
            data = [[rng.randrange(8) for _ in range(d)] for _ in range(d + 2)]
            if rep == 0:
                for row in data:
                    row[-1] = row[0]
            T = scatter(data)
            for k in range(1, min(4, d - 1) + 1):
                for x, y in combinations(range(d), 2):
                    for S in [(), *sets(d, x, y, k)]:
                        assert retain(T, x, y, S, F(1, 10)) == retain_schur(T, x, y, S, F(1, 10))
                        independent += 1
                assert graph(T, k, F(1, 10)) == graph(T, k, F(1, 10), retain_schur)
    worlds = 0
    certified = 0
    for index in range(10):
        d, n = (3, 4) if index < 4 else (4, 5) if index < 8 else (4, 4)
        data = [[rng.randrange(8) for _ in range(d)] for _ in range(n)]
        depths = [3] * d
        depths[-1] = 2
        if index >= 8:
            depths[-2] = 2
        k = min(2, d - 2)
        a, lo, widths = boxes(data, 3, depths)
        decisions = partial_graph(a, k, F(1, 10))
        certified += sum((value is not None for value in decisions.values()))
        unknown = [(r, i) for r in range(n) for i, w in enumerate(widths) if w]
        for bits in product((0, 1), repeat=len(unknown)):
            completion = [row[:] for row in lo]
            for (r, i), bit in zip(unknown, bits):
                completion[r][i] += bit
            actual = set(graph(scatter(completion), k, F(1, 10)))
            assert all((value is None or value == ((x, y) in actual) for (x, y), value in decisions.items()))
            worlds += 1
    geometric = []
    for tau in (F(1, 100), F(1, 10), F(1, 2), F(9, 10), F(99, 100)):
        for k in (2, 3, 4):
            for ill in (False, True):
                data = walsh_fixture(k, None, ill)
                n, d = (len(data), len(data[0]))
                N = 2 ** 14 * (k + 1) ** 2
                D = max(F(2 ** 14, tau * tau * (1 - tau)), F(32 * N, 1 - tau))
                B = (int(D) * 16 + 1024).bit_length() + 3
                data = [[2 ** (B - 1) + v for v in row] for row in data]
                T = scatter(data)
                width = 2 ** B - 1
                assert all((width * width >= D * D * F(T[i][i], n * n) for i in (0, 1)))
                amplitude = 2 ** (B - 1)
                signs = [[-1 if row[i] >= amplitude else 1 for i in (0, 1)] for row in data]
                retaining = [row[:] for row in data]
                for i in (0, 1):
                    retaining[0][i] += amplitude * signs[0][i]
                assert all((0 <= v < 2 ** B for row in retaining for v in row))
                rt = scatter(retaining)
                assert all((retain(rt, 0, 1, S, tau) for S in [(), *sets(d, 0, 1, k)]))
                assert (0, 1) in graph(rt, k, tau)
                r, t = next(((r, t) for r, t in combinations(range(n), 2) if signs[r][1] == signs[t][1]))
                deleting = [row[:] for row in data]
                deleting[r][0] += amplitude * signs[r][0]
                deleting[t][1] += amplitude * signs[t][1]
                rounded = (2 * amplitude + n - 1) // (2 * (n - 1))
                deleting[r][1] += rounded * signs[r][1]
                assert all((0 <= v < 2 ** B for row in deleting for v in row))
                assert not retain(scatter(deleting), 0, 1, (), tau)
                geometric.append({'tau': str(tau), 'k': k, 'ill_conditioned': ill, 'n': n, 'B': B})
    data = walsh_fixture(2, None)
    T = scatter(data)
    R = [[F(v, len(data) ** 2) for v in row] for row in T]
    expected = [[3, 2, 1, 1], [2, 3, 1, 1], [1, 1, 1, 0], [1, 1, 0, 1]]
    assert R == expected and det(expected) == 1
    assert (0, 1) in graph(T, 1, F(1, 10)) and (0, 1) not in graph(T, 2, F(1, 10))
    margin_atoms = 0
    for k in range(1, 5):
        d = k + 2
        K = d + 1
        C = factorial(k + 1) * (k + 1) * 2 ** k
        beta = F(1, 100)
        eta = beta / (36 * C * K ** (2 * k))
        error = 3 * eta
        for rho in (F(0), F(1, 3)):
            a = [[I((1 if i == j else rho) - error, (1 if i == j else rho) + error) for j in range(d)] for i in range(d)]
            for x, y in combinations(range(d), 2):
                for S in [(), *sets(d, x, y, k)]:
                    actual_rho = rho / (1 + len(S) * rho)
                    assert abs(actual_rho ** 2 - F(1, 100)) >= beta
                    assert partial_atom(a, x, y, S, F(1, 10)) == (rho != 0)
                    margin_atoms += 1
    traces = []
    for k in (1, 2, 3):
        for B in (8, 24, 80):
            data = walsh_fixture(k, B)
            d = len(data[0])
            history = [0] * d
            for depth in range(B + 1):
                a, _, _ = boxes(data, B, [depth] * d)
                for i in range(d):
                    history[i] = max(history[i], a[i][i].lo)
                    a[i][i] = I(history[i], a[i][i].hi)
                decisions = partial_graph(a, k, F(1, 10))
                if all((v is not None for v in decisions.values())):
                    answer = tuple((edge for edge, value in decisions.items() if value))
                    break
            else:
                raise AssertionError('Full precision did not determine the graph')
            assert answer == graph(scatter(data), k, F(1, 10))
            traces.append({'k': k, 'B': B, 'n': len(data), 'd': d, 'depth': depth, 'queries': d * depth, 'graph': answer, 'transport': 'mathematical prefixes; no file codec'})
    net_polynomial = [62, 116, 60]
    assert all((v > 0 for v in net_polynomial))
    report = {'status': 'passed', 'independent_determinant_schur_comparisons': independent, 'exhaustive_small_box_worlds': worlds, 'certified_small_box_edges': certified, 'promised_precision_interval_atoms': margin_atoms, 'coarse_column_instances': geometric, 'uniform_traces': traces, 'order_two_example': {'covariance': expected, 'k1': graph(T, 1, F(1, 10)), 'k2': graph(T, 2, F(1, 10))}, 'net_bound_positive_polynomial_coefficients': net_polynomial, 'limitations': ['Finite checks do not prove the subspace-net statement.', 'This suite checks upper-bound constructions and interval decisions.', 'Uniform traces do not compute exact optimal certificate costs.', 'k is fixed for the polynomial-time implementation claim.'], 'sources': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),)}}
    (out / 'extension_checks.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'passed', 'det_schur': independent, 'worlds': worlds, 'certified_edges': certified, 'margin_atoms': margin_atoms, 'coarse_cases': len(geometric), 'traces': traces}, indent=2))
if __name__ == '__main__':
    main()
