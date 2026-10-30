# Copyright (c) 2026 Siyu Cai. SPDX-License-Identifier: MIT
import argparse
from fractions import Fraction as F
from itertools import combinations, product
from pathlib import Path
import hashlib
import json
import random
import sys
import traceback
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pacd.codec import PrefixReader, encode
from pacd.whole_column_stopping_hardness import construct, parameters
SOURCES = ['pacd/whole_column_stopping_hardness.py', 'scripts/check_whole_column_stopping_hardness.py', 'pacd/hardness_fixed_threshold.py', 'pacd/intervals.py', 'pacd/codec.py']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def scalar_graph(scatter, tau):

    def retains(x, y, z=None):
        a, b, c = (scatter[x, x], scatter[y, y], scatter[x, y])
        if z is not None:
            zz = scatter[z, z]
            if zz <= 0:
                return True
            a, b, c = (a * zz - scatter[x, z] ** 2, b * zz - scatter[y, z] ** 2, c * zz - scatter[x, z] * scatter[y, z])
        if a <= 0 or b <= 0:
            return True
        return tau.denominator ** 2 * c * c > tau.numerator ** 2 * a * b
    marginal = {(x, y) for x, y in combinations(range(3), 2) if retains(x, y)}
    final = set(marginal)
    for x, y in marginal:
        z = 3 - x - y
        eligible = tuple(sorted((x, z))) in marginal or tuple(sorted((y, z))) in marginal
        if eligible and (not retains(x, y, z)):
            final.remove((x, y))
    return (sorted(final), marginal, [retains(x, y, 3 - x - y) for x, y in combinations(range(3), 2)])

def check_case(weights, target, tau, rng, all_single_flips=False):
    lo, hi, nominal, W, info = construct(weights, target, tau)
    n, m, A, a = (info[key] for key in ('n', 'm', 'A', 'a'))
    p, q = (tau.numerator, tau.denominator)
    offset = info['offset']
    X = [int(v) - offset for v in nominal[:, 0]]
    Y = [int(v) - offset for v in nominal[:, 1]]
    z = [int(v) - offset - 1 for v in nominal[:, 2]]
    W = list(map(int, W))
    U = [(x + y) // 2 for x, y in zip(X, Y)]
    V = [(x - y) // 2 for x, y in zip(X, Y)]
    assert sum(X) == sum(Y) == sum(z) == sum(W) == 0
    assert sum((v * v for v in X)) == sum((v * v for v in Y)) == q * q * a
    assert sum((v * w for v, w in zip(U, V))) == 0
    assert sum((v * w for v, w in zip(z, W))) == 0
    assert sum((v * w for v, w in zip(z, V))) == 4 * A * target
    assert all((int(hi[i, 2]) - int(lo[i, 2]) == 1 and int(lo[i, 2]) % 2 == 0 for i in range(n)))
    assert np.array_equal(lo[:, :2], hi[:, :2])
    base = {(0, 0): n * sum((x * x for x in X)), (1, 1): n * sum((y * y for y in Y)), (0, 1): n * sum((x * y for x, y in zip(X, Y)))}
    base[1, 0] = base[0, 1]
    checks = 0
    solutions = 0

    def world(bits, flips):
        nonlocal checks, solutions
        e = [-int(b) for b in bits] + [0] * (n - m)
        for row in flips:
            assert row >= m
            e[row] = 1 if W[row] > 0 else -1
        Z = [v + change for v, change in zip(z, e)]
        szz = n * sum((v * v for v in Z)) - sum(Z) ** 2
        scatter = dict(base)
        scatter[2, 2] = szz
        scatter[0, 2] = scatter[2, 0] = n * sum((x * v for x, v in zip(X, Z)))
        scatter[1, 2] = scatter[2, 1] = n * sum((y * v for y, v in zip(Y, Z)))
        graph, marginal, conditional = scalar_graph(scatter, tau)
        hit = not flips and sum((w * b for w, b in zip(weights, bits))) == target
        assert graph == ([(0, 1)] if hit else [])
        assert (0, 1) in marginal and (not any(conditional))
        assert ((0, 2) not in marginal and (1, 2) not in marginal) == hit
        delta = F(n * sum((v * v for v in e)) - sum(e) ** 2, n)
        we = sum((w * v for w, v in zip(W, e)))
        assert 0 <= delta <= n
        if flips:
            assert we >= A and 2 * we > p * delta
        else:
            assert we == 0 and sum((w * v for w, v in zip(z, e))) == 0
            assert sum((v * w for v, w in zip(V, Z))) == 4 * A * (target - sum((w * b for w, b in zip(weights, bits))))
        xx, yy, zz, xy, xz, yz = (scatter[t] for t in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)))
        assert xx > 0 and xx * yy - xy * xy > 0
        assert xx * yy * zz + 2 * xy * xz * yz - xx * yz * yz - yy * xz * xz - zz * xy * xy > 0
        checks += 1
        solutions += int(hit)
    for bits in product((0, 1), repeat=m):
        world(bits, ())
    subset_solutions = solutions
    rows = list(range(m, n)) if all_single_flips else rng.sample(range(m, n), min(32, n - m))
    for row in rows:
        for bits in product((0, 1), repeat=m) if all_single_flips else [tuple((rng.randrange(2) for _ in weights))]:
            world(bits, (row,))
    for _ in range(32):
        world(tuple((rng.randrange(2) for _ in weights)), tuple((row for row in range(m, n) if rng.randrange(2))))
    world((1,) * m, tuple((i for i in range(m, n) if W[i] < 0)))
    world((0,) * m, tuple((i for i in range(m, n) if W[i] > 0)))
    assert solutions == subset_solutions
    return ({**info, 'scalar_world_checks': checks, 'subset_solutions': subset_solutions, 'all_auxiliary_single_flips': all_single_flips, 'total_legal_completions_log2': n, 'scope': 'All item assignments, selected nuisance worlds, and quantified proof; not exhaustive enumeration of all 2^n column completions.'}, (lo, hi, nominal))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/whole_column_stopping_hardness/verified')
    out = parser.parse_args().output_dir
    out.mkdir(parents=True, exist_ok=False)
    try:
        rng = random.Random(2026103006)
        records, codec = ([], [])
        for tau in (F(1, 100), F(1, 10), F(1, 4), F(1, 2), F(2, 3), F(9, 10), F(99, 100)):
            for weights, target in [([2, 2], 1), ([2, 2], 2), ([2, 3, 7], 5), ([2, 3, 7], 6)]:
                record, arrays = check_case(weights, target, tau, rng, all_single_flips=tau == F(1, 10) and weights == [2, 2])
                records.append(record)
                if tau in (F(1, 10), F(1, 2)) and weights == [2, 2]:
                    lo, hi, nominal = arrays
                    path = out / f'tau_{tau.numerator}_{tau.denominator}_T{target}.bp'
                    assert record['bits'] <= 32 and record['n'] <= 65535
                    encode(path, np.asarray(nominal, dtype=np.int64), record['bits'], record['n'])
                    with PrefixReader(path) as reader:
                        for column, depth in enumerate(record['depths']):
                            for _ in range(depth):
                                reader.read_next(column)
                        assert np.array_equal(reader.lower, lo) and np.array_equal(reader.upper, hi)
                        assert reader.depths == record['depths']
                        assert all((reader.width(i) == (0 if i < 2 else 1) for i in range(3)))
                        codec.append({'file': path.name, 'sha256': sha(path), 'n': reader.n, 'bits': reader.bits, 'depths': reader.depths, 'unknown_cells': reader.n, 'queries': len(reader.actions)})
        for exponent in (16, 64, 256):
            scale = 2 ** exponent
            for target in (5 * scale, 5 * scale + 1):
                record, _ = check_case([2 * scale, 3 * scale, 7 * scale], target, F(1, 10), rng)
                records.append(record)
        report = {'status': 'passed', 'thresholds': ['1/100', '1/10', '1/4', '1/2', '2/3', '9/10', '99/100'], 'cases': records, 'case_count': len(records), 'scalar_world_checks': sum((r['scalar_world_checks'] for r in records)), 'codec_checks': codec, 'maximum_bits': max((r['bits'] for r in records)), 'maximum_rows': max((r['n'] for r in records)), 'sources': {name: sha(ROOT / name) for name in SOURCES}, 'scope': 'Scalar integer graph evaluation and aligned-prefix codec checks.'}
        (out / 'checks.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({k: report[k] for k in ('status', 'case_count', 'scalar_world_checks', 'maximum_bits', 'maximum_rows', 'codec_checks')}, indent=2))
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
