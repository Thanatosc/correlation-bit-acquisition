# Copyright (c) 2026 Siyu Cai. SPDX-License-Identifier: MIT
from fractions import Fraction
from itertools import combinations
import math
import numpy as np
from .column_monotone import ColumnMonotoneEvaluator
from .edge_formula import edge_formula, leaves, plan, value
from .intervals import Interval, ci_certificate
COARSE_DEPTH = 12

def base(m):
    if not isinstance(m, int) or m < 1:
        raise ValueError('m must be a positive integer')
    n = 1 << (5 + 4 * m).bit_length()
    h = np.array([[1 - 2 * ((r & c).bit_count() % 2) for c in range(1, 6 + 4 * m)] for r in range(n)], dtype=object)
    common = h[:, :3].sum(axis=1)
    contrast = h[:, 3:5].sum(axis=1)
    columns = [9 * (common + contrast), 9 * (common - contrast)]
    for i in range(m):
        noise = h[:, 5 + 4 * i:9 + 4 * i]
        columns.append(5 * common + 18 * noise[:, 0] + 2 * noise[:, 1] + noise[:, 2] + noise[:, 3])
    return (np.column_stack(columns), h[:, 0])

def instance(labels, bits):
    labels = tuple(labels)
    if not labels or any((b not in (0, 1) for b in labels)) or (not isinstance(bits, int)) or (bits < 13):
        raise ValueError('Nonempty binary labels and B >= 13 required')
    centered, u1 = base(len(labels))
    scale = 1 << bits - 7
    data = 64 * scale + scale * centered
    for i, bit in enumerate(labels):
        data[:, i + 2] += (1 + (2 * bit - 1) * u1) // 2
    return data

def normalized_scatter4(labels, bits):
    labels = tuple(labels)
    if not labels or any((b not in (0, 1) for b in labels)) or (not isinstance(bits, int)) or (bits < 13):
        raise ValueError('Nonempty binary labels and B >= 13 required')
    scale = 1 << bits - 7
    a = 1620 * scale * scale
    scatter = {(0, 0): a, (1, 1): a, (0, 1): 324 * scale * scale, (1, 0): 324 * scale * scale}
    signs = [2 * b - 1 for b in labels]
    for i, sign in enumerate(signs, 2):
        scatter[i, i] = a + 20 * sign * scale + 1
        cross = 540 * scale * scale + 18 * sign * scale
        for endpoint in (0, 1):
            scatter[endpoint, i] = scatter[i, endpoint] = cross
    for i, j in combinations(range(len(labels)), 2):
        cross = 300 * scale * scale + 10 * (signs[i] + signs[j]) * scale + signs[i] * signs[j]
        scatter[i + 2, j + 2] = scatter[j + 2, i + 2] = cross
    return scatter

def expected_edges(labels):
    labels = tuple(labels)
    m = len(labels)
    result = [(endpoint, i + 2) for endpoint in (0, 1) for i in range(m)]
    if not any(labels):
        result.append((0, 1))
    return sorted(result)

def fixed_envelope(d):
    result = {}
    for a in range(d):
        for b in range(d):
            center = 1620 if a == b else 324 if {a, b} == {0, 1} else 540 if min(a, b) < 2 else 300
            result[a, b] = Interval(center - 6, center + 6)
    return result

def structural_tests(m):
    tests = {(0, 1, ()): 'dependent'}
    d = m + 2
    for endpoint in (0, 1):
        for z in range(2, d):
            tests[endpoint, z, ()] = 'dependent'
            for condition in range(d):
                if condition not in (endpoint, z):
                    tests[endpoint, z, (condition,)] = 'dependent'
    for z, w in combinations(range(2, d), 2):
        tests[z, w, (0,)] = 'independent'
    return tests

def certify_prefix(reader):
    evaluator = ColumnMonotoneEvaluator(reader, origin_depths=(0,) * reader.d)
    formulas = {e: edge_formula(reader.d, *e) for e in combinations(range(reader.d), 2)}
    tests = sorted({t for f in formulas.values() for t in leaves(f)})
    known, trace = ({}, [])
    for test in tests:
        decision, _ = evaluator.evaluate(test)
        if decision != 'unknown':
            known[test] = decision != 'independent'
            trace.append(evaluator.record(test))
    values = {e: value(f, known) for e, f in formulas.items()}
    complete = None not in values.values()
    proofs = []
    if complete:
        for edge, formula in formulas.items():
            target = values[edge]
            cost, requirements = plan(formula, known, target, lambda *_: math.inf)
            assert cost == 0
            proofs.append({'edge': edge, 'present': target, 'requirements': [{'test': t, 'non_deleting': bit} for t, bit in requirements]})
    return {'status': 'certified' if complete else 'unresolved', 'edges': [e for e, present in values.items() if present] if complete else None, 'origin_depths': (0,) * reader.d, 'threshold': str(Fraction(1, 10)), 'max_order': 1, 'bound': 'column_monotone', 'trace': trace, 'edge_proofs': proofs, 'actions': list(reader.actions), 'depths': list(reader.depths), 'payload_bits': reader.payload_bits, 'wire_bytes': reader.wire_bytes, 'read_bytes': reader.read_bytes, 'requests': reader.requests, 'file_bytes': reader.file_bytes, 'certificate_calls': evaluator.calls, 'scatter_calls': evaluator.scatter_calls, 'column_calls': evaluator.column_calls}
