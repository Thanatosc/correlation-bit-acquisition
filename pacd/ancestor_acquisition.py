# Copyright (c) 2026 Siyu Cai. SPDX-License-Identifier: MIT
from fractions import Fraction
from itertools import combinations, product
import math
import time
from types import SimpleNamespace
import numpy as np
from .edge_formula import edge_formula, leaves, plan, value
from .residual import BoundEvaluator

class CertificateTimeout(Exception):
    pass

def project_local(reader, test, support, depths):
    columns = (test[0], test[1], *test[2])
    if len(support) != len(depths):
        raise ValueError('One depth per support block required')
    expected = tuple((i for i, block in enumerate(reader.blocks) if block[0] in columns))
    if tuple(support) != expected:
        raise ValueError('Support must contain all involved blocks in reader order')
    lo = np.empty((reader.n, len(columns)), dtype=object)
    hi = np.empty_like(lo)
    mapping = {column: j for j, column in enumerate(columns)}
    for block_id, depth in zip(support, depths):
        if not 0 <= depth <= reader.depths[block_id]:
            raise ValueError('Cannot project to unread precision')
        column, start, end, *_ = reader.blocks[block_id]
        step = 1 << reader.bits - depth
        values = reader.lower[start:end, column] // step * step
        lo[start:end, mapping[column]] = values
        hi[start:end, mapping[column]] = values + step - 1
    return (lo, hi)

class AncestorFormula:

    def __init__(self, reader, bound='interval', threshold=Fraction(1, 10), max_order=1, origin_depths=None):
        if bound not in ('interval', 'residual') or max_order not in (0, 1):
            raise ValueError('Unsupported bound or conditioning order')
        self.reader, self.bound = (reader, bound)
        self.threshold = Fraction(threshold)
        if not 0 <= self.threshold < 1:
            raise ValueError('Threshold must be in [0,1)')
        self.origin = tuple(reader.depths if origin_depths is None else origin_depths)
        if len(self.origin) != len(reader.depths) or any((not isinstance(k, int) or not 0 <= k <= now for k, now in zip(self.origin, reader.depths))):
            raise ValueError('Origin must be a known ancestor depth vector')
        self.formulas = {edge: edge_formula(reader.d, *edge, max_order) for edge in combinations(range(reader.d), 2)}
        self.tests = sorted(set((t for f in self.formulas.values() for t in leaves(f))), key=lambda t: (len(t[2]), t))
        self.supports = {t: tuple((i for i, block in enumerate(reader.blocks) if block[0] in (t[0], t[1], *t[2]))) for t in self.tests}
        self.states, self.facts = ({}, [])
        self.seen = {t: set() for t in self.tests}
        self.calls = self.residual_calls = 0

    def refresh(self, deadline=math.inf):
        for test in self.tests:
            if test in self.states:
                continue
            support = self.supports[test]
            ranges = [range(self.reader.depths[b], self.origin[b] - 1, -1) for b in support]
            for depths in product(*ranges):
                if time.perf_counter() >= deadline:
                    raise CertificateTimeout
                if depths in self.seen[test]:
                    continue
                lo, hi = project_local(self.reader, test, support, depths)
                local = (0, 1, (2,) if test[2] else ())
                evaluator = BoundEvaluator(SimpleNamespace(lower=lo, upper=hi, version=0), self.threshold, self.bound)
                decision, margin, source = evaluator.evaluate(local)
                self.calls += evaluator.calls
                self.residual_calls += evaluator.residual_calls
                self.seen[test].add(depths)
                if decision != 'unknown':
                    self.states[test] = decision != 'independent'
                    self.facts.append({'test': test, 'decision': decision, 'bound': source, 'margin': None if margin is None else [margin.lo, margin.hi], 'support': support, 'ancestor_depths': depths, 'acquired_depths': list(self.reader.depths), 'payload_bits': self.reader.payload_bits})
                    break

    def graph(self):
        values = {edge: value(f, self.states) for edge, f in self.formulas.items()}
        return None if any((v is None for v in values.values())) else [e for e, v in values.items() if v]

    def proofs(self):
        proofs = []
        for edge, f in self.formulas.items():
            target = value(f, self.states)
            if target is None:
                continue
            cost, requirements = plan(f, self.states, target, lambda *_: math.inf)
            assert cost == 0
            proofs.append({'edge': edge, 'present': target, 'requirements': [{'test': t, 'non_deleting': v} for t, v in requirements]})
        return proofs

def next_balanced_block(reader, origin_depths):
    return min(reader.available(), key=lambda b: ((reader.depths[b] - origin_depths[b] + 1) * reader.action_cost(b), b))

def discover_ancestors(reader, bound='interval', threshold=Fraction(1, 10), max_order=1, max_additional_actions=None, timeout_s=300):
    if max_additional_actions is not None and max_additional_actions < 0:
        raise ValueError('Additional action budget must be nonnegative')
    if timeout_s < 0:
        raise ValueError('Timeout must be nonnegative')
    started, cpu_started = (time.perf_counter(), time.process_time())
    origin = tuple(reader.depths)
    initial_wire, initial_payload = (reader.wire_bytes, reader.payload_bits)
    initial_actions = len(reader.actions)
    checker = AncestorFormula(reader, bound, threshold, max_order)
    deadline = started + timeout_s
    timestamps, status, edges = ([], None, None)
    while True:
        try:
            checker.refresh(deadline)
        except CertificateTimeout:
            status = 'timeout'
            break
        edges = checker.graph()
        if edges is not None:
            status = 'certified'
            break
        if time.perf_counter() >= deadline:
            status = 'timeout'
            break
        if max_additional_actions is not None and len(timestamps) >= max_additional_actions:
            status = 'unresolved'
            break
        if reader.complete:
            raise RuntimeError('Fully read data must resolve the reference formulas')
        block = next_balanced_block(reader, origin)
        timestamps.append((reader.depths[block] - origin[block] + 1) * reader.action_cost(block))
        reader.read_next(block)
    return {'status': status, 'edges': edges if status == 'certified' else None, 'mode': 'ancestor_balanced', 'bound': bound, 'threshold': str(Fraction(threshold)), 'max_order': max_order, 'origin_depths': origin, 'initial_wire_bytes': initial_wire, 'initial_payload_bits': initial_payload, 'added_wire_bytes': reader.wire_bytes - initial_wire, 'added_payload_bits': reader.payload_bits - initial_payload, 'initial_actions': initial_actions, 'actions': list(reader.actions), 'completion_costs': timestamps, 'depths': list(reader.depths), 'payload_bits': reader.payload_bits, 'read_bytes': reader.read_bytes, 'wire_bytes': reader.wire_bytes, 'requests': reader.requests, 'file_bytes': reader.file_bytes, 'trace': checker.facts, 'edge_proofs': checker.proofs(), 'certificate_calls': checker.calls, 'residual_calls': checker.residual_calls, 'seconds': time.perf_counter() - started, 'cpu_seconds': time.process_time() - cpu_started}
