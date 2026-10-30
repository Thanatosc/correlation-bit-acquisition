# Copyright (c) 2026 Siyu Cai. SPDX-License-Identifier: MIT
from fractions import Fraction
from itertools import combinations
import math
import time
import numpy as np
from .ancestor_acquisition import next_balanced_block
from .edge_formula import edge_formula, leaves, plan, value
from .intervals import Interval, ci_certificate, scatter_bounds

class ColumnMonotoneEvaluator:

    def __init__(self, reader, threshold=Fraction(1, 10), origin_depths=None):
        if len(reader.blocks) != reader.d or any(((c, s, e) != (i, 0, reader.n) for i, (c, s, e, *_) in enumerate(reader.blocks))):
            raise ValueError('Requires exactly one full-row block per column, in column order')
        self.reader = reader
        self.threshold = Fraction(threshold)
        if not 0 <= self.threshold < 1:
            raise ValueError('Threshold must be in [0,1)')
        self.origin = tuple(reader.depths if origin_depths is None else origin_depths)
        if len(self.origin) != reader.d or any((not isinstance(s, int) or not 0 <= s <= k for s, k in zip(self.origin, reader.depths))):
            raise ValueError('Origin must be a known column-depth ancestor')
        self.processed = [s - 1 for s in self.origin]
        self.floor = [0] * reader.d
        self.floor_depth = list(self.origin)
        self.current_depths = None
        self.cache = {}
        self.calls = self.scatter_calls = self.column_calls = 0

    def refresh(self):
        depths = tuple(self.reader.depths)
        if depths == self.current_depths:
            return
        current = scatter_bounds(self.reader.lower, self.reader.upper)
        self.scatter_calls += 1
        for col, k in enumerate(depths):
            if k < self.processed[col]:
                raise ValueError('Reader precision must never decrease')
            for q in range(self.processed[col] + 1, k + 1):
                if q == k:
                    lo = current[col, col].lo
                else:
                    step = 1 << self.reader.bits - q
                    lower = self.reader.lower[:, col:col + 1] // step * step
                    upper = lower + step - 1
                    lo = scatter_bounds(lower, upper)[0, 0].lo
                    self.column_calls += 1
                if lo > self.floor[col]:
                    self.floor[col] = lo
                    self.floor_depth[col] = q
            self.processed[col] = k
            current[col, col] = Interval(self.floor[col], current[col, col].hi)
        self.scatter = current
        self.current_depths = depths
        self.cache.clear()

    def evaluate(self, test):
        self.refresh()
        if test not in self.cache:
            self.calls += 1
            decision, margin = ci_certificate(self.scatter, *test, self.threshold)
            self.cache[test] = (decision, margin)
        return self.cache[test]

    def record(self, test):
        decision, margin = self.evaluate(test)
        return {'test': test, 'decision': decision, 'bound': 'column_monotone', 'margin': None if margin is None else [margin.lo, margin.hi], 'depths': list(self.reader.depths), 'floor_depths': list(self.floor_depth), 'variance_floors': list(self.floor), 'payload_bits': self.reader.payload_bits}

def discover_column_balanced(reader, threshold=Fraction(1, 10), max_order=1, max_additional_actions=None, timeout_s=300):
    if max_order not in (0, 1) or timeout_s < 0 or (max_additional_actions is not None and max_additional_actions < 0):
        raise ValueError('Unsupported order or negative budget')
    started, cpu = (time.perf_counter(), time.process_time())
    evaluator = ColumnMonotoneEvaluator(reader, threshold)
    origin = tuple(reader.depths)
    initial_wire, initial_payload = (reader.wire_bytes, reader.payload_bits)
    initial_actions = len(reader.actions)
    formulas = {e: edge_formula(reader.d, *e, max_order) for e in combinations(range(reader.d), 2)}
    tests = sorted({t for f in formulas.values() for t in leaves(f)}, key=lambda t: (len(t[2]), t))
    states, trace, stamps = ({}, [], [])
    edges = None
    while True:
        if time.perf_counter() - started >= timeout_s:
            status = 'timeout'
            break
        for test in tests:
            if test not in states:
                decision, _ = evaluator.evaluate(test)
                if decision != 'unknown':
                    states[test] = decision != 'independent'
                    trace.append(evaluator.record(test))
        values = {e: value(f, states) for e, f in formulas.items()}
        if None not in values.values():
            status = 'certified'
            edges = [e for e, v in values.items() if v]
            break
        if time.perf_counter() - started >= timeout_s:
            status = 'timeout'
            break
        if max_additional_actions is not None and len(stamps) >= max_additional_actions:
            status = 'unresolved'
            break
        b = next_balanced_block(reader, origin)
        stamps.append((reader.depths[b] - origin[b] + 1) * reader.action_cost(b))
        reader.read_next(b)
    proofs = []
    if status == 'certified':
        for e, f in formulas.items():
            target = value(f, states)
            cost, requirements = plan(f, states, target, lambda *_: math.inf)
            assert cost == 0
            proofs.append({'edge': e, 'present': target, 'requirements': [{'test': t, 'non_deleting': v} for t, v in requirements]})
    return {'status': status, 'edges': edges, 'mode': 'column_balanced', 'bound': 'column_monotone', 'threshold': str(Fraction(threshold)), 'max_order': max_order, 'origin_depths': origin, 'initial_wire_bytes': initial_wire, 'initial_payload_bits': initial_payload, 'initial_actions': initial_actions, 'added_wire_bytes': reader.wire_bytes - initial_wire, 'added_payload_bits': reader.payload_bits - initial_payload, 'wire_bytes': reader.wire_bytes, 'read_bytes': reader.read_bytes, 'requests': reader.requests, 'payload_bits': reader.payload_bits, 'file_bytes': reader.file_bytes, 'depths': list(reader.depths), 'actions': list(reader.actions), 'completion_costs': stamps, 'trace': trace, 'edge_proofs': proofs, 'certificate_calls': evaluator.calls, 'scatter_calls': evaluator.scatter_calls, 'column_calls': evaluator.column_calls, 'seconds': time.perf_counter() - started, 'cpu_seconds': time.process_time() - cpu}
