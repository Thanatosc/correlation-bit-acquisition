# Copyright (c) 2026 Siyu Cai. SPDX-License-Identifier: MIT
from fractions import Fraction
from itertools import combinations
import math
import time
from .column_monotone import ColumnMonotoneEvaluator
from .edge_formula import edge_formula, leaves, live_tests, plan, value

def support_mask(test, unfinished_mask):
    x, y, condition = test
    return sum((1 << c for c in (x, y, *condition))) & unfinished_mask

def forceable(formula, states, frozen_mask, unfinished_mask):
    if formula.kind == 'literal':
        known = states.get(formula.test)
        if known is not None:
            return 1 << int(known == formula.positive)
        support = support_mask(formula.test, unfinished_mask)
        return 0 if support & ~frozen_mask == 0 else 3
    children = [forceable(c, states, frozen_mask, unfinished_mask) for c in formula.children]
    if formula.kind == 'and':
        return int(any((v & 1 for v in children))) | (2 if all((v & 2 for v in children)) else 0)
    return int(all((v & 1 for v in children))) | (2 if any((v & 2 for v in children)) else 0)

def choose_witness(formulas, states, depths, bits):
    unfinished = sum((1 << c for c, k in enumerate(depths) if k < bits))
    candidates = []
    for edge, formula in formulas.items():
        if value(formula, states) is not None:
            continue
        mask = 0
        for test in live_tests(formula, states):
            mask |= support_mask(test, unfinished)
        initial = mask
        assert mask and forceable(formula, states, mask, unfinished) == 0
        for col in reversed(range(len(depths))):
            if mask & 1 << col:
                trial = mask & ~(1 << col)
                if forceable(formula, states, trial, unfinished) == 0:
                    mask = trial
        columns = tuple((c for c in range(len(depths)) if mask & 1 << c))
        if not columns:
            raise RuntimeError('Unresolved full-precision atoms violate verifier completeness')
        candidates.append((len(columns), edge, columns, initial.bit_count()))
    if not candidates:
        return None
    _, edge, columns, initial_size = min(candidates)
    return {'edge': edge, 'columns': columns, 'initial_live_columns': initial_size}

def discover_formula_witness(reader, threshold=Fraction(1, 10), max_order=1, max_additional_actions=None, timeout_s=300):
    if max_order not in (0, 1) or timeout_s < 0 or (max_additional_actions is not None and max_additional_actions < 0):
        raise ValueError('Unsupported order or negative budget')
    started, cpu = (time.perf_counter(), time.process_time())
    evaluator = ColumnMonotoneEvaluator(reader, threshold)
    costs = [reader.action_cost(b) for b in range(reader.d)]
    if len(set(costs)) != 1 or costs[0] <= 0:
        raise ValueError('Witness rounds require equal positive column-plane costs')
    origin = tuple(reader.depths)
    initial_wire, initial_payload = (reader.wire_bytes, reader.payload_bits)
    initial_actions = len(reader.actions)
    formulas = {e: edge_formula(reader.d, *e, max_order) for e in combinations(range(reader.d), 2)}
    tests = sorted({t for f in formulas.values() for t in leaves(f)}, key=lambda t: (len(t[2]), t))
    states, trace, rounds, queue = ({}, [], [], [])
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
        added_actions = len(reader.actions) - initial_actions
        if max_additional_actions is not None and added_actions >= max_additional_actions:
            status = 'unresolved'
            break
        if not queue:
            witness = choose_witness(formulas, states, reader.depths, reader.bits)
            if witness is None:
                raise RuntimeError('Unresolved graph must have a witness')
            rounds.append({**witness, 'depths': list(reader.depths), 'fact_count': len(trace), 'action_start': len(reader.actions), 'action_end': len(reader.actions)})
            queue = list(witness['columns'])
        reader.read_next(queue.pop(0))
        rounds[-1]['action_end'] = len(reader.actions)
    proofs = []
    if status == 'certified':
        for edge, formula in formulas.items():
            target = value(formula, states)
            cost, requirements = plan(formula, states, target, lambda *_: math.inf)
            assert cost == 0
            proofs.append({'edge': edge, 'present': target, 'requirements': [{'test': t, 'non_deleting': v} for t, v in requirements]})
    lower_bound = len(rounds) * costs[0]
    return {'status': status, 'edges': edges, 'mode': 'formula_witness', 'bound': 'column_monotone', 'threshold': str(Fraction(threshold)), 'max_order': max_order, 'origin_depths': origin, 'initial_wire_bytes': initial_wire, 'initial_payload_bits': initial_payload, 'initial_actions': initial_actions, 'added_wire_bytes': reader.wire_bytes - initial_wire, 'added_payload_bits': reader.payload_bits - initial_payload, 'wire_bytes': reader.wire_bytes, 'read_bytes': reader.read_bytes, 'requests': reader.requests, 'payload_bits': reader.payload_bits, 'file_bytes': reader.file_bytes, 'depths': list(reader.depths), 'actions': list(reader.actions), 'trace': trace, 'edge_proofs': proofs, 'witness_rounds': rounds, 'certificate_lower_bound_wire': lower_bound, 'max_witness_size': max((len(r['columns']) for r in rounds), default=0), 'certificate_calls': evaluator.calls, 'scatter_calls': evaluator.scatter_calls, 'column_calls': evaluator.column_calls, 'seconds': time.perf_counter() - started, 'cpu_seconds': time.process_time() - cpu}
