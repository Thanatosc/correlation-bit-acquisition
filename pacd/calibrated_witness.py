# Copyright (c) 2026 Siyu Cai. SPDX-License-Identifier: MIT
from fractions import Fraction
from itertools import combinations
import math
import time
from .column_monotone import ColumnMonotoneEvaluator
from .edge_formula import edge_formula, leaves, plan, value
from .formula_witness import choose_witness

def calibration_round_limit(n, threshold=Fraction(1, 10)):
    tau = Fraction(threshold)
    if n < 2 or not 0 < tau < 1:
        raise ValueError('n >= 2 and 0 < threshold < 1 required')
    target = 2048 * n * tau.denominator ** 2
    power, rounds = (tau.numerator ** 2, 0)
    while power < target:
        power <<= 2
        rounds += 1
    return rounds

def is_calibrated(n, width, floor, threshold=Fraction(1, 10)):
    tau = Fraction(threshold)
    return (32 * tau.denominator * n * width) ** 2 <= tau.numerator ** 2 * floor

def discover_calibrated(reader, threshold=Fraction(1, 10), max_order=1, max_additional_actions=None, timeout_s=300):
    tau = Fraction(threshold)
    if not 0 < tau < 1 or max_order not in (0, 1) or timeout_s < 0 or (max_additional_actions is not None and max_additional_actions < 0):
        raise ValueError('Require positive threshold, order zero/one and nonnegative budgets')
    if any(reader.depths) or reader.actions:
        raise ValueError('This policy is defined from an empty prefix')
    started, cpu = (time.perf_counter(), time.process_time())
    evaluator = ColumnMonotoneEvaluator(reader, tau, origin_depths=(0,) * reader.d)
    costs = [reader.action_cost(c) for c in range(reader.d)]
    if len(set(costs)) != 1 or costs[0] <= 0:
        raise ValueError('Equal positive whole-column plane costs required')
    initial_wire = reader.wire_bytes
    formulas = {e: edge_formula(reader.d, *e, max_order) for e in combinations(range(reader.d), 2)}
    tests = sorted({t for f in formulas.values() for t in leaves(f)}, key=lambda t: (len(t[2]), t))
    states, trace, calibration, witnesses, queue, action_phases = ({}, [], [], [], [], [])
    mandatory = [None] * reader.d
    phase, transition, edges = ('mandatory', None, None)
    while True:
        if time.perf_counter() - started >= timeout_s:
            status = 'timeout'
            break
        evaluator.refresh()
        for c in range(reader.d):
            if mandatory[c] is None and (evaluator.floor[c] > 0 or reader.depths[c] == reader.bits):
                mandatory[c] = reader.depths[c]
        for test in tests:
            if test not in states:
                decision, _ = evaluator.evaluate(test)
                if decision != 'unknown':
                    states[test] = decision != 'independent'
                    trace.append(evaluator.record(test))
        values = {e: value(f, states) for e, f in formulas.items()}
        if None not in values.values():
            status = 'certified'
            edges = [e for e, present in values.items() if present]
            break
        if time.perf_counter() - started >= timeout_s:
            status = 'timeout'
            break
        if max_additional_actions is not None and len(reader.actions) >= max_additional_actions:
            status = 'unresolved'
            break
        if phase == 'mandatory':
            pending = [c for c, r in enumerate(mandatory) if r is None]
            if not pending:
                phase = 'calibration'
                continue
            column = min(pending, key=lambda c: (reader.depths[c], c))
            reader.read_next(column)
            action_phases.append(phase)
            continue
        if phase == 'calibration' and (not queue):
            if all((is_calibrated(reader.n, (1 << reader.bits - reader.depths[c]) - 1, evaluator.floor[c], tau) for c in range(reader.d))):
                possible = [e for e in formulas if states.get((*e, ())) is not False]
                transition = {'depths': list(reader.depths), 'fact_count': len(trace), 'variance_floors': list(evaluator.floor), 'possible_marginal_edges': possible, 'action_start': len(reader.actions)}
                phase = 'witness'
                continue
            queue = list(reader.available())
            if not queue:
                raise RuntimeError('Full precision must certify the graph')
            calibration.append({'columns': list(queue), 'depths': list(reader.depths), 'fact_count': len(trace), 'action_start': len(reader.actions), 'action_end': len(reader.actions)})
        if phase == 'witness' and (not queue):
            witness = choose_witness(formulas, states, reader.depths, reader.bits)
            if witness is None:
                raise RuntimeError('Unresolved graph needs a witness')
            witnesses.append({**witness, 'depths': list(reader.depths), 'fact_count': len(trace), 'action_start': len(reader.actions), 'action_end': len(reader.actions)})
            queue = list(witness['columns'])
        column = queue.pop(0)
        reader.read_next(column)
        action_phases.append(phase)
        (calibration if phase == 'calibration' else witnesses)[-1]['action_end'] = len(reader.actions)
    proofs = []
    if status == 'certified':
        assert all((r is not None for r in mandatory))
        for edge, formula in formulas.items():
            target = values[edge]
            cost, requirements = plan(formula, states, target, lambda *_: math.inf)
            assert cost == 0
            proofs.append({'edge': edge, 'present': target, 'requirements': [{'test': t, 'non_deleting': bit} for t, bit in requirements]})
    lower_queries = sum((r for r in mandatory if r is not None)) + len(calibration) + len(witnesses)
    k = max((len(r['columns']) for r in witnesses), default=0)
    factor = min(reader.d, max(len(calibration) + 1, k)) if status == 'certified' else None
    if status == 'certified':
        assert len(reader.actions) <= factor * lower_queries
    return {'status': status, 'edges': edges, 'mode': 'calibrated_witness', 'bound': 'column_monotone', 'threshold': str(tau), 'max_order': max_order, 'origin_depths': [0] * reader.d, 'initial_wire_bytes': initial_wire, 'initial_payload_bits': 0, 'initial_actions': 0, 'added_wire_bytes': reader.wire_bytes - initial_wire, 'added_payload_bits': reader.payload_bits, 'wire_bytes': reader.wire_bytes, 'read_bytes': reader.read_bytes, 'requests': reader.requests, 'payload_bits': reader.payload_bits, 'file_bytes': reader.file_bytes, 'actions': list(reader.actions), 'depths': list(reader.depths), 'trace': trace, 'edge_proofs': proofs, 'mandatory_depths': mandatory, 'calibration_rounds': calibration, 'calibration_transition': transition, 'calibration_round_limit': calibration_round_limit(reader.n, tau), 'witness_rounds': witnesses, 'action_phases': action_phases, 'phase_at_stop': phase, 'certificate_lower_bound_queries': lower_queries, 'certificate_lower_bound_wire': lower_queries * costs[0], 'max_witness_size': k, 'instance_competitive_factor': factor, 'certificate_calls': evaluator.calls, 'scatter_calls': evaluator.scatter_calls, 'column_calls': evaluator.column_calls, 'seconds': time.perf_counter() - started, 'cpu_seconds': time.process_time() - cpu}
