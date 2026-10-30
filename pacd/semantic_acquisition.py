# Copyright (c) 2026 Siyu Cai. SPDX-License-Identifier: MIT
from fractions import Fraction
from itertools import combinations
from .calibrated_witness import is_calibrated
from .column_monotone import ColumnMonotoneEvaluator
from .intervals import ci_certificate
from .semantic_moments import MomentLimits, certify_graph

def discover_semantic(reader, threshold=Fraction(1, 10), max_order=1, limits=MomentLimits(), on_limit='read', max_actions=None):
    tau = Fraction(threshold)
    if not 0 < tau < 1 or max_order not in (0, 1) or on_limit not in ('read', 'return'):
        raise ValueError('Invalid threshold, order, or resource-limit policy')
    if max_actions is not None and (not isinstance(max_actions, int) or max_actions < 0):
        raise ValueError('max_actions must be nonnegative or None')
    if any(reader.depths) or reader.actions:
        raise ValueError('Acquisition starts from empty whole-column prefixes')
    evaluator = ColumnMonotoneEvaluator(reader, tau)
    prices = [reader.action_cost(i) for i in range(reader.d)]
    if len(set(prices)) != 1 or prices[0] <= 0:
        raise ValueError('Equal positive column-plane prices required')
    phase = 'calibration'
    queue = []
    neighbors = None
    history = []
    resources = 0
    transition = None
    rounds = []
    result = None
    while True:
        result = certify_graph(reader.lower, reader.upper, tau, max_order, limits)
        entry = {'depths': list(reader.depths), 'status': result['status'], 'edge': result['edge'], 'reason': result['reason'], 'moment_calls': sum((r['method'] == 'moment_dp' for r in result['decisions'])), 'moment_transitions': sum((r.get('stats', {}).get('transitions', 0) for r in result['decisions']))}
        history.append(entry)
        if result['status'] == 'certified':
            status = 'certified'
            break
        if result['status'] == 'resource_limit':
            resources += 1
            if on_limit == 'return':
                status = 'resource_limit'
                break
        if max_actions is not None and len(reader.actions) >= max_actions:
            status = 'action_limit'
            break
        evaluator.refresh()
        if not queue and phase == 'calibration':
            if all((is_calibrated(reader.n, (1 << reader.bits - reader.depths[i]) - 1, evaluator.floor[i], tau) for i in range(reader.d))):
                neighbors = {i: set() for i in range(reader.d)}
                for x, y in combinations(range(reader.d), 2):
                    if ci_certificate(evaluator.scatter, x, y, threshold=tau)[0] != 'independent':
                        neighbors[x].add(y)
                        neighbors[y].add(x)
                transition = {'depths': list(reader.depths), 'neighbors': {i: sorted(v) for i, v in neighbors.items()}, 'variance_floors': list(evaluator.floor)}
                phase = 'witness'
            else:
                queue = [i for i in range(reader.d) if reader.depths[i] < reader.bits]
                rounds.append({'phase': 'calibration', 'columns': list(queue)})
        if not queue and phase == 'witness':
            if result['status'] == 'ambiguous':
                x, y = result['edge']
                support = {x, y} | neighbors[x] | neighbors[y]
                kind = 'semantic_witness'
            else:
                support = set(range(reader.d))
                kind = 'resource_fallback'
            queue = sorted((i for i in support if reader.depths[i] < reader.bits))
            rounds.append({'phase': kind, 'columns': list(queue), 'edge': result['edge']})
        if not queue:
            raise RuntimeError('A full-precision input must have a fixed output')
        reader.read_next(queue.pop(0))
    return {'status': status, 'edges': result['edges'] if status == 'certified' else None, 'threshold': str(tau), 'max_order': max_order, 'queries': len(reader.actions), 'depths': list(reader.depths), 'actions': list(reader.actions), 'wire_bytes': reader.wire_bytes, 'history': history, 'rounds': rounds, 'calibration_transition': transition, 'resource_limited_checks': resources, 'all_semantic_checks_completed': resources == 0, 'semantic_competitive_bound': 'conditional_on_theorem_assumptions' if resources == 0 and status == 'certified' else 'not_claimed', 'final_reason': result['reason']}
