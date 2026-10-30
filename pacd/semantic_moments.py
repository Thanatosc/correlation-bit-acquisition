# Copyright (c) 2026 Siyu Cai. SPDX-License-Identifier: MIT
from dataclasses import dataclass
from fractions import Fraction
from itertools import chain, combinations, product
from math import prod
from numbers import Integral
import numpy as np
from .intervals import ci_certificate, exact_ci, scatter_bounds
from .positive_correlation_corner import positive_corner_certificate
from .skeleton import reference_skeleton

@dataclass(frozen=True)
class MomentLimits:
    max_states: int | None = 50000
    max_transitions: int | None = 500000
    max_row_options: int | None = 10000
    max_nodes: int | None = 500000

    def __post_init__(self):
        for number in (self.max_states, self.max_transitions, self.max_row_options, self.max_nodes):
            if number is not None and (not isinstance(number, int) or isinstance(number, bool) or number < 1):
                raise ValueError('Limits must be positive integers or None')
UNLIMITED = MomentLimits(None, None, None, None)

def _bounds(lower, upper):
    lo, hi = (np.asarray(lower, dtype=object), np.asarray(upper, dtype=object))
    if lo.ndim != 2 or lo.shape != hi.shape or lo.shape[0] < 2 or (lo.shape[1] < 2):
        raise ValueError('Ordered n-by-d integer bounds, n,d >= 2, required')
    if any((not isinstance(v, Integral) or isinstance(v, bool) for v in chain(lo.flat, hi.flat))):
        raise ValueError('Bounds must contain integers')
    lo = np.array([[int(v) for v in row] for row in lo], dtype=object)
    hi = np.array([[int(v) for v in row] for row in hi], dtype=object)
    if np.any(lo > hi):
        raise ValueError('Reversed bounds')
    return (lo, hi)

def _scatter(state, n, k, pairs):
    result = {}
    for index, (i, j) in enumerate(pairs, k):
        result[i, j] = result[j, i] = n * state[index] - state[i] * state[j]
    return result

def _edge(scatter, k, tau, max_order):
    if exact_ci(scatter, 0, 1, threshold=tau) == 'independent':
        return False
    if max_order:
        for z in range(2, k):
            eligible = exact_ci(scatter, 0, z, threshold=tau) != 'independent' or exact_ci(scatter, 1, z, threshold=tau) != 'independent'
            if eligible and exact_ci(scatter, 0, 1, (z,), tau) == 'independent':
                return False
    return True

def moment_edge(lower, upper, threshold=Fraction(1, 10), max_order=1, limits=MomentLimits()):
    lo, hi = _bounds(lower, upper)
    tau = Fraction(threshold)
    if not 0 < tau < 1 or max_order not in (0, 1):
        raise ValueError('Require 0 < threshold < 1 and order zero or one')
    n, k = lo.shape
    shifts = [min(lo[:, i]) for i in range(k)]
    lo = lo - np.array(shifts, dtype=object)
    hi = hi - np.array(shifts, dtype=object)
    pairs = list(combinations(range(k), 2))
    pairs = sorted([(i, i) for i in range(k)] + pairs)
    zero = (0,) * (k + len(pairs))
    states = {zero: None}
    stats = {'rows_completed': 0, 'peak_states': 1, 'terminal_states': None, 'transitions': 0, 'nodes_created': 0, 'columns': k, 'raw_completions': prod((int(b - a + 1) for a, b in zip(lo.flat, hi.flat)))}

    def limited(reason):
        return {'status': 'resource_limit', 'possible_values': None, 'witnesses': [], 'reason': reason, 'stats': stats}
    for row in range(n):
        options = prod((int(hi[row, i] - lo[row, i] + 1) for i in range(k)))
        if limits.max_row_options is not None and options > limits.max_row_options:
            return limited('row_options')
        if limits.max_transitions is not None and stats['transitions'] + len(states) * options > limits.max_transitions:
            return limited('transitions')
        vectors = []
        for values in product(*(range(int(lo[row, i]), int(hi[row, i]) + 1) for i in range(k))):
            phi = values + tuple((values[i] * values[j] for i, j in pairs))
            vectors.append((values, phi))
        following = {}
        for state, node in states.items():
            for values, phi in vectors:
                target = tuple((a + b for a, b in zip(state, phi)))
                stats['transitions'] += 1
                if target in following:
                    continue
                if limits.max_states is not None and len(following) >= limits.max_states:
                    return limited('states')
                if limits.max_nodes is not None and stats['nodes_created'] >= limits.max_nodes:
                    return limited('nodes')
                following[target] = (node, values)
                stats['nodes_created'] += 1
        states = following
        stats['rows_completed'] = row + 1
        stats['peak_states'] = max(stats['peak_states'], len(states))
    stats['terminal_states'] = len(states)
    witnesses = {}
    for state, node in states.items():
        present = _edge(_scatter(state, n, k, pairs), k, tau, max_order)
        if present not in witnesses:
            rows = []
            while node is not None:
                node, values = node
                rows.append([int(values[i] + shifts[i]) for i in range(k)])
            witnesses[present] = list(reversed(rows))
            if len(witnesses) == 2:
                break
    return {'status': 'ambiguous' if len(witnesses) == 2 else 'certified', 'possible_values': sorted(witnesses), 'witnesses': [{'present': v, 'data': witnesses[v]} for v in sorted(witnesses)], 'reason': None, 'stats': stats}

def certify_graph(lower, upper, threshold=Fraction(1, 10), max_order=1, limits=MomentLimits(), use_shortcuts=True):
    lo, hi = _bounds(lower, upper)
    tau = Fraction(threshold)
    if not 0 < tau < 1 or max_order not in (0, 1):
        raise ValueError('Require 0 < threshold < 1 and order zero or one')
    n, d = lo.shape
    if use_shortcuts:
        baseline = tuple(reference_skeleton(lo, tau, max_order)[0])
        for seed in range(1, 5):
            candidate = np.array([[hi[r, c] if (r & c + seed).bit_count() % 2 else lo[r, c] for c in range(d)] for r in range(n)], dtype=object)
            graph = tuple(reference_skeleton(candidate, tau, max_order)[0])
            if graph != baseline:
                edge = min(set(graph) ^ set(baseline))
                return {'status': 'ambiguous', 'edges': None, 'edge': list(edge), 'reason': None, 'witnesses': [{'edge_present': edge in baseline, 'data': lo.tolist()}, {'edge_present': edge in graph, 'data': candidate.tolist()}], 'decisions': [{'method': 'completion_probes', 'completions_checked': seed + 1}]}
    box = scatter_bounds(lo, hi)
    marginal = {e: ci_certificate(box, *e, threshold=tau)[0] for e in combinations(range(d), 2)}

    def m(a, b):
        return marginal[min(a, b), max(a, b)]
    edges = []
    decisions = []
    for x, y in combinations(range(d), 2):
        fixed = None
        candidates = []
        if use_shortcuts and m(x, y) == 'independent':
            fixed = False
        else:
            for z in range(d):
                if z in (x, y) or not max_order:
                    continue
                conditional = ci_certificate(box, x, y, (z,), tau)[0]
                if use_shortcuts:
                    if conditional in ('dependent', 'singular') or m(x, z) == m(y, z) == 'independent':
                        continue
                    if conditional == 'independent' and (m(x, z) in ('dependent', 'singular') or m(y, z) in ('dependent', 'singular')):
                        fixed = False
                        break
                candidates.append(z)
            if fixed is None and use_shortcuts and (not candidates) and (m(x, y) in ('dependent', 'singular')):
                fixed = True
            if fixed is None and use_shortcuts and (not candidates):
                corner = positive_corner_certificate(lo[:, [x, y]], hi[:, [x, y]], tau)
                if corner['decision'] == 'dependent':
                    fixed = True
        if fixed is not None:
            decisions.append({'edge': [x, y], 'method': 'interval_or_corner', 'present': fixed})
            if fixed:
                edges.append((x, y))
            continue
        support = [x, y, *candidates]
        result = moment_edge(lo[:, support], hi[:, support], tau, max_order, limits)
        decisions.append({'edge': [x, y], 'method': 'moment_dp', 'support': support, 'status': result['status'], 'stats': result['stats']})
        if result['status'] == 'resource_limit':
            return {'status': 'resource_limit', 'edges': None, 'edge': [x, y], 'reason': result['reason'], 'witnesses': [], 'decisions': decisions}
        if result['status'] == 'ambiguous':
            witnesses = []
            for item in result['witnesses']:
                data = lo.copy()
                data[:, support] = np.array(item['data'], dtype=object)
                witnesses.append({'edge_present': item['present'], 'data': data.tolist()})
            return {'status': 'ambiguous', 'edges': None, 'edge': [x, y], 'reason': None, 'witnesses': witnesses, 'decisions': decisions}
        present = result['possible_values'][0]
        decisions[-1]['present'] = present
        if present:
            edges.append((x, y))
    return {'status': 'certified', 'edges': edges, 'edge': None, 'reason': None, 'witnesses': [], 'decisions': decisions}
