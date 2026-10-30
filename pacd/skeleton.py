# Copyright (c) 2026 Siyu Cai. SPDX-License-Identifier: MIT
from fractions import Fraction
from itertools import combinations
from .intervals import exact_ci, exact_scatter

def skeleton_program(d, max_order=1):
    if max_order not in (0, 1):
        raise ValueError('This prototype supports max_order 0 or 1')
    edges = set(combinations(range(d), 2))
    for order in range(max_order + 1):
        neighbors = {v: set() for v in range(d)}
        for x, y in edges:
            neighbors[x].add(y)
            neighbors[y].add(x)
        for x, y in sorted(edges):
            conditions = sorted(set(combinations(sorted(neighbors[x] - {y}), order)) | set(combinations(sorted(neighbors[y] - {x}), order)))
            for condition in conditions:
                decision = (yield (x, y, condition))
                if decision not in ('independent', 'dependent', 'singular'):
                    raise ValueError('Cannot advance the program on an unresolved test')
                if decision == 'independent':
                    edges.remove((x, y))
                    break
    return sorted(edges)

def reference_skeleton(data, threshold=Fraction(1, 10), max_order=1):
    scatter = exact_scatter(data)
    program = skeleton_program(len(data[0]), max_order)
    trace = []
    try:
        test = next(program)
        while True:
            decision = exact_ci(scatter, *test, threshold=threshold)
            trace.append((*test, decision))
            test = program.send(decision)
    except StopIteration as end:
        return (end.value, trace)
