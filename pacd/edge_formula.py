# Copyright (c) 2026 Siyu Cai. SPDX-License-Identifier: MIT
from dataclasses import dataclass
from math import inf

@dataclass(frozen=True)
class Formula:
    kind: str
    children: tuple = ()
    test: tuple = ()
    positive: bool = True

def literal(x, y, condition=(), positive=True):
    x, y = sorted((x, y))
    return Formula('literal', test=(x, y, tuple(condition)), positive=positive)

def edge_formula(d, x, y, max_order=1):
    if max_order not in (0, 1) or not 0 <= x < y < d:
        raise ValueError('Require x < y and maximum order zero or one')
    marginal = literal(x, y)
    if max_order == 0:
        return marginal
    factors = [marginal]
    for z in range(d):
        if z not in (x, y):
            no_neighbor = Formula('and', (literal(x, z, positive=False), literal(y, z, positive=False)))
            factors.append(Formula('or', (literal(x, y, (z,)), no_neighbor)))
    return Formula('and', tuple(factors))

def leaves(formula):
    if formula.kind == 'literal':
        return [formula.test]
    return [test for child in formula.children for test in leaves(child)]

def value(formula, states):
    if formula.kind == 'literal':
        bit = states.get(formula.test)
        return None if bit is None else bit == formula.positive
    values = [value(child, states) for child in formula.children]
    controlling = formula.kind == 'or'
    if controlling in values:
        return controlling
    if None in values:
        return None
    return not controlling

def live_tests(formula, states):
    if value(formula, states) is not None:
        return set()
    if formula.kind == 'literal':
        return {formula.test}
    return set().union(*(live_tests(child, states) for child in formula.children))

def plan(formula, states, target, weight):
    if formula.kind == 'literal':
        wanted = formula.positive if target else not formula.positive
        known = states.get(formula.test)
        if known is not None and known != wanted:
            return (inf, [])
        cost = 0 if known is not None else weight(formula.test, wanted)
        return (cost, [(formula.test, wanted)])
    all_needed = (formula.kind == 'and') == target
    choices = [plan(child, states, target, weight) for child in formula.children]
    if all_needed:
        return (sum((cost for cost, _ in choices)), [item for _, requirements in choices for item in requirements])
    return min(choices, key=lambda item: (item[0], item[1]))
