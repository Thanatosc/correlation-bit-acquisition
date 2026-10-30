# Copyright (c) 2026 Siyu Cai. SPDX-License-Identifier: MIT
from fractions import Fraction as F
from itertools import product
from math import comb
import unittest
import numpy as np
from pacd.hardness_any_fixed_threshold import construct
from pacd.semantic_moments import MomentLimits, UNLIMITED, certify_graph, moment_edge
from pacd.skeleton import reference_skeleton

def brute_graphs(lo, hi, tau=F(1, 10), order=1):
    choices = [range(int(a), int(b) + 1) for a, b in zip(lo.flat, hi.flat)]
    return {tuple(reference_skeleton(np.array(values, dtype=object).reshape(lo.shape), tau, order)[0]) for values in product(*choices)}

class SemanticMomentTests(unittest.TestCase):

    def compare(self, lo, hi, tau=F(1, 10), order=1):
        graphs = brute_graphs(lo, hi, tau, order)
        for shortcuts in (False, True):
            result = certify_graph(lo, hi, tau, order, UNLIMITED, shortcuts)
            self.assertEqual(result['status'], 'certified' if len(graphs) == 1 else 'ambiguous')
            if len(graphs) == 1:
                self.assertEqual(tuple(result['edges']), next(iter(graphs)))
            else:
                actual = []
                for witness in result['witnesses']:
                    data = np.array(witness['data'], dtype=object)
                    self.assertTrue(np.all(lo <= data) and np.all(data <= hi))
                    actual.append(tuple(reference_skeleton(data, tau, order)[0]))
                self.assertEqual(len(set(actual)), 2)
                self.assertTrue(set(actual) <= graphs)

    def test_small_boxes_against_complete_matrix_enumeration(self):
        rng = np.random.default_rng(9031)
        for d in (2, 3):
            for index in range(12):
                lo = rng.integers(-3, 4, size=(3, d)).astype(object)
                hi = lo + rng.integers(0, 2, size=(3, d)).astype(object)
                self.compare(lo, hi, F(1, 10) if index % 2 else F(1, 2), index % 2)
        lo = np.zeros((3, 3), dtype=object)
        self.compare(lo, lo + 1, F(1, 2))

    def test_all_binary_whole_column_prefix_depths(self):
        data = np.array([[0, 1, 0], [1, 0, 1], [1, 1, 0]], dtype=object)
        for depths in product((0, 1), repeat=3):
            step = np.array([2 ** (1 - k) for k in depths], dtype=object)
            lo = data // step * step
            self.compare(lo, lo + step - 1, F(1, 2))

    def test_threshold_equality_and_singular_retention(self):
        equality = np.array([[0, 0], [1, 2], [2, 1]], dtype=object)
        self.assertEqual(certify_graph(equality, equality, F(1, 2))['edges'], [])
        singular = np.array([[3, 0], [3, 1], [3, 2]], dtype=object)
        self.assertEqual(certify_graph(singular, singular)['edges'], [(0, 1)])

    def test_moment_count_for_large_binary_completion_set(self):
        lo = np.zeros((32, 2), dtype=object)
        result = moment_edge(lo, lo + 1, limits=UNLIMITED)
        self.assertEqual(result['stats']['raw_completions'], 2 ** 64)
        self.assertEqual(result['stats']['terminal_states'], comb(35, 3))
        self.assertEqual(result['status'], 'ambiguous')
        graphs = [reference_skeleton(w['data'])[0] for w in result['witnesses']]
        self.assertNotEqual(*graphs)

    def test_joint_integer_dependencies_in_cellwise_control(self):
        for target, status in ((1, 'certified'), (2, 'ambiguous')):
            lo, hi, _ = construct([2, 2], target, F(1, 10))
            result = certify_graph(lo, hi, limits=UNLIMITED)
            self.assertEqual(result['status'], status)
            if status == 'certified':
                self.assertEqual(result['edges'], [])
            else:
                a, b = [reference_skeleton(w['data'])[0] for w in result['witnesses']]
                self.assertNotEqual(a, b)

    def test_resource_limit_never_asserts_ambiguity_or_constancy(self):
        lo = np.zeros((4, 2), dtype=object)
        for limits in (MomentLimits(max_states=1), MomentLimits(max_transitions=1), MomentLimits(max_row_options=1), MomentLimits(max_nodes=1)):
            result = moment_edge(lo, lo + 1, limits=limits)
            self.assertEqual(result['status'], 'resource_limit')
            self.assertIsNone(result['possible_values'])
            self.assertEqual(result['witnesses'], [])
        huge = np.zeros((2, 2), dtype=object)
        result = certify_graph(huge, huge + 2 ** 256, limits=MomentLimits(max_row_options=2), use_shortcuts=False)
        self.assertEqual(result['status'], 'resource_limit')
        self.assertIsNone(result['edges'])

    def test_bounds_contract(self):
        for lo, hi in (([[0, 0]], [[1, 1]]), ([[0.0, 0], [0, 0]], [[1, 1], [1, 1]]), ([[1, 0], [0, 0]], [[0, 1], [1, 1]])):
            with self.assertRaises(ValueError):
                certify_graph(lo, hi)
        with self.assertRaises(ValueError):
            MomentLimits(max_states=0)
if __name__ == '__main__':
    unittest.main()
