# Copyright (c) 2026 Siyu Cai. SPDX-License-Identifier: MIT
from fractions import Fraction
import unittest
import numpy as np
from pacd.column_monotone import ColumnMonotoneEvaluator, discover_column_balanced
from pacd.intervals import ci_certificate, scatter_bounds
from pacd.skeleton import reference_skeleton
from support import MemoryReader

class ColumnMonotoneTests(unittest.TestCase):

    def test_layout_boundary(self):
        reader = MemoryReader(np.zeros((5, 3), dtype=int), 3, 2)
        with self.assertRaises(ValueError):
            ColumnMonotoneEvaluator(reader)
        self.assertEqual(reader.actions, [])

    def test_repairs_actual_column_counterexample_without_joint_enumeration(self):
        data = np.full((106, 2), 512, dtype=np.int64)
        data[0, 0] = 768
        data[1, 0] = 260
        data[2:54, 0] += 4
        data[2:54, 1] += 1
        data[54:, 1] -= 1
        reader = MemoryReader(data, 10, 106)
        for b, k in enumerate((8, 10)):
            for _ in range(k):
                reader.read_next(b)
        self.assertEqual(ci_certificate(scatter_bounds(reader.lower, reader.upper), 0, 1)[0], 'unknown')
        old = reader.wire_bytes
        ev = ColumnMonotoneEvaluator(reader, origin_depths=(7, 10))
        self.assertEqual(ev.evaluate((0, 1, ()))[0], 'independent')
        self.assertEqual(ev.floor_depth, [7, 10])
        self.assertEqual(ev.column_calls, 1)
        self.assertEqual(reader.wire_bytes, old)

    def test_incremental_and_late_boxes_match(self):
        data = np.array([[1, 7, 4], [3, 4, 5], [7, 2, 1], [5, 1, 7], [2, 6, 3]])
        a = MemoryReader(data, 3, 5)
        b = MemoryReader(data, 3, 5)
        ev = ColumnMonotoneEvaluator(a)
        ev.refresh()
        for col in (2, 0, 1, 1, 0, 2):
            prev = ev.scatter.copy()
            a.read_next(col)
            ev.refresh()
            self.assertTrue(all((ev.scatter[t].lo >= v.lo and ev.scatter[t].hi <= v.hi for t, v in prev.items())))
        for col in (0, 0, 1, 1, 2, 2):
            b.read_next(col)
        late = ColumnMonotoneEvaluator(b, origin_depths=(0, 0, 0))
        late.refresh()
        self.assertEqual(ev.scatter, late.scatter)
        self.assertEqual(ev.floor_depth, late.floor_depth)

    def test_budget_timeout_singular_and_equality(self):
        data = np.array([[0, 0], [0, 2], [2, 0], [2, 2]])
        for tau in (Fraction(0), Fraction(1, 10)):
            reader = MemoryReader(data, 2, 4, [2, 7])
            incomplete = discover_column_balanced(reader, threshold=tau, max_additional_actions=0)
            self.assertEqual(incomplete['status'], 'unresolved')
            self.assertIsNone(incomplete['edges'])
            timed = discover_column_balanced(reader, threshold=tau, timeout_s=0)
            self.assertEqual(timed['status'], 'timeout')
            self.assertIsNone(timed['edges'])
            result = discover_column_balanced(reader, threshold=tau)
            self.assertEqual(result['edges'], reference_skeleton(data, tau)[0])
        reader = MemoryReader(np.zeros((4, 3), dtype=int), 2, 4)
        self.assertEqual(discover_column_balanced(reader)['edges'], [(0, 1), (0, 2), (1, 2)])
if __name__ == '__main__':
    unittest.main()
