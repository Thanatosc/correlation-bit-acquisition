# Copyright (c) 2026 Siyu Cai. SPDX-License-Identifier: MIT
from fractions import Fraction as F
from pathlib import Path
from uuid import uuid4
import unittest
import numpy as np
from pacd.codec import PrefixReader, encode
from pacd.semantic_acquisition import discover_semantic
from pacd.semantic_moments import MomentLimits, UNLIMITED
from pacd.skeleton import reference_skeleton

def gap_data(bits):
    h = np.array([[1 - 2 * ((r & c).bit_count() % 2) for c in (1, 2, 4, 8)] for r in range(16)], dtype=object)
    u = 3 * h[:, 0] + h[:, 1] + h[:, 2]
    v = 3 * h[:, 3]
    x, y = (u + v, u - v)
    sx, sy = (np.sign(10 * y - x), np.sign(10 * x - y))
    L = 1 << bits - 5
    return np.column_stack((16 * L + L * x - (sx < 0), 16 * L + L * y - (sy < 0)))

class SemanticAcquisitionTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.folder = Path(__file__).resolve().parents[1] / 'artifacts/semantic_moment_oracle' / ('test_' + uuid4().hex[:10])
        cls.folder.mkdir(parents=True)

    def run_reader(self, data, bits, **kwargs):
        path = self.folder / (uuid4().hex + '.bp')
        encode(path, np.array(data, dtype=np.int64), bits, len(data))
        with PrefixReader(path) as reader:
            return discover_semantic(reader, **kwargs)

    def test_exact_acquisition_on_small_whole_column_data(self):
        for data in ([[0, 0, 1], [1, 0, 0], [1, 1, 0]], [[0, 1, 0], [1, 0, 1], [0, 0, 0], [1, 1, 1]]):
            result = self.run_reader(data, 1, threshold=F(1, 2), limits=UNLIMITED)
            self.assertEqual(result['status'], 'certified')
            self.assertEqual(result['edges'], reference_skeleton(data, F(1, 2))[0])
            self.assertTrue(result['all_semantic_checks_completed'])
            self.assertLessEqual(result['queries'], len(data[0]))

    def test_gap_family_stops_by_fourteen_planes(self):
        for bits in (12, 16, 20, 28, 32):
            data = gap_data(bits)
            result = self.run_reader(data, bits)
            self.assertEqual(result['status'], 'certified')
            self.assertEqual(result['edges'], [(0, 1)])
            self.assertLessEqual(result['queries'], 14)
            self.assertEqual(result['wire_bytes'], 26 + 16 * result['queries'])

    def test_resource_return_and_action_limits(self):
        data = [[0, 0], [1, 1]]
        result = self.run_reader(data, 2, limits=MomentLimits(max_row_options=1), on_limit='return')
        self.assertEqual(result['status'], 'resource_limit')
        self.assertIsNone(result['edges'])
        self.assertEqual(result['queries'], 0)
        self.assertEqual(result['semantic_competitive_bound'], 'not_claimed')
        fallback = self.run_reader(data, 2, limits=MomentLimits(max_row_options=1))
        self.assertEqual(fallback['status'], 'certified')
        self.assertEqual(fallback['edges'], [(0, 1)])
        self.assertFalse(fallback['all_semantic_checks_completed'])
        self.assertEqual(fallback['semantic_competitive_bound'], 'not_claimed')
        limited = self.run_reader([[0, 0], [0, 1], [1, 0], [1, 1]], 1, max_actions=0)
        self.assertEqual(limited['status'], 'action_limit')
        self.assertIsNone(limited['edges'])
if __name__ == '__main__':
    unittest.main()
