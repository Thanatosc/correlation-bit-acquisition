# Copyright (c) 2026 Siyu Cai. SPDX-License-Identifier: MIT
import numpy as np

class MemoryReader:

    def __init__(self, data, bits, block_rows, costs=None):
        self._data = np.asarray(data, dtype=object)
        self.n, self.d = self._data.shape
        self.bits = bits
        self.blocks = [(c, s, min(self.n, s + block_rows), 0, 1) for c in range(self.d) for s in range(0, self.n, block_rows)]
        self._costs = costs or [1] * len(self.blocks)
        self.depths = [0] * len(self.blocks)
        self.lower = np.zeros(self._data.shape, dtype=object)
        self.upper = np.full(self._data.shape, 2 ** bits - 1, dtype=object)
        self.payload_bits = self.read_bytes = self.wire_bytes = self.requests = 0
        self.actions = []
        self.file_bytes = bits * sum(self._costs)

    def action_cost(self, b):
        return self._costs[b]

    def available(self):
        return [b for b, k in enumerate(self.depths) if k < self.bits]

    @property
    def complete(self):
        return not self.available()

    def read_next(self, b):
        self.depths[b] += 1
        c, s, e, *_ = self.blocks[b]
        step = 1 << self.bits - self.depths[b]
        self.lower[s:e, c] = self._data[s:e, c] // step * step
        self.upper[s:e, c] = self.lower[s:e, c] + step - 1
        self.payload_bits += e - s
        self.read_bytes += self._costs[b]
        self.wire_bytes += self._costs[b]
        self.requests += 1
        self.actions.append((b, self.depths[b]))
