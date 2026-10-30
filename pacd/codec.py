# Copyright (c) 2026 Siyu Cai. SPDX-License-Identifier: MIT
from pathlib import Path
import struct
import numpy as np
MAGIC = b'PACDBP01'
HEADER = struct.Struct('<8sIHHH')
RECORD = struct.Struct('<IH')

def encode(path, data, bits=16, block_rows=64):
    raw = np.asarray(data)
    if raw.ndim != 2 or raw.dtype.kind not in 'iu' or raw.shape[0] < 2 or (raw.shape[1] < 2):
        raise ValueError('data must be an integer n-by-d matrix with n,d >= 2')
    if not 1 <= bits <= 32 or not 1 <= block_rows <= 65535:
        raise ValueError('Unsupported bit depth or block size')
    if np.any(raw < 0) or np.any(raw >= 2 ** bits):
        raise ValueError('Values must fit the public unsigned integer range')
    n, d = raw.shape
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as stream:
        stream.write(HEADER.pack(MAGIC, n, d, bits, block_rows))
        block_id = 0
        for col in range(d):
            for start in range(0, n, block_rows):
                values = raw[start:start + block_rows, col].astype(np.uint64)
                for plane in range(bits):
                    binary = (values >> bits - plane - 1 & 1).astype(np.uint8)
                    stream.write(RECORD.pack(block_id, plane))
                    stream.write(np.packbits(binary, bitorder='big').tobytes())
                block_id += 1
    return path

class PrefixReader:

    def __init__(self, path, request_bytes=8):
        if request_bytes < 0:
            raise ValueError('request_bytes must be nonnegative')
        self._stream = Path(path).open('rb')
        header = self._stream.read(HEADER.size)
        try:
            magic, self.n, self.d, self.bits, self.block_rows = HEADER.unpack(header)
            if magic != MAGIC or self.n < 2 or self.d < 2 or (not 1 <= self.bits <= 32) or (not self.block_rows):
                raise ValueError('Invalid bit-plane header')
        except Exception:
            self._stream.close()
            raise
        self.request_bytes = request_bytes
        self.read_bytes = HEADER.size
        self.requests = 1
        self.payload_bits = 0
        self.version = 0
        self.lower = np.zeros((self.n, self.d), dtype=object)
        self.upper = np.full((self.n, self.d), 2 ** self.bits - 1, dtype=object)
        self.blocks = []
        offset = HEADER.size
        for col in range(self.d):
            for start in range(0, self.n, self.block_rows):
                end = min(self.n, start + self.block_rows)
                size = RECORD.size + (end - start + 7) // 8
                self.blocks.append((col, start, end, offset, size))
                offset += self.bits * size
        self.file_bytes = offset
        self.depths = [0] * len(self.blocks)
        self.actions = []

    @property
    def wire_bytes(self):
        return self.read_bytes + self.requests * self.request_bytes

    @property
    def complete(self):
        return all((k == self.bits for k in self.depths))

    def available(self, columns=None):
        return [i for i, (col, *_rest) in enumerate(self.blocks) if self.depths[i] < self.bits and (columns is None or col in columns)]

    def action_cost(self, block_id):
        return self.blocks[block_id][-1] + self.request_bytes

    def width(self, block_id):
        return 2 ** (self.bits - self.depths[block_id]) - 1

    def read_next(self, block_id):
        plane = self.depths[block_id]
        if plane >= self.bits:
            raise ValueError('Block is already fully decoded')
        col, start, end, offset, size = self.blocks[block_id]
        self._stream.seek(offset + plane * size)
        record = self._stream.read(size)
        if len(record) != size or RECORD.unpack(record[:RECORD.size]) != (block_id, plane):
            raise ValueError('Truncated or corrupt plane record')
        binary = np.unpackbits(np.frombuffer(record[RECORD.size:], dtype=np.uint8), bitorder='big')[:end - start]
        remaining = self.bits - plane - 1
        self.lower[start:end, col] += binary.astype(object) * 2 ** remaining
        self.upper[start:end, col] = self.lower[start:end, col] + 2 ** remaining - 1
        self.depths[block_id] += 1
        self.payload_bits += end - start
        self.read_bytes += size
        self.requests += 1
        self.version += 1
        self.actions.append((block_id, plane + 1))

    def read_all(self):
        self._stream.seek(HEADER.size)
        body = self._stream.read(self.file_bytes - HEADER.size)
        if len(body) != self.file_bytes - HEADER.size:
            raise ValueError('Truncated file')
        for i, (col, start, end, offset, size) in enumerate(self.blocks):
            values = np.zeros(end - start, dtype=object)
            for plane in range(self.bits):
                position = offset - HEADER.size + plane * size
                record = body[position:position + size]
                if RECORD.unpack(record[:RECORD.size]) != (i, plane):
                    raise ValueError('Corrupt plane record')
                binary = np.unpackbits(np.frombuffer(record[RECORD.size:], dtype=np.uint8), bitorder='big')[:end - start]
                values += binary.astype(object) * 2 ** (self.bits - plane - 1)
            self.lower[start:end, col] = self.upper[start:end, col] = values
        self.payload_bits = self.n * self.d * self.bits
        self.depths = [self.bits] * len(self.blocks)
        self.read_bytes += len(body)
        self.requests += 1
        self.version += 1

    def close(self):
        self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
