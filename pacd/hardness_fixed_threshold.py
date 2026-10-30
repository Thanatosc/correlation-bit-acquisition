# Copyright (c) 2026 Siyu Cai. SPDX-License-Identifier: MIT
def squares(number):
    if number < 0:
        raise ValueError('Nonnegative square sum required')
    result = []
    for bit in range(number.bit_length()):
        if number >> bit & 1:
            result.extend([2 ** (bit // 2)] * (1 if bit % 2 == 0 else 2))
    assert sum((x * x for x in result)) == number
    return result
