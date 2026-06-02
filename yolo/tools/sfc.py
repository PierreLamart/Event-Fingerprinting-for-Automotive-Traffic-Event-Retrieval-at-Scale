#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools.sfc

The original repository used the third-party `zCurve` package to compute a
Morton / Z-order code by interleaving bits.

That dependency is not always available in notebook environments, so this
refactor includes a small, dependency-free implementation.

This module is *optional* for the main request (the required output is the 6-D
cell activations CSV), but keeping it makes the pipeline backwards compatible
with the original output format.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence


def _interleave_bits(values: Sequence[int]) -> int:
    """Interleave bits of non-negative integers (Morton code).

    Bits are interleaved from least-significant to most-significant.

    Example (2D)
    ------------
    x = 5  (0b0101)
    y = 3  (0b0011)
    morton = 0b00011011

    Notes
    -----
    This implementation is generic for any dimension count.
    """

    if not values:
        return 0

    if any(v < 0 for v in values):
        raise ValueError("Morton interleave expects non-negative integers")

    n_dims = len(values)
    max_bits = max(int(v).bit_length() for v in values)
    code = 0

    for bit in range(max_bits):
        for dim in range(n_dims):
            code |= ((int(values[dim]) >> bit) & 1) << (bit * n_dims + dim)

    return code


def calculate_morton(values: Iterable[float]) -> int:
    """Compute a Morton code for a vector of floats.

    The original code:
        1. capped floats to one decimal,
        2. multiplied by 10,
        3. converted to int,
        4. used zCurve.interlace(...).

    We preserve that behaviour so existing downstream code keeps working.

    Parameters
    ----------
    values:
        Iterable of numbers (length is typically 6).

    Returns
    -------
    int
        Morton code (bit-interleaved integer).
    """

    int_values: List[int] = [int(round(float(v), 1) * 10) for v in values]
    return _interleave_bits(int_values)
