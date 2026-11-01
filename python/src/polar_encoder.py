"""
Polar encoder
"""


import numpy as np
from numba import njit

from polar_sequence import VALID_N


@njit(cache=True)
def _polar_encode_kernel(c: np.ndarray, fro: np.ndarray) -> np.ndarray:
    N = len(fro)
    u = np.zeros(N, dtype=np.int32)
    info_index = 0
    for i in range(N):
        if fro[i] == 0:
            u[i] = c[info_index]
            info_index += 1

    d = u.copy()
    stage = 1
    while stage < N:
        for i in range(0, N, 2 * stage):
            for j in range(stage):
                d[i + j] ^= d[i + stage + j]
        stage <<= 1
    return d


def polar_encode_core(c: np.ndarray, fro: np.ndarray) -> np.ndarray:
    """Polar encoder for input information bits
    Input:
        c: information data bits, value is 0 or 1
        fro: frozen bit mask, value 1 for frozen index

    Return:
        encoded data bits, value is 0 or 1
    """

    c = np.ascontiguousarray(c, dtype=np.int32)
    fro = np.ascontiguousarray(fro)

    # Check parameters
    N = len(fro)
    if N not in VALID_N:
        raise ValueError(f"Unsupported code length {N}.")
    K = len(c)
    if not 1 <= K <= N:
        raise ValueError(f"K must be between 1 and {N}.")
    if np.count_nonzero(fro == 0) != K:
        raise ValueError("Length of `c` must match the number of information positions in `fro`.")

    return _polar_encode_kernel(c, fro)
