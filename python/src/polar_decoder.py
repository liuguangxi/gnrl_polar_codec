"""
Polar decoder
"""

import numpy as np
from numba import njit

from polar_sequence import VALID_N


@njit(cache=True)
def _polar_decode_sc_kernel(llr: np.ndarray, fro: np.ndarray) -> np.ndarray:
    N = len(llr)
    depth = 0
    size = N
    while size > 1:
        depth += 1
        size >>= 1
    alpha = np.empty((depth + 1, N), dtype=np.float64)
    beta = np.empty((depth + 1, N), dtype=np.int32)
    left_beta = np.empty((depth, N), dtype=np.int32)
    decoded_u = np.empty(N, dtype=np.int32)
    alpha[0] = llr

    for phi in range(N):
        for level in range(depth):
            size = N >> level
            half = size >> 1
            branch = (phi >> (depth - level - 1)) & 1

            if branch == 0:
                for i in range(half):
                    left = alpha[level, i]
                    right = alpha[level, half + i]
                    sign = 1.0
                    if left < 0:
                        sign = -sign
                    if right < 0:
                        sign = -sign
                    alpha[level + 1, i] = sign * min(abs(left), abs(right))
            else:
                lower_bits = depth - level - 1
                if phi & ((1 << lower_bits) - 1) == 0:
                    for i in range(half):
                        left_beta[level, i] = beta[level + 1, i]
                for i in range(half):
                    alpha[level + 1, i] = (
                        alpha[level, half + i]
                        + (1 - 2 * left_beta[level, i]) * alpha[level, i]
                    )

        if fro[phi] != 0:
            decoded_u[phi] = 0
        else:
            decoded_u[phi] = int(alpha[depth, 0] < 0)
        beta[depth, 0] = decoded_u[phi]

        for level in range(depth - 1, -1, -1):
            lower_bits = depth - level - 1
            lower_mask = (1 << lower_bits) - 1
            if phi & lower_mask != lower_mask:
                break

            half = N >> (level + 1)
            branch = (phi >> (depth - level - 1)) & 1

            if branch == 0:
                for i in range(half):
                    beta[level, i] = beta[level + 1, i]
                break

            for i in range(half):
                beta[level, i] = left_beta[level, i] ^ beta[level + 1, i]
                beta[level, half + i] = beta[level + 1, i]

    return decoded_u


def polar_decode_sc_core(llr: np.ndarray, fro: np.ndarray) -> np.ndarray:
    """Polar decode LLR data with SC and the min-sum approximation
    Input:
        llr: LLR data, positive value for bit 0, negative value for bit 1
        fro: frozen mask vector for input data, value 1 for frozen index

    Return:
        decoded information bits at non-frozen positions, value is 0 or 1
    """

    llr = np.ascontiguousarray(llr, dtype=np.float64)
    fro = np.ascontiguousarray(fro)

    # Check parameters
    N = len(llr)
    if N not in VALID_N:
        raise ValueError(f"Unsupported code length {N}.")
    if len(fro) != N:
        raise ValueError(f"Length of `fro` is not equal to {N}.")

    decoded_u = _polar_decode_sc_kernel(llr, fro)
    return decoded_u[fro == 0]


@njit(cache=True)
def _lazy_unique_slot(
    lazy_copy: np.ndarray, layer: int, path: int, path_count: int, L: int
) -> int:
    """Return a private storage slot for a path and layer without copying data."""

    slot = lazy_copy[layer, path]
    shared = False
    for other_path in range(path_count):
        if other_path != path and lazy_copy[layer, other_path] == slot:
            shared = True
            break

    if shared:
        for candidate_slot in range(L):
            occupied = False
            for other_path in range(path_count):
                if other_path != path and lazy_copy[layer, other_path] == candidate_slot:
                    occupied = True
                    break
            if not occupied:
                lazy_copy[layer, path] = candidate_slot
                return candidate_slot

    return slot


@njit(cache=True)
def _polar_decode_scl_kernel(llr: np.ndarray, fro: np.ndarray, L: int) -> np.ndarray:
    if L == 1:
        return _polar_decode_sc_kernel(llr, fro)

    N = len(llr)
    depth = 0
    size = N
    while size > 1:
        depth += 1
        size >>= 1

    alpha = np.zeros((depth + 1, L, N), dtype=np.float64)
    beta = np.zeros((depth + 1, L, N), dtype=np.int32)
    left_beta = np.zeros((depth, L, N), dtype=np.int32)
    lazy_copy = np.zeros((depth + 1, L), dtype=np.int32)
    lazy_copy_beta = np.zeros((depth + 1, L), dtype=np.int32)
    lazy_copy_left_beta = np.zeros((depth, L), dtype=np.int32)
    decoded_u = np.zeros((L, N), dtype=np.int32)
    next_lazy_copy = np.empty_like(lazy_copy)
    next_lazy_copy_beta = np.empty_like(lazy_copy_beta)
    next_lazy_copy_left_beta = np.empty_like(lazy_copy_left_beta)
    next_decoded_u = np.empty_like(decoded_u)
    path_metric = np.full(L, np.inf, dtype=np.float64)
    next_path_metric = np.empty_like(path_metric)
    path_metric[0] = 0.0
    path_count = 1
    alpha[0, 0] = llr

    candidate_metric = np.empty(2 * L, dtype=np.float64)
    candidate_path = np.empty(2 * L, dtype=np.int32)
    candidate_bit = np.empty(2 * L, dtype=np.int32)

    for phi in range(N):
        for path in range(path_count):
            for level in range(depth):
                size = N >> level
                half = size >> 1
                branch = (phi >> (depth - level - 1)) & 1

                if branch == 0:
                    input_slot = lazy_copy[level, path]
                    output_slot = _lazy_unique_slot(
                        lazy_copy, level + 1, path, path_count, L
                    )
                    for i in range(half):
                        left = alpha[level, input_slot, i]
                        right = alpha[level, input_slot, half + i]
                        sign = 1.0
                        if left < 0:
                            sign = -sign
                        if right < 0:
                            sign = -sign
                        alpha[level + 1, output_slot, i] = sign * min(abs(left), abs(right))
                else:
                    lower_bits = depth - level - 1
                    if phi & ((1 << lower_bits) - 1) == 0:
                        source_slot = lazy_copy_beta[level + 1, path]
                        output_slot = _lazy_unique_slot(
                            lazy_copy_left_beta, level, path, path_count, L
                        )
                        for i in range(half):
                            left_beta[level, output_slot, i] = beta[
                                level + 1, source_slot, i
                            ]
                    input_slot = lazy_copy[level, path]
                    output_slot = _lazy_unique_slot(
                        lazy_copy, level + 1, path, path_count, L
                    )
                    for i in range(half):
                        left_sum_slot = lazy_copy_left_beta[level, path]
                        alpha[level + 1, output_slot, i] = (
                            alpha[level, input_slot, half + i]
                            + (1 - 2 * left_beta[level, left_sum_slot, i])
                            * alpha[level, input_slot, i]
                        )

        candidate_count = 0
        for path in range(path_count):
            bit_llr = alpha[depth, lazy_copy[depth, path], 0]
            if fro[phi] != 0:
                bit = 0
                penalty = 0.0
                if bit_llr < 0:
                    penalty += abs(bit_llr)
                candidate_metric[candidate_count] = path_metric[path] + penalty
                candidate_path[candidate_count] = path
                candidate_bit[candidate_count] = bit
                candidate_count += 1
            else:
                for bit in range(2):
                    penalty = 0.0
                    favored_bit = 0
                    if bit_llr < 0:
                        favored_bit = 1
                    if bit != favored_bit:
                        penalty += abs(bit_llr)
                    candidate_metric[candidate_count] = path_metric[path] + penalty
                    candidate_path[candidate_count] = path
                    candidate_bit[candidate_count] = bit
                    candidate_count += 1

        order = np.argsort(candidate_metric[:candidate_count])
        kept = min(L, candidate_count)

        for new_path in range(kept):
            candidate = order[new_path]
            source = candidate_path[candidate]
            bit = candidate_bit[candidate]
            next_decoded_u[new_path] = decoded_u[source]
            next_path_metric[new_path] = candidate_metric[candidate]
            next_lazy_copy[:, new_path] = lazy_copy[:, source]
            next_lazy_copy_beta[:, new_path] = lazy_copy_beta[:, source]
            next_lazy_copy_left_beta[:, new_path] = lazy_copy_left_beta[:, source]
            next_decoded_u[new_path, phi] = bit

        for new_path in range(kept):
            candidate = order[new_path]
            bit = candidate_bit[candidate]
            beta_slot = _lazy_unique_slot(
                next_lazy_copy_beta, depth, new_path, kept, L
            )
            beta[depth, beta_slot, 0] = bit

            for level in range(depth - 1, -1, -1):
                lower_bits = depth - level - 1
                lower_mask = (1 << lower_bits) - 1
                if phi & lower_mask != lower_mask:
                    break

                half = N >> (level + 1)
                branch = (phi >> (depth - level - 1)) & 1
                output_slot = _lazy_unique_slot(
                    next_lazy_copy_beta, level, new_path, kept, L
                )
                child_slot = next_lazy_copy_beta[level + 1, new_path]
                if branch == 0:
                    for i in range(half):
                        beta[level, output_slot, i] = beta[
                            level + 1, child_slot, i
                        ]
                    break

                left_sum_slot = next_lazy_copy_left_beta[level, new_path]
                for i in range(half):
                    beta[level, output_slot, i] = (
                        left_beta[level, left_sum_slot, i]
                        ^ beta[level + 1, child_slot, i]
                    )
                    beta[level, output_slot, half + i] = beta[
                        level + 1, child_slot, i
                    ]

        lazy_copy, next_lazy_copy = next_lazy_copy, lazy_copy
        lazy_copy_beta, next_lazy_copy_beta = next_lazy_copy_beta, lazy_copy_beta
        lazy_copy_left_beta, next_lazy_copy_left_beta = (
            next_lazy_copy_left_beta,
            lazy_copy_left_beta,
        )
        decoded_u, next_decoded_u = next_decoded_u, decoded_u
        path_metric, next_path_metric = next_path_metric, path_metric
        path_count = kept

    best_path = 0
    for path in range(1, path_count):
        if path_metric[path] < path_metric[best_path]:
            best_path = path
    return decoded_u[best_path]


def polar_decode_scl_core(llr: np.ndarray, fro: np.ndarray, L: int) -> np.ndarray:
    """Polar decode LLR data with SCL and the min-sum approximation
    Input:
        llr: LLR data, positive value for bit 0, negative value for bit 1
        fro: frozen mask vector for input data, value 1 for frozen index
        L: maximum number of candidate paths retained

    Return:
        decoded information bits at non-frozen positions, value is 0 or 1
    """

    llr = np.ascontiguousarray(llr, dtype=np.float64)
    fro = np.ascontiguousarray(fro)

    N = len(llr)
    if N not in VALID_N:
        raise ValueError(f"Unsupported code length {N}.")
    if len(fro) != N:
        raise ValueError(f"Length of `fro` is not equal to {N}.")
    if not isinstance(L, (int, np.integer)):
        raise TypeError("List length `L` must be an integer.")
    if L < 1:
        raise ValueError("List length `L` must be positive.")

    information_bits = int(np.count_nonzero(fro == 0))
    max_paths = 1
    for _ in range(information_bits):
        if max_paths >= L:
            break
        max_paths = min(L, max_paths * 2)
    L = max_paths

    decoded_u = _polar_decode_scl_kernel(llr, fro, L)
    return decoded_u[fro == 0]
