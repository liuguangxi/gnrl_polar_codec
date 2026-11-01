"""
BER and BLER simulation for the Polar successive-cancellation list decoder
"""


import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polar_decoder import polar_decode_scl_core
from polar_encoder import polar_encode_core
from polar_sequence import VALID_N, make_frozen_mask


def parse_snr(value: str) -> np.ndarray:
    """Parse comma-separated SNR values or start:step:stop notation."""

    try:
        if ":" in value:
            start, step, stop = (float(item) for item in value.split(":"))
            if step <= 0 or stop < start:
                raise ValueError
            return np.arange(start, stop + step * 1e-9, step)

        snr = np.asarray([float(item) for item in value.split(",")], dtype=np.float64)
        if snr.size == 0:
            raise ValueError
        return snr
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "SNR must be comma-separated values or start:step:stop."
        ) from error


def simulate(
    K: int,
    N: int,
    L: int,
    vec_snr: np.ndarray,
    error_bit_threshold: int,
    total_bit_threshold: int,
    seed: int,
) -> None:
    """Simulate BPSK over AWGN and print BER and BLER per SNR."""

    if N not in VALID_N:
        raise ValueError(f"Unsupported code length {N}.")
    if not 1 <= K <= N:
        raise ValueError(f"K must be between 1 and {N}.")
    if L < 1:
        raise ValueError("List length must be positive.")
    if error_bit_threshold <= 0:
        raise ValueError("Error-bit threshold must be positive.")
    if total_bit_threshold < K:
        raise ValueError("Total-bit threshold must be at least K.")

    frozen = make_frozen_mask(K, N)
    rng = np.random.default_rng(seed)

    print(f"K = {K}  N = {N}  L = {L}  Rate = {K / N:.4f}")
    print(f"VecSnr (dB) = {', '.join(f'{snr:g}' for snr in vec_snr)}")
    print(f"Error-bit threshold = {error_bit_threshold}")
    print(f"Total-bit threshold = {total_bit_threshold}")
    print(f"Seed = {seed}")
    print()

    simulation_start = time.perf_counter()
    for snr in vec_snr:
        noise_variance = 10.0 ** (-snr / 10.0)
        noise_stddev = np.sqrt(noise_variance)
        bit_errors = 0
        total_bits = 0
        block_errors = 0
        total_blocks = 0

        while bit_errors < error_bit_threshold and total_bits < total_bit_threshold:
            tx_bits = rng.integers(0, 2, size=K, dtype=np.int32)
            encoded_bits = polar_encode_core(tx_bits, frozen)
            tx_symbols = 1.0 - 2.0 * encoded_bits
            rx_symbols = tx_symbols + rng.normal(0.0, noise_stddev, size=N)
            llr = 2.0 * rx_symbols / noise_variance
            rx_bits = polar_decode_scl_core(llr, frozen, L)

            errors = int(np.count_nonzero(tx_bits != rx_bits))
            bit_errors += errors
            total_bits += K
            block_errors += int(errors > 0)
            total_blocks += 1

        ber = bit_errors / total_bits
        bler = block_errors / total_blocks
        print(
            f"SNR (dB) = {snr:6.2f}  "
            f"BER = {ber:.10f}  ({bit_errors} / {total_bits})  "
            f"BLER = {bler:.10f}  ({block_errors} / {total_blocks})"
        )

    elapsed_seconds = time.perf_counter() - simulation_start
    print(f"\nElapsed time is {elapsed_seconds:.3f} seconds.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Polar SCL min-sum decoder BER/BLER simulation over BPSK AWGN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-k", "--K", "--k", dest="K", type=int, default=64, help="Information length"
    )
    parser.add_argument(
        "-n", "--N", "--n", dest="N", type=int, default=128, help="Codeword length"
    )
    parser.add_argument(
        "-l",
        "--L",
        "--list-length",
        dest="L",
        type=int,
        default=8,
        help="Maximum number of candidate paths retained by the SCL decoder",
    )
    parser.add_argument(
        "--snr",
        type=parse_snr,
        default=parse_snr("0:0.5:4"),
        metavar="VALUES",
        help="SNR in dB: start:step:stop or comma-separated values",
    )
    parser.add_argument(
        "--err-bits",
        dest="error_bit_threshold",
        type=int,
        default=1_000,
        help="Stop a SNR point after this many bit errors",
    )
    parser.add_argument(
        "--total-bits",
        dest="total_bit_threshold",
        type=int,
        default=1_000_000,
        help="Stop a SNR point after this many information bits",
    )
    parser.add_argument("--seed", type=int, default=0, help="NumPy random seed")
    args = parser.parse_args()

    simulate(
        args.K,
        args.N,
        args.L,
        args.snr,
        args.error_bit_threshold,
        args.total_bit_threshold,
        args.seed,
    )


if __name__ == "__main__":
    main()
