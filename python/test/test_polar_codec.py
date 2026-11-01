"""
Polar encoder and successive-cancellation decoder tests
"""


import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polar_decoder import polar_decode_sc_core, polar_decode_scl_core
from polar_encoder import polar_encode_core
from polar_sequence import VALID_N, make_frozen_mask


class TestPolarCodec(unittest.TestCase):
    def test_sc_decodes_noiseless_codewords(self):
        rng = np.random.default_rng(7)

        parameters = [
            (N, K)
            for N in VALID_N
            for K in sorted({1, 2, N // 4, N // 2, 3 * N // 4, N - 1, N})
        ]

        for N, K in parameters:
            with self.subTest(N=N, K=K):
                frozen = make_frozen_mask(K, N)
                info = rng.integers(0, 2, size=K, dtype=np.int32)
                codeword = polar_encode_core(info, frozen)
                llr = 20.0 * (1 - 2 * codeword)
                decoded_info = polar_decode_sc_core(llr, frozen)

                self.assertEqual(len(decoded_info), K)
                np.testing.assert_array_equal(decoded_info, info)

    def test_scl_decodes_noiseless_codewords(self):
        rng = np.random.default_rng(11)
        for N in sorted(VALID_N):
            K = N // 2
            with self.subTest(N=N, K=K):
                frozen = make_frozen_mask(K, N)
                info = rng.integers(0, 2, size=K, dtype=np.int32)
                codeword = polar_encode_core(info, frozen)
                llr = 20.0 * (1 - 2 * codeword)
                decoded_info = polar_decode_scl_core(llr, frozen, L=4)
                np.testing.assert_array_equal(decoded_info, info)

    def test_scl_list_length_one_matches_sc(self):
        rng = np.random.default_rng(19)
        for N in sorted(VALID_N):
            frozen = make_frozen_mask(N // 2, N)
            llr = rng.normal(size=N)
            with self.subTest(N=N):
                np.testing.assert_array_equal(
                    polar_decode_scl_core(llr, frozen, L=1),
                    polar_decode_sc_core(llr, frozen),
                )

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            polar_decode_sc_core(np.zeros(16), np.ones(16))
        with self.assertRaises(ValueError):
            polar_decode_sc_core(np.zeros(32), np.ones(31))
        with self.assertRaises(ValueError):
            polar_decode_scl_core(np.zeros(16), np.ones(16), L=2)
        with self.assertRaises(ValueError):
            polar_decode_scl_core(np.zeros(32), np.ones(31), L=2)
        with self.assertRaises(ValueError):
            polar_decode_scl_core(np.zeros(32), np.ones(32), L=0)
        with self.assertRaises(TypeError):
            polar_decode_scl_core(np.zeros(32), np.ones(32), L=1.5)
        with self.assertRaises(ValueError):
            polar_encode_core(np.zeros(1), np.zeros(32))


if __name__ == "__main__":
    unittest.main()
