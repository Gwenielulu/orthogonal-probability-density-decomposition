import unittest

import numpy as np

from opd import decompose, reconstruct, transform


class TestNumpyOPD(unittest.TestCase):
    def test_rank_blocks_are_mapped_back_to_original_locations(self):
        field = np.array([3.0, 1.0, 4.0, 2.0])
        expected = np.array([3.5, 1.5, 3.5, 1.5])
        np.testing.assert_allclose(transform(field, level=1), expected)

    def test_transform_preserves_nd_shape(self):
        field = np.arange(8 * 8 * 8, dtype=float).reshape(8, 8, 8)
        self.assertEqual(transform(field, level=5).shape, field.shape)

    def test_multilevel_components_reconstruct_the_input(self):
        rng = np.random.default_rng(4)
        field = rng.normal(size=(16, 16))
        coarse, details = decompose(field, level=8)
        np.testing.assert_allclose(reconstruct(coarse, details), field, atol=1e-12)

    def test_multilevel_components_are_orthogonal(self):
        rng = np.random.default_rng(5)
        field = rng.normal(size=(16, 16))
        coarse, details = decompose(field, level=8)
        components = [coarse] + details
        for i, first in enumerate(components):
            for second in components[i + 1 :]:
                self.assertAlmostEqual(float(np.vdot(first, second)), 0.0, places=10)


class TestTorchOPD(unittest.TestCase):
    def test_torch_matches_numpy(self):
        try:
            import torch
            from opd.torch_opd import transform as torch_transform
        except ImportError:
            self.skipTest("PyTorch is not installed")

        rng = np.random.default_rng(6)
        field = rng.normal(size=(16, 16)).astype(np.float32)
        expected = transform(field, level=5)
        result = torch_transform(torch.from_numpy(field), level=5).numpy()
        np.testing.assert_allclose(result, expected, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

