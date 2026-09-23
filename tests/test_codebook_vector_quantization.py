"""Tests for mathematical codebook and product-vector quantization."""

from __future__ import annotations

import unittest

import numpy as np
from tensorflow import keras

from src.quantization.custom_quantization import (
    build_universal_codebook_kde,
    codebook_vectorize_model,
    codebook_vectorize_model_n_bits,
    compress_product_codebooks,
    estimate_gradient_importance,
    fit_importance_weighted_kmeans_codebook,
    fit_kmeans_codebook,
    nearest_codeword_indices,
    nearest_codeword_indices_weighted,
    product_quantize_array,
    quantization_mse,
    reconstruct_codebook_model_weights,
    reconstruct_codebook_tensor,
    reconstruct_pooled_codebook_tensor,
    vector_quantize_array,
)


class CodebookVectorQuantizationTest(unittest.TestCase):
    def test_vector_quantization_uses_integer_assignments_and_lookup(self) -> None:
        values = np.array(
            [[0.0, 0.1, 10.0, 10.1], [0.1, 0.0, 9.9, 10.0]],
            dtype=np.float32,
        )

        result = vector_quantize_array(
            values,
            vector_dim=2,
            codebook_size=2,
            kmeans_iterations=20,
            seed=3,
        )
        reconstructed = reconstruct_codebook_tensor(result)

        self.assertEqual(result.assignments.dtype, np.uint8)
        self.assertEqual(result.assignments.shape, (2, 2))
        self.assertEqual(result.codebooks.shape, (2, 2))
        self.assertEqual(reconstructed.shape, values.shape)
        self.assertLess(quantization_mse(values, result), 0.01)

    def test_product_quantization_builds_one_codebook_per_subspace(self) -> None:
        values = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [0.2, 0.1, 9.8, 10.1],
                [5.0, 5.0, -4.0, -4.0],
            ],
            dtype=np.float32,
        )

        result = product_quantize_array(
            values,
            vector_dim=2,
            codebook_size=2,
            codebook_dtype=np.float32,
            kmeans_iterations=20,
            seed=5,
        )

        self.assertEqual(result.mode, "product")
        self.assertEqual(result.codebooks.shape, (2, 2, 2))
        self.assertEqual(result.assignments.shape, (3, 2))
        self.assertEqual(reconstruct_codebook_tensor(result).shape, values.shape)

    def test_non_divisible_axis_is_padded_and_restored(self) -> None:
        values = np.arange(10, dtype=np.float32).reshape(2, 5)
        result = vector_quantize_array(
            values,
            vector_dim=3,
            codebook_size=4,
            kmeans_iterations=5,
        )

        self.assertEqual(result.layout.original_width, 5)
        self.assertEqual(result.layout.padded_width, 6)
        self.assertEqual(reconstruct_codebook_tensor(result).shape, (2, 5))

    def test_universal_kde_codebook_is_balanced_and_deterministic(self) -> None:
        first = np.zeros((20, 4), dtype=np.float32)
        second = np.full((2, 4), 10.0, dtype=np.float32)

        one = build_universal_codebook_kde(
            [first, second],
            vector_dim=2,
            codebook_size=32,
            samples_per_array=20,
            bandwidth=0.0,
            seed=11,
        )
        two = build_universal_codebook_kde(
            [first, second],
            vector_dim=2,
            codebook_size=32,
            samples_per_array=20,
            bandwidth=0.0,
            seed=11,
        )

        np.testing.assert_array_equal(one, two)
        self.assertEqual(one.shape, (32, 2))
        self.assertTrue(np.all(np.logical_or(one == 0.0, one == 10.0)))
        self.assertTrue(np.any(one == 0.0))
        self.assertTrue(np.any(one == 10.0))

    def test_frequency_distance_pool_has_requested_size(self) -> None:
        values = np.array(
            [
                [0.0, 0.1, 5.0, 5.1],
                [0.1, 0.0, 5.1, 5.0],
                [1.0, 1.1, 6.0, 6.1],
                [1.1, 1.0, 6.1, 6.0],
            ],
            dtype=np.float32,
        )
        product = product_quantize_array(
            values,
            vector_dim=2,
            codebook_size=3,
            codebook_dtype=np.float32,
            kmeans_iterations=10,
        )
        pooled = compress_product_codebooks(
            product,
            target_pool_size=2,
            distance_threshold=0.05,
        )

        self.assertEqual(pooled.pool.shape, (2, 2))
        self.assertEqual(pooled.local_to_pool.shape, (2, 3))
        self.assertEqual(reconstruct_pooled_codebook_tensor(pooled).shape, values.shape)

    def test_model_function_quantizes_rank_two_and_skips_bias(self) -> None:
        model = keras.Sequential(
            [keras.Input(shape=(4,)), keras.layers.Dense(4, use_bias=True)]
        )
        result = codebook_vectorize_model(
            model,
            vector_dim=2,
            codebook_size=2,
            mode="product",
            kmeans_iterations=5,
        )

        self.assertEqual(len(result.tensors), 1)
        self.assertEqual(len(result.skipped_tensor_names), 1)
        self.assertIn("bias", result.skipped_tensor_names[0])
        self.assertLess(
            result.estimated_compressed_size_bytes,
            result.original_size_bytes,
        )
        reconstructed = reconstruct_codebook_model_weights(model, result)
        self.assertEqual(len(reconstructed), len(model.weights))
        self.assertEqual(reconstructed[0].shape, model.weights[0].shape)
        np.testing.assert_array_equal(reconstructed[1], model.weights[1].numpy())

    def test_model_function_can_preserve_selected_kernel(self) -> None:
        model = keras.Sequential(
            [keras.Input(shape=(4,)), keras.layers.Dense(4, use_bias=True)]
        )
        kernel_name = getattr(model.weights[0], "path", model.weights[0].name)
        result = codebook_vectorize_model(
            model,
            vector_dim=2,
            codebook_size=2,
            mode="vector",
            excluded_tensor_names={kernel_name},
            kmeans_iterations=5,
        )

        self.assertEqual(len(result.tensors), 0)
        self.assertIn(kernel_name, result.skipped_tensor_names)
        self.assertEqual(
            result.estimated_compressed_size_bytes,
            result.original_size_bytes,
        )

    def test_n_bit_model_function_derives_codebook_size(self) -> None:
        model = keras.Sequential(
            [keras.Input(shape=(8,)), keras.layers.Dense(8, use_bias=False)]
        )
        result = codebook_vectorize_model_n_bits(
            model,
            num_bits=3,
            vector_dim=2,
            mode="vector",
            kmeans_iterations=5,
        )

        self.assertEqual(len(result.tensors), 1)
        self.assertEqual(result.tensors[0].assignment_bits, 3)
        self.assertEqual(result.tensors[0].codebooks.shape, (8, 2))

    def test_model_codebook_can_quantize_one_specific_kernel(self) -> None:
        model = keras.Sequential(
            [
                keras.Input(shape=(3,)),
                keras.layers.Dense(4, name="first"),
                keras.layers.Dense(2, name="second"),
            ]
        )
        selected = model.layers[1].kernel
        selected_name = getattr(selected, "path", selected.name)

        result = codebook_vectorize_model(
            model,
            vector_dim=1,
            codebook_size=4,
            mode="vector",
            included_tensor_names={selected_name},
            kmeans_iterations=5,
        )

        self.assertEqual(len(result.tensors), 1)
        self.assertEqual(result.tensors[0].name, selected_name)

    def test_importance_weighted_kmeans_reduces_weighted_distortion(self) -> None:
        values = np.array([[0.0], [2.0], [3.0], [10.0]], dtype=np.float32)
        importance = np.array([[100.0], [1.0], [1.0], [1.0]], dtype=np.float32)
        ordinary = fit_kmeans_codebook(values, 2, iterations=20, seed=7)
        weighted = fit_importance_weighted_kmeans_codebook(
            values, importance, 2, iterations=20, seed=7
        )
        ordinary_assignment = nearest_codeword_indices(values, ordinary)
        weighted_assignment = nearest_codeword_indices_weighted(
            values, weighted, importance
        )
        ordinary_error = np.sum(
            importance * (values - ordinary[ordinary_assignment]) ** 2
        )
        weighted_error = np.sum(
            importance * (values - weighted[weighted_assignment]) ** 2
        )

        self.assertLessEqual(weighted_error, ordinary_error + 1e-6)

    def test_scalar_nearest_assignment_matches_explicit_distances(self) -> None:
        values = np.array([[-2.0], [-0.5], [0.0], [0.5], [3.0]], dtype=np.float32)
        centers = np.array([[1.0], [-1.0], [1.0], [4.0]], dtype=np.float32)
        expected = np.argmin((values - centers.T) ** 2, axis=1)

        actual = nearest_codeword_indices(values, centers)

        np.testing.assert_array_equal(actual, expected)

    def test_gradient_importance_includes_frozen_weight_shapes(self) -> None:
        model = keras.Sequential(
            [keras.Input(shape=(2,)), keras.layers.Dense(2)]
        )
        model.layers[0].trainable = False
        inputs = np.array([[1.0, -1.0], [0.5, 0.25]], dtype=np.float32)
        labels = np.array([0, 1], dtype=np.int32)
        kernel = model.layers[0].kernel
        kernel_name = getattr(kernel, "path", kernel.name)
        importance = estimate_gradient_importance(
            model,
            [(inputs, labels)],
            included_tensor_names={kernel_name},
        )

        self.assertFalse(model.trainable_weights)
        self.assertEqual(set(importance), {kernel_name})
        self.assertEqual(importance[kernel_name].shape, tuple(kernel.shape))
        self.assertTrue(np.all(importance[kernel_name] > 0))


if __name__ == "__main__":
    unittest.main()
