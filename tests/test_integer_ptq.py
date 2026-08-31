"""Tests for the explicit integer PTQ mathematics."""

from __future__ import annotations

import unittest

import numpy as np
from tensorflow import keras

from src.quantization.custom_quantization import (
    custom_ptq,
    quantize_weights_symmetric_int8,
)
from src.quantization.custom_quantization.advanced_ptq import (
    QuantizationParams,
    asymmetric_quantization_params,
    integer_conv2d_nhwc,
    integer_dense,
    requantize_int32,
)


class IntegerPTQTest(unittest.TestCase):
    def test_custom_ptq_can_quantize_one_specific_kernel(self) -> None:
        model = keras.Sequential(
            [
                keras.Input(shape=(3,)),
                keras.layers.Dense(4, name="first"),
                keras.layers.Dense(2, name="second"),
            ]
        )
        selected = model.layers[1].kernel
        selected_name = getattr(selected, "path", selected.name)

        result = quantize_weights_symmetric_int8(
            model, included_tensor_names={selected_name}
        )

        self.assertEqual(len(result.tensors), 1)
        self.assertEqual(result.tensors[0].name, selected_name)
        self.assertEqual(result.tensors[0].original_shape, tuple(selected.shape))

    def test_fixed_point_requantization_does_not_change_unit_scale(self) -> None:
        params = asymmetric_quantization_params(-128, 127, dtype="int8")
        accumulator = np.array([-128, -3, 0, 7, 127], dtype=np.int32)

        actual = requantize_int32(accumulator, params.scale, params)

        np.testing.assert_array_equal(actual.values, accumulator.astype(np.int8))

    def test_integer_dense_uses_per_output_channel_weight_scales(self) -> None:
        input_params = QuantizationParams(
            scale=0.5,
            zero_point=0,
            qmin=-127,
            qmax=127,
            num_bits=8,
            dtype="int8",
            symmetric=True,
        )
        weight_params = QuantizationParams(
            scale=np.array([0.25, 0.5], dtype=np.float32),
            zero_point=0,
            qmin=-127,
            qmax=127,
            num_bits=8,
            dtype="int8",
            symmetric=True,
        )
        output_params = QuantizationParams(
            scale=0.125,
            zero_point=0,
            qmin=-128,
            qmax=127,
            num_bits=8,
            dtype="int8",
            symmetric=False,
        )

        result = integer_dense(
            inputs=np.array([[1.0, -1.0]], dtype=np.float32),
            weights=np.array([[0.5, 1.0], [-0.25, 0.5]], dtype=np.float32),
            bias=np.array([0.125, -0.25], dtype=np.float32),
            input_params=input_params,
            weight_params=weight_params,
            output_params=output_params,
        )

        np.testing.assert_array_equal(result.accumulator_int32, [[7, 1]])
        np.testing.assert_array_equal(result.output_int, [[7, 2]])
        self.assertEqual(result.accumulator_int32.dtype, np.int32)
        self.assertEqual(result.output_int.dtype, np.int8)

    def test_custom_ptq_calibrates_asymmetric_uint8_activations(self) -> None:
        model = keras.Sequential(
            [
                keras.Input(shape=(2,)),
                keras.layers.Dense(
                    1,
                    use_bias=False,
                    name="dense",
                    kernel_initializer=keras.initializers.Constant([[1.0], [1.0]]),
                ),
            ]
        )

        result = custom_ptq(
            model,
            representative_samples=[
                np.array([[-2.0, 0.0]], dtype=np.float32),
                np.array([[1.0, 2.0]], dtype=np.float32),
            ],
        )
        activations = result["activations"]

        self.assertEqual(activations.ranges["dense"], (-2.0, 3.0))
        self.assertAlmostEqual(activations.scales["dense"], 5.0 / 255.0)
        self.assertEqual(activations.zero_points["dense"], 102)
        self.assertEqual(activations.dtype, "uint8")

    def test_integer_conv2d_keeps_accumulation_and_output_integer(self) -> None:
        input_params = QuantizationParams(
            scale=0.5,
            zero_point=0,
            qmin=-127,
            qmax=127,
            num_bits=8,
            dtype="int8",
            symmetric=True,
        )
        weight_params = QuantizationParams(
            scale=np.array([0.25, 0.25], dtype=np.float32),
            zero_point=0,
            qmin=-127,
            qmax=127,
            num_bits=8,
            dtype="int8",
            symmetric=True,
        )
        output_params = QuantizationParams(
            scale=0.125,
            zero_point=0,
            qmin=-128,
            qmax=127,
            num_bits=8,
            dtype="int8",
            symmetric=False,
        )

        result = integer_conv2d_nhwc(
            inputs=np.array([[[[1.0], [-1.0]], [[0.5], [0.0]]]], dtype=np.float32),
            kernels=np.array([[[[0.5, -0.25]]]], dtype=np.float32),
            bias=np.array([0.0, 0.125], dtype=np.float32),
            input_params=input_params,
            weight_params=weight_params,
            output_params=output_params,
        )

        expected = np.array([[[[4, -1], [-4, 3]], [[2, 0], [0, 1]]]])
        np.testing.assert_array_equal(result.accumulator_int32, expected)
        np.testing.assert_array_equal(result.output_int, expected.astype(np.int8))


if __name__ == "__main__":
    unittest.main()
