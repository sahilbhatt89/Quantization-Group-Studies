"""Tests for the custom post-training weight quantizer."""

from __future__ import annotations

import unittest

import numpy as np
from tensorflow import keras

from src.quantization.custom_quantization import (
    custom_ptq,
    quantize_weights_symmetric_int8,
)


class CustomPTQTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
