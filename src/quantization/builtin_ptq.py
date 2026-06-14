"""TensorFlow Lite post-training quantization utilities."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import tensorflow as tf
from tensorflow import keras


RepresentativeDataset = Callable[[], Iterable[list[np.ndarray]]]


def build_fixed_input_model(
    model: keras.Model,
    input_shape: tuple[int, int, int] = (224, 224, 3),
) -> keras.Model:
    """Wrap a model with a fixed input shape for TFLite conversion."""
    inputs = keras.Input(shape=input_shape, dtype=tf.float32, name="images")
    outputs = model(inputs, training=False)
    return keras.Model(inputs, outputs, name=f"{model.name}_fixed_input")


def convert_float(model: keras.Model) -> bytes:
    """Convert a Keras model to an unquantized float TFLite model."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    return converter.convert()


def convert_dynamic_range(model: keras.Model) -> bytes:
    """Quantize model weights to INT8 using dynamic-range PTQ."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    return converter.convert()


def convert_full_integer(
    model: keras.Model,
    representative_dataset: RepresentativeDataset,
) -> bytes:
    """Quantize weights and intermediate activations to INT8."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    return converter.convert()
