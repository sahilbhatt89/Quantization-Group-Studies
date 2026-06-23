"""Custom 8-bit post-training quantization.

This module intentionally does not use TensorFlow Lite, TensorFlow Model
Optimization Toolkit, ONNX Runtime, or Hugging Face quantization helpers.
The goal is to keep the quantization math visible for the project study.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow import keras


INT8_QMIN = -127
INT8_QMAX = 127


@dataclass(frozen=True)
class QuantizedTensor:
    """INT8 tensor plus metadata needed to reconstruct an FP32 approximation."""

    name: str
    values: np.ndarray
    scale: float
    original_shape: tuple[int, ...]
    original_dtype: str


@dataclass(frozen=True)
class WeightQuantizationResult:
    """Custom weight-quantization report."""

    tensors: list[QuantizedTensor]
    fp32_size_bytes: int
    int8_size_bytes: int
    compression_ratio: float
    memory_reduction_percent: float


@dataclass(frozen=True)
class ActivationQuantizationResult:
    """Custom activation-quantization calibration report."""

    ranges: dict[str, tuple[float, float]]
    scales: dict[str, float]
    fp32_size_bytes: int
    int8_size_bytes: int
    compression_ratio: float
    memory_reduction_percent: float


def custom_ptq(
    model: keras.Model,
    representative_samples: list[np.ndarray] | None = None,
) -> dict[str, WeightQuantizationResult | ActivationQuantizationResult]:
    """Run the custom PTQ pipeline.

    Weight quantization can run without data because it only depends on stored
    model parameters. Activation quantization needs representative samples so
    intermediate layer ranges can be calibrated after training.
    """
    results: dict[str, WeightQuantizationResult | ActivationQuantizationResult] = {
        "weights": quantize_weights_symmetric_int8(model),
    }

    if representative_samples:
        results["activations"] = calibrate_activation_ranges(
            model,
            representative_samples,
        )

    return results


def quantize_weights_symmetric_int8(model: keras.Model) -> WeightQuantizationResult:
    """Quantize every floating-point Keras weight tensor to signed INT8.

    Symmetric quantization uses zero as the exact center point. For each tensor,
    the largest absolute FP32 value defines the scale:

        scale = max(abs(weight)) / 127

    Each FP32 value is then mapped to an INT8 value:

        q = round(weight / scale)

    The scale is stored because INT8 values alone are not enough to reconstruct
    an FP32 approximation.
    """
    quantized_tensors: list[QuantizedTensor] = []
    fp32_size_bytes = 0
    int8_size_bytes = 0

    for weight in model.weights:
        weight_array = weight.numpy()

        # Only floating-point tensors are quantized. Integer counters or other
        # non-floating tensors are copied into the size accounting unchanged.
        if not np.issubdtype(weight_array.dtype, np.floating):
            fp32_size_bytes += weight_array.nbytes
            int8_size_bytes += weight_array.nbytes
            continue

        quantized, scale = _quantize_array_symmetric_int8(weight_array)

        fp32_size_bytes += weight_array.nbytes
        int8_size_bytes += quantized.nbytes
        quantized_tensors.append(
            QuantizedTensor(
                name=weight.name,
                values=quantized,
                scale=scale,
                original_shape=tuple(weight_array.shape),
                original_dtype=str(weight_array.dtype),
            )
        )

    return WeightQuantizationResult(
        tensors=quantized_tensors,
        fp32_size_bytes=fp32_size_bytes,
        int8_size_bytes=int8_size_bytes,
        compression_ratio=fp32_size_bytes / int8_size_bytes,
        memory_reduction_percent=(1 - int8_size_bytes / fp32_size_bytes) * 100,
    )


def calibrate_activation_ranges(
    model: keras.Model,
    representative_samples: list[np.ndarray],
) -> ActivationQuantizationResult:
    """Estimate per-layer activation ranges for custom INT8 quantization.

    The function builds a temporary model that returns all layer outputs. It
    then runs representative samples through the pretrained model and records
    min/max activation values for each layer output. Those ranges define the
    scale that would be used to quantize each intermediate activation tensor.
    """
    activation_model, activation_names = _build_activation_model(model)
    ranges: dict[str, tuple[float, float]] = {}
    fp32_size_bytes = 0
    int8_size_bytes = 0

    for sample in representative_samples:
        outputs = activation_model(sample, training=False)
        if not isinstance(outputs, list):
            outputs = [outputs]

        for layer_name, output in zip(activation_names, outputs):
            output_array = output.numpy()

            # Some layers may return non-floating tensors. They are not part of
            # activation quantization, so skip them.
            if not np.issubdtype(output_array.dtype, np.floating):
                continue

            current_min = float(np.min(output_array))
            current_max = float(np.max(output_array))
            previous_min, previous_max = ranges.get(
                layer_name,
                (current_min, current_max),
            )
            ranges[layer_name] = (
                min(previous_min, current_min),
                max(previous_max, current_max),
            )

            fp32_size_bytes += output_array.size * np.dtype(np.float32).itemsize
            int8_size_bytes += output_array.size * np.dtype(np.int8).itemsize

    scales = {
        layer_name: _scale_from_min_max(min_value, max_value)
        for layer_name, (min_value, max_value) in ranges.items()
    }

    return ActivationQuantizationResult(
        ranges=ranges,
        scales=scales,
        fp32_size_bytes=fp32_size_bytes,
        int8_size_bytes=int8_size_bytes,
        compression_ratio=fp32_size_bytes / int8_size_bytes,
        memory_reduction_percent=(1 - int8_size_bytes / fp32_size_bytes) * 100,
    )


def dequantize_tensor(quantized_tensor: QuantizedTensor) -> np.ndarray:
    """Convert a custom INT8 tensor back to an FP32 approximation."""
    return quantized_tensor.values.astype(np.float32) * quantized_tensor.scale


def _quantize_array_symmetric_int8(array: np.ndarray) -> tuple[np.ndarray, float]:
    max_abs = float(np.max(np.abs(array)))
    scale = max_abs / INT8_QMAX if max_abs else 1.0

    # Clipping guarantees that rounding never creates a value outside the
    # signed INT8 range used by this custom quantizer.
    quantized = np.round(array / scale)
    quantized = np.clip(quantized, INT8_QMIN, INT8_QMAX).astype(np.int8)
    return quantized, scale


def _scale_from_min_max(min_value: float, max_value: float) -> float:
    max_abs = max(abs(min_value), abs(max_value))
    return max_abs / INT8_QMAX if max_abs else 1.0


def _build_activation_model(model: keras.Model) -> tuple[keras.Model, list[str]]:
    outputs = []
    output_names = []

    for layer in model.layers:
        # Nested models, such as the ResNet backbone, expose one output tensor
        # here. We keep the first implementation simple and measure the visible
        # layer outputs; deeper layer-wise hooks can be added later if needed.
        try:
            outputs.append(layer.output)
            output_names.append(layer.name)
        except AttributeError:
            continue

    activation_model = keras.Model(
        inputs=model.input,
        outputs=outputs,
        name=f"{model.name}_activation_calibrator",
    )
    return activation_model, output_names
