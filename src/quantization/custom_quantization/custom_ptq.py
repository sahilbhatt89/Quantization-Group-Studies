"""Custom post-training quantization for configurable bit widths.

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
    """Quantized tensor plus metadata needed to reconstruct an FP32 approximation."""

    name: str
    values: np.ndarray
    scale: float | np.ndarray
    original_shape: tuple[int, ...]
    original_dtype: str
    quantization_axis: int | None = None
    num_bits: int = 8
    qmin: int = INT8_QMIN
    qmax: int = INT8_QMAX


@dataclass(frozen=True)
class WeightQuantizationResult:
    """Custom weight-quantization report."""

    tensors: list[QuantizedTensor]
    fp32_size_bytes: int
    quantized_size_bytes: int
    compression_ratio: float
    memory_reduction_percent: float
    num_bits: int = 8

    @property
    def int8_size_bytes(self) -> int:
        """Backward-compatible alias for earlier 8-bit-only reports."""
        return self.quantized_size_bytes


@dataclass(frozen=True)
class ActivationQuantizationResult:
    """Per-tensor affine parameters for integer activation tensors."""

    ranges: dict[str, tuple[float, float]]
    scales: dict[str, float]
    zero_points: dict[str, int]
    fp32_size_bytes: int
    int8_size_bytes: int
    compression_ratio: float
    memory_reduction_percent: float
    qmin: int = 0
    qmax: int = 255
    dtype: str = "uint8"


def custom_ptq(
    model: keras.Model,
    representative_samples: list[np.ndarray] | None = None,
    quantize_min_rank: int = 2,
    per_channel: bool = True,
) -> dict[str, WeightQuantizationResult | ActivationQuantizationResult]:
    """Run the custom PTQ pipeline.

    Weight quantization can run without data because it only depends on stored
    model parameters. Activation quantization needs representative samples so
    intermediate layer ranges can be calibrated after training.
    """
    results: dict[str, WeightQuantizationResult | ActivationQuantizationResult] = {
        "weights": quantize_weights_symmetric_int8(
            model,
            quantize_min_rank=quantize_min_rank,
            per_channel=per_channel,
        ),
    }

    if representative_samples:
        results["activations"] = calibrate_activation_ranges(
            model,
            representative_samples,
        )

    return results


def custom_ptq_n_bits(
    model: keras.Model,
    num_bits: int,
    representative_samples: list[np.ndarray] | None = None,
    quantize_min_rank: int = 2,
    per_channel: bool = True,
) -> dict[str, WeightQuantizationResult | ActivationQuantizationResult]:
    """Run the custom PTQ pipeline for a selected integer bit width.

    This function supports 1-bit through 8-bit symmetric signed weight
    quantization. Quantized values are still stored in NumPy int8 arrays because
    NumPy has no native 1-bit, 2-bit, or 3-bit signed integer dtype. Reported
    sizes use theoretical packed storage for the selected bit width.
    """
    results: dict[str, WeightQuantizationResult | ActivationQuantizationResult] = {
        "weights": quantize_weights_symmetric_n_bits(
            model,
            num_bits=num_bits,
            quantize_min_rank=quantize_min_rank,
            per_channel=per_channel,
        ),
    }

    if representative_samples:
        results["activations"] = calibrate_activation_ranges(
            model,
            representative_samples,
        )

    return results


def quantize_weights_symmetric_int8(
    model: keras.Model,
    quantize_min_rank: int = 2,
    per_channel: bool = True,
) -> WeightQuantizationResult:
    """Quantize kernel-like floating-point Keras weight tensors to signed INT8.

    Symmetric quantization uses zero as the exact center point. For each tensor
    or output channel, the largest absolute FP32 value defines the scale:

        scale = max(abs(weight)) / 127

    Each FP32 value is then mapped to an INT8 value:

        q = round(weight / scale)

    The scale is stored because INT8 values alone are not enough to reconstruct
    an FP32 approximation.

    By default, only tensors with rank 2 or higher are quantized. Rank-1
    tensors, such as biases and batch-normalization statistics, are kept in
    their original precision because quantizing them can change predictions
    disproportionately while saving little memory.

    By default, rank-2-or-higher tensors use per-output-channel scales on the
    last axis. This is closer to common TFLite PTQ behavior than one scale for
    the whole tensor.
    """
    return quantize_weights_symmetric_n_bits(
        model,
        num_bits=8,
        quantize_min_rank=quantize_min_rank,
        per_channel=per_channel,
    )


def quantize_weights_symmetric_n_bits(
    model: keras.Model,
    num_bits: int,
    quantize_min_rank: int = 2,
    per_channel: bool = True,
) -> WeightQuantizationResult:
    """Quantize kernel-like floating-point Keras weights to signed N-bit values."""
    qmin, qmax = _signed_symmetric_range(num_bits)
    quantized_tensors: list[QuantizedTensor] = []
    fp32_size_bytes = 0
    quantized_size_bytes = 0

    for weight in model.weights:
        weight_array = weight.numpy()

        # Integer tensors and sensitive rank-1 floating tensors are copied into
        # the size accounting unchanged.
        if (
            not np.issubdtype(weight_array.dtype, np.floating)
            or weight_array.ndim < quantize_min_rank
        ):
            fp32_size_bytes += weight_array.nbytes
            quantized_size_bytes += weight_array.nbytes
            continue

        quantized, scale, quantization_axis = _quantize_weight_array_symmetric_n_bits(
            weight_array,
            num_bits=num_bits,
            qmin=qmin,
            qmax=qmax,
            per_channel=per_channel,
        )

        fp32_size_bytes += weight_array.nbytes
        quantized_size_bytes += _packed_size_bytes(weight_array.size, num_bits)
        quantized_tensors.append(
            QuantizedTensor(
                name=weight.name,
                values=quantized,
                scale=scale,
                original_shape=tuple(weight_array.shape),
                original_dtype=str(weight_array.dtype),
                quantization_axis=quantization_axis,
                num_bits=num_bits,
                qmin=qmin,
                qmax=qmax,
            )
        )

    return WeightQuantizationResult(
        tensors=quantized_tensors,
        fp32_size_bytes=fp32_size_bytes,
        quantized_size_bytes=quantized_size_bytes,
        compression_ratio=fp32_size_bytes / quantized_size_bytes,
        memory_reduction_percent=(1 - quantized_size_bytes / fp32_size_bytes) * 100,
        num_bits=num_bits,
    )


def calibrate_activation_ranges(
    model: keras.Model,
    representative_samples: list[np.ndarray],
) -> ActivationQuantizationResult:
    """Calibrate per-layer UINT8 affine activation quantizers.

    The function builds a temporary model that returns all layer outputs. It
    then runs representative samples through the pretrained model and records
    min/max activation values for each layer output.  Each range is expanded to
    contain real zero exactly and mapped to ``[0, 255]`` using

        scale = (r_max - r_min) / 255
        zero_point = clip(round(-r_min / scale), 0, 255)

    This is calibration only: no fake-quantization nodes are inserted into the
    Keras model and no dequantized activations are fed to later layers.
    """
    if not representative_samples:
        raise ValueError("representative_samples must contain at least one sample.")

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

    params = {
        layer_name: _asymmetric_uint8_params(min_value, max_value)
        for layer_name, (min_value, max_value) in ranges.items()
    }
    scales = {layer_name: scale for layer_name, (scale, _) in params.items()}
    zero_points = {
        layer_name: zero_point for layer_name, (_, zero_point) in params.items()
    }

    if int8_size_bytes == 0:
        raise ValueError("Calibration did not observe any floating-point activations.")

    return ActivationQuantizationResult(
        ranges=ranges,
        scales=scales,
        zero_points=zero_points,
        fp32_size_bytes=fp32_size_bytes,
        int8_size_bytes=int8_size_bytes,
        compression_ratio=fp32_size_bytes / int8_size_bytes,
        memory_reduction_percent=(1 - int8_size_bytes / fp32_size_bytes) * 100,
    )


def dequantize_tensor(quantized_tensor: QuantizedTensor) -> np.ndarray:
    """Convert a custom INT8 tensor back to an FP32 approximation."""
    scale = quantized_tensor.scale
    if isinstance(scale, np.ndarray):
        scale_shape = [1] * quantized_tensor.values.ndim
        axis = quantized_tensor.quantization_axis
        if axis is None:
            raise ValueError("Per-channel scale requires quantization_axis.")
        scale_shape[axis] = scale.shape[0]
        scale = scale.reshape(scale_shape)
    return quantized_tensor.values.astype(np.float32) * scale


def _quantize_weight_array_symmetric_n_bits(
    array: np.ndarray,
    num_bits: int,
    qmin: int,
    qmax: int,
    per_channel: bool,
) -> tuple[np.ndarray, float | np.ndarray, int | None]:
    if per_channel and array.ndim >= 2:
        quantized, scale = _quantize_array_symmetric_n_bits_per_channel(
            array,
            num_bits=num_bits,
            qmin=qmin,
            qmax=qmax,
            axis=-1,
        )
        return quantized, scale, -1

    quantized, scale = _quantize_array_symmetric_n_bits(
        array,
        num_bits=num_bits,
        qmin=qmin,
        qmax=qmax,
    )
    return quantized, scale, None


def _quantize_array_symmetric_int8(array: np.ndarray) -> tuple[np.ndarray, float]:
    return _quantize_array_symmetric_n_bits(
        array,
        num_bits=8,
        qmin=INT8_QMIN,
        qmax=INT8_QMAX,
    )


def _quantize_array_symmetric_n_bits(
    array: np.ndarray,
    num_bits: int,
    qmin: int,
    qmax: int,
) -> tuple[np.ndarray, float]:
    max_abs = float(np.max(np.abs(array)))
    scale = max_abs / qmax if max_abs else 1.0

    if num_bits == 1:
        quantized = np.where(array >= 0, 1, -1).astype(np.int8)
        return quantized, scale

    # Clipping guarantees that rounding never creates a value outside the
    # signed integer range used by this custom quantizer.
    quantized = np.round(array / scale)
    quantized = np.clip(quantized, qmin, qmax).astype(np.int8)
    return quantized, scale


def _quantize_array_symmetric_n_bits_per_channel(
    array: np.ndarray,
    num_bits: int,
    qmin: int,
    qmax: int,
    axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    axis = axis % array.ndim
    reduce_axes = tuple(index for index in range(array.ndim) if index != axis)
    max_abs = np.max(np.abs(array), axis=reduce_axes)
    scale = np.where(max_abs > 0, max_abs / qmax, 1.0).astype(np.float32)

    scale_shape = [1] * array.ndim
    scale_shape[axis] = scale.shape[0]
    if num_bits == 1:
        quantized = np.where(array >= 0, 1, -1).astype(np.int8)
        return quantized, scale

    quantized = np.round(array / scale.reshape(scale_shape))
    quantized = np.clip(quantized, qmin, qmax).astype(np.int8)
    return quantized, scale


def _quantize_array_symmetric_int8_per_channel(
    array: np.ndarray,
    axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    return _quantize_array_symmetric_n_bits_per_channel(
        array,
        num_bits=8,
        qmin=INT8_QMIN,
        qmax=INT8_QMAX,
        axis=axis,
    )


def _signed_symmetric_range(num_bits: int) -> tuple[int, int]:
    if not 1 <= num_bits <= 8:
        raise ValueError(f"num_bits must be between 1 and 8, got {num_bits}.")
    if num_bits == 1:
        return -1, 1
    qmax = (2 ** (num_bits - 1)) - 1
    return -qmax, qmax


def _packed_size_bytes(num_values: int, num_bits: int) -> int:
    return (num_values * num_bits + 7) // 8


def _asymmetric_uint8_params(min_value: float, max_value: float) -> tuple[float, int]:
    range_min = min(float(min_value), 0.0)
    range_max = max(float(max_value), 0.0)
    if range_max == range_min:
        return 1.0, 0
    scale = (range_max - range_min) / 255
    zero_point = int(np.clip(np.rint(-range_min / scale), 0, 255))
    return float(scale), zero_point


def _build_activation_model(model: keras.Model) -> tuple[keras.Model, list[str]]:
    replayed = _build_sequential_activation_model(model)
    if replayed is not None:
        return replayed

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


def _build_sequential_activation_model(
    model: keras.Model,
) -> tuple[keras.Model, list[str]] | None:
    if len(model.inputs) != 1:
        return None

    input_shape = tuple(model.inputs[0].shape[1:])
    inputs = keras.Input(
        shape=input_shape,
        dtype=model.inputs[0].dtype,
        name=f"{model.name}_activation_input",
    )

    outputs = []
    output_names = []
    x = inputs

    for layer in model.layers:
        if isinstance(layer, keras.layers.InputLayer):
            continue

        try:
            if isinstance(layer, keras.layers.Dropout):
                x = layer(x, training=False)
            else:
                x = layer(x)
        except Exception:
            return None

        outputs.append(x)
        output_names.append(layer.name)

    if not outputs:
        return None

    activation_model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=f"{model.name}_activation_calibrator",
    )
    return activation_model, output_names
