"""Experimental PTQ building blocks beyond the simple custom weight quantizer.

This module keeps the math explicit for study purposes. It is intentionally a
reference implementation, not an optimized runtime. The functions here cover:

- asymmetric activation quantization with a zero-point
- symmetric weight quantization
- MSE-based range search
- small integer Dense and Conv2D inference references with INT32 accumulation

The implementation follows the affine mapping and integer-only requantization
in Jacob et al. (arXiv:1712.05877), the per-channel PTQ recommendations in
Krishnamoorthi (arXiv:1806.08342), and the W8A8 pipeline summarized by Nagel et
al. (arXiv:2106.08295).  Floating point is used to choose calibration constants
offline; the layer MAC, bias addition, and requantization path is integer-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from tensorflow import keras


QuantizedDType = Literal["int8", "uint8"]
Padding = Literal["valid", "same"]


@dataclass(frozen=True)
class QuantizationParams:
    """Scale/zero-point parameters for affine quantization."""

    scale: float | np.ndarray
    zero_point: int | np.ndarray
    qmin: int
    qmax: int
    num_bits: int
    dtype: QuantizedDType
    symmetric: bool


@dataclass(frozen=True)
class QuantizedArray:
    """Quantized values plus the parameters needed to dequantize them."""

    values: np.ndarray
    params: QuantizationParams


@dataclass(frozen=True)
class IntegerLayerResult:
    """Reference integer layer result."""

    accumulator_int32: np.ndarray
    output_int: np.ndarray
    output_dequantized: np.ndarray
    input_params: QuantizationParams
    weight_params: QuantizationParams
    output_params: QuantizationParams


def asymmetric_quantization_params(
    min_value: float,
    max_value: float,
    num_bits: int = 8,
    dtype: QuantizedDType = "uint8",
) -> QuantizationParams:
    """Create asymmetric affine quantization params with a zero-point."""
    qmin, qmax = _integer_range(num_bits, dtype)
    min_value = float(min_value)
    max_value = float(max_value)

    if max_value < min_value:
        raise ValueError("max_value must be greater than or equal to min_value.")

    # Exact real zero must be representable for zero padding and ReLU.  Relax
    # one-sided observed ranges to contain it, as prescribed by the affine
    # quantizers in all three references.
    min_value = min(min_value, 0.0)
    max_value = max(max_value, 0.0)
    if max_value == min_value:
        return QuantizationParams(
            scale=1.0,
            zero_point=0 if dtype == "int8" else qmin,
            qmin=qmin,
            qmax=qmax,
            num_bits=num_bits,
            dtype=dtype,
            symmetric=False,
        )

    scale = (max_value - min_value) / (qmax - qmin)
    zero_point = int(round(qmin - min_value / scale))
    zero_point = int(np.clip(zero_point, qmin, qmax))

    return QuantizationParams(
        scale=float(scale),
        zero_point=zero_point,
        qmin=qmin,
        qmax=qmax,
        num_bits=num_bits,
        dtype=dtype,
        symmetric=False,
    )


def symmetric_quantization_params(
    max_abs: float | np.ndarray,
    num_bits: int = 8,
    dtype: QuantizedDType = "int8",
) -> QuantizationParams:
    """Create signed symmetric quantization params with zero-point 0."""
    if dtype != "int8":
        raise ValueError("Symmetric weights are expected to use dtype='int8'.")

    qmin, qmax = _signed_symmetric_range(num_bits)
    max_abs_array = np.asarray(max_abs, dtype=np.float64)
    scale_array = np.where(max_abs_array > 0, max_abs_array / qmax, 1.0)
    scale: float | np.ndarray
    if scale_array.ndim == 0:
        scale = float(scale_array)
    else:
        scale = scale_array.astype(np.float32)

    return QuantizationParams(
        scale=scale,
        zero_point=0,
        qmin=qmin,
        qmax=qmax,
        num_bits=num_bits,
        dtype=dtype,
        symmetric=True,
    )


def quantize_tensor(array: np.ndarray, params: QuantizationParams) -> QuantizedArray:
    """Quantize a tensor using affine quantization parameters."""
    quantized = np.round(array / params.scale + params.zero_point)
    quantized = np.clip(quantized, params.qmin, params.qmax)
    return QuantizedArray(
        values=quantized.astype(_numpy_dtype(params.dtype)),
        params=params,
    )


def dequantize_tensor(quantized: QuantizedArray) -> np.ndarray:
    """Reconstruct an FP32 approximation from quantized values."""
    params = quantized.params
    return (
        quantized.values.astype(np.float32) - params.zero_point
    ) * params.scale


def mse_optimal_symmetric_range(
    array: np.ndarray,
    num_bits: int = 8,
    num_candidates: int = 100,
    min_clip_ratio: float = 0.5,
) -> tuple[float, float]:
    """Find a symmetric clipping range that minimizes reconstruction MSE."""
    max_abs = float(np.max(np.abs(array)))
    if max_abs == 0:
        return 1.0, 0.0

    best_clip = max_abs
    best_error = float("inf")

    for clip_abs in np.linspace(max_abs * min_clip_ratio, max_abs, num_candidates):
        params = symmetric_quantization_params(clip_abs, num_bits=num_bits)
        clipped = np.clip(array, -clip_abs, clip_abs)
        reconstructed = dequantize_tensor(quantize_tensor(clipped, params))
        error = float(np.mean((array - reconstructed) ** 2))
        if error < best_error:
            best_clip = float(clip_abs)
            best_error = error

    return best_clip, best_error


def mse_optimal_asymmetric_range(
    array: np.ndarray,
    num_bits: int = 8,
    dtype: QuantizedDType = "uint8",
    num_candidates: int = 80,
    max_percentile: float = 99.999,
) -> tuple[float, float, float]:
    """Find an asymmetric activation range that minimizes reconstruction MSE."""
    if array.size == 0:
        return 0.0, 1.0, 0.0

    lower_bound = float(np.min(array))
    upper_bound = float(np.percentile(array, max_percentile))
    if upper_bound <= lower_bound:
        upper_bound = float(np.max(array))
    if upper_bound <= lower_bound:
        return lower_bound, lower_bound + 1.0, 0.0

    best_min = lower_bound
    best_max = upper_bound
    best_error = float("inf")

    candidate_max_values = np.linspace(upper_bound, float(np.max(array)), num_candidates)
    for candidate_max in candidate_max_values:
        params = asymmetric_quantization_params(
            lower_bound,
            candidate_max,
            num_bits=num_bits,
            dtype=dtype,
        )
        clipped = np.clip(array, lower_bound, candidate_max)
        reconstructed = dequantize_tensor(quantize_tensor(clipped, params))
        error = float(np.mean((array - reconstructed) ** 2))
        if error < best_error:
            best_max = float(candidate_max)
            best_error = error

    return best_min, best_max, best_error


def calibrate_activation_ranges_mse(
    model: keras.Model,
    representative_samples: list[np.ndarray],
    num_bits: int = 8,
    dtype: QuantizedDType = "uint8",
    num_candidates: int = 80,
) -> dict[str, QuantizationParams]:
    """Calibrate activation params with MSE-based asymmetric ranges."""
    activation_model, activation_names = _build_activation_model(model)
    activations: dict[str, list[np.ndarray]] = {name: [] for name in activation_names}

    for sample in representative_samples:
        outputs = activation_model(sample, training=False)
        if not isinstance(outputs, list):
            outputs = [outputs]

        for name, output in zip(activation_names, outputs):
            output_array = output.numpy()
            if np.issubdtype(output_array.dtype, np.floating):
                activations[name].append(output_array.reshape(-1))

    params_by_layer: dict[str, QuantizationParams] = {}
    for name, chunks in activations.items():
        if not chunks:
            continue
        values = np.concatenate(chunks)
        min_value, max_value, _ = mse_optimal_asymmetric_range(
            values,
            num_bits=num_bits,
            dtype=dtype,
            num_candidates=num_candidates,
        )
        params_by_layer[name] = asymmetric_quantization_params(
            min_value,
            max_value,
            num_bits=num_bits,
            dtype=dtype,
        )

    return params_by_layer


def integer_dense(
    inputs: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray | None = None,
    input_params: QuantizationParams | None = None,
    weight_params: QuantizationParams | None = None,
    output_params: QuantizationParams | None = None,
) -> IntegerLayerResult:
    """Reference Dense layer using integer matmul and INT32 accumulation."""
    if input_params is None:
        input_params = asymmetric_quantization_params(
            float(np.min(inputs)),
            float(np.max(inputs)),
            dtype="uint8",
        )
    if weight_params is None:
        # One symmetric quantizer per output unit.  This is the per-output-
        # channel scheme recommended for PTQ in the cited white papers.
        weight_params = symmetric_quantization_params(
            np.max(np.abs(weights), axis=0)
        )

    input_q = quantize_tensor(inputs, input_params)
    weight_q = quantize_tensor(weights, weight_params)

    input_centered = input_q.values.astype(np.int32) - input_params.zero_point
    weight_centered = weight_q.values.astype(np.int32) - weight_params.zero_point
    accumulator = input_centered @ weight_centered

    if bias is not None:
        bias_int32 = np.round(
            bias / (input_params.scale * weight_params.scale)
        ).astype(np.int32)
        accumulator = _saturating_add_int32(accumulator, bias_int32)

    accumulator_scale = np.asarray(input_params.scale) * np.asarray(
        weight_params.scale
    )

    if output_params is None:
        # Parameter selection is an offline/calibration operation.  The actual
        # requantization below remains integer-only.
        output_float = accumulator.astype(np.float64) * accumulator_scale
        output_params = asymmetric_quantization_params(
            float(np.min(output_float)),
            float(np.max(output_float)),
            dtype="uint8",
        )

    output_q = requantize_int32(
        accumulator,
        accumulator_scale=accumulator_scale,
        output_params=output_params,
    )

    return IntegerLayerResult(
        accumulator_int32=accumulator.astype(np.int32),
        output_int=output_q.values,
        output_dequantized=dequantize_tensor(output_q),
        input_params=input_params,
        weight_params=weight_params,
        output_params=output_params,
    )


def integer_conv2d_nhwc(
    inputs: np.ndarray,
    kernels: np.ndarray,
    bias: np.ndarray | None = None,
    strides: tuple[int, int] = (1, 1),
    padding: Padding = "valid",
    input_params: QuantizationParams | None = None,
    weight_params: QuantizationParams | None = None,
    output_params: QuantizationParams | None = None,
) -> IntegerLayerResult:
    """Reference NHWC Conv2D with integer multiply and INT32 accumulation."""
    if inputs.ndim != 4:
        raise ValueError("inputs must have shape (batch, height, width, channels).")
    if kernels.ndim != 4:
        raise ValueError("kernels must have shape (kernel_h, kernel_w, in_ch, out_ch).")

    if input_params is None:
        input_params = asymmetric_quantization_params(
            float(np.min(inputs)),
            float(np.max(inputs)),
            dtype="uint8",
        )
    if weight_params is None:
        weight_params = symmetric_quantization_params(
            np.max(np.abs(kernels), axis=(0, 1, 2))
        )

    input_q = quantize_tensor(inputs, input_params)
    weight_q = quantize_tensor(kernels, weight_params)

    input_centered = input_q.values.astype(np.int32) - input_params.zero_point
    weight_centered = weight_q.values.astype(np.int32) - weight_params.zero_point
    accumulator = _conv2d_nhwc_int32(
        input_centered,
        weight_centered,
        strides=strides,
        padding=padding,
    )

    if bias is not None:
        bias_int32 = np.round(
            bias / (input_params.scale * weight_params.scale)
        ).astype(np.int32)
        accumulator = _saturating_add_int32(
            accumulator,
            bias_int32.reshape((1, 1, 1, -1)),
        )

    accumulator_scale = np.asarray(input_params.scale) * np.asarray(
        weight_params.scale
    )

    if output_params is None:
        output_float = accumulator.astype(np.float64) * accumulator_scale
        output_params = asymmetric_quantization_params(
            float(np.min(output_float)),
            float(np.max(output_float)),
            dtype="uint8",
        )

    output_q = requantize_int32(
        accumulator,
        accumulator_scale=accumulator_scale,
        output_params=output_params,
    )

    return IntegerLayerResult(
        accumulator_int32=accumulator.astype(np.int32),
        output_int=output_q.values,
        output_dequantized=dequantize_tensor(output_q),
        input_params=input_params,
        weight_params=weight_params,
        output_params=output_params,
    )


def requantize_int32(
    accumulator: np.ndarray,
    accumulator_scale: float | np.ndarray,
    output_params: QuantizationParams,
) -> QuantizedArray:
    """Requantize an INT32 accumulator using fixed-point integer arithmetic.

    The real multiplier ``accumulator_scale / output_scale`` is converted
    offline to a Q31 multiplier and a base-two shift.  Runtime evaluation then
    consists solely of integer multiplication, rounded shifts, zero-point
    addition and saturation, following Jacob et al. (2017), equations 4--6.
    """
    real_multiplier = np.asarray(accumulator_scale, dtype=np.float64) / np.asarray(
        output_params.scale,
        dtype=np.float64,
    )
    multiplier, shift = _quantize_multiplier(real_multiplier)
    scaled = _multiply_by_quantized_multiplier(accumulator, multiplier, shift)
    shifted = scaled + np.asarray(output_params.zero_point, dtype=np.int64)
    clipped = np.clip(shifted, output_params.qmin, output_params.qmax)
    return QuantizedArray(
        values=clipped.astype(_numpy_dtype(output_params.dtype)),
        params=output_params,
    )


def _quantize_multiplier(
    real_multiplier: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Represent positive real multipliers as Q31 integers and binary shifts."""
    values = np.asarray(real_multiplier, dtype=np.float64)
    if np.any(values < 0) or np.any(~np.isfinite(values)):
        raise ValueError("real_multiplier must contain finite non-negative values.")

    significand, shift = np.frexp(values)
    q31 = np.rint(significand * (1 << 31)).astype(np.int64)

    overflow = q31 == (1 << 31)
    q31 = np.where(overflow, q31 // 2, q31)
    shift = np.where(overflow, shift + 1, shift)
    q31 = np.where(values == 0, 0, q31)
    shift = np.where(values == 0, 0, shift)
    return q31, shift.astype(np.int32)


def _multiply_by_quantized_multiplier(
    values: np.ndarray,
    multiplier: np.ndarray,
    shift: np.ndarray,
) -> np.ndarray:
    """Apply a Q31 multiplier and rounded power-of-two shift using integers."""
    values64 = np.asarray(values, dtype=np.int64)
    multiplier64 = _broadcast_output_parameter(multiplier, values64.ndim)
    shift64 = _broadcast_output_parameter(shift, values64.ndim)

    left_shift = np.maximum(shift64, 0)
    right_shift = np.maximum(-shift64, 0)
    shifted_values = values64 * np.left_shift(np.int64(1), left_shift)
    product = shifted_values * multiplier64
    scaled = _rounding_divide_by_power_of_two(product, 31)
    return _rounding_divide_by_power_of_two(scaled, right_shift)


def _rounding_divide_by_power_of_two(
    values: np.ndarray,
    exponent: int | np.ndarray,
) -> np.ndarray:
    """Signed round-to-nearest division by ``2**exponent``."""
    exponent_array = np.asarray(exponent, dtype=np.int64)
    divisor = np.left_shift(np.int64(1), exponent_array)
    magnitude = np.abs(values)
    rounded = (magnitude + divisor // 2) // divisor
    return np.where(values < 0, -rounded, rounded)


def _broadcast_output_parameter(parameter: np.ndarray, rank: int) -> np.ndarray:
    parameter = np.asarray(parameter, dtype=np.int64)
    if parameter.ndim == 0:
        return parameter
    if parameter.ndim != 1:
        raise ValueError("Per-channel parameters must be scalar or rank one.")
    return parameter.reshape((1,) * (rank - 1) + (parameter.shape[0],))


def _saturating_add_int32(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = left.astype(np.int64) + np.asarray(right, dtype=np.int64)
    limits = np.iinfo(np.int32)
    return np.clip(result, limits.min, limits.max).astype(np.int32)


def _integer_range(num_bits: int, dtype: QuantizedDType) -> tuple[int, int]:
    if not 1 <= num_bits <= 8:
        raise ValueError(f"num_bits must be between 1 and 8, got {num_bits}.")
    if dtype == "uint8":
        return 0, (2**num_bits) - 1
    if dtype == "int8":
        return -(2 ** (num_bits - 1)), (2 ** (num_bits - 1)) - 1
    raise ValueError(f"Unsupported dtype: {dtype}.")


def _signed_symmetric_range(num_bits: int) -> tuple[int, int]:
    if not 1 <= num_bits <= 8:
        raise ValueError(f"num_bits must be between 1 and 8, got {num_bits}.")
    if num_bits == 1:
        return -1, 1
    qmax = (2 ** (num_bits - 1)) - 1
    return -qmax, qmax


def _numpy_dtype(dtype: QuantizedDType) -> type[np.int8] | type[np.uint8]:
    return np.uint8 if dtype == "uint8" else np.int8


def _conv2d_nhwc_int32(
    inputs: np.ndarray,
    kernels: np.ndarray,
    strides: tuple[int, int],
    padding: Padding,
) -> np.ndarray:
    batch, in_h, in_w, in_ch = inputs.shape
    kernel_h, kernel_w, kernel_in_ch, out_ch = kernels.shape
    stride_h, stride_w = strides

    if in_ch != kernel_in_ch:
        raise ValueError("Input channels must match kernel input channels.")
    if padding not in {"valid", "same"}:
        raise ValueError("padding must be 'valid' or 'same'.")

    if padding == "same":
        out_h = int(np.ceil(in_h / stride_h))
        out_w = int(np.ceil(in_w / stride_w))
        pad_along_h = max((out_h - 1) * stride_h + kernel_h - in_h, 0)
        pad_along_w = max((out_w - 1) * stride_w + kernel_w - in_w, 0)
        pad_top = pad_along_h // 2
        pad_bottom = pad_along_h - pad_top
        pad_left = pad_along_w // 2
        pad_right = pad_along_w - pad_left
        inputs = np.pad(
            inputs,
            ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
        )
    else:
        out_h = (in_h - kernel_h) // stride_h + 1
        out_w = (in_w - kernel_w) // stride_w + 1

    output = np.zeros((batch, out_h, out_w, out_ch), dtype=np.int32)
    for out_y in range(out_h):
        in_y = out_y * stride_h
        for out_x in range(out_w):
            in_x = out_x * stride_w
            window = inputs[:, in_y : in_y + kernel_h, in_x : in_x + kernel_w, :]
            output[:, out_y, out_x, :] = np.tensordot(
                window,
                kernels,
                axes=([1, 2, 3], [0, 1, 2]),
            )
    return output


def _build_activation_model(model: keras.Model) -> tuple[keras.Model, list[str]]:
    replayed = _build_sequential_activation_model(model)
    if replayed is not None:
        return replayed

    outputs = []
    output_names = []

    for layer in model.layers:
        try:
            outputs.append(layer.output)
            output_names.append(layer.name)
        except AttributeError:
            continue

    return keras.Model(
        inputs=model.input,
        outputs=outputs,
        name=f"{model.name}_advanced_activation_calibrator",
    ), output_names


def _build_sequential_activation_model(
    model: keras.Model,
) -> tuple[keras.Model, list[str]] | None:
    if len(model.inputs) != 1:
        return None

    input_shape = tuple(model.inputs[0].shape[1:])
    inputs = keras.Input(
        shape=input_shape,
        dtype=model.inputs[0].dtype,
        name=f"{model.name}_advanced_activation_input",
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

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=f"{model.name}_advanced_activation_calibrator",
    ), output_names
