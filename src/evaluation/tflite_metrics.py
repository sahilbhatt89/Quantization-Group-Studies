"""Size, tensor-type, and latency metrics for TFLite models."""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema_fb


def size_metrics(baseline_path: Path, quantized_path: Path) -> dict[str, float | int]:
    """Compare serialized model sizes."""
    baseline_bytes = baseline_path.stat().st_size
    quantized_bytes = quantized_path.stat().st_size
    return {
        "size_bytes": quantized_bytes,
        "size_mib": quantized_bytes / (1024**2),
        **reduction_metrics(baseline_bytes, quantized_bytes),
    }


def reduction_metrics(
    baseline_bytes: int,
    quantized_bytes: int,
) -> dict[str, float]:
    """Compute compression and memory-reduction metrics."""
    return {
        "compression_ratio": baseline_bytes / quantized_bytes,
        "memory_reduction_percent": (1 - quantized_bytes / baseline_bytes) * 100,
    }


def inspect_tflite(model_path: Path) -> dict[str, object]:
    """Report tensor types and graph storage estimates."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    tensor_details = interpreter.get_tensor_details()
    dtype_counts = Counter(np.dtype(detail["dtype"]).name for detail in tensor_details)
    storage = _graph_storage(model_path, tensor_details)
    return {
        "input_dtype": np.dtype(input_detail["dtype"]).name,
        "output_dtype": np.dtype(output_detail["dtype"]).name,
        "tensor_dtype_counts": dict(sorted(dtype_counts.items())),
        **storage,
    }


def benchmark_tflite(
    model_path: Path,
    sample: np.ndarray,
    warmup_runs: int = 5,
    measured_runs: int = 30,
) -> dict[str, float | int]:
    """Measure single-sample TFLite inference latency."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    input_detail = interpreter.get_input_details()[0]

    if tuple(input_detail["shape"]) != tuple(sample.shape):
        interpreter.resize_tensor_input(input_detail["index"], sample.shape)
    interpreter.allocate_tensors()

    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    model_input = _quantize_input(sample, input_detail)

    for _ in range(warmup_runs):
        interpreter.set_tensor(input_detail["index"], model_input)
        interpreter.invoke()

    times_ms = []
    for _ in range(measured_runs):
        interpreter.set_tensor(input_detail["index"], model_input)
        start = time.perf_counter()
        interpreter.invoke()
        times_ms.append((time.perf_counter() - start) * 1000)

    output = interpreter.get_tensor(output_detail["index"])
    return {
        "runs": measured_runs,
        "mean_latency_ms": float(np.mean(times_ms)),
        "median_latency_ms": float(np.median(times_ms)),
        "std_latency_ms": float(np.std(times_ms)),
        "top_class_index": int(np.argmax(output[0])),
    }


def _quantize_input(sample: np.ndarray, input_detail: dict) -> np.ndarray:
    dtype = input_detail["dtype"]
    if dtype == np.float32:
        return sample.astype(np.float32)

    scale, zero_point = input_detail["quantization"]
    if scale == 0:
        raise ValueError("Quantized model input has an invalid zero scale.")

    quantized = np.round(sample / scale + zero_point)
    limits = np.iinfo(dtype)
    return np.clip(quantized, limits.min, limits.max).astype(dtype)


def _graph_storage(
    model_path: Path,
    tensor_details: list[dict],
) -> dict[str, int]:
    """Estimate constant-weight and runtime-activation tensor storage."""
    model_content = model_path.read_bytes()
    model = schema_fb.Model.GetRootAsModel(model_content, 0)
    subgraph = model.Subgraphs(0)
    details_by_index = {detail["index"]: detail for detail in tensor_details}

    constant_buffer_ids = set()
    activation_sizes = []

    for tensor_index in range(subgraph.TensorsLength()):
        tensor = subgraph.Tensors(tensor_index)
        buffer_id = tensor.Buffer()
        buffer_size = model.Buffers(buffer_id).DataLength()
        if buffer_size:
            constant_buffer_ids.add(buffer_id)
            continue

        detail = details_by_index.get(tensor_index)
        if detail is None:
            continue
        elements = int(np.prod(detail["shape"], dtype=np.int64))
        activation_sizes.append(elements * np.dtype(detail["dtype"]).itemsize)

    weight_bytes = sum(
        model.Buffers(buffer_id).DataLength() for buffer_id in constant_buffer_ids
    )
    return {
        "weight_storage_bytes": weight_bytes,
        "activation_tensor_storage_bytes": sum(activation_sizes),
        "largest_activation_tensor_bytes": max(activation_sizes, default=0),
    }
