"""Mathematical codebook and product-vector weight quantization.

Research provenance
-------------------
FINAL-RESULT PATH (tensor-local scalar codebooks, ``vector_dim=1``):

* Gray (1984): finite codebooks, nearest-codeword assignments, and lookup
  reconstruction form the classical vector-quantization foundation.
* Arthur and Vassilvitskii (2007): ``fit_kmeans_codebook`` uses fixed-seed
  k-means++ D-squared initialization.
* Lloyd (1982): the same function alternates nearest-centroid assignment and
  centroid-mean updates until convergence.
* Gong et al. (2014): motivates applying vector quantization to neural-network
  parameter tensors.
* Han et al. (ICLR 2016): motivates weight sharing and accounting for packed
  assignment indices plus stored centroids.  Pruning, retraining, and Huffman
  coding from Deep Compression are not implemented here.

OPTIONAL LIBRARY PATHS, IMPLEMENTED BUT NOT USED IN FINAL RESULTS:

* Jegou et al. (2011): product quantization in ``product_quantize_array``.
* Choi et al. (2017): importance-weighted distortion and gradient-squared
  importance proxies in the importance-aware functions.
* Shao et al. (2024), DPQ: usage-prioritized product-codebook pooling and
  distance pruning in ``compress_product_codebooks``.  No diffusion model,
  DDPM loss, or activation-aware DPQ training is implemented.
* Deng et al. (2024), VQ4ALL: balanced multi-tensor sampling and Gaussian-KDE
  universal-codebook generation in ``build_universal_codebook_kde``.  No
  differentiable assignments, task-loss optimization, or progressive network
  construction is implemented.

The final report reconstructs codebook weights for FP32 Keras execution.
Following the deployment caution discussed by Abushahla et al. (2025), stored
assignment compression is not presented as measured latency, energy, or
runtime-memory improvement.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2
from typing import Collection, Literal

import numpy as np
import tensorflow as tf
from tensorflow import keras


CodebookMode = Literal["vector", "product"]


@dataclass(frozen=True)
class VectorLayout:
    """Metadata required to restore grouped row vectors to a tensor."""

    original_shape: tuple[int, ...]
    moved_shape: tuple[int, ...]
    axis: int
    vector_dim: int
    row_count: int
    original_width: int
    padded_width: int


@dataclass(frozen=True)
class CodebookQuantizedTensor:
    """Gray/Han-style assignments, shared centroids, and layout metadata."""

    name: str
    assignments: np.ndarray
    codebooks: np.ndarray
    layout: VectorLayout
    mode: CodebookMode
    original_dtype: str
    assignment_bits: int

    @property
    def original_size_bytes(self) -> int:
        return int(np.prod(self.layout.original_shape)) * np.dtype(
            self.original_dtype
        ).itemsize

    @property
    def packed_assignment_size_bytes(self) -> int:
        return _packed_size_bytes(self.assignments.size, self.assignment_bits)

    @property
    def estimated_compressed_size_bytes(self) -> int:
        return self.packed_assignment_size_bytes + self.codebooks.nbytes

    @property
    def physical_numpy_size_bytes(self) -> int:
        return self.assignments.nbytes + self.codebooks.nbytes

    @property
    def compression_ratio(self) -> float:
        return self.original_size_bytes / self.estimated_compressed_size_bytes

    @property
    def memory_reduction_percent(self) -> float:
        return (
            1 - self.estimated_compressed_size_bytes / self.original_size_bytes
        ) * 100


@dataclass(frozen=True)
class PooledCodebookTensor:
    """Product assignments whose local codebooks map into one shared pool."""

    quantized: CodebookQuantizedTensor
    pool: np.ndarray
    local_to_pool: np.ndarray
    pool_index_bits: int

    @property
    def estimated_compressed_size_bytes(self) -> int:
        mapping_bytes = _packed_size_bytes(
            self.local_to_pool.size,
            self.pool_index_bits,
        )
        return (
            self.quantized.packed_assignment_size_bytes
            + self.pool.nbytes
            + mapping_bytes
        )

    @property
    def compression_ratio(self) -> float:
        return self.quantized.original_size_bytes / self.estimated_compressed_size_bytes

    @property
    def memory_reduction_percent(self) -> float:
        return (
            1
            - self.estimated_compressed_size_bytes
            / self.quantized.original_size_bytes
        ) * 100


@dataclass(frozen=True)
class ModelCodebookQuantizationResult:
    """Codebook quantization report for all selected model parameters."""

    tensors: list[CodebookQuantizedTensor]
    skipped_tensor_names: list[str]
    original_size_bytes: int
    estimated_compressed_size_bytes: int

    @property
    def compression_ratio(self) -> float:
        return self.original_size_bytes / self.estimated_compressed_size_bytes

    @property
    def memory_reduction_percent(self) -> float:
        return (
            1 - self.estimated_compressed_size_bytes / self.original_size_bytes
        ) * 100


def vector_quantize_array(
    array: np.ndarray,
    vector_dim: int = 4,
    codebook_size: int = 256,
    *,
    axis: int = -1,
    codebook: np.ndarray | None = None,
    codebook_dtype: np.dtype | str = np.float32,
    max_kmeans_samples: int | None = 100_000,
    kmeans_iterations: int = 50,
    seed: int = 42,
    name: str = "tensor",
) -> CodebookQuantizedTensor:
    """Quantize all sub-vectors using one shared nearest-centroid codebook.

    If ``codebook`` is supplied, it is treated as a frozen universal codebook;
    otherwise a fixed-seed NumPy k-means codebook is fitted to the tensor.

    Gray supplies the classical nearest-codeword formulation.  Gong and Han
    motivate applying it to neural-network weights and storing shared centroid
    indices.  The report calls this with ``vector_dim=1``, so its evaluated
    configuration is scalar rather than multi-value vector quantization.
    """
    vectors, layout = vectorize_array(array, vector_dim, axis=axis)
    flat_vectors = vectors.reshape(-1, vector_dim)

    if codebook is None:
        fit_vectors = _sample_rows(flat_vectors, max_kmeans_samples, seed)
        fitted = fit_kmeans_codebook(
            fit_vectors,
            codebook_size,
            iterations=kmeans_iterations,
            seed=seed,
        )
    else:
        fitted = np.asarray(codebook, dtype=np.float32)
        if fitted.ndim != 2 or fitted.shape[1] != vector_dim:
            raise ValueError(
                "codebook must have shape (codebook_size, vector_dim); "
                f"received {fitted.shape}."
            )
        codebook_size = fitted.shape[0]

    stored_codebook = fitted.astype(codebook_dtype)
    assignment_values = nearest_codeword_indices(flat_vectors, stored_codebook)
    assignments = assignment_values.reshape(vectors.shape[:2]).astype(
        _index_dtype(codebook_size)
    )
    return CodebookQuantizedTensor(
        name=name,
        assignments=assignments,
        codebooks=stored_codebook,
        layout=layout,
        mode="vector",
        original_dtype=str(np.asarray(array).dtype),
        assignment_bits=_index_bits(codebook_size),
    )


def product_quantize_array(
    array: np.ndarray,
    vector_dim: int = 4,
    codebook_size: int = 256,
    *,
    axis: int = -1,
    codebook_dtype: np.dtype | str = np.float16,
    max_kmeans_samples: int | None = 100_000,
    kmeans_iterations: int = 50,
    seed: int = 42,
    name: str = "tensor",
) -> CodebookQuantizedTensor:
    """Apply Jegou-style product quantization with one codebook per position.

    For a matrix with width ``n`` and vector dimension ``d``, this produces
    ``n / d`` codebooks. Assignment ``(i, j)`` selects the nearest codeword in
    codebook ``j`` for row ``i`` and subspace ``j``.  This optional path is
    unit-tested but is not used by the final report experiments.
    """
    vectors, layout = vectorize_array(array, vector_dim, axis=axis)
    group_count = vectors.shape[1]
    codebooks = np.empty(
        (group_count, codebook_size, vector_dim),
        dtype=np.dtype(codebook_dtype),
    )
    assignments = np.empty(
        (layout.row_count, group_count),
        dtype=_index_dtype(codebook_size),
    )

    for group_index in range(group_count):
        group_vectors = vectors[:, group_index, :]
        fit_vectors = _sample_rows(
            group_vectors,
            max_kmeans_samples,
            seed + group_index,
        )
        fitted = fit_kmeans_codebook(
            fit_vectors,
            codebook_size,
            iterations=kmeans_iterations,
            seed=seed + group_index,
        ).astype(codebook_dtype)
        codebooks[group_index] = fitted
        assignments[:, group_index] = nearest_codeword_indices(
            group_vectors,
            fitted,
        ).astype(assignments.dtype)

    return CodebookQuantizedTensor(
        name=name,
        assignments=assignments,
        codebooks=codebooks,
        layout=layout,
        mode="product",
        original_dtype=str(np.asarray(array).dtype),
        assignment_bits=_index_bits(codebook_size),
    )


def build_universal_codebook_kde(
    arrays: list[np.ndarray],
    vector_dim: int,
    codebook_size: int,
    *,
    axis: int = -1,
    samples_per_array: int = 10_000,
    bandwidth: float | None = None,
    seed: int = 42,
    codebook_dtype: np.dtype | str = np.float32,
) -> np.ndarray:
    """Sample a VQ4ALL-inspired universal codebook from a Gaussian KDE.

    Each source array contributes exactly ``samples_per_array`` sub-vectors
    (sampling with replacement when needed), preventing a large model/tensor
    from dominating the shared distribution. Sampling a Gaussian KDE is
    equivalent to selecting an observed vector and adding Gaussian noise with
    the selected bandwidth.  This is only the reusable KDE/codebook component;
    it is not a reproduction of VQ4ALL's training procedure and is not used in
    the final scalar-codebook results.
    """
    if not arrays:
        raise ValueError("arrays must contain at least one tensor.")
    _validate_codebook_configuration(vector_dim, codebook_size)
    if samples_per_array < 1:
        raise ValueError("samples_per_array must be positive.")

    rng = np.random.default_rng(seed)
    balanced_samples = []
    for array in arrays:
        vectors, _ = vectorize_array(array, vector_dim, axis=axis)
        flat_vectors = vectors.reshape(-1, vector_dim)
        chosen = rng.choice(
            len(flat_vectors),
            size=samples_per_array,
            replace=len(flat_vectors) < samples_per_array,
        )
        balanced_samples.append(flat_vectors[chosen])
    observations = np.concatenate(balanced_samples).astype(np.float32)

    if bandwidth is None:
        # Scott's rule with one pooled scalar bandwidth, matching the paper's
        # isotropic Gaussian-kernel formulation.
        scale = float(np.sqrt(np.mean(np.var(observations, axis=0))))
        bandwidth = scale * len(observations) ** (-1.0 / (vector_dim + 4))
    if bandwidth < 0:
        raise ValueError("bandwidth must be non-negative.")

    mixture_indices = rng.integers(0, len(observations), size=codebook_size)
    noise = rng.normal(0.0, bandwidth, size=(codebook_size, vector_dim))
    return (observations[mixture_indices] + noise).astype(codebook_dtype)


def compress_product_codebooks(
    quantized: CodebookQuantizedTensor,
    target_pool_size: int | None = None,
    *,
    distance_threshold: float = 0.0,
    pool_dtype: np.dtype | str = np.float16,
) -> PooledCodebookTensor:
    """Apply DPQ-inspired usage-priority and distance-based codebook pooling.

    Centroid importance is its assignment frequency. Centroids are considered
    redundant when their RMS L2 distance from a selected pool vector is at or
    below ``distance_threshold``. If pruning yields fewer vectors than the
    target, the remaining most-used centroids fill the pool, as in DPQ.  This
    optional utility does not implement DPQ's diffusion or activation-aware
    optimization and is not used in the final experiments.
    """
    if quantized.mode != "product" or quantized.codebooks.ndim != 3:
        raise ValueError("compress_product_codebooks requires product quantization.")
    if distance_threshold < 0:
        raise ValueError("distance_threshold must be non-negative.")

    group_count, codebook_size, vector_dim = quantized.codebooks.shape
    if target_pool_size is None:
        m = quantized.layout.row_count
        n = quantized.layout.padded_width
        target_pool_size = max(1, ceil(m * n / (16 * vector_dim**2)))
    target_pool_size = min(int(target_pool_size), group_count * codebook_size)
    if target_pool_size < 1:
        raise ValueError("target_pool_size must be positive.")

    usage = np.zeros((group_count, codebook_size), dtype=np.int64)
    for group_index in range(group_count):
        usage[group_index] = np.bincount(
            quantized.assignments[:, group_index].astype(np.int64),
            minlength=codebook_size,
        )

    candidates = quantized.codebooks.reshape(-1, vector_dim).astype(np.float32)
    flat_usage = usage.reshape(-1)
    order = np.argsort(-flat_usage, kind="stable")
    selected: list[int] = []
    selected_set: set[int] = set()

    for candidate_index in order:
        if len(selected) >= target_pool_size:
            break
        if not selected:
            selected.append(int(candidate_index))
            selected_set.add(int(candidate_index))
            continue
        distances = _rms_distances(candidates[candidate_index], candidates[selected])
        if float(np.min(distances)) > distance_threshold:
            selected.append(int(candidate_index))
            selected_set.add(int(candidate_index))

    if len(selected) < target_pool_size:
        for candidate_index in order:
            value = int(candidate_index)
            if value not in selected_set:
                selected.append(value)
                selected_set.add(value)
            if len(selected) >= target_pool_size:
                break

    pool = candidates[selected].astype(pool_dtype)
    local_to_pool = nearest_codeword_indices(candidates, pool).reshape(
        group_count,
        codebook_size,
    ).astype(_index_dtype(len(pool)))
    return PooledCodebookTensor(
        quantized=quantized,
        pool=pool,
        local_to_pool=local_to_pool,
        pool_index_bits=_index_bits(len(pool)),
    )


def reconstruct_codebook_tensor(
    quantized: CodebookQuantizedTensor,
) -> np.ndarray:
    """Reconstruct a floating tensor using explicit codebook lookups."""
    assignments = quantized.assignments.astype(np.int64)
    if quantized.mode == "vector":
        vectors = quantized.codebooks[assignments]
    else:
        group_indices = np.arange(assignments.shape[1])[None, :]
        vectors = quantized.codebooks[group_indices, assignments]
    return restore_vectorized_array(vectors, quantized.layout).astype(
        quantized.original_dtype
    )


def reconstruct_pooled_codebook_tensor(
    pooled: PooledCodebookTensor,
) -> np.ndarray:
    """Reconstruct product-quantized weights through the compressed pool."""
    assignments = pooled.quantized.assignments.astype(np.int64)
    group_indices = np.arange(assignments.shape[1])[None, :]
    pool_indices = pooled.local_to_pool[group_indices, assignments]
    vectors = pooled.pool[pool_indices.astype(np.int64)]
    return restore_vectorized_array(vectors, pooled.quantized.layout).astype(
        pooled.quantized.original_dtype
    )


def codebook_vectorize_model(
    model: keras.Model,
    vector_dim: int = 4,
    codebook_size: int = 256,
    *,
    mode: CodebookMode = "product",
    quantize_min_rank: int = 2,
    excluded_tensor_names: Collection[str] | None = None,
    included_tensor_names: Collection[str] | None = None,
    importance_by_name: dict[str, np.ndarray] | None = None,
    codebook_dtype: np.dtype | str = np.float16,
    max_kmeans_samples: int | None = 100_000,
    kmeans_iterations: int = 50,
    seed: int = 42,
) -> ModelCodebookQuantizationResult:
    """Apply Gong/Han-style codebook weight sharing to selected model tensors.

    The original model is not mutated. Biases and normalization vectors are
    retained in their original representation and included in size accounting.
    Names in ``excluded_tensor_names`` are also retained exactly. This is useful
    for mixed-precision PTQ, where accuracy-sensitive input/output tensors stay
    in FP32 while the remaining kernels use codebook assignments.

    Final experiments pass ``mode='vector'`` and ``vector_dim=1``. Product and
    importance-weighted branches are optional research-informed capabilities.
    """
    if mode not in {"vector", "product"}:
        raise ValueError("mode must be 'vector' or 'product'.")
    tensors: list[CodebookQuantizedTensor] = []
    skipped_names: list[str] = []
    original_size = 0
    compressed_size = 0
    excluded = set(excluded_tensor_names or ())
    included = None if included_tensor_names is None else set(included_tensor_names)

    for tensor_index, weight in enumerate(model.weights):
        values = weight.numpy()
        name = getattr(weight, "path", weight.name)
        original_size += values.nbytes
        if (
            not np.issubdtype(values.dtype, np.floating)
            or values.ndim < quantize_min_rank
            or name in excluded
            or (included is not None and name not in included)
        ):
            skipped_names.append(name)
            compressed_size += values.nbytes
            continue

        kwargs = dict(
            array=values,
            vector_dim=vector_dim,
            codebook_size=codebook_size,
            codebook_dtype=codebook_dtype,
            max_kmeans_samples=max_kmeans_samples,
            kmeans_iterations=kmeans_iterations,
            seed=seed + tensor_index,
            name=name,
        )
        importance = None if importance_by_name is None else importance_by_name.get(name)
        if importance is not None and np.asarray(importance).shape != values.shape:
            raise ValueError(
                f"Importance for {name} has shape {np.asarray(importance).shape}; "
                f"expected {values.shape}."
            )
        if mode == "product" and importance is not None:
            result = importance_weighted_product_quantize_array(
                importance=importance, **kwargs
            )
        elif mode == "product":
            result = product_quantize_array(**kwargs)
        elif importance is not None:
            result = importance_weighted_vector_quantize_array(
                importance=importance, **kwargs
            )
        else:
            result = vector_quantize_array(**kwargs)
        tensors.append(result)
        compressed_size += result.estimated_compressed_size_bytes

    return ModelCodebookQuantizationResult(
        tensors=tensors,
        skipped_tensor_names=skipped_names,
        original_size_bytes=original_size,
        estimated_compressed_size_bytes=compressed_size,
    )


def codebook_vectorize_model_n_bits(
    model: keras.Model,
    num_bits: int,
    vector_dim: int = 4,
    **kwargs,
) -> ModelCodebookQuantizationResult:
    """Codebook-quantize model weights using ``2 ** num_bits`` entries.

    ``num_bits`` is the assignment-index width.  For vector dimension ``d``,
    the assignment payload is therefore ``num_bits / d`` bits per scalar
    weight, before accounting for the stored centroids.
    """
    if not isinstance(num_bits, (int, np.integer)) or not 1 <= int(num_bits) <= 32:
        raise ValueError("num_bits must be an integer between 1 and 32.")
    return codebook_vectorize_model(
        model,
        vector_dim=vector_dim,
        codebook_size=2 ** int(num_bits),
        **kwargs,
    )


def reconstruct_codebook_model_weights(
    model: keras.Model,
    result: ModelCodebookQuantizationResult,
) -> list[np.ndarray]:
    """Return model-order weights reconstructed from a codebook report.

    The returned list can be passed to ``clone.set_weights(...)`` for accuracy
    evaluation. It does not mutate ``model`` and, consistent with the hardware
    caveat emphasized by Abushahla et al., does not pretend that reconstructed
    FP32 execution is an indexed-codebook hardware runtime.
    """
    tensors_by_name = {tensor.name: tensor for tensor in result.tensors}
    if len(tensors_by_name) != len(result.tensors):
        raise ValueError("Quantized tensor names must be unique.")
    reconstructed_weights = []
    consumed: set[str] = set()
    for weight in model.weights:
        name = getattr(weight, "path", weight.name)
        values = weight.numpy()
        quantized = tensors_by_name.get(name)
        if quantized is None:
            reconstructed_weights.append(values)
            continue
        reconstructed = reconstruct_codebook_tensor(quantized)
        if reconstructed.shape != values.shape:
            raise ValueError(
                f"Reconstructed shape for {name} is {reconstructed.shape}; "
                f"expected {values.shape}."
            )
        reconstructed_weights.append(reconstructed.astype(values.dtype))
        consumed.add(name)
    unused = set(tensors_by_name) - consumed
    if unused:
        raise ValueError(f"Quantized tensors do not belong to model: {sorted(unused)}")
    return reconstructed_weights


def vectorize_array(
    array: np.ndarray,
    vector_dim: int,
    *,
    axis: int = -1,
) -> tuple[np.ndarray, VectorLayout]:
    """Move ``axis`` last and split every row into contiguous sub-vectors."""
    values = np.asarray(array)
    if values.ndim == 0:
        raise ValueError("array must have rank one or greater.")
    if vector_dim < 1:
        raise ValueError("vector_dim must be positive.")
    axis = axis % values.ndim
    moved = np.moveaxis(values, axis, -1)
    original_width = moved.shape[-1]
    padded_width = ceil(original_width / vector_dim) * vector_dim
    row_count = int(np.prod(moved.shape[:-1])) if moved.ndim > 1 else 1
    matrix = moved.reshape(row_count, original_width).astype(np.float32)
    if padded_width != original_width:
        matrix = np.pad(matrix, ((0, 0), (0, padded_width - original_width)))
    vectors = matrix.reshape(row_count, padded_width // vector_dim, vector_dim)
    layout = VectorLayout(
        original_shape=tuple(values.shape),
        moved_shape=tuple(moved.shape),
        axis=axis,
        vector_dim=vector_dim,
        row_count=row_count,
        original_width=original_width,
        padded_width=padded_width,
    )
    return vectors, layout


def restore_vectorized_array(
    vectors: np.ndarray,
    layout: VectorLayout,
) -> np.ndarray:
    """Undo :func:`vectorize_array`, including removal of zero padding."""
    matrix = np.asarray(vectors).reshape(layout.row_count, layout.padded_width)
    moved = matrix[:, : layout.original_width].reshape(layout.moved_shape)
    return np.moveaxis(moved, -1, layout.axis)


def fit_kmeans_codebook(
    vectors: np.ndarray,
    codebook_size: int,
    *,
    iterations: int = 50,
    tolerance: float = 1e-6,
    seed: int = 42,
) -> np.ndarray:
    """Fit centroids with Arthur--Vassilvitskii initialization and Lloyd updates."""
    data = np.asarray(vectors, dtype=np.float32)
    if data.ndim != 2 or not len(data):
        raise ValueError("vectors must be a non-empty rank-2 array.")
    _validate_codebook_configuration(data.shape[1], codebook_size)
    if iterations < 1:
        raise ValueError("iterations must be positive.")

    rng = np.random.default_rng(seed)
    centers = np.empty((codebook_size, data.shape[1]), dtype=np.float32)
    # Arthur and Vassilvitskii (k-means++): choose later centers with
    # probability proportional to squared distance from the nearest center.
    centers[0] = data[rng.integers(len(data))]
    closest_distance = np.sum((data - centers[0]) ** 2, axis=1)
    for center_index in range(1, codebook_size):
        total = float(np.sum(closest_distance))
        if total == 0.0:
            selected = rng.integers(len(data))
        else:
            selected = rng.choice(len(data), p=closest_distance / total)
        centers[center_index] = data[selected]
        distance = np.sum((data - centers[center_index]) ** 2, axis=1)
        closest_distance = np.minimum(closest_distance, distance)

    # Lloyd iteration: nearest-center assignment followed by centroid means.
    for _ in range(iterations):
        assignment = nearest_codeword_indices(data, centers)
        updated = centers.copy()
        minimum_distance = np.sum((data - centers[assignment]) ** 2, axis=1)
        farthest_order = np.argsort(-minimum_distance, kind="stable")
        empty_counter = 0
        for center_index in range(codebook_size):
            members = data[assignment == center_index]
            if len(members):
                updated[center_index] = np.mean(members, axis=0)
            else:
                updated[center_index] = data[
                    farthest_order[empty_counter % len(farthest_order)]
                ]
                empty_counter += 1
        shift = float(np.max(np.sqrt(np.sum((updated - centers) ** 2, axis=1))))
        centers = updated
        if shift <= tolerance:
            break
    return centers


def fit_importance_weighted_kmeans_codebook(
    vectors: np.ndarray,
    importance: np.ndarray,
    codebook_size: int,
    *,
    iterations: int = 50,
    tolerance: float = 1e-6,
    seed: int = 42,
) -> np.ndarray:
    """Fit centroids with Choi-inspired importance-weighted distortion.

    This optional path is implemented and tested but is not used by the final
    unweighted scalar-codebook experiments.
    """
    data = np.asarray(vectors, dtype=np.float32)
    weights = np.asarray(importance, dtype=np.float32)
    if data.ndim != 2 or not len(data) or weights.shape != data.shape:
        raise ValueError("vectors and importance must be equal, non-empty rank-2 arrays.")
    if np.any(weights < 0) or not np.all(np.isfinite(weights)):
        raise ValueError("importance must contain finite, non-negative values.")
    _validate_codebook_configuration(data.shape[1], codebook_size)
    if iterations < 1:
        raise ValueError("iterations must be positive.")

    # A small floor makes completely insensitive coordinates numerically safe.
    safe_weights = np.maximum(weights, np.finfo(np.float32).eps)
    rng = np.random.default_rng(seed)
    centers = np.empty((codebook_size, data.shape[1]), dtype=np.float32)
    centers[0] = data[rng.integers(len(data))]
    closest = np.sum(safe_weights * (data - centers[0]) ** 2, axis=1)
    for center_index in range(1, codebook_size):
        total = float(np.sum(closest))
        selected = (
            rng.integers(len(data))
            if total == 0.0
            else rng.choice(len(data), p=closest / total)
        )
        centers[center_index] = data[selected]
        distance = np.sum(
            safe_weights * (data - centers[center_index]) ** 2, axis=1
        )
        closest = np.minimum(closest, distance)

    for _ in range(iterations):
        assignment = nearest_codeword_indices_weighted(data, centers, safe_weights)
        updated = centers.copy()
        assigned_error = np.sum(
            safe_weights * (data - centers[assignment]) ** 2, axis=1
        )
        farthest = np.argsort(-assigned_error, kind="stable")
        empty_counter = 0
        for center_index in range(codebook_size):
            mask = assignment == center_index
            if np.any(mask):
                member_weights = safe_weights[mask]
                updated[center_index] = np.sum(
                    member_weights * data[mask], axis=0
                ) / np.sum(member_weights, axis=0)
            else:
                updated[center_index] = data[
                    farthest[empty_counter % len(farthest)]
                ]
                empty_counter += 1
        shift = float(np.max(np.linalg.norm(updated - centers, axis=1)))
        centers = updated
        if shift <= tolerance:
            break
    return centers


def nearest_codeword_indices(
    vectors: np.ndarray,
    codebook: np.ndarray,
    *,
    chunk_size: int = 16_384,
) -> np.ndarray:
    """Return nearest-centroid indices without forming one huge distance matrix."""
    data = np.asarray(vectors, dtype=np.float32)
    centers = np.asarray(codebook, dtype=np.float32)
    if data.ndim != 2 or centers.ndim != 2 or data.shape[1] != centers.shape[1]:
        raise ValueError("vectors and codebook must be compatible rank-2 arrays.")
    result = np.empty(len(data), dtype=np.int64)
    center_norm = np.sum(centers * centers, axis=1)
    for start in range(0, len(data), chunk_size):
        chunk = data[start : start + chunk_size]
        distances = (
            np.sum(chunk * chunk, axis=1, keepdims=True)
            + center_norm[None, :]
            - 2.0 * chunk @ centers.T
        )
        result[start : start + len(chunk)] = np.argmin(distances, axis=1)
    return result


def nearest_codeword_indices_weighted(
    vectors: np.ndarray,
    codebook: np.ndarray,
    importance: np.ndarray,
    *,
    chunk_size: int = 16_384,
) -> np.ndarray:
    """Return assignments minimizing importance-weighted squared distance."""
    data = np.asarray(vectors, dtype=np.float32)
    centers = np.asarray(codebook, dtype=np.float32)
    weights = np.asarray(importance, dtype=np.float32)
    if (
        data.ndim != 2
        or centers.ndim != 2
        or data.shape[1] != centers.shape[1]
        or weights.shape != data.shape
    ):
        raise ValueError("vectors, codebook, and importance have incompatible shapes.")
    if np.any(weights < 0):
        raise ValueError("importance must be non-negative.")
    result = np.empty(len(data), dtype=np.int64)
    for start in range(0, len(data), chunk_size):
        chunk = data[start : start + chunk_size]
        chunk_weights = weights[start : start + chunk_size]
        distances = np.sum(
            chunk_weights[:, None, :] * (chunk[:, None, :] - centers[None, :, :]) ** 2,
            axis=2,
        )
        result[start : start + len(chunk)] = np.argmin(distances, axis=1)
    return result


def importance_weighted_vector_quantize_array(
    array: np.ndarray,
    importance: np.ndarray,
    vector_dim: int = 4,
    codebook_size: int = 256,
    *,
    axis: int = -1,
    codebook_dtype: np.dtype | str = np.float32,
    max_kmeans_samples: int | None = 100_000,
    kmeans_iterations: int = 50,
    seed: int = 42,
    name: str = "tensor",
) -> CodebookQuantizedTensor:
    """Vector-quantize an array using a gradient-derived distortion weight."""
    vectors, layout = vectorize_array(array, vector_dim, axis=axis)
    importance_vectors, importance_layout = vectorize_array(
        importance, vector_dim, axis=axis
    )
    if importance_layout != layout:
        raise ValueError("importance must have the same shape as array.")
    flat = vectors.reshape(-1, vector_dim)
    flat_importance = np.maximum(
        importance_vectors.reshape(-1, vector_dim), np.finfo(np.float32).eps
    )
    if max_kmeans_samples is not None and len(flat) > max_kmeans_samples:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(flat), size=max_kmeans_samples, replace=False)
        fit_vectors, fit_importance = flat[chosen], flat_importance[chosen]
    else:
        fit_vectors, fit_importance = flat, flat_importance
    codebook = fit_importance_weighted_kmeans_codebook(
        fit_vectors,
        fit_importance,
        codebook_size,
        iterations=kmeans_iterations,
        seed=seed,
    ).astype(codebook_dtype)
    assignments = nearest_codeword_indices_weighted(
        flat, codebook, flat_importance
    ).reshape(vectors.shape[:2]).astype(_index_dtype(codebook_size))
    return CodebookQuantizedTensor(
        name=name,
        assignments=assignments,
        codebooks=codebook,
        layout=layout,
        mode="vector",
        original_dtype=str(np.asarray(array).dtype),
        assignment_bits=_index_bits(codebook_size),
    )


def importance_weighted_product_quantize_array(
    array: np.ndarray,
    importance: np.ndarray,
    vector_dim: int = 4,
    codebook_size: int = 256,
    *,
    axis: int = -1,
    codebook_dtype: np.dtype | str = np.float16,
    max_kmeans_samples: int | None = 100_000,
    kmeans_iterations: int = 50,
    seed: int = 42,
    name: str = "tensor",
) -> CodebookQuantizedTensor:
    """Product-quantize each subspace with importance-weighted k-means."""
    vectors, layout = vectorize_array(array, vector_dim, axis=axis)
    importance_vectors, importance_layout = vectorize_array(
        importance, vector_dim, axis=axis
    )
    if importance_layout != layout:
        raise ValueError("importance must have the same shape as array.")
    group_count = vectors.shape[1]
    codebooks = np.empty(
        (group_count, codebook_size, vector_dim), dtype=np.dtype(codebook_dtype)
    )
    assignments = np.empty(
        (layout.row_count, group_count), dtype=_index_dtype(codebook_size)
    )
    for group_index in range(group_count):
        data = vectors[:, group_index, :]
        weights = np.maximum(
            importance_vectors[:, group_index, :], np.finfo(np.float32).eps
        )
        if max_kmeans_samples is not None and len(data) > max_kmeans_samples:
            rng = np.random.default_rng(seed + group_index)
            chosen = rng.choice(len(data), size=max_kmeans_samples, replace=False)
            fit_data, fit_weights = data[chosen], weights[chosen]
        else:
            fit_data, fit_weights = data, weights
        fitted = fit_importance_weighted_kmeans_codebook(
            fit_data,
            fit_weights,
            codebook_size,
            iterations=kmeans_iterations,
            seed=seed + group_index,
        ).astype(codebook_dtype)
        codebooks[group_index] = fitted
        assignments[:, group_index] = nearest_codeword_indices_weighted(
            data, fitted, weights
        ).astype(assignments.dtype)
    return CodebookQuantizedTensor(
        name=name,
        assignments=assignments,
        codebooks=codebooks,
        layout=layout,
        mode="product",
        original_dtype=str(np.asarray(array).dtype),
        assignment_bits=_index_bits(codebook_size),
    )


def estimate_gradient_importance(
    model: keras.Model,
    calibration_batches,
    *,
    included_tensor_names: Collection[str] | None = None,
    use_square_root: bool = False,
    epsilon: float = 1e-12,
) -> dict[str, np.ndarray]:
    """Estimate a Choi-style diagonal sensitivity proxy from squared gradients.

    All selected floating weights are watched explicitly, so frozen pretrained
    backbones are supported as well as trainable layers.  This optional proxy
    is not used in the final reported codebook sweep or sensitivity ranking.
    """
    included = None if included_tensor_names is None else set(included_tensor_names)
    selected = []
    for weight in model.weights:
        name = getattr(weight, "path", weight.name)
        if included is not None and name not in included:
            continue
        if np.issubdtype(np.dtype(weight.dtype), np.floating):
            selected.append((name, weight))
    totals = [np.zeros(tuple(weight.shape), dtype=np.float64) for _, weight in selected]
    batch_count = 0
    watched = [getattr(weight, "value", weight) for _, weight in selected]
    for inputs, labels in calibration_batches:
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(watched)
            predictions = model(inputs, training=False)
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    labels, predictions, from_logits=True
                )
            )
        gradients = tape.gradient(loss, watched)
        for index, gradient in enumerate(gradients):
            if gradient is not None:
                totals[index] += np.square(np.asarray(gradient, dtype=np.float64))
        batch_count += 1
    if batch_count == 0:
        raise ValueError("calibration_batches must yield at least one batch.")
    result = {}
    for (name, weight), total in zip(selected, totals):
        importance = total / batch_count
        if use_square_root:
            importance = np.sqrt(importance)
        result[name] = np.maximum(importance, epsilon).astype(np.float32)
    return result


def quantization_mse(
    original: np.ndarray,
    quantized: CodebookQuantizedTensor | PooledCodebookTensor,
) -> float:
    """Calculate mean squared reconstruction error."""
    if isinstance(quantized, PooledCodebookTensor):
        reconstructed = reconstruct_pooled_codebook_tensor(quantized)
    else:
        reconstructed = reconstruct_codebook_tensor(quantized)
    return float(
        np.mean(
            (np.asarray(original, dtype=np.float32) - reconstructed.astype(np.float32))
            ** 2
        )
    )


def _sample_rows(
    rows: np.ndarray,
    maximum: int | None,
    seed: int,
) -> np.ndarray:
    if maximum is None or len(rows) <= maximum:
        return rows
    rng = np.random.default_rng(seed)
    return rows[rng.choice(len(rows), size=maximum, replace=False)]


def _rms_distances(vector: np.ndarray, others: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((others - vector[None, :]) ** 2, axis=1))


def _validate_codebook_configuration(vector_dim: int, codebook_size: int) -> None:
    if vector_dim < 1:
        raise ValueError("vector_dim must be positive.")
    if codebook_size < 2:
        raise ValueError("codebook_size must be at least two.")
    if codebook_size > 2**32:
        raise ValueError("codebook_size exceeds the supported index range.")


def _index_bits(codebook_size: int) -> int:
    return max(1, ceil(log2(codebook_size)))


def _index_dtype(codebook_size: int) -> np.dtype:
    maximum_index = codebook_size - 1
    if maximum_index <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    if maximum_index <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


def _packed_size_bytes(number_of_values: int, number_of_bits: int) -> int:
    return (number_of_values * number_of_bits + 7) // 8
