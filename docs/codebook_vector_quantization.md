# Custom codebook vector quantization

The implementation is in
`src/quantization/custom_quantization/codebook_vector_quantization.py`. It uses
explicit NumPy mathematics and integer codebook indices. It does not use fake
quantization, TensorFlow Model Optimization, or TFLite conversion.

## Core representation

For a weight matrix `W` whose last dimension has width `n`, contiguous groups
of `d` values form sub-vectors:

```text
W row = [w0 ... w(d-1) | wd ... w(2d-1) | ...]
```

For a codebook `C` with `k` vectors, ordinary vector quantization stores the
nearest-codeword index:

```text
A[i, j] = argmin_p ||w[i, j] - C[p]||²
W_hat[i, j] = C[A[i, j]]
```

The assignment needs `ceil(log2(k))` bits instead of storing `d` FP32 values.
The implementation includes both assignment and codebook storage in its memory
report rather than assuming that codebook overhead is negligible.

## Paper-to-code mapping

| Implemented component | Source | Function |
|---|---|---|
| Contiguous weight sub-vectors and nearest-centroid integer assignments | DPQ and VQ4ALL | `vectorize_array()`, `nearest_codeword_indices()` |
| K-means codebook construction | DPQ background/method | `fit_kmeans_codebook()` |
| One shared codebook for all vector positions | Classical VQ described by both papers | `vector_quantize_array()` |
| One codebook per subspace/position | DPQ product quantization, Eq. 3 | `product_quantize_array()` |
| FP16 product-codebook storage | DPQ codebook compression | `codebook_dtype=np.float16` |
| Importance measured by assignment frequency | DPQ codebook pool | `compress_product_codebooks()` |
| Distance-based redundant-centroid removal | DPQ codebook pool, Eqs. 4–5 | `compress_product_codebooks()` |
| Default DPQ pool-size target `ceil(mn/(16d²))` | DPQ Section 3.2 | `compress_product_codebooks(target_pool_size=None)` |
| Equal sub-vector sampling from multiple tensors/models | VQ4ALL Section 4.1 | `build_universal_codebook_kde()` |
| Gaussian KDE codebook sampling | VQ4ALL Eqs. 3–4 | `build_universal_codebook_kde()` |
| Frozen universal codebook assignment | VQ4ALL | pass `codebook=` to `vector_quantize_array()` |
| Exact assignment/codebook storage accounting | DPQ and MCU deployment concerns | result dataclass properties |
| Rank-based model weight selection | Existing project convention | `codebook_vectorize_model()` |

The microcontroller survey explains the hardware trade-off behind the storage
accounting: non-uniform/codebook quantization can substantially reduce model
storage, but lookup-table decoding is not automatically compatible with fast
fixed-point matrix multiplication. Compression ratio therefore must not be
reported as inference acceleration without a runtime that directly supports
indexed codebook operations.

## Product quantization example

```python
from src.quantization.custom_quantization import (
    product_quantize_array,
    quantization_mse,
    reconstruct_codebook_tensor,
)

quantized = product_quantize_array(
    weight_array,
    vector_dim=4,
    codebook_size=256,
    codebook_dtype="float16",
)

integer_assignments = quantized.assignments
product_codebooks = quantized.codebooks
reconstructed_weight = reconstruct_codebook_tensor(quantized)

print(quantized.compression_ratio)
print(quantized.memory_reduction_percent)
print(quantization_mse(weight_array, quantized))
```

With `codebook_size=256`, every assignment is stored in a NumPy `uint8`. One
assignment represents `vector_dim` original scalar values.

## Universal KDE codebook example

```python
from src.quantization.custom_quantization import (
    build_universal_codebook_kde,
    vector_quantize_array,
)

universal_codebook = build_universal_codebook_kde(
    arrays=[resnet_weight, vit_weight, distilbert_weight],
    vector_dim=4,
    codebook_size=4096,
    samples_per_array=10_000,
    seed=42,
)

quantized = vector_quantize_array(
    another_weight,
    vector_dim=4,
    codebook=universal_codebook,
)
```

Each source tensor contributes the same number of sampled vectors. Sampling
from the fitted Gaussian KDE is implemented as sampling an observed vector from
the equal-weight empirical mixture and adding isotropic Gaussian noise. When
`bandwidth=0`, the codewords are direct samples of the observed vectors.

## DPQ-style codebook pool

```python
from src.quantization.custom_quantization import (
    compress_product_codebooks,
    reconstruct_pooled_codebook_tensor,
)

pooled = compress_product_codebooks(
    quantized_product_tensor,
    target_pool_size=1024,
    distance_threshold=0.01,
)

reconstructed = reconstruct_pooled_codebook_tensor(pooled)
print(pooled.compression_ratio)
```

The pool first considers centroids in descending assignment-frequency order.
A centroid is retained when its root-mean-square L2 distance from all selected
pool vectors exceeds the threshold. If this leaves the pool under-filled, the
remaining most-used centroids fill it. Every original product-codebook centroid
is then represented by an integer index into the shared pool.

## Model-wide use

```python
import keras

from src.quantization.custom_quantization import codebook_vectorize_model
from src.quantization.custom_quantization import reconstruct_codebook_model_weights

report = codebook_vectorize_model(
    model,
    vector_dim=4,
    codebook_size=256,
    mode="product",
    quantize_min_rank=2,
)

print(len(report.tensors))
print(report.compression_ratio)
print(report.memory_reduction_percent)

evaluation_model = keras.models.clone_model(model)
evaluation_model.set_weights(reconstruct_codebook_model_weights(model, report))
```

This function does not mutate the model. As in the existing custom PTQ path,
rank-1 bias and normalization tensors remain unchanged and are included in the
total storage calculation.

## Deliberate limitations

The following paper components are not claimed by this generic data-free PTQ
module:

- DPQ's activation-aware assignment update minimizing
  `||W x - W_hat x||²`.
- DPQ's backward codebook optimization with the diffusion/DDPM loss.
- VQ4ALL's differentiable top-n soft assignments and ratio optimization.
- VQ4ALL's task loss, block-wise distillation, ratio regularizer, and
  Progressive Network Construction.
- A hardware kernel that multiplies directly from packed codebook indices.

Those methods require task-specific calibration or training loops. The current
module implements the reusable mathematical weight vectorization, codebook
construction, hard assignment, pooling, reconstruction, and memory analysis
needed before such calibration is added.
