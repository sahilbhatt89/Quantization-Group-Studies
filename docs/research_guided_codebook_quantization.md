# Research-guided codebook quantization

The improved N-bit path is implemented in
`src/quantization/custom_quantization/codebook_vector_quantization.py` and used
by the completed ResNet sweep in Notebook 26.

## Paper-to-code mapping

### Gong et al., *Compressing Deep Convolutional Networks Using Vector Quantization*

- The N-bit CNN sweep uses learned k-means codebooks for model parameters.
- The paper reports that scalar k-means is a strong CNN baseline, while product
  and residual quantization can exploit additional structure.
- Notebook 26 uses scalar codebooks because this makes `num_bits`
  equal to bits per weight and was substantially more accurate for this
  ResNet-18 than the earlier two-weight codebook.

### Choi et al., *Towards the Limit of Network Quantization*

- `estimate_gradient_importance()` computes a square-root gradient
  second-moment proxy from representative labeled samples.
- `fit_importance_weighted_kmeans_codebook()` minimizes component-wise
  sensitivity-weighted distortion rather than treating every error equally.
- The proxy is normalized, bounded, and blended with uniform weights. The
  bounding is an engineering stabilization: a small calibration set does not
  provide the exact diagonal Hessian used in the paper.

### Jégou et al., *Product Quantization for Nearest Neighbor Search*

- `product_quantize_array()` and
  `importance_weighted_product_quantize_array()` learn independent subspace
  codebooks whose Cartesian product represents a much larger implicit
  codebook.
- Product mode remains available through `mode="product"`.
- It is not the final configuration in Notebook 26 because an `n`-bit assignment for a
  two-weight subvector costs only `n / 2` assignment bits per weight. That
  earlier sweep was therefore not comparable to N-bit scalar PTQ and measured
  poorly on this CNN.

## N-bit definition

`codebook_vectorize_model_n_bits(model, num_bits=n, vector_dim=d)` derives:

```text
K = 2**n
assignment bits per scalar = n / d
```

Notebook 26 sets `d = 1`, so `n` is the actual index width per quantized
weight. Codebook values and preserved tensors add overhead beyond the packed
assignment stream.

## Evaluation limitation

Assignments and codebooks are reconstructed into floating-point Keras weights
for accuracy evaluation. This measures approximation accuracy and theoretical
packed parameter storage; it is not a native indexed-codebook inference
runtime. Activations remain FP32. Quantization is simulated by reconstructing weights before evaluation, without quantization-aware training or inserted fake-quantization nodes.
