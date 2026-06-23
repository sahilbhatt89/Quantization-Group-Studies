# Project Scope

## Architecture-Aware Quantization Study

This project studies post-training quantization across pretrained neural network architectures using TensorFlow/Keras.

The final study compares quantization behavior across three model families:

1. CNN-based image classification models
2. Vision Transformer-based image classification models
3. NLP Transformer-based text classification models

The planned final scope is 9 pretrained models total:

- 3 CNN/image models
- 3 Vision Transformer models
- 3 NLP Transformer models

The current implementation starts with a pretrained ResNet-18 model as the first CNN baseline.

## Quantization Scope

Based on supervisor feedback, the project includes both weight quantization and activation quantization.

### Weight Quantization

Weight quantization converts model parameters from FP32 to 8-bit values.

This is used to measure:

- Model-size reduction
- Weight memory reduction
- Compression ratio for stored model parameters

### Activation Quantization

Activation quantization measures or simulates 8-bit quantization of intermediate layer outputs during inference.

This is included because intermediate activations are important for hardware implementation. Even if model weights are compressed, hardware memory pressure can still be strongly affected by activation tensors produced between layers.

Activation quantization is reported separately from weight quantization.

## Output Criteria

The project reports two final criteria separately for weights and intermediate activations:

1. M/M reduction
2. Compression

Accuracy, F1-score, latency, and throughput may be used as sanity checks, but they are not the final output criteria.

## Metrics

### M/M Reduction

```text
M/M Reduction (%) = (1 - Quantized Size / FP32 Size) * 100
```

For weights:

```text
Weight M/M Reduction (%) = (1 - Quantized Weight Size / FP32 Weight Size) * 100
```

For activations:

```text
Activation M/M Reduction (%) = (1 - Quantized Activation Size / FP32 Activation Size) * 100
```

### Compression Ratio

```text
Compression Ratio = FP32 Size / Quantized Size
```

For weights:

```text
Weight Compression Ratio = FP32 Weight Size / Quantized Weight Size
```

For activations:

```text
Activation Compression Ratio = FP32 Activation Size / Quantized Activation Size
```

## Learning Baseline

The project may first use built-in TensorFlow/TFLite post-training quantization as a learning and reference baseline.

The main custom implementation will then quantize weights and activations manually so the project can compare:

1. FP32 baseline
2. Built-in PTQ baseline
3. Custom 8-bit quantization
