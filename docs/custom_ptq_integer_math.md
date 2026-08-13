# Custom PTQ integer mathematics

The custom PTQ implementation follows three supplied references:

- Jacob et al., *Quantization and Training of Neural Networks for Efficient
  Integer-Arithmetic-Only Inference* (arXiv:1712.05877)
- Krishnamoorthi, *Quantizing deep convolutional networks for efficient
  inference* (arXiv:1806.08342)
- Nagel et al., *A White Paper on Neural Network Quantization*
  (arXiv:2106.08295)

## Quantization scheme

For a real tensor `r`, scale `S`, integer zero-point `Z`, and integer tensor
`q`, the affine mapping is

```text
r = S (q - Z)
q = clamp(round(r / S) + Z, qmin, qmax)
```

Weights use signed symmetric quantization (`Z = 0`) with one scale per output
channel. Activations use one asymmetric UINT8 quantizer per tensor, calibrated
from representative data. Real zero is always included in the activation range
so padding and ReLU zero remain exact.

## Integer layer execution

Dense and convolution kernels multiply centered 8-bit operands and accumulate
into INT32:

```text
acc = sum((q_input - Z_input) * q_weight) + q_bias
q_bias = round(bias / (S_input * S_weight))
```

For per-channel weights, `S_weight`, `q_bias`, and the accumulator scale are
vectors indexed by output channel. The accumulator is converted to the next
8-bit activation using

```text
M = (S_input * S_weight) / S_output
q_output = clamp(Z_output + round(M * acc), qmin, qmax)
```

`M` is computed during conversion/calibration and encoded as a Q31 integer
multiplier plus a power-of-two shift. Runtime requantization uses integer
multiplication, rounded bit shifts, zero-point addition, and saturation. It does
not convert the accumulator to floating point.

The `output_dequantized` field returned by the reference layer functions is a
diagnostic view for error measurement only. It is not fed into a subsequent
layer. The custom path does not insert simulated-quantization operations into
the Keras graph.
