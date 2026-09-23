# Speaker notes — 20 minutes

The deck has 17 content slides plus the title and questions slides. The target below totals about 20 minutes. The slides use short labels and diagrams; these notes carry the explanation that should be spoken rather than printed.

## 1. Title — 0:30

- State the question: how do uniform PTQ and codebook quantization behave from 2 to 8 bits on three trained checkpoints?
- Name the checkpoints: ResNet-18, DistilBERT and ViT-B/16.
- Say that the conclusions are specific to these checkpoints and this pipeline.

## 2. Why compress neural networks? — 1:00

- FP32 gives each weight 32 bits, so model parameters dominate storage for these experiments.
- Edge deployment creates memory, bandwidth, energy and sometimes latency constraints.
- Quantization stores each selected weight with fewer bits. The scientific question is where the accuracy/storage trade-off changes sharply.

## 3. Neural network: what and how — 1:05

- A layer multiplies inputs by learned weights, adds a bias and applies a non-linearity.
- During training, the loss updates weights. During evaluation, those weights are fixed.
- In this study only the stored weights are quantized. Activations and the reconstructed evaluation model remain floating point.

## 4. CNN architecture — 1:00

- Convolutional filters reuse small local kernels across the image, building edges, textures and higher-level shapes.
- ResNet skip connections allow features to bypass a block and help train deep networks.
- The first convolution is special because every later stage consumes its output; an error there can propagate through the whole network.

## 5. Transformer architecture — 1:05

- Text is split into tokens; ViT splits an image into patches. Both become embeddings with position information.
- Attention lets each token combine information from other tokens. Feed-forward layers then transform each token.
- Residual paths and normalization surround these operations. DistilBERT and ViT share this block pattern but solve different tasks.

## 6. Quantization — 0:55

- The blue points represent many FP32 weight values. The orange marks are a small set of representable values.
- Quantization projects each weight to one of these values. Lower bits reduce storage but increase projection error.
- The effect on accuracy depends on both the error and where it occurs in the network.

## 7. PTQ versus codebook — 1:10

- Uniform PTQ uses a regular signed grid and a separate scale per output channel.
- Codebook quantization learns tensor-wide centroids with fixed-seed k-means++ and assigns every weight to its nearest centroid.
- This is a comparison of full configurations. Besides uniform versus learned placement, granularity and the number of levels differ.

## 8. Implementation — 1:10

- Start from the trained FP32 checkpoint and select rank-two-or-higher weight tensors.
- Quantize the selected tensors at one bit width, reconstruct them as floating-point values, insert them at the same indices and copy every other tensor unchanged.
- Codebook fitting uses up to 100,000 sampled values, one initialization, 30 iterations, tolerance (10^{-6}), and a seed tied to the tensor index.
- This simulated evaluation measures accuracy and storage accounting. It does not measure native low-bit speed.

## 9. Checkpoints and evaluation settings — 1:05

- Give the task, evaluation-set size, parameter count and FP32 baseline for each model.
- The ResNet backbone was adapted end-to-end, while the ViT backbone was frozen during adaptation.
- Therefore, differences cannot be assigned to architecture alone; training procedure and checkpoint quality are also present.

## 10. Experiment map — 1:10

- Experiment 1 establishes the FP32 baseline. Experiment 2 checks W8 custom quantization against the framework path.
- Experiment 3 sweeps the full model from W2 to W8 for both methods.
- Experiment 4 replaces one tensor or selected ViT module at a time at W8 and W4.
- Experiment 5 compares W4 and W8 sensitivity ranks. Experiment 6 explores selective protection for the ResNet W4 failure.
- Mention the tensor counts shown on the slide.

## 11. Metrics — 0:55

- Accuracy change is reported in percentage points relative to FP32.
- Compression ratio and memory reduction are two mathematically linked views of the same bit-width accounting.
- Saved paired predictions support exact McNemar tests; sensitivity ranks support Spearman correlation with permutation inference.

## 12. Full-model results — 1:30

- Read by row: ResNet-18, DistilBERT, ViT-B/16. The left column is accuracy and the right is compression.
- W8 remains close to baseline for every checkpoint; the largest custom drop is 0.59 percentage points.
- At W4, DistilBERT and ViT stay within 0.6 points, while this ResNet checkpoint collapses.
- Compression follows the bit width, so the scientific difference is in retained accuracy.

## 13. Low-bit comparison — 1:10

- ResNet W4 is poor under both methods.
- DistilBERT gains strongly from codebook quantization at W2 and W3.
- ViT shows the largest method difference at W3: 89.77% codebook versus 32.22% PTQ, a 57.55-point observation.
- Treat centroid adaptation as a plausible hypothesis, not a proven cause.

## 14. ResNet sensitivity — 1:10

- The full-range panel exposes the first-layer collapse; the detail panel keeps the other kernels legible.
- Quantizing `conv1` alone at W4 loses 60.65 points with PTQ and 78.62 points with codebook.
- This localizes the failure to early rounding sensitivity and forward error propagation.
- The study does not prove that ResNet or CNNs generally fail at W4, and it does not include the proposed `conv1`-at-W8 mixed-precision ablation.

## 15. Transformer sensitivity — 1:05

- Colors are centered at zero: blue means a decrease, orange means an increase relative to the same FP32 checkpoint.
- Model-specific limits are labeled because DistilBERT changes are much smaller than ResNet changes.
- Sensitivity is distributed rather than dominated by one catastrophic tensor in the selected transformer sets.

## 16. Cross-bit statistics — 1:00

- DistilBERT codebook sensitivity ranks are positively related across W4 and W8: rho 0.693 with permutation p 0.00551.
- The accuracy-only robustness calculation gives a similar result.
- Exact McNemar tests have a resolution floor: with very few discordant predictions, a small p-value is arithmetically impossible.
- Therefore, zero significant DistilBERT tensors is not evidence that the model is insensitive.

## 17. Comparison boundary — 1:05

- Use the rows to compare the three observed checkpoint behaviors.
- W8 is consistently benign; lower-bit behavior and the benefit of a codebook vary sharply.
- Training, checkpoint and quantizer configuration all change together. The data do not isolate architecture as the cause.

## 18. Conclusions — 0:55

- W8 is the conservative starting point in this study.
- Low-bit deployment needs checkpoint-specific sweeps and tensor-level diagnosis.
- Codebooks can recover large W2–W3 accuracy gaps, but not every collapse.
- Native low-bit hardware measurements and causal ablations remain future work.

## 19. Questions — 0:55

- Return to the practical message: measure the checkpoint, find the breakpoint, and protect the tensors that dominate error.
- Likely questions: why ViT was frozen, why the quantizers use different granularity, what McNemar can resolve on 872 examples, and whether `conv1` protection recovers ResNet W4.
