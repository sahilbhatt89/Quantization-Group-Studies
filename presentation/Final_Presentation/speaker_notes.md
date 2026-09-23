# Speaker notes — 20 minutes

The main deck is paced for 20 minutes; numerical tables and the BatchNorm mechanism slide can be treated as supporting material when time is tight. These notes carry the explanation that should be spoken rather than printed.

## 1. Title — 0:30

- State the question: how do uniform PTQ and codebook quantization behave from 2 to 8 bits on three trained checkpoints?
- Name the checkpoints: ResNet-18, DistilBERT and ViT-B/16.
- Say that the conclusions are specific to these checkpoints and this pipeline.

## 2. Research questions and contribution — 0:40

- Introduce all four research questions before referring to RQ4 later.
- State the concrete contribution: two custom post-training weight quantizers, 42 full-model sweep configurations, and W4/W8 isolated sensitivity studies.
- Emphasize that this compares complete checkpoint–quantizer configurations rather than isolating architecture.

## 3. What is a neural network? — 0:45

- A neural network learns weighted connections that transform input data into predictions.
- Hidden layers build intermediate features. Training adjusts the connection weights to reduce prediction error.

## 4. Information flow and quantization — 0:55

- In the FP32 path, each layer transforms the preceding activations using learned weights.
- Quantization replaces those weights with nearby low-bit values. The resulting perturbations can accumulate through later layers and alter the final class.
- Activations remain floating point in this study.

## 5. CNN architecture — 0:50

- Convolutional filters reuse small local kernels across the image, building edges, textures and higher-level shapes.
- ResNet skip connections allow features to bypass a block and help train deep networks.
- The first convolution is special because every later stage consumes its output; an error there can propagate through the whole network.

## 6. Transformer architecture — 0:50

- Text is split into tokens; ViT splits an image into patches. Both become embeddings with position information.
- Attention lets each token combine information from other tokens. Feed-forward layers then transform each token.
- Residual paths and normalization surround these operations. The two variants here are DistilBERT for text and ViT-B/16 for images.

## 7. DistilBERT — 0:50

- DistilBERT is a smaller model distilled from BERT. It uses six encoder blocks to form contextual text representations.
- The classifier uses the sentence-level representation for SST-2 sentiment prediction.
- The evaluated checkpoint has 88.42% FP32 accuracy on 872 examples.

## 8. ViT-B/16 — 0:50

- ViT divides an image into 16-by-16 patches and treats the projected patches as tokens.
- Twelve encoder blocks exchange information across patches; the class token feeds the image classifier.
- This checkpoint reaches 95.73% FP32 accuracy on 10,000 CIFAR-10 images, with its backbone frozen during adaptation.

## 9. What is quantization? — 0:35

- Quantization represents high-precision values using a smaller discrete set of low-bit values.
- It reduces weight storage and memory traffic, while rounding error can reduce model accuracy.
- This study quantizes weights after training and keeps activations in FP32.

## 10. Quantization diagram — 0:35

- The blue points represent many FP32 weight values. The orange marks are a small set of representable values.
- Quantization projects each weight to one of these values. Lower bits reduce storage but increase projection error.
- The effect on accuracy depends on both the error and where it occurs in the network.

## 11. Uniform PTQ definition — 0:45

- Uniform PTQ maps trained FP32 weights to evenly spaced low-bit values without retraining.
- The scale controls the distance between levels; rounding produces the stored integer and multiplying by the scale reconstructs the evaluated weight.
- Clipping keeps the integer within the signed range.

## 12. Uniform PTQ implementation — 0:55

- Select each rank-two-or-higher weight tensor and process it by output channel.
- For every channel, derive the symmetric scale from its largest absolute weight and the signed integer range.
- Divide, round, clip and reconstruct the weights, then evaluate the model with FP32 activations.
- Repeat this procedure from W2 through W8.

## 13. Scalar codebook quantization — 0:50

- Codebook quantization learns representative centroid values from each tensor's weight distribution.
- Every weight stores the index of its nearest centroid; reconstruction replaces the index with that centroid.
- The study uses scalar values, one codebook per tensor and (2^b) centroids.

## 14. Codebook implementation — 0:55

- Sample at most 100,000 values, use fixed-seed k-means++ initialization and refine the centroids with Lloyd updates.
- Use one initialization, at most 30 iterations and tolerance (10^{-6}).
- Assign every tensor weight to its nearest learned centroid, reconstruct the tensor in FP32 and evaluate without retraining.
- The seed is 42 plus the weight index; every unselected tensor is copied unchanged.

## 15. Tensor-selection methodology — 0:55

- PTQ and codebook quantization use the same selected tensors within every experiment.
- Every eligible tensor must be floating point and rank two or higher.
- ResNet selects all 21 eligible kernels. DistilBERT selects 40 embedding or kernel tensors. ViT selects 76 embedding or kernel tensors plus the class token.
- For sensitivity analysis, ViT's 76 tensors are grouped into 14 logical modules; seven predefined modules are tested at W4.
- No biases are selected. Rank-one biases, normalization vectors and rank-two transformer attention biases remain FP32.

## 16. Checkpoints and datasets — 0:45

- Give the dataset, evaluation-set size, parameter count, FP32 storage and baseline accuracy for each checkpoint.
- Explain that FP32 parameter storage is approximately four bytes per parameter.
- Every quantized result is measured against the FP32 baseline of the same saved checkpoint.
- State the training confound before the results: ResNet was adapted end-to-end, DistilBERT was fine-tuned, and the ViT backbone was frozen while its head was trained.

## 17. Experiment map — 0:55

- Experiment 1 establishes the FP32 baseline. Experiment 2 checks W8 custom quantization against the framework path.
- Experiment 3 sweeps the full model from W2 to W8 for both methods.
- Experiment 4 quantizes one analysis unit at a time: a ResNet kernel, DistilBERT tensor or ViT logical module.
- Experiment 5 compares W4 and W8 sensitivity ranks and answers RQ4.
- Mention the tensor counts shown on the slide.

## 18. FP32 and built-in baselines — 0:55

- First evaluate each saved checkpoint unchanged on the complete labelled split; this establishes the FP32 reference.
- Then apply the framework-provided W8 path as a software validation baseline.
- ResNet uses executable W8A8 full INT8. DistilBERT and ViT use TFLite dynamic-range quantization; do not claim that all internal activation arithmetic remains FP32.
- These paths are not numerically identical to the custom weight-only methods.

## 19. Custom PTQ full sweep — 0:55

- Read each model as accuracy, compression ratio and parameter-memory reduction from W2 through W8.
- Compression is governed mainly by the shared bit width.
- DistilBERT and ViT recover close to their FP32 baselines at W4; this ResNet checkpoint recovers much later, near W7.

## 20. Custom codebook full sweep — 0:55

- The table uses the same accuracy, compression and reduction columns for the learned codebook configuration.
- DistilBERT retains 78.33% at W2, and ViT retains 89.77% at W3.
- ResNet remains severely degraded through W6 and returns near baseline at W7 and W8.

## 21. PTQ accuracy graph — 0:45

- The dashed horizontal lines show each checkpoint's FP32 reference.
- DistilBERT and ViT return close to baseline at W4, while ResNet remains near chance through W4 and recovers near W7.
- The W4 callout makes the checkpoint-specific breakpoint visible.

## 22. Codebook accuracy comparison — 0:45

- Codebook quantization retains 78.33% for DistilBERT at W2 and 89.77% for ViT at W3.
- ResNet still collapses at W4 and recovers near W7.
- The largest method difference is ViT W3: codebook exceeds custom PTQ by 57.55 percentage points.
- This is a comparison of complete configurations: codebook is per-tensor with $2^b$ centroids, while PTQ is per-channel with $2^b-1$ levels.

## 23. Accuracy and storage trade-off — 0:40

- Read the three accuracy panels by model; these curves carry the model- and method-specific result.
- The compact table retains the storage result without repeating six near-identical plots.
- Memory reduction and compression are mathematically linked by $S=100(1-1/R)$, so they are two views of the same storage result.
- These are estimated packed parameter sizes; accuracy was evaluated with FP32-reconstructed weights rather than native packed execution.

## 24. Layer-by-layer comparison — 0:55

- Each run starts from the original FP32 checkpoint and quantizes exactly one analysis unit; every other weight stays FP32.
- ResNet uses individual kernels, DistilBERT uses individual tensors, and ViT groups tensors into logical modules.
- Sensitivity is the FP32 accuracy minus the isolated-run accuracy. Positive values are losses; negative values are observed gains.
- The transformer W4 studies cover predefined architectural subsets, while ResNet is exhaustive.

## 25. ResNet-18 layer sensitivity — 1:05

- Red means loss and blue means an observed gain. The report heatmap uses a common $\pm80$-point scale, so the smaller effects appear pale.
- The three largest W4 losses are the first convolution, stack 0 block 0 conv 1, and stack 0 block 1 conv 1.
- Quantizing the first convolution alone loses 78.62 points with codebook and 60.65 points with PTQ.
- The result localizes the W4 failure to early kernels; it does not establish a general CNN limitation.

## 26. DistilBERT tensor sensitivity — 0:55

- The displayed W4 comparison covers 15 predefined tensors from the input, early, middle, late and output regions.
- Three codebook cases tie at a 0.229-point loss: layer 2 intermediate FFN, layer 0 attention output, and layer 2 output FFN. Prediction disagreement orders the tie.
- The effects are small and distributed, but zero FDR rejections does not prove equivalence because the exact McNemar test has limited resolution on 872 examples.

## 27. ViT-B/16 module sensitivity — 0:55

- The displayed W4 comparison covers seven predefined modules.
- Transformer block 1 has the largest sampled loss at 0.20 points with codebook, followed by block 6 and block 3.
- Every sampled W4 effect remains within plus or minus 0.20 points, and none passes the FDR threshold.
- The overall analysis shows that W8 is robust for all three checkpoints, while lower-bit behavior depends strongly on the checkpoint and complete quantization configuration.
- ResNet's W4 failure is concentrated in early kernels; the sampled DistilBERT and ViT units remain close to FP32. These results do not isolate architecture as the cause.

## 28. Experiment 5: rank stability — 1:00

- This answers RQ4: whether sensitivity measured at W8 predicts sensitivity at W4.
- The table reports Spearman rank correlations and two-sided permutation p-values for both methods and all three checkpoints.
- The ranks use accuracy degradation and prediction disagreement as the tie-breaker.
- DistilBERT codebook is the only significant positive association: rho 0.693 and permutation p 0.00551; it remains below 0.05 after a six-test Bonferroni correction.
- The other five results provide insufficient evidence of transfer. They do not prove that no relationship exists.
- General W8-to-W4 transfer was not established; the positive association is limited to the tested DistilBERT codebook subset.

## 29. Comparison boundary — 0:50

- Use the rows to compare the three observed checkpoint behaviors.
- W8 is consistently benign; lower-bit behavior and the benefit of a codebook vary sharply.
- Training, checkpoint and quantizer configuration all change together. The data do not isolate architecture as the cause.
- Headline numbers: custom W8 loses at most 0.59 points with about four-times estimated compression; the two transformer W4 results stay within 0.6 points with about 7.9-times estimated compression.

## 30. Conclusions — 0:40

- W8 is the conservative starting point in this study.
- Low-bit deployment needs checkpoint-specific sweeps and tensor-level diagnosis.
- Selected codebook configurations retain more W2–W3 accuracy, but the methods also differ in granularity and level count.
- Native low-bit hardware measurements and causal ablations remain future work.

## 31. Selected references — 0:10

- Do not read the references aloud; use this slide to show that the architecture and quantization definitions are sourced.

## 32. Questions — 0:20

- Return to the practical message: measure the checkpoint, find its breakpoint, and use layer sensitivity to nominate mixed-precision candidates for future validation.
- Likely questions: why ViT was frozen, why the quantizers use different granularity, what McNemar can resolve on 872 examples, and whether BatchNorm explains the ResNet W4 collapse.

## 33. Backup: BatchNorm hypothesis — Q\&A only

- The recorded runs retain saved FP32 BatchNorm parameters and moving statistics and evaluate with `training=False`.
- No folding, recalibration, bias correction, equalization or adaptive rounding was performed.
- A mismatch between quantized early-layer activations and saved statistics is plausible, but the study did not isolate or test that mechanism.
- The measured result is the location of the failure: the first convolution is exceptionally sensitive at W4.
