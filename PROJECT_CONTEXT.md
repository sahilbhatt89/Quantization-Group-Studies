# Quantization project context

Last checked against the repository: 7 September 2026.

Report revision, 20 September 2026: both report sources now use checkpoint-specific claims, disclose training/granularity/level-count confounds, document fixed-seed codebook settings and the unchanged BatchNorm policy, and explicitly leave the ResNet W4 collapse unexplained. The custom evaluation is simulated weight quantization through FP32 reconstruction without QAT. Choi and exact-binomial McNemar references are corrected; transformer PTQ context is added. The duplicate baseline table and redundant overview storage column are removed. Notebooks 20, 21 and 27 regenerate zero-centered heatmaps from saved measurements (ResNet ±80 pp; both transformers ±0.35 pp). Notebook 25 plots accuracy and compression. `scripts/refresh_report_notebook_plots.py` executes only saved-data plotting cells; `scripts/sync_report_notebook_figures.py --check` verifies exact notebook image copies and supporting tables. Measurements and permutation p-values are unchanged. See `report/REVISION_NOTES.md` and `report/README.md`.

This document provides a project handoff and a reference for explaining the work to the supervisor. It describes the reported experiments, distinguishes optional implementation features from evaluated methods, and records outstanding discussion points. Numerical results below are the values reported in the repository report; creating this document did not rerun inference.

## Purpose and scope

The project studies weight quantization in DistilBERT and ViT-B/16, with ResNet-18 as a CNN reference. It compares custom symmetric per-channel post-training quantization (PTQ) with custom scalar codebook quantization across two to eight bits. It also measures isolated component sensitivity at W8 and W4 and compares sensitivity rankings across those precisions.

The implementations are research-informed project code. The project does not claim a new quantization algorithm. Task-specific model training precedes quantization; the reported custom experiments simulate weight quantization through floating-point reconstruction with FP32 activations and use no quantization-aware training.

Some introductory text in `README.md` and `docs/project_scope.md` describes an earlier, broader plan involving custom activation quantization. That plan must not be treated as evidence that activations were quantized in the final custom accuracy experiments.

## Models and evaluation

| Model | Evaluation split | Examples | FP32 parameters | FP32 accuracy |
|---|---|---:|---:|---:|
| ResNet-18 | CIFAR-10 test | 10,000 | 11,191,242 | 90.47% |
| DistilBERT | SST-2 labelled development | 872 | 66,955,010 | 88.42% |
| ViT-B/16 | CIFAR-10 test | 10,000 | 85,806,346 | 95.73% |

ResNet uses the KerasHub `resnet_18_imagenet` preset with a CIFAR-10 classifier. DistilBERT uses `distil_bert_base_en_uncased` with a sentiment classifier and sequence length 128. ViT uses `vit_base_patch16_224_imagenet`; its backbone remained frozen while the CIFAR-10 head was trained.

CIFAR-10 images are resized from 32 × 32 to 224 × 224 and normalized with the model's preprocessing. Resizing changes spatial resolution; it is not an 8-bit quantization operation. Test images receive no augmentation.

Each quantized configuration starts from the same saved FP32 checkpoint for that model. Compare degradation against that checkpoint, rather than comparing absolute accuracies across different tasks. One SST-2 example corresponds to approximately 0.115 percentage points (pp).

## Implementation and execution precision

| Path | Role |
|---|---|
| [custom_ptq.py](src/quantization/custom_quantization/custom_ptq.py) | Custom PTQ used for final results |
| [codebook_vector_quantization.py](src/quantization/custom_quantization/codebook_vector_quantization.py) | Custom scalar codebook path and optional vector/product features |
| [builtin_ptq.py](src/quantization/inbuilt_quantization/builtin_ptq.py) | Separate TensorFlow Lite comparisons |
| [src/models](src/models) | Model loading and construction |
| [src/evaluation](src/evaluation) | TFLite evaluation utilities |

For the reported custom experiments, quantized weights are reconstructed into floating-point arrays and installed in a Keras evaluation model. Keras executes floating-point operators with FP32 activations. W4 or W8 describes the intended parameter representation, not native low-bit execution during these evaluations.

The built-in ResNet full-integer TFLite baseline is described as W8A8: both weights and activations are quantized. The built-in transformer dynamic-range baselines retain FP32 activations. Do not label the custom experiments W8A8.

Optional activation-range calibration in `custom_ptq.py` records calibration parameters; it is not used in the reported custom accuracy sweeps. The unused `advanced_ptq.py` implementation and Notebook 05 were removed from the working tree. Final results use `custom_ptq.py`.

### Custom PTQ mathematics

For the reported bit widths b = 2, ..., 8:

```text
Q = 2^(b - 1) - 1
scale[channel] = max(abs(weights[channel])) / Q
q = clip(round(weight / scale[channel]), -Q, Q)
reconstructed_weight = scale[channel] * q
```

The last tensor axis supplies the channel dimension. All-zero channels use a safe nonzero scale and reconstruct to zero. The zero point is zero. At W8 the integer range is [-127, 127]; at W4 it is [-7, 7]; at W2 it is [-1, 1]. Thus the symmetric grid uses 2^b - 1 levels, whereas the codebook uses 2^b centroids.

Low-bit PTQ values remain in NumPy INT8 arrays during processing. Reported sizes account for an estimated packed b-bit representation and FP32 scale metadata, rather than treating those working arrays as physically packed storage.

### Custom codebook mathematics

The main experiments use scalar groups (`vector_dim=1`), with K = 2^b centroids per selected tensor. Fixed-seed k-means++ initialization and up to 30 Lloyd updates fit the centroids. Each tensor uses all values up to 100,000, or 100,000 sampled without replacement above that threshold. One initialization is used; the maximum centroid-displacement tolerance is 1e-6. The seed is 42 plus the tensor index in the complete model.weights list. Sampling and initialization each create a generator with that seed. Nearest-centroid assignment covers every selected value.

```text
assignment[i] = index of the centroid nearest to weight[i]
reconstructed_weight[i] = codebook[assignment[i]]
```

Storage accounting includes b-bit assignments and FP32 centroids. Optional vector/product quantization, importance weighting and shared-codebook features are not the main reported scalar-codebook experiments.

### Tensor selection

Both custom methods use the same eligible tensors within an experiment:

| Model | Selection | Eligible tensors |
|---|---|---:|
| ResNet-18 | Floating-point tensors of rank at least two | 21 |
| DistilBERT | Rank at least two, path ending `/kernel` or `/embeddings` | 40 |
| ViT-B/16 | Eligible kernels, embeddings and learned class token | 76 |

Biases and normalization parameters remain FP32 under the experimental selection policy. Transformer attention biases remain FP32 even when their implementation gives them rank two.

DistilBERT has six transformer blocks, not 40 transformer layers. Its 40 eligible tensors comprise two embeddings, six kernels per block, a pooled dense kernel and the classifier kernel. ViT's 76 tensors are grouped into 14 logical modules for sensitivity analysis: input embedding, 12 transformer blocks and classifier.

## Notebook map

| Notebook | Purpose |
|---|---|
| 06 | Train/adapt ResNet-18 for CIFAR-10 |
| 07 | Compare trained ResNet FP32 and PTQ accuracy |
| 09 | DistilBERT SST-2 checkpoint and PTQ comparisons |
| 10 | ViT CIFAR-10 checkpoint and PTQ comparisons |
| 11–13 | Scikit-learn versus custom codebook comparisons |
| [17](notebooks/17_resnet18_full_cifar10_ptq_codebook_layer_sensitivity.ipynb) | ResNet exhaustive W8 kernel sensitivity |
| [18](notebooks/18_distilbert_full_sst2_ptq_codebook_layer_sensitivity.ipynb) | DistilBERT exhaustive W8 tensor sensitivity |
| [19](notebooks/19_vit_full_cifar10_ptq_codebook_module_sensitivity.ipynb) | ViT exhaustive W8 module sensitivity |
| [20](notebooks/20_resnet18_full_cifar10_4bit_ptq_codebook_layer_sensitivity.ipynb) | ResNet exhaustive W4 sensitivity and W4/W8 comparison |
| [21](notebooks/21_distilbert_sst2_4bit_stratified_sensitivity_w4_w8.ipynb) | DistilBERT stratified W4 sensitivity and W4/W8 comparison |
| [22](notebooks/22_vit_cifar10_4bit_stratified_sensitivity_w4_w8.ipynb) | ViT stratified W4 sensitivity and W4/W8 comparison |
| [23](notebooks/23_distilbert_sst2_ptq_codebook_2_to_8_bit_sweep.ipynb) | Complete DistilBERT W2–W8 sweep |
| [24](notebooks/24_vit_cifar10_ptq_codebook_2_to_8_bit_sweep.ipynb) | Complete ViT W2–W8 sweep |
| [25](notebooks/25_consolidated_full_model_ptq_codebook_2_to_8_bit_results.ipynb) | Consolidates saved results; performs no inference |
| [26](notebooks/26_resnet18_full_cifar10_ptq_codebook_2_to_8_bit_sweep.ipynb) | Authoritative complete ResNet PTQ/codebook W2–W8 sweep |

Notebook 15 was removed as an unexecuted/superseded codebook sweep. Notebook numbering therefore contains intentional gaps. Renaming notebooks requires checking references in documentation and other notebooks.

## Isolated sensitivity and cross-bit comparison

For each run, start from FP32, quantize exactly one eligible kernel/tensor/module, and evaluate the complete labelled split. All other parameters stay FP32. Repeat separately for PTQ and codebook quantization.

```text
accuracy_degradation_pp = FP32_accuracy_percent - quantized_accuracy_percent
prediction_disagreement_percent = 100 * mean(FP32_prediction != quantized_prediction)
```

Positive degradation means worse measured accuracy; negative degradation means better measured accuracy. Disagreement includes cases where both models are wrong but predict different classes.

Rank units within each method by descending accuracy degradation, then descending prediction disagreement to break ties. Weight MSE/NMSE and other reconstruction errors are supporting metrics, not the main sensitivity ranking criterion.

| Model | W8 units tested per method | W4 units tested per method | Matched cross-bit units |
|---|---:|---:|---:|
| ResNet-18 | 21/21 kernels | 21/21 kernels | 21 |
| DistilBERT | 40/40 tensors | 15/40 tensors | 15 |
| ViT-B/16 | 14/14 modules | 7/14 modules | 7 |

The transformer W4 subsets are fixed lists chosen for architectural coverage. They are neither random samples nor selections of the most/least sensitive W8 units. “Stratified” here means a deliberate architecture-spanning subset, not probability sampling within strata.

### Exact DistilBERT W4 subset

Notebook 21's `STRATIFIED_TENSORS` contains:

| Region (zero-based block numbering) | Tensors | Count |
|---|---|---:|
| Input | Token and position embeddings | 2 |
| Block 0 | Query, attention output, feed-forward output | 3 |
| Block 2 | Query, value, attention output, feed-forward intermediate, feed-forward output | 5 |
| Block 5 | Query, attention output, feed-forward output | 3 |
| Classification head | `pooled_dense/kernel`, `logits/kernel` | 2 |

This totals 15 tensors and 30 isolated W4 evaluations across two methods. It does not mean that all 15 were quantized together in an isolated sensitivity run.

Notebook 22 selects `input_embedding`, `transformer_block_1`, `transformer_block_3`, `transformer_block_6`, `transformer_block_9`, `transformer_block_11`, and `classifier`.

### Cross-bit statistics

Notebooks 20–22 match W4 units to saved W8 results from Notebooks 17–19, respectively, then calculate Spearman correlation between the matched sensitivity ranks. The implemented ranks include the disagreement tie-break; they need not give the same correlation as ranking accuracy degradation alone with averaged ties.

| Model | Method | Reported Spearman rho | Reported p |
|---|---|---:|---:|
| ResNet-18 | PTQ | -0.094 | 0.687 |
| ResNet-18 | Codebook | -0.292 | 0.199 |
| DistilBERT | PTQ | 0.143 | 0.612 |
| DistilBERT | Codebook | 0.693 | 0.004 |
| ViT-B/16 | PTQ | -0.321 | 0.482 |
| ViT-B/16 | Codebook | -0.357 | 0.432 |

Only DistilBERT codebook has a significant positive relationship at the reported 0.05 threshold. Nonsignificant results do not prove that no relationship exists, particularly for small subsets. These correlation p-values are distinct from the FDR-adjusted McNemar values used for individual sensitivity tests.

Pearson correlations elsewhere compare numerical reconstruction errors with model responses. Pearson measures linear association; Spearman measures rank association. Those error-versus-response tables answer a different question from W4/W8 rank stability.

## Bootstrap and McNemar interpretation

Let CC mean both predictions are correct, CW mean only FP32 is correct, WC mean only the quantized model is correct, and WW mean both are wrong.

```text
accuracy_drop_pp = 100 * (CW - WC) / N
```

The paired bootstrap uses the per-example correctness differences [-1, 0, +1], with counts [WC, CC + WW, CW]. It draws N outcomes from the empirical multinomial distribution 2,000 times, calculates the drop for each draw and uses its 2.5th and 97.5th percentiles as the 95% confidence interval. This resamples the observed paired outcomes; it does not retrain or rerun the model 2,000 times.

An interval [-0.05, 0.17] pp spans a possible 0.05 pp improvement to a 0.17 pp degradation. It includes zero, so the bootstrap interval does not establish a directional change. It measures evaluation-sample uncertainty conditional on the fitted models, not variability across training seeds.

The exact two-sided McNemar p-value comes from a binomial test of CW successes in CW + WC discordant pairs under probability 0.5. A continuity-corrected McNemar statistic is also stored, but the stored exact p-value is computed from the binomial test, not from that statistic.

Benjamini–Hochberg adjusts the McNemar p-values across units within each method. An adjusted p below 0.05 is the study's significance criterion; inspect CW versus WC or the sign of degradation for direction. A value such as 0.632791 is not significant and does not mean a 63% probability that the model is unchanged. Neither significance nor nonsignificance gives the size of the effect; inspect the accuracy drop and interval as well.

## Main reported results and limitations

| Model | FP32 accuracy | W4 PTQ | W4 codebook | W8 PTQ | W8 codebook |
|---|---:|---:|---:|---:|---:|
| ResNet-18 | 90.47% | 11.56% | 8.43% | 89.88% | 90.25% |
| DistilBERT | 88.42% | 88.99% | 88.30% | 88.42% | 88.07% |
| ViT-B/16 | 95.73% | 95.23% | 95.20% | 95.79% | 95.73% |

All 42 sweep configurations are reported as measured: three models × two methods × seven bit widths. Each bit width is generated directly from FP32. W8 reduces estimated parameter storage by about 75%; W4 gives about 7.9× compression. Quantizing only ResNet's first convolution at W4 causes losses of 60.65 pp with PTQ and 78.62 pp with codebook quantization.

Compression is FP32 parameter bytes divided by estimated quantized representation bytes. Reduction is `100 * (1 - quantized_bytes / FP32_bytes)`. Include preserved FP32 parameters and scales/codebooks in the latter representation. These measurements do not establish custom inference latency, throughput, energy or peak runtime-memory improvements.

The architecture comparison applies to these checkpoints, selection policies and datasets. Different pretraining/adaptation procedures, the small SST-2 split and partial transformer W4 coverage limit generalization. Small observed accuracy increases are not evidence of improved generalization.

## Research foundations as attributed by the project

The report bibliography and source comments provide the detailed references. This mapping describes their role in the project rather than claiming complete reproductions of their algorithms.

| Reference | Role |
|---|---|
| Jacob et al. (2018) | Affine quantization framework; custom weights use its symmetric zero-point-zero form |
| Krishnamoorthi (2018) | Per-channel weight quantization and optional calibration background |
| Nagel et al. (2021) | PTQ framework and interpretation of low-bit failure |
| Arthur and Vassilvitskii (2007); Lloyd (1982) | Codebook initialization and centroid updates |
| Gray (1984); Gong et al. (2014); Han et al. (2016) | Quantization, neural-network compression and weight-sharing foundations |
| Jégou et al. (2011) | Optional product-quantization capabilities |
| Choi et al. (2017) | Sensitivity-weighted distortion motivation and optional importance weighting |
| DPQ; VQ4ALL | Optional codebook capabilities; their full methods are not the final evaluated path |
| Dietterich (1998) | Paired classifier-comparison background, not the original source of McNemar's test |
| Benjamini and Hochberg (1995) | FDR correction of per-unit McNemar tests |

The one-unit-at-a-time accuracy ablation is the project's empirical sensitivity protocol. Do not describe it as a direct implementation of Choi's sensitivity-weighted quantization algorithm.

## Saved data, dependencies and report files

Candidate tables hold tensor metadata and indices; the checkpoint and `baseline_weights` hold actual FP32 arrays. Notebook caches hold reconstructed weights and predictions, and artifact directories hold CSV results, figures and run metadata.

Sensitivity output directories are `artifacts/resnet18_full_layer_sensitivity`, `artifacts/distilbert_full_layer_sensitivity`, `artifacts/vit_full_layer_sensitivity`, `artifacts/resnet18_4bit_layer_sensitivity`, `artifacts/distilbert_4bit_stratified_sensitivity`, and `artifacts/vit_4bit_stratified_sensitivity`.

Some notebooks also contain exploratory sensitivity-guided protection and random-control experiments that are not reported conclusions. Before deleting these cells, account for downstream dependencies: Notebooks 20 and 21 read W8 full-model rows from `selective_quantization_all_configurations.csv`, and Notebook 22 reads `module_selective_quantization.csv`.

The report package contains [quantization_study_reference.tex](report/quantization_study_reference.tex), [Quantization_Report_with_page_numbers.tex](report/Quantization_Report_with_page_numbers.tex) and [images](report/images). Both TeX files currently enable page numbers. The two TeX sources and their compiled PDFs are synchronized.

Setup is documented in [README.md](README.md), with dependencies in [requirements.txt](requirements.txt). Experiments require their saved task checkpoints, evaluation data and, for cross-bit comparisons, prior W8 artifacts. Respect cache versions and checkpoint identity when resuming runs. Notebook 25 is the starting point for reading consolidated results without inference.

## Pending experiment discussed with the supervisor

A follow-up was discussed to select the 15 or 20 least-sensitive DistilBERT tensors according to W8. No Notebook 27 is present at this context snapshot, and those results must not be presented as completed.

Two protocols answer different questions: quantizing the selected tensors individually measures their isolated W4 sensitivity; quantizing all selected tensors together measures a sensitivity-guided mixed-precision configuration. Record the intended protocol, ranking source and method-specific selection before running it. Selecting only low-sensitivity W8 units restricts the population being studied and cannot replace a claim about rank transfer across all 40 tensors. Using the same labelled split for selection and evaluation also makes a deployment/generalization claim exploratory.

The existing stratified experiment remains a valid measurement of its predefined subset. A new selection policy requires separate outputs and an accurate description in any updated report.
