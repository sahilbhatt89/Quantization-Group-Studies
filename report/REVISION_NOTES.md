# Report revision — 20 September 2026

This revision changes writing, documentation and presentation. It does not run
training, quantization or model inference. Recorded measurement CSVs, accuracy
tables, rank correlations and p-values are preserved. The cause of the ResNet
W4 collapse is unresolved; the report makes no claim of having repaired it.

| Requested correction | Implemented change |
| --- | --- |
| Broad architecture conclusion | Abstract, RQ2, scope, discussion and conclusion now describe the selected checkpoints and evaluated policies. |
| Severe ResNet W4 collapse | Measurements are retained; the cause is explicitly unresolved and no general CNN limitation is inferred. |
| Training confound | Frozen ViT backbone versus end-to-end ResNet adaptation is prominent in the abstract, model description and limitations. |
| Method confounds | Per-channel versus per-tensor granularity and symmetric `2^b-1` versus codebook `2^b` levels are both disclosed. |
| ViT W3 explanation | The 57.55-pp difference remains an observation; centroid adaptation is an untested hypothesis confounded by granularity and level count. |
| Simulated evaluation | Wording distinguishes floating-point reconstruction with FP32 activations from native low-bit inference and from QAT. |
| Codebook reproducibility | Fixed-seed k-means++, one initialization, at most 30 iterations, 100,000-value fitting cap, sampling without replacement, tolerance 1e-6, and seed `42 + model.weights index` are documented. |
| BatchNorm | Recorded inference retains saved FP32 parameters/statistics, uses `training=False`, and applies no folding, recalibration, bias correction, equalization or adaptive rounding. |
| Citations and exact test | Choi authors/venue corrected; Hessian-weighted terminology used. The exact conditional binomial McNemar calculation is specified and supported by Fagerland et al., rather than attributing exactness to Dietterich's approximate-test paper. |
| Transformer literature | GPTQ, AWQ, SmoothQuant, LLM.int8() and PTQ4ViT are discussed with explicit differences in models, tasks and quantization scope; none is claimed as an evaluated baseline. |
| Redundancy | Repeated standalone FP32 table removed; baseline values remain in the model table/descriptions. Notebook 25's overview now contains accuracy and compression columns; memory reduction remains in tables and is explicitly derived from compression. |
| Heatmaps | Notebook-produced plots retain all measured values and use zero-centered colors. ResNet has a ±80 pp full panel and an annotated ±7 pp detail panel; DistilBERT and ViT share ±0.35 pp. Every caption states the scale differences. |
| Spacing | Explicit braces after the `pp` macro preserve word spacing, including “60.65 pp with”, “78.62 pp with” and “0.41 pp for”. |

## Statistical follow-up — 20 September 2026

- The report now gives the exact McNemar resolution floor, `2^(1-n)`, from each recorded discordant count. For DistilBERT, 40/40 W8 PTQ tests, 39/40 W8 codebook tests, and 12/15 W4 tests for each method cannot attain raw `p < 0.05`. Zero FDR rejections are no longer described as evidence of insensitivity or equivalence.
- The significant DistilBERT codebook rank result is not created by its prediction-disagreement tie-break. Accuracy degradation with average ties gives Spearman rho 0.713 and permutation `p = 0.003765`; accuracy-only leave-one-tensor-out rho remains 0.645–0.766. Disagreement is also associated, but it is not the sole source of the accuracy association.
- Notebook 20 now produces a dual-scale ResNet figure: a ±80 pp full-range panel plus an annotated ±7 pp detail panel for the remaining 18 kernels. Dots identify FDR-adjusted McNemar significance. No measurements were recomputed.

## Evidence for documented settings

- Notebook 06 marks the ResNet backbone trainable; Notebook 10 and
  `src/models/transformer_based/pretrained_vit.py` use a frozen ViT backbone.
- Notebooks 17–24 and 26 specify the evaluated codebook iteration cap,
  sampling cap and base seed. `codebook_vector_quantization.py` supplies the
  single-initialization fitter, tolerance, sampling behavior and tensor-index
  seed offset. The report distinguishes notebook overrides from fitter defaults.
- The custom sensitivity and sweep notebooks call their models with
  `training=False`, retaining unselected checkpoint tensors. The documented
  BatchNorm policy is not a diagnosis of the ResNet failure.
- The sensitivity notebooks calculate their exact p-values with
  `scipy.stats.binomtest(correct_to_wrong, discordant, 0.5)`; no statistical
  results were recalculated or replaced in this revision.

## Verified literature

- [Choi, El-Khamy and Lee — Towards the Limit of Network Quantization](https://arxiv.org/abs/1612.01543): authors, ICLR 2017 conference attribution and Hessian-weighted distortion.
- [Fagerland, Lydersen and Laake — The McNemar test for binary matched-pairs data](https://doi.org/10.1186/1471-2288-13-91): exact conditional binomial testing and its conservatism with discrete data. This supports the description of the test used, not a claim that exact testing is universally preferable.
- [Nagel et al. — A White Paper on Neural Network Quantization](https://arxiv.org/html/2106.08295v1#S3.SS5): the more elaborate PTQ pipeline, including adaptive rounding, and its different ImageNet evaluation context.
- [GPTQ](https://arxiv.org/abs/2210.17323): second-order weight quantization for generative transformers.
- [AWQ](https://arxiv.org/abs/2306.00978): activation-aware channel scaling for weight-only LLM quantization.
- [SmoothQuant](https://arxiv.org/abs/2211.10438): activation-outlier rescaling for W8A8 LLM inference.
- [LLM.int8()](https://arxiv.org/abs/2208.07339): higher-precision treatment of outlier feature dimensions alongside INT8 multiplication.
- [PTQ4ViT](https://arxiv.org/abs/2111.12293): twin uniform quantization and Hessian-guided calibration for vision transformers on ImageNet.

## Reproduction and checks

Run `scripts/refresh_report_notebook_plots.py` with the project Python, followed
by `scripts/sync_report_notebook_figures.py`. Only the explicitly selected
saved-data plotting cells are executed. The refresh preserves embedded notebook
outputs; the sync copies the resulting images and supporting value/provenance
files into the report. See `README.md` for cell indices and compilation commands.

The publication package includes the notebook-image checksum manifest and
unrounded heatmap value tables. A checksum comparison against the pre-revision
snapshot verifies that existing measurement CSVs are unchanged. Both TeX
sources and PDFs are synchronized, and the Overleaf archive contains the
current referenced images only.
