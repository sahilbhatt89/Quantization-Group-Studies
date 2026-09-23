# Current report package

`quantization_study_reference.tex` is the current project-specific IEEE-style report draft. Its numerical claims are based on measurements saved under `artifacts/`; it does not reuse unrelated values from the supplied dummy report.

Updated 20 September 2026: both report sources now frame the results as observations of three selected checkpoints and the evaluated pipeline. The report discloses the training and method confounds, documents the recorded codebook and BatchNorm policies, corrects the citations and simulated-evaluation terminology, and adds transformer PTQ context. The cause of the ResNet W4 collapse remains unresolved. All recorded measurements and statistical results are preserved. See [REVISION_NOTES.md](REVISION_NOTES.md) for the changes and verified sources.

The separate duplicate FP32 table is consolidated into the model/dataset table and existing model descriptions. Notebook 25's overview now plots accuracy and compression only: memory reduction is the derived quantity `100 * (1 - 1 / compression_ratio)` and remains in the tables.

All report images are unmodified copies of notebook-produced PNGs. Heatmaps in Notebooks 20, 21 and 27 use zero-centered colors. ResNet's Notebook 20 figure retains a ±80 pp full-range panel and adds an annotated ±7 pp detail panel for the 18 kernels outside the three largest PTQ W4 losses; dots mark FDR-adjusted McNemar significance. DistilBERT and ViT share ±0.35 pp. Every heatmap caption identifies its scale. Notebook 27 reads the matched measurements saved by Notebook 22.

Regenerate only the report plotting cells from saved CSVs, then copy and verify their outputs:

```bash
.venv/bin/python scripts/refresh_report_notebook_plots.py
python3 scripts/sync_report_notebook_figures.py
python3 scripts/sync_report_notebook_figures.py --check
```

Run these commands from the repository root. The refresh script executes only an explicit list of cells tagged `report-plot-only`: Notebook 20 cell 34, Notebook 21 cell 26, Notebook 25 cells 1, 13 and 17, and Notebook 27 cells 1, 3, 5 and 7 (zero-based). It preserves their executed notebook outputs and performs no model loading, training, quantization or inference. The heatmap cells in Notebooks 20 and 21 can also be run independently in Jupyter; there is no need to rerun their experiment cells.

`data/notebook_figure_manifest.json` records every included image's source notebook, plotting cell, artifact path and SHA-256 checksum. The `notebook20_*`, `notebook21_*` and `notebook27_*` files in `data/` contain the heatmap value tables and source manifests. Verification checks both TeX sources and requires byte-identical image and supporting-data copies.

The cross-bit table reports permutation p-values, with the original approximate values retained alongside them in `data/cross_bit_rank_validation.csv`. Accuracy measurements and Spearman coefficients are unchanged. ViT uses all 5,040 permutations; the other models use 199,999 random permutations with fixed seeds. The DistilBERT codebook association remains significant (permutation p = 0.00551). `data/distilbert_codebook_rank_robustness.csv` removes the disagreement tie-break and evaluates accuracy degradation and disagreement separately; `data/distilbert_codebook_accuracy_leave_one_out.csv` records the accuracy-only leave-one-tensor-out checks.

`data/mcnemar_resolution_by_unit.csv` records every exact test's discordant count and minimum attainable two-sided p-value. `data/mcnemar_resolution_summary.csv` summarizes the resolution limits by model, precision and method. In particular, the report no longer interprets DistilBERT's zero rejected hypotheses as evidence of insensitivity or equivalence.

Regenerate the supporting sensitivity values, McNemar resolution diagnostics, and rank-robustness checks from saved notebook artifacts with:

```bash
.venv/bin/python scripts/build_report_sensitivity.py
```

Run this command from the repository root. It performs no model inference and creates no images. `data/sensitivity_figure_manifest.json` records the source CSV checksums and analysis definitions for these supporting tables; the notebook figure manifest above governs the report images. The sensitivity CSVs retain their validation ordering, which can differ from the original notebook plots.

The report contains:

1. Abstract
2. Introduction and research questions
3. Models and datasets
4. Quantization methods and research foundations
5. FP32 reference evaluation (consolidated baseline table)
6. Eight-bit full-model comparison
7. Two-to-eight-bit sweep tables and figures
8. Isolated W4/W8 sensitivity, notebook-produced heatmaps for all three models, and rank validation
9. Limitations and conclusion

Before submission:

1. Confirm the author email address and department wording.
2. Regenerate tables if a notebook is rerun with a different checkpoint or configuration.
3. Confirm whether the university expects IEEE conference layout or a longer thesis/report template.
4. Do not claim native INT4/codebook speed-up from the current reconstructed-FP32 evaluation.
5. Use Notebook 26 as the authoritative completed ResNet PTQ and codebook 2--8-bit sweep.

## Local compilation

Compile from this directory so the `images/` paths resolve:

```bash
cd report
pdflatex quantization_study_reference.tex
pdflatex quantization_study_reference.tex
```

The first pass creates cross-references; the second resolves them. Alternatively, run `tectonic quantization_study_reference.tex`; Tectonic resolves cross-references automatically.

## Overleaf upload

The simplest option is to upload `quantization_report_overleaf.zip` as a new Overleaf project. Alternatively, upload the following while preserving the folder name:

- `quantization_study_reference.tex`
- the complete `images/` directory
- the `data/` directory for the accompanying plotted values and validation results

Set `quantization_study_reference.tex` as the Overleaf main document. The bibliography is embedded in the `.tex` file, so no separate `.bib` file is required.

## Saved sweep status

- ResNet PTQ and codebook 2--8 bits: complete in Notebook 26.
- DistilBERT PTQ and codebook 2--8 bits: complete.
- ViT PTQ and codebook 2--8 bits: complete.

Notebook 25 consolidates the completed architecture sweeps without rerunning
quantization or inference.
