# Current report package

`quantization_study_reference.tex` is the current project-specific IEEE-style report draft. Its numerical claims are based on measurements saved under `artifacts/`; it does not reuse unrelated values from the supplied dummy report.

The report contains:

1. Abstract
2. Introduction and research questions
3. Models and datasets
4. Quantization methods and research foundations
5. FP32 full-model evaluation
6. Eight-bit full-model comparison
7. Two-to-eight-bit sweep tables and figures

Before submission:

1. Confirm the author email address and department wording.
2. Regenerate tables if a notebook is rerun with a different checkpoint or configuration.
3. Confirm whether the university expects IEEE conference layout or a longer thesis/report template.
4. Do not claim native INT4/codebook speed-up from the current reconstructed-FP32 evaluation.
5. Execute Notebook 15 if a complete ResNet codebook 2--8-bit curve is required; the current report does not infer the missing curve.

## Local compilation

Compile from this directory so the `images/` paths resolve:

```bash
cd report
pdflatex quantization_study_reference.tex
pdflatex quantization_study_reference.tex
```

The first pass creates cross-references; the second resolves them.

## Overleaf upload

The simplest option is to upload `quantization_report_overleaf.zip` as a new Overleaf project. Alternatively, upload both of the following while preserving the folder name:

- `quantization_study_reference.tex`
- the complete `images/` directory

Set `quantization_study_reference.tex` as the Overleaf main document. The bibliography is embedded in the `.tex` file, so no separate `.bib` file is required.

## Saved sweep status

- ResNet PTQ 2--8 bits: complete.
- DistilBERT PTQ and codebook 2--8 bits: complete.
- ViT PTQ and codebook 2--8 bits: complete.
- ResNet codebook: W4 and W8 results complete; Notebook 15 has no saved 2--8-bit execution output.

The report records the last limitation explicitly rather than filling the missing curve from unrelated data.
