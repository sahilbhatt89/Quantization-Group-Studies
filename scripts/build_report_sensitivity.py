"""Export sensitivity values and validate cross-bit ranks from saved CSVs.

No model loading, training, quantization or inference is performed. Original
notebook artifacts are read-only. Run with .venv/bin/python from the repo root.
Report figures are copied separately by sync_report_notebook_figures.py.
"""

from itertools import permutations
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "data"
SEED = 42
RESAMPLES = 199_999
SPECS = [
    ("ResNet-18", "resnet18", "resnet18_4bit_layer_sensitivity/layer_sensitivity_w4_vs_w8.csv",
     "tensor", "sensitivity_rank_w4", "sensitivity_rank_w8", 21),
    ("DistilBERT", "distilbert", "distilbert_4bit_stratified_sensitivity/matched_tensor_sensitivity_w4_vs_w8.csv",
     "tensor", "subset_sensitivity_rank_w4", "subset_sensitivity_rank_w8", 15),
    ("ViT-B/16", "vit", "vit_4bit_stratified_sensitivity/stratified_module_w4_vs_w8.csv",
     "module", "sensitivity_rank_w4", "sample_rank_w8", 7),
]
COLS = [("ptq", "w4"), ("ptq", "w8"), ("codebook", "w4"), ("codebook", "w8")]
MCNEMAR_SPECS = [
    ("ResNet-18", 4, "resnet18_4bit_layer_sensitivity/layerwise_sensitivity_all_methods.csv"),
    ("ResNet-18", 8, "resnet18_full_layer_sensitivity/layerwise_sensitivity_all_methods.csv"),
    ("DistilBERT", 4, "distilbert_4bit_stratified_sensitivity/w4_stratified_tensor_sensitivity.csv"),
    ("DistilBERT", 8, "distilbert_full_layer_sensitivity/layerwise_sensitivity_all_methods.csv"),
    ("ViT-B/16", 4, "vit_4bit_stratified_sensitivity/w4_stratified_module_sensitivity.csv"),
    ("ViT-B/16", 8, "vit_full_layer_sensitivity/module_sensitivity_all_methods.csv"),
]


def permutation_p(x, y, seed):
    """Two-sided absolute-rho test; exact for n=7, Monte Carlo otherwise."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    y = np.asarray(y, dtype=float) - np.mean(y)
    norm = np.linalg.norm(x) * np.linalg.norm(y)
    observed = abs(x @ y / norm)
    extreme = 0
    if len(x) <= 7:
        draws = np.array(list(permutations(y)))
        extreme = np.count_nonzero(np.abs(draws @ x / norm) >= observed - 1e-12)
        return extreme / len(draws), len(draws), "exact"
    rng = np.random.default_rng(seed)
    remaining = RESAMPLES
    while remaining:
        n = min(10_000, remaining)
        indices = np.argsort(rng.random((n, len(y))), axis=1)
        extreme += np.count_nonzero(np.abs(y[indices] @ x / norm) >= observed - 1e-12)
        remaining -= n
    # Include the observed arrangement in the randomized test.
    return (extreme + 1) / (RESAMPLES + 1), RESAMPLES, "Monte Carlo"


def spearman_permutation(x, y, seed):
    """Spearman rho with a permutation p-value that preserves observed ties."""
    ranked_x = rankdata(np.asarray(x, dtype=float), method="average")
    ranked_y = rankdata(np.asarray(y, dtype=float), method="average")
    rho = float(spearmanr(x, y).statistic)
    p_value, count, kind = permutation_p(ranked_x, ranked_y, seed)
    return rho, p_value, count, kind


def export_mcnemar_resolution():
    """Export attainable exact-test floors from recorded discordant counts."""
    rows = []
    for model, bits, rel in MCNEMAR_SPECS:
        path = ROOT / "artifacts" / rel
        frame = pd.read_csv(path)
        if "method_family" not in frame:
            frame["method_family"] = frame["method"].str.replace(
                r"_w[48]$", "", regex=True
            )
        family_sizes = frame.groupby("method_family").size().to_dict()
        for _, row in frame.iterrows():
            discordant = int(row.correct_to_wrong + row.wrong_to_correct)
            log2_floor = 0 if discordant == 0 else 1 - discordant
            floor = 1.0 if discordant == 0 else float(np.exp2(log2_floor))
            tests = int(family_sizes[row.method_family])
            rows.append({
                "model": model,
                "bit_width": bits,
                "method": row.method_family,
                "unit": row.get("tensor", row.get("analysis_unit", row.get("module"))),
                "correct_to_wrong": int(row.correct_to_wrong),
                "wrong_to_correct": int(row.wrong_to_correct),
                "discordant_pairs": discordant,
                "minimum_two_sided_exact_p": floor,
                "log2_minimum_exact_p": log2_floor,
                "raw_0_05_attainable": floor < 0.05,
                "first_bh_cutoff_attainable": floor <= 0.05 / tests,
                "observed_exact_p": float(row.mcnemar_p_value),
                "observed_fdr_p": float(row.mcnemar_fdr_p_value),
            })
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "mcnemar_resolution_by_unit.csv", index=False)
    summary = []
    for (model, bits, method), group in detail.groupby(
        ["model", "bit_width", "method"], sort=False
    ):
        summary.append({
            "model": model,
            "bit_width": bits,
            "method": method,
            "tests": len(group),
            "discordant_min": int(group.discordant_pairs.min()),
            "discordant_median": float(group.discordant_pairs.median()),
            "discordant_max": int(group.discordant_pairs.max()),
            "raw_0_05_unattainable": int((~group.raw_0_05_attainable).sum()),
            "first_bh_cutoff_unattainable": int((~group.first_bh_cutoff_attainable).sum()),
            "fdr_significant": int((group.observed_fdr_p < 0.05).sum()),
        })
    result = pd.DataFrame(summary)
    result.to_csv(OUT / "mcnemar_resolution_summary.csv", index=False)
    return result


def export_distilbert_rank_robustness():
    """Test whether the codebook association survives removal of its tie-break."""
    path = ROOT / "artifacts/distilbert_4bit_stratified_sensitivity/matched_tensor_sensitivity_w4_vs_w8.csv"
    frame = pd.read_csv(path)
    frame = frame[frame.method_family.eq("codebook")].reset_index(drop=True)
    metrics = [
        ("reported ranks (accuracy, then disagreement tie-break)",
         "subset_sensitivity_rank_w4", "subset_sensitivity_rank_w8"),
        ("accuracy degradation only (average ties)",
         "accuracy_degradation_pp_w4", "accuracy_degradation_pp_w8"),
        ("prediction disagreement only (average ties)",
         "prediction_disagreement_percent_w4", "prediction_disagreement_percent_w8"),
    ]
    rows = []
    for offset, (analysis, x_col, y_col) in enumerate(metrics):
        rho, p_value, count, kind = spearman_permutation(
            frame[x_col], frame[y_col], SEED + 100 + offset
        )
        tau = kendalltau(frame[x_col], frame[y_col], variant="b")
        rows.append({
            "analysis": analysis,
            "units": len(frame),
            "unique_w4_values": int(frame[x_col].nunique()),
            "unique_w8_values": int(frame[y_col].nunique()),
            "spearman_rho": rho,
            "permutation_p": p_value,
            "permutations": count,
            "test": kind,
            "kendall_tau_b": float(tau.statistic),
            "kendall_asymptotic_p": float(tau.pvalue),
        })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "distilbert_codebook_rank_robustness.csv", index=False)

    leave_one_out = []
    for index, omitted in frame.iterrows():
        subset = frame.drop(index)
        rho, p_value, count, kind = spearman_permutation(
            subset.accuracy_degradation_pp_w4,
            subset.accuracy_degradation_pp_w8,
            SEED + 200 + index,
        )
        leave_one_out.append({
            "omitted_tensor": omitted.tensor,
            "units": len(subset),
            "accuracy_only_spearman_rho": rho,
            "permutation_p": p_value,
            "permutations": count,
            "test": kind,
        })
    pd.DataFrame(leave_one_out).to_csv(
        OUT / "distilbert_codebook_accuracy_leave_one_out.csv", index=False
    )
    return result, pd.DataFrame(leave_one_out)


def main():
    OUT.mkdir(exist_ok=True)
    statistics, sources = [], []
    for model, slug, rel, unit, rank4, rank8, expected in SPECS:
        path = ROOT / "artifacts" / rel
        frame = pd.read_csv(path)
        assert set(frame.method_family) == {"ptq", "codebook"}
        assert not frame.duplicated(["method_family", unit]).any()
        order = frame[frame.method_family == "ptq"].sort_values(rank4)[unit].tolist()
        assert len(order) == expected
        columns = {}
        for family, bit in COLS:
            group = frame[frame.method_family == family].set_index(unit)
            assert set(group.index) == set(order)
            columns[f"{family}_{bit}"] = group.loc[order, f"accuracy_degradation_pp_{bit}"]
        matrix = pd.DataFrame(columns)
        assert np.isfinite(matrix.to_numpy()).all()
        matrix.to_csv(OUT / f"{slug}_heatmap_values.csv", index_label="unit")
        for family in ("ptq", "codebook"):
            group = frame[frame.method_family == family].sort_values(unit)
            x, y = group[rank4].to_numpy(), group[rank8].to_numpy()
            assert set(x) == set(range(1, expected + 1))
            assert set(y) == set(range(1, expected + 1))
            corr = spearmanr(x, y)
            seed = SEED + len(statistics)
            p, count, kind = permutation_p(x, y, seed)
            statistics.append(dict(model=model, method=family, units=expected, rho=corr.statistic,
                                   approximate_p=corr.pvalue, permutation_p=p, test=kind,
                                   permutations=count, seed=seed))
        sources.append(dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    result = pd.DataFrame(statistics)
    result.to_csv(OUT / "cross_bit_rank_validation.csv", index=False)
    mcnemar = export_mcnemar_resolution()
    robustness, leave_one_out = export_distilbert_rank_robustness()
    (OUT / "sensitivity_figure_manifest.json").write_text(json.dumps({
        "sources": sources, "column_order": [f"{a}_{b}" for a, b in COLS],
        "row_order": "Descending PTQ W4 degradation, then disagreement, using saved ranks",
        "purpose": "Supporting sensitivity tables and rank validation; no report figures are generated",
        "report_figures": "notebook_figure_manifest.json",
        "p_value": "Two-sided absolute Spearman-rho permutation tail; plus-one correction for Monte Carlo",
        "mcnemar_resolution": "Minimum attainable two-sided exact p is 1 for zero discordances and 2^(1-n) otherwise",
        "rank_robustness": "DistilBERT codebook: reported ranks, accuracy-only average ties, disagreement-only average ties, and accuracy-only leave-one-out",
        "inference_rerun": False,
    }, indent=2) + "\n")
    print(result.to_string(index=False))
    print("\nMcNemar resolution summary:\n", mcnemar.to_string(index=False))
    print("\nDistilBERT codebook robustness:\n", robustness.to_string(index=False))
    print("\nAccuracy-only leave-one-out rho range:",
          leave_one_out.accuracy_only_spearman_rho.min(),
          leave_one_out.accuracy_only_spearman_rho.max())


if __name__ == "__main__":
    main()
