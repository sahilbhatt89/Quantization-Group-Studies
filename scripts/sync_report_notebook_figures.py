"""Copy original notebook figures into the report, or verify them with --check.

No plotting, image processing, or notebook execution is performed.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


ROOT = Path(__file__).resolve().parents[1]
# Report filename, notebook number, zero-based plotting cell, saved artifact.
FIGURES = [
    ("ptq_accuracy_all_models_2_to_8_bit.png", "25", 15,
     "consolidated_full_model_bit_sweeps/ptq_accuracy_all_models_2_to_8_bit.png"),
    ("codebook_accuracy_all_models_2_to_8_bit.png", "25", 15,
     "consolidated_full_model_bit_sweeps/codebook_accuracy_all_models_2_to_8_bit.png"),
    ("all_models_accuracy_compression_2_to_8_bit.png", "25", 13,
     "consolidated_full_model_bit_sweeps/all_models_accuracy_compression_2_to_8_bit.png"),
    ("resnet18_w4_w8_layer_sensitivity.png", "20", 34,
     "resnet18_4bit_layer_sensitivity/figure6_layer_sensitivity_heatmap_w4_w8.png"),
    ("distilbert_w4_w8_layer_sensitivity.png", "21", 26,
     "distilbert_4bit_stratified_sensitivity/figure2_w4_w8_heatmap.png"),
    ("vit_w4_w8_module_sensitivity_heatmap.png", "27", 5,
     "vit_w4_w8_heatmap/vit_w4_w8_module_sensitivity_heatmap.png"),
]
SUPPORTING_FILES = [
    ("resnet18_4bit_layer_sensitivity/report_heatmap_values.csv", "notebook20_resnet_heatmap_values.csv"),
    ("resnet18_4bit_layer_sensitivity/report_heatmap_significance.csv", "notebook20_resnet_heatmap_significance.csv"),
    ("resnet18_4bit_layer_sensitivity/report_heatmap_manifest.json", "notebook20_resnet_heatmap_manifest.json"),
    ("distilbert_4bit_stratified_sensitivity/report_heatmap_values.csv", "notebook21_distilbert_heatmap_values.csv"),
    ("distilbert_4bit_stratified_sensitivity/report_heatmap_manifest.json", "notebook21_distilbert_heatmap_manifest.json"),
    ("vit_w4_w8_heatmap/vit_w4_w8_heatmap_values.csv", "notebook27_vit_heatmap_values.csv"),
    ("vit_w4_w8_heatmap/heatmap_manifest.json", "notebook27_vit_heatmap_manifest.json"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify without writing files")
    args = parser.parse_args()
    expected = {entry[0] for entry in FIGURES}
    for tex in (ROOT / "report").glob("*.tex"):
        referenced = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex.read_text()))
        if referenced != expected:
            raise ValueError(f"{tex.name}: figure references differ from notebook manifest: {referenced ^ expected}")
    entries = []
    for filename, number, cell_index, artifact in FIGURES:
        notebook, = (ROOT / "notebooks").glob(f"{number}_*.ipynb")
        cells = json.loads(notebook.read_text())["cells"]
        source = "".join(cells[cell_index]["source"])
        notebook_source = "\n".join("".join(cell["source"]) for cell in cells)
        original = ROOT / "artifacts" / artifact
        if "savefig" not in source or original.name not in notebook_source:
            raise ValueError(f"Notebook plotting source changed: {notebook.name}, cell {cell_index}")
        destination = ROOT / "report" / "images" / filename
        if not args.check:
            shutil.copyfile(original, destination)
        if original.read_bytes() != destination.read_bytes():
            raise ValueError(f"Report image differs from notebook artifact: {filename}")
        entries.append({
            "report_image": str(destination.relative_to(ROOT)),
            "notebook": str(notebook.relative_to(ROOT)),
            "plot_cell_index": cell_index,
            "artifact": str(original.relative_to(ROOT)),
            "sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        })
    for artifact, filename in SUPPORTING_FILES:
        original = ROOT / "artifacts" / artifact
        destination = ROOT / "report" / "data" / filename
        if not args.check:
            shutil.copyfile(original, destination)
        if original.read_bytes() != destination.read_bytes():
            raise ValueError(f"Report data differs from notebook artifact: {filename}")
    manifest = {"policy": "Unmodified copies of figures saved by notebooks only", "figures": entries}
    path = ROOT / "report" / "data" / "notebook_figure_manifest.json"
    if args.check:
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Notebook figure manifest is stale")
    else:
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified {len(entries)} report images: byte-identical to their notebook artifacts.")


if __name__ == "__main__":
    main()
