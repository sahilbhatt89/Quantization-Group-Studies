"""Execute only saved-data report plotting cells and retain notebook outputs.

Run with .venv/bin/python from the repository root. This allowlist deliberately
excludes all model-loading, training, quantization and inference cells.
"""

from pathlib import Path
import json
import os
import tempfile

# Keep plotting caches out of the user's global configuration directories.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "quantization-matplotlib"))
os.environ.setdefault("IPYTHONDIR", str(Path(tempfile.gettempdir()) / "quantization-ipython"))
os.environ.setdefault("MPLBACKEND", "Agg")

from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output


ROOT = Path(__file__).resolve().parents[1]
CELLS = {"20": [34], "21": [26], "25": [1, 13, 17], "27": [1, 3, 5, 7]}


def main():
    os.chdir(ROOT)
    for number, indices in CELLS.items():
        path, = (ROOT / "notebooks").glob(f"{number}_*.ipynb")
        notebook = json.loads(path.read_text())
        shell = InteractiveShell.instance()
        shell.reset(new_session=True)
        for index in indices:
            cell = notebook["cells"][index]
            if cell["cell_type"] != "code" or "report-plot-only" not in cell["metadata"].get("tags", []):
                raise ValueError(f"Unapproved plotting cell: {path.name}:{index}")
            with capture_output() as captured:
                result = shell.run_cell("".join(cell["source"]), store_history=True)
            if not result.success:
                raise RuntimeError(f"{path.name}:{index}\n{captured.stdout}\n{captured.stderr}\n"
                                   f"{result.error_in_exec or result.error_before_exec}")
            outputs = []
            for name, stream in (("stdout", captured.stdout), ("stderr", captured.stderr)):
                if stream:
                    outputs.append({"output_type": "stream", "name": name,
                                    "text": stream.splitlines(keepends=True)})
            outputs.extend({"output_type": "display_data", "data": output.data,
                            "metadata": output.metadata} for output in captured.outputs)
            cell["outputs"] = outputs
            cell["execution_count"] = result.execution_count
        notebook["metadata"]["report_plot_refresh"] = {
            "executed_cell_indices": indices,
            "source": "saved CSV measurements only",
            "inference_rerun": False,
        }
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
        print(f"Refreshed {path.name}: cells {indices}")


if __name__ == "__main__":
    main()
