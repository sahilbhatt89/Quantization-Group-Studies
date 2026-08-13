"""Convert and measure built-in TFLite PTQ variants of ResNet-18."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import (
    inspect_tflite,
    predict_tflite,
    reduction_metrics,
    size_metrics,
)
from src.models import load_pretrained_resnet18
from src.quantization import (
    build_fixed_input_model,
    convert_dynamic_range,
    convert_float,
    convert_full_integer,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Directory containing representative calibration images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "resnet18_builtin_ptq",
        help="Directory for converted models and results.",
    )
    return parser.parse_args()


def load_images(image_dir: Path) -> list[np.ndarray]:
    image_paths = sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise ValueError(f"No calibration images found in {image_dir}")

    images = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        images.append(np.asarray(image, dtype=np.uint8)[None, ...])
    return images


def preprocess_images(model, images: list[np.ndarray]) -> list[np.ndarray]:
    return [
        np.asarray(model.preprocessor(image), dtype=np.float32)
        for image in images
    ]


def write_model(path: Path, model_content: bytes) -> None:
    path.write_bytes(model_content)
    print(f"Wrote {path.name}: {path.stat().st_size / (1024**2):.2f} MiB")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading pretrained ResNet-18...")
    model = load_pretrained_resnet18()
    raw_images = load_images(args.image_dir)
    samples = preprocess_images(model, raw_images)
    fixed_model = build_fixed_input_model(model)

    if len(samples) < 50:
        print(
            f"Warning: only {len(samples)} calibration image(s) found. "
            "Use 50-200 representative images for meaningful full INT8 calibration."
        )

    def representative_dataset():
        for sample in samples:
            yield [sample]

    paths = {
        "float32": args.output_dir / "resnet18_float32.tflite",
        "dynamic_int8": args.output_dir / "resnet18_dynamic_int8.tflite",
        "full_int8": args.output_dir / "resnet18_full_int8.tflite",
    }

    write_model(paths["float32"], convert_float(fixed_model))
    write_model(paths["dynamic_int8"], convert_dynamic_range(fixed_model))
    write_model(
        paths["full_int8"],
        convert_full_integer(fixed_model, representative_dataset),
    )

    results = {}
    for name, path in paths.items():
        metrics = {
            **inspect_tflite(path),
            **predict_tflite(path, samples[0]),
        }
        if name == "float32":
            size_bytes = path.stat().st_size
            metrics.update(
                {
                    "size_bytes": size_bytes,
                    "size_mib": size_bytes / (1024**2),
                    "compression_ratio": 1.0,
                    "memory_reduction_percent": 0.0,
                }
            )
        else:
            metrics.update(size_metrics(paths["float32"], path))
        results[name] = metrics

    baseline = results["float32"]
    for metrics in results.values():
        metrics["weight_memory"] = {
            "size_bytes": metrics["weight_storage_bytes"],
            **reduction_metrics(
                baseline["weight_storage_bytes"],
                metrics["weight_storage_bytes"],
            ),
        }
        metrics["activation_memory"] = {
            "tensor_storage_bytes": metrics["activation_tensor_storage_bytes"],
            "largest_tensor_bytes": metrics["largest_activation_tensor_bytes"],
            **reduction_metrics(
                baseline["activation_tensor_storage_bytes"],
                metrics["activation_tensor_storage_bytes"],
            ),
        }

    results_path = args.output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print("\nResults")
    for name, metrics in results.items():
        print(
            f"{name:>12}: {metrics['size_mib']:.2f} MiB, "
            f"{metrics['compression_ratio']:.2f}x compression, "
            f"{metrics['memory_reduction_percent']:.2f}% reduction, "
            f"top class {metrics['top_class_index']}"
        )
        print(
            f"{'':>14}weight reduction "
            f"{metrics['weight_memory']['memory_reduction_percent']:.2f}%, "
            f"activation tensor reduction "
            f"{metrics['activation_memory']['memory_reduction_percent']:.2f}%"
        )
    print(f"\nDetailed results: {results_path}")


if __name__ == "__main__":
    main()
