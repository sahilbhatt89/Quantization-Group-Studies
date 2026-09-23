"""Load and summarize the pretrained ResNet-18 model."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import RESNET18_IMAGENET_PRESET, load_pretrained_resnet18


def main() -> None:
    print(f"Loading KerasHub preset: {RESNET18_IMAGENET_PRESET}")
    model = load_pretrained_resnet18()
    model.summary()


if __name__ == "__main__":
    main()
