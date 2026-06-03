"""Pretrained TensorFlow/Keras ResNet-18 model loader."""

from __future__ import annotations

import keras_hub


RESNET18_IMAGENET_PRESET = "resnet_18_imagenet"


def load_pretrained_resnet18():
    """Load the ImageNet-pretrained ResNet-18 classifier from KerasHub."""
    return keras_hub.models.ResNetImageClassifier.from_preset(
        RESNET18_IMAGENET_PRESET,
        load_weights=True,
    )
