"""Pretrained KerasHub Vision Transformer model loaders."""

from __future__ import annotations

import keras_hub
from tensorflow import keras


VIT_BASE_PATCH16_224_IMAGENET_PRESET = "vit_base_patch16_224_imagenet"


def load_pretrained_vit_backbone() -> keras.Model:
    """Load the ImageNet-pretrained ViT-B/16 backbone."""
    return keras_hub.models.ViTBackbone.from_preset(
        VIT_BASE_PATCH16_224_IMAGENET_PRESET,
        load_weights=True,
    )


def build_vit_cifar10_classifier(
    num_classes: int = 10,
    freeze_backbone: bool = True,
) -> keras.Model:
   
    backbone = load_pretrained_vit_backbone()
    backbone.trainable = not freeze_backbone
    return keras_hub.models.ViTImageClassifier(
        backbone=backbone,
        num_classes=num_classes,
        preprocessor=None,
        activation=None,
        name="vit_cifar10_classifier",
    )


def build_vit_image_preprocessor() -> keras.layers.Layer:
    """Load the resize and normalization pipeline matching ViT-B/16."""
    return keras_hub.models.ViTImageClassifierPreprocessor.from_preset(
        VIT_BASE_PATCH16_224_IMAGENET_PRESET,
    )
