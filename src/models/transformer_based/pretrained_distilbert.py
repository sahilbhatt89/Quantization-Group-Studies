"""Pretrained KerasHub DistilBERT model loaders."""

from __future__ import annotations

import keras_hub
from tensorflow import keras


DISTILBERT_BASE_EN_UNCASED_PRESET = "distil_bert_base_en_uncased"
SST2_CLASS_NAMES = ("negative", "positive")


def load_pretrained_distilbert_backbone() -> keras.Model:
    """Load the English uncased DistilBERT backbone and pretrained weights."""
    return keras_hub.models.DistilBertBackbone.from_preset(
        DISTILBERT_BASE_EN_UNCASED_PRESET,
        load_weights=True,
    )


def build_distilbert_text_classifier(
    num_classes: int = 2,
    freeze_backbone: bool = False,
) -> keras.Model:
    """Build a classifier using the existing pretrained DistilBERT weights.

    The pretrained backbone is retained and a new task-specific classification
    head is attached. Inputs must already be tokenized dictionaries containing
    ``token_ids`` and ``padding_mask``; text preprocessing intentionally stays
    outside the model that will later be converted and quantized.
    """
    backbone = load_pretrained_distilbert_backbone()
    backbone.trainable = not freeze_backbone
    return keras_hub.models.DistilBertTextClassifier(
        backbone=backbone,
        num_classes=num_classes,
        preprocessor=None,
        name="distilbert_text_classifier",
    )


def build_distilbert_text_preprocessor(
    sequence_length: int = 128,
) -> keras.layers.Layer:
    """Load the matching tokenizer and build a fixed-length preprocessor."""
    return keras_hub.models.DistilBertTextClassifierPreprocessor.from_preset(
        DISTILBERT_BASE_EN_UNCASED_PRESET,
        sequence_length=sequence_length,
    )
