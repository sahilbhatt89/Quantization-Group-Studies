"""Pretrained TensorFlow/Keras ResNet-18 model loader.

Research provenance: He et al. (CVPR 2016) defines residual learning, basic
blocks, identity shortcuts, and projection shortcuts.  This repository does
not reimplement those blocks; KerasHub supplies the executable
``resnet_18_imagenet`` architecture and pretrained weights.  Project code
reuses its backbone and replaces the task head for CIFAR-10.
"""

from __future__ import annotations

import keras_hub
import tensorflow as tf
from tensorflow import keras


RESNET18_IMAGENET_PRESET = "resnet_18_imagenet"
CIFAR10_CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def load_pretrained_resnet18():
    """Load the ImageNet-pretrained ResNet-18 classifier from KerasHub."""
    return keras_hub.models.ResNetImageClassifier.from_preset(
        RESNET18_IMAGENET_PRESET,
        load_weights=True,
    )


def build_resnet18_cifar10_classifier(
    num_classes: int = 10,
    freeze_backbone: bool = True,
) -> keras.Model:
   
    # He et al. supplies the architecture; KerasHub is the software source.
    # The CIFAR-10 dataset facts/labels are attributed to Krizhevsky (2009).
    imagenet_model = load_pretrained_resnet18()
    backbone = imagenet_model.get_layer("res_net_backbone")
    pooler = imagenet_model.get_layer("pooler")
    dropout = imagenet_model.get_layer("output_dropout")

    backbone.trainable = not freeze_backbone

    inputs = keras.Input(shape=(224, 224, 3), dtype=tf.float32, name="images")
    if freeze_backbone:
        x = backbone(inputs, training=False)
    else:
        x = backbone(inputs)
    x = pooler(x)
    x = dropout(x)
    outputs = keras.layers.Dense(num_classes, name="cifar10_predictions")(x)

    model = keras.Model(inputs, outputs, name="resnet18_cifar10_classifier")
    model.preprocessor = imagenet_model.preprocessor
    return model
