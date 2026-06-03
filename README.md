# Quantization Group Studies

This project studies post-training quantization using an already pretrained ResNet-18 model.

## Project Steps

1. Load a pretrained ResNet-18 model.
2. Apply post-training quantization and evaluate the result.
3. Implement a custom PTQ method and compare the result.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Pretrained ResNet-18

Load the pretrained TensorFlow/Keras ResNet-18 model and print its summary:

```bash
python3 scripts/load_pretrained_resnet18.py
```

You can also test the model in the baseline notebook:

```text
notebooks/01_test_pretrained_resnet18.ipynb
```

This project uses the KerasHub `resnet_18_imagenet` preset:

- Model documentation: https://keras.io/keras_hub/api/models/resnet/resnet_image_classifier/
- Preset: `resnet_18_imagenet`

The first run may download the ImageNet pretrained weights through KerasHub.
