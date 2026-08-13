# Quantization Group Studies

This project studies post-training quantization using pretrained TensorFlow/Keras models.

The current implementation starts with a pretrained ResNet-18 model. The full project scope includes both weight quantization and activation quantization, with M/M reduction and compression reported separately for model weights and intermediate activations.

See the detailed scope document:

- [Project scope](docs/project_scope.md)

## Project Steps

1. Load a pretrained ResNet-18 model.
2. Measure FP32 baseline weight and activation memory.
3. Apply built-in TensorFlow/TFLite PTQ as a learning baseline.
4. Implement custom 8-bit weight quantization.
5. Implement custom 8-bit activation quantization.
6. Compare M/M reduction and compression for weights and activations.

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

## Built-In PTQ Baseline

Convert ResNet-18 to an unquantized TFLite baseline, dynamic-range INT8, and
full INT8 weights/activations/input/output:

```bash
python3 scripts/benchmark_builtin_ptq_resnet18.py
```

The script uses images in `data/` for full INT8 activation calibration and
writes generated models and `results.json` under
`artifacts/resnet18_builtin_ptq/`.

The report includes:

- Serialized model size and compression
- Constant weight-buffer memory and compression
- Intermediate activation tensor-storage estimate and compression
- Input/output and intermediate tensor data types

The activation value is the sum of nonconstant tensor sizes represented in the
TFLite graph. It is useful for a consistent comparison, but it is not the exact
peak interpreter arena memory because the runtime may reuse tensor buffers.

For meaningful calibration, place approximately 50-200 representative images
in `data/`. The current sample image is sufficient only for a conversion smoke
test.

## DistilBERT PTQ on SST-2

The DistilBERT investigation loads the KerasHub
`distil_bert_base_en_uncased` pretrained weights, fine-tunes and caches an
SST-2 sentiment classifier, then compares the FP32 model with built-in TFLite
dynamic-range PTQ and custom per-channel INT8 weight PTQ:

```text
notebooks/09_compare_distilbert_sst2_ptq.ipynb
```

The notebook downloads SST-2 on its first run and reports prediction accuracy,
parameter ranks, quantized tensor/value counts, and parameter-memory reduction.
