# Quantization Group Studies

This project studies post-training quantization using pretrained TensorFlow/Keras models.

The current implementation starts with a pretrained ResNet-18 model. The full project scope includes both weight quantization and activation quantization, with M/M reduction and compression reported separately for model weights and intermediate activations.

See the detailed scope document:

- [Project scope](docs/project_scope.md)
- [Codebook vector quantization](docs/codebook_vector_quantization.md)

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

## Vision Transformer PTQ on CIFAR-10

The Vision Transformer investigation loads the KerasHub
`vit_base_patch16_224_imagenet` pretrained weights, trains and caches a CIFAR-10
classification head, then compares the FP32 model with built-in TFLite
dynamic-range PTQ and custom per-channel INT8 weight PTQ:

```text
notebooks/10_compare_vit_cifar10_ptq.ipynb
```

The notebook reports accuracy, parameter ranks, quantized tensor/value counts,
TFLite graph dtypes, and parameter-memory reduction.

The full and compute-bounded sensitivity studies are:

- `notebooks/17_resnet18_full_cifar10_ptq_codebook_layer_sensitivity.ipynb` — ResNet-18 W8 layer sensitivity.
- `notebooks/18_distilbert_full_sst2_ptq_codebook_layer_sensitivity.ipynb` — DistilBERT W8 tensor sensitivity.
- `notebooks/19_vit_full_cifar10_ptq_codebook_module_sensitivity.ipynb` — ViT W8 module sensitivity.
- `notebooks/20_resnet18_full_cifar10_4bit_ptq_codebook_layer_sensitivity.ipynb` — ResNet-18 W4 analysis and W4/W8 comparison.
- `notebooks/21_distilbert_sst2_4bit_stratified_sensitivity_w4_w8.ipynb` — compute-bounded DistilBERT W4/W8 study.
- `notebooks/22_vit_cifar10_4bit_stratified_sensitivity_w4_w8.ipynb` — compute-bounded ViT W4/W8 study using all 10,000 CIFAR-10 test images.
- `notebooks/23_distilbert_sst2_ptq_codebook_2_to_8_bit_sweep.ipynb` — full DistilBERT PTQ/codebook 2–8-bit accuracy and compression sweep.
- `notebooks/24_vit_cifar10_ptq_codebook_2_to_8_bit_sweep.ipynb` — resumable full-test ViT PTQ/codebook 2–8-bit accuracy and compression sweep.
- `notebooks/25_consolidated_full_model_ptq_codebook_2_to_8_bit_results.ipynb` — read-only report-data loader for FP32 and W8 tables plus saved 2–8-bit accuracy, compression, and memory-reduction results; it performs no model loading, quantization, training, or inference.
- `notebooks/26_resnet18_full_cifar10_ptq_codebook_2_to_8_bit_sweep.ipynb` — resumable full-test ResNet-18 PTQ and scalar-codebook W2–W8 sweep across all 21 eligible tensors, producing accuracy, compression-ratio, and parameter-memory-reduction tables and graphs.
- `notebooks/27_vit_w4_w8_sensitivity_heatmap_from_notebook22.ipynb` — executed ViT W4/W8 heatmap notebook using Notebook 22's saved matched results; exports PNG/PDF, the 28 plotted values and a source-checksum manifest without rerunning inference.

## Codebook VQ: library baseline vs custom implementation

The first codebook comparison uses the trained CIFAR-10 ResNet-18 and compares
scikit-learn's built-in `KMeans` codebook construction with the project's
custom mathematical NumPy implementation under identical vector dimensions,
codebook sizes, tensor coverage, and evaluation conditions:

```text
notebooks/11_compare_sklearn_custom_codebook_vq_resnet18.ipynb
```


Jacob, B., Kligys, S., Chen, B., Zhu, M., Tang, M., Howard, A., Adam, H.,
  and Kalenichenko, D. (2018). Quantization and training of neural networks
  for efficient integer-arithmetic-only inference. CVPR, 2704–2713.
  arXiv:1712.05877

Krishnamoorthi, R. (2018). Quantizing deep convolutional networks for
  efficient inference: A whitepaper. arXiv:1806.08342

Nagel, M., Fournarakis, M., Amjad, R.A., Bondarenko, Y., van Baalen, M.,
  and Blankevoort, T. (2021). A white paper on neural network quantization.
  arXiv:2106.08295

  PTQ research papers used



  
