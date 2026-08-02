# gastrohun-frozen-cnn-vit-benchmark

Reproducibility code for a benchmark of **frozen** CNN and Vision
Transformer (ViT) backbones as feature extractors, coupled with an SVM
classifier, for multi-class anatomical endoscopic image classification on
the [GastroHUN](https://doi.org/10.1038/s41597-025-04401-5) dataset.

> Under an identical *frozen extraction + SVM* protocol, which architecture
> family — CNN or ViT — produces more discriminative embeddings for
> endoscopic image classification?

The pipeline compares 5 ViT-family backbones (ViT-Base, ViT-Large, DeiT
variants, ViT-MAE) against 9 CNN backbones (ConvNeXt-Tiny/Base/Large,
ResNet-18/50/101/152, VGG-16/19), all frozen, followed by an SVM (RBF
kernel) classifier tuned via nested cross-validation, with Friedman/Nemenyi
statistical testing and a final held-out test evaluation of the best model
per family.

## Repository structure

```
gastrohun-frozen-cnn-vit-benchmark/
├── README.md                      <- you are here
├── LICENSE                        <- MIT (code only, see note in LICENSE)
├── CITATION.cff                   <- GitHub "Cite this repository" metadata
├── requirements.txt               <- pinned dependency versions
├── .gitignore                     <- excludes data, embeddings, weights, secrets
├── src/
│   └── cnn_vs_vit_pipeline_pytorch.py   <- full pipeline (extraction -> CV -> stats -> figures)
├── notebooks/
│   └── cnn_vs_vit_pipeline_pytorch.ipynb  <- same pipeline, as a notebook
├── configs/
│   └── model_checkpoints.yaml     <- exact backbone IDs, seed, SVM grid, CV protocol
└── docs/
    └── reproduction_instructions.md  <- step-by-step reproduction guide
```

## Quickstart

```bash
git clone https://github.com/<your-username>/gastrohun-frozen-cnn-vit-benchmark.git
cd gastrohun-frozen-cnn-vit-benchmark
pip install -r requirements.txt

# 1. Obtain GastroHUN from the official source (see docs/reproduction_instructions.md)
#    and point these at your local copy:
export GASTROHUN_DATA_DIR=/path/to/gastrohun/images
export GASTROHUN_SPLIT_CSV=/path/to/gastrohun/official_splits/image_classification.csv

# 2. Run the full pipeline
python src/cnn_vs_vit_pipeline_pytorch.py
```

Outputs (embeddings, CV results, statistical tests, paper-ready tables and
figures) are written under `outputs/` by default — see
[`docs/reproduction_instructions.md`](docs/reproduction_instructions.md)
for the full breakdown of what each stage produces and how to resume an
interrupted run.

## What is / isn't in this repository

**Included:** full pipeline code, exact hyperparameters and seeds, library
version pins, dataset-acquisition instructions, and instructions to
regenerate every table and figure.

**Deliberately excluded** (see `.gitignore`): GastroHUN images, any
extracted embeddings (`.npy`), pretrained model weights/checkpoints, and
anything resembling credentials or personal file paths. Pretrained weights
are fetched at run time from the Hugging Face Hub / torchvision using the
identifiers pinned in `configs/model_checkpoints.yaml`.

## Methodological note

Frozen-feature + SVM results here are **not directly comparable** to
full fine-tuning numbers (e.g., the ~88% macro-F1 reported for full
fine-tuning in the original GastroHUN paper). The fair comparison enabled
by this repository is CNN-frozen vs. ViT-frozen under one identical
protocol — see `docs/reproduction_instructions.md` for details.

## Two functions were reconstructed during cleanup — please verify

`build_torchvision_cnn_extractor` and `critical_difference` were called in
the original notebook export but their definitions were missing (most
likely lost in an earlier Colab edit). Standard reimplementations are
included and clearly marked with `NOTE ON RECONSTRUCTION` comments in
`src/cnn_vs_vit_pipeline_pytorch.py`. Please re-run and diff against your
original CNN embeddings / critical-difference figure before treating this
as the final camera-ready release — see
[`docs/reproduction_instructions.md`](docs/reproduction_instructions.md#5-two-reconstructed-helper-functions--please-verify).

## Citation

If you use this code, please cite it — see [`CITATION.cff`](CITATION.cff)
(GitHub will surface a "Cite this repository" button automatically once
this file is present). Update the `preferred-citation` block with the
final journal/DOI once the paper is published.

## License

Code is released under the [MIT License](LICENSE). This does **not**
extend to the GastroHUN dataset or to pretrained model weights, which
retain their own upstream licenses/terms.
