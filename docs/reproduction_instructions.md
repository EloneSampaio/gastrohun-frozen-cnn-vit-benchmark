# Reproduction Instructions

This document explains, step by step, how to reproduce the tables and
figures reported in the paper from a clean environment.

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

GPU strongly recommended (the paper's runs used an NVIDIA GPU on Google
Colab Pro; an L4 or A100-class GPU is roughly what feature extraction was
timed against). CPU-only extraction of ~5,300 images across 13 backbones
will run but is considerably slower.

## 2. Obtain the GastroHUN dataset

GastroHUN is **not** redistributed in this repository. Obtain it from the
official source described in:

> Bravo, D. et al. *GastroHUN: an endoscopy dataset of complete systematic
> screening protocol for the stomach.* Sci. Data 12, 102 (2025).
> DOI: 10.1038/s41597-025-04401-5

Follow the data-access procedure described in that paper / its associated
repository to obtain:

1. The image files (`Labeled Images`, or equivalent).
2. The official split CSV, `image_classification.csv`, containing at least
   the columns `filename`, `set_type` (`Train` / `Validation` / `Test`),
   and the annotator-agreement columns used as labels (`FG agreement` for
   train/validation, `Complete agreement` for test).

Once downloaded, arrange the files locally (any layout works, as long as
you point the environment variables below at it), e.g.:

```
data/gastrohun/
├── images/                          # flat folder, one file per image
└── official_splits/
    └── image_classification.csv
```

Then set:

```bash
export GASTROHUN_DATA_DIR="$(pwd)/data/gastrohun/images"
export GASTROHUN_SPLIT_CSV="$(pwd)/data/gastrohun/official_splits/image_classification.csv"
export GASTROHUN_FEATURES_DIR="$(pwd)/outputs/frozen_features"   # cache for embeddings
export GASTROHUN_RESULTS_DIR="$(pwd)/outputs/results"            # metrics/figures/tables
```

(On Windows/PowerShell use `$env:GASTROHUN_DATA_DIR = "..."`, etc.)

If `image_classification.csv` uses a different column name for the
train/validation consensus label in your download, edit
`LABEL_COLUMN_TRAINVAL` near the top of `src/cnn_vs_vit_pipeline_pytorch.py`
accordingly (the script prints the class distribution early on so you can
 sanity-check it against the official CSV: 5,162 train+val / 803 test).

## 3. Run the pipeline

The script is organized as sequential stages. The checked-in Python script is
the canonical reproducibility entry point; the notebook under `notebooks/` is
a legacy interactive export and is not the source of truth for the corrected
patient-grouped protocol. Running the script top to bottom reproduces
everything:

```bash
python src/cnn_vs_vit_pipeline_pytorch.py
```

Stages, in order, and what they produce:

| Stage | What it does | Key outputs |
|---|---|---|
| Data loading | Loads the official split, maps labels, prints class distribution | console only |
| ViT feature extraction | Frozen forward pass through 5 ViT/DeiT backbones, saves `[CLS]` embeddings and patient IDs | `outputs/frozen_features/<model_slug>/{train,validation,test}_{features,labels,patients}.npy` |
| CNN feature extraction | Frozen forward pass through 2 ConvNeXt + 6 torchvision CNNs (ResNet-18/50/101/152, VGG-16/19), global-average-pooled embeddings | same layout under `outputs/frozen_features/` |
| Nested 10-fold CV | Ranks all 13 backbones with a patient-grouped SVM (RBF) classifier, checkpointed per fold | `outputs/results/cv_f1macro_per_fold_all_backbones_grouped.csv` |
| Friedman + Nemenyi | Statistical significance test across backbones | `outputs/results/nemenyi_pvalues.csv`, `critical_difference_diagram.png` |
| Final test evaluation | Refits best ViT and best CNN on train+val, evaluates once on held-out test | `outputs/results/test_report_<slug>.csv` |
| Article tables/figures | Formats everything above into paper-ready tables (`.tex` + `.csv`) and 300dpi figures | `outputs/results/article_tables/`, `outputs/results/article_figures/` |

Feature extraction is the slow part and is checkpointed: if
`outputs/frozen_features/<model_slug>/` already contains the feature, label and
patient-ID files for all three splits, that backbone is skipped on rerun.
The nested-CV stage is checkpointed per fold via
`cv_f1macro_per_fold_all_backbones_grouped.csv`, so an interrupted run can be
resumed by simply re-running the script.

## 4. Hyperparameters, seed, and exact model identifiers

All of these are also listed machine-readably in
`configs/model_checkpoints.yaml`:

- Global seed: `42` (`numpy.random.seed`, `torch.manual_seed`,
  `random_state=42` on every `StratifiedGroupKFold` / `SVC` / `GridSearchCV`).
- SVM: RBF kernel, `class_weight="balanced"`, `probability=False`, and grid
  `C in {1, 10, 100}`, `gamma in {"scale"}` during ranking. The final refit
  expands gamma to `{"scale", 0.1}`. Scores for ROC/AUC are obtained from
  `decision_function`.
- Outer CV: `StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)`
  using `num patient` as the group, with the same fold partition reused across
  all 13 backbones so the Friedman test's paired-block assumption holds.
- Standardization is part of the SVM pipeline and is fitted separately inside
  each training fold.
- Backbone identifiers: see `configs/model_checkpoints.yaml`
  (`vit_backbones`, `cnn_backbones_huggingface`,
  `cnn_backbones_torchvision`). All are pulled from the Hugging Face Hub /
  torchvision at run time — no weights are stored in this repository.

## 5. Reconstructed helper functions — please verify

While preparing this repository, two functions were referenced in the
original script but were not present in the exported source
(`build_torchvision_cnn_extractor`, `critical_difference`, `plot_roc_curves`,
and `save_clinical_table_latex`). Standard, clearly-marked implementations
have been added directly in
`src/cnn_vs_vit_pipeline_pytorch.py` (search for "NOTE ON RECONSTRUCTION").
Before treating this repository as the final, camera-ready reproduction
package:

1. Re-run the CNN feature-extraction stage and confirm the resulting
   embedding shapes/values match your original `cnn_features/*` outputs.
2. Confirm the critical-difference value printed for Figure 4 matches the
   value reported in the paper / your original figure.

If either check fails, replace the reconstructed function with your
original implementation before publishing the release.

## 6. Known methodological caveat (carry into the README / paper text)

Frozen-feature + SVM numbers in this pipeline are **not directly
comparable** to the ~88% macro-F1 achieved by full fine-tuning in the
original GastroHUN benchmark paper (Bravo et al., 2025) or in this group's
own end-to-end fine-tuning study. The fair comparison here is CNN-frozen
vs. ViT-frozen under an identical extraction + SVM protocol, not
frozen vs. fine-tuned.
