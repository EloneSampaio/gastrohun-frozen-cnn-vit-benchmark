# -*- coding: utf-8 -*-
"""cnn_vs_vit_pipeline_pytorch.py

Reproducibility code for:
    "Visual Transformers for Endoscopic Image Classification"
    Sampaio, E. I.; Balsanello, V. G. B.; Luizelli, M. C.; Flores, E. S.
    Federal University of Pampa (UNIPAMPA)

Frozen CNN vs. ViT backbones (PyTorch) as feature extractors + SVM
classifier, on GastroHUN. Stages: (1) load/inspect split, (2) ViT feature
extraction (HuggingFace), (3) CNN feature extraction (HuggingFace ConvNeXt +
torchvision ResNet/VGG), (4) SVM classifier, (5) nested 10-fold CV ranking
across all 13 backbones, (6) Friedman/Nemenyi statistical test, (7) final
test-set evaluation of best ViT vs. best CNN, (8) tables/figures export.

Seed fixed to 42 throughout (numpy, torch, all CV splits, SVM). See
README.md and docs/reproduction_instructions.md for setup and dataset
acquisition.
"""

import os
import time
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import (
    resnet18, resnet50, resnet101, resnet152, vgg16, vgg19,
    ResNet18_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights,
    VGG16_Weights, VGG19_Weights,
)

# `scikit-posthocs` is required for the Nemenyi test in the statistical-testing
# stage. Install it up front via `pip install -r requirements.txt` (it is NOT
# installed automatically by this script anymore -- see README.md).

np.random.seed(42)
torch.manual_seed(42)

# --- GPU check ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}"
      + (" - GPU active, extraction should be fast" if DEVICE.type == "cuda"
         else " - WARNING: no GPU detected, extraction will be slow."))

# -----------------------------------------------------------------------
# Path configuration (env vars; see docs/reproduction_instructions.md)
# -----------------------------------------------------------------------
#   GASTROHUN_DATA_DIR     -> folder with GastroHUN images (flat, one file each)
#   GASTROHUN_SPLIT_CSV    -> path to official split CSV (image_classification.csv)
#   GASTROHUN_FEATURES_DIR -> cache dir for extracted embeddings (.npy)
#   GASTROHUN_RESULTS_DIR  -> output dir for metrics/figures/tables
DATA_DIR = os.environ.get("GASTROHUN_DATA_DIR", "./data/gastrohun/images")
OFFICIAL_SPLIT = os.environ.get(
    "GASTROHUN_SPLIT_CSV",
    "./data/gastrohun/official_splits/image_classification.csv",
)
FEATURES_DIR = os.environ.get("GASTROHUN_FEATURES_DIR", "./outputs/frozen_features")
RESULTS_DIR = os.environ.get("GASTROHUN_RESULTS_DIR", "./outputs/results")

os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("Data directory:", DATA_DIR)
print("Official split CSV:", OFFICIAL_SPLIT)
print("Features directory:", FEATURES_DIR)
print("Results directory:", RESULTS_DIR)

"""## Data loading

Label column by split (per GastroHUN protocol): train/validation use
Fellow Gastroenterologist (FG) consensus; test uses full consensus
("Complete agreement"). 23 classes: A1-A6, L1-L6, P1-P6, G1-G4, OTHERCLASS.

If the exact FG-consensus column name differs in your CSV download, adjust
LABEL_COLUMN_TRAINVAL below.
"""

MAP_CATEGORIES = {
    'A1': 0, 'L1': 1, 'P1': 2, 'G1': 3,
    'A2': 4, 'L2': 5, 'P2': 6, 'G2': 7,
    'A3': 8, 'L3': 9, 'P3': 10, 'G3': 11,
    'A4': 12, 'L4': 13, 'P4': 14, 'G4': 15,
    'A5': 16, 'L5': 17, 'P5': 18,
    'A6': 19, 'L6': 20, 'P6': 21,
    'OTHERCLASS': 22
}

CLASS_NAMES = list(MAP_CATEGORIES.keys())
NUM_CLASSES = len(MAP_CATEGORIES)


def load_and_process_csv(official_split_path, label_column):
    """Loads the official split CSV and maps string labels (e.g. 'A1') to
    integers via MAP_CATEGORIES. Rows with no valid label are dropped."""
    if not os.path.exists(official_split_path):
        raise FileNotFoundError(f"CSV not found at: {official_split_path}")

    df = pd.read_csv(official_split_path, index_col=0)
    df[label_column] = df[label_column].replace(MAP_CATEGORIES).astype('Int64')
    df.dropna(subset=[label_column], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


LABEL_COLUMN_TRAINVAL = "FG agreement"   # adjust if your CSV uses a different name
LABEL_COLUMN_TEST = "Complete agreement"

df_trainval_full = load_and_process_csv(OFFICIAL_SPLIT, LABEL_COLUMN_TRAINVAL)
df_test_full = load_and_process_csv(OFFICIAL_SPLIT, LABEL_COLUMN_TEST)

train_df = df_trainval_full[df_trainval_full['set_type'] == 'Train'].copy()
val_df = df_trainval_full[df_trainval_full['set_type'] == 'Validation'].copy()
test_df = df_test_full[df_test_full['set_type'] == 'Test'].copy()

print(f"Train:      {len(train_df)} images")
print(f"Validation: {len(val_df)} images")
print(f"Test:       {len(test_df)} images (Complete agreement)")

dist = pd.DataFrame({
    "Class": CLASS_NAMES,
    "Train": [(train_df[LABEL_COLUMN_TRAINVAL] == i).sum() for i in range(NUM_CLASSES)],
    "Validation": [(val_df[LABEL_COLUMN_TRAINVAL] == i).sum() for i in range(NUM_CLASSES)],
    "Test": [(test_df[LABEL_COLUMN_TEST] == i).sum() for i in range(NUM_CLASSES)],
})
dist["Total"] = dist["Train"] + dist["Validation"] + dist["Test"]
print("\nClass distribution:")
print(dist.to_string(index=False))
# Expected: 5,162 train+val / 803 test in the official CSV used for this
# reproduction. If these don't match, check LABEL_COLUMN_TRAINVAL.

"""### Dataset

GastroHUN is NOT redistributed in this repository (see docs/reproduction_instructions.md
for the official download procedure and required data-use agreement). This
script expects the dataset to already be extracted locally, as a flat folder
of images (`DATA_DIR`) plus the official split CSV (`OFFICIAL_SPLIT`).
"""

if not os.path.exists(OFFICIAL_SPLIT):
    raise FileNotFoundError(
        f"Official split CSV not found at: {OFFICIAL_SPLIT}\n"
        "Download GastroHUN from the official source and set GASTROHUN_SPLIT_CSV "
        "(see docs/reproduction_instructions.md)."
    )
if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(
        f"Image directory not found at: {DATA_DIR}\n"
        "Download and extract GastroHUN, then set GASTROHUN_DATA_DIR "
        "(see docs/reproduction_instructions.md)."
    )

"""## ViT feature extraction

Frozen forward pass (no fine-tuning) through each pretrained ViT/DeiT
backbone; the [CLS] token embedding is saved as the image representation.

Backbones: google/vit-base-patch16-224, google/vit-large-patch32-384,
facebook/deit-base-distilled-patch16-224, facebook/deit-base-patch16-384,
facebook/vit-mae-large.

`get_hf_preprocessing_params` resolves input resolution per-model (handles
HuggingFace `SizeDict` objects, 224 vs 384 variants, and DeiT's strict
224px requirement despite the processor suggesting 256px).
"""

from transformers import AutoModel, AutoImageProcessor, AutoConfig
import collections.abc

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_batch_size(model_key):
    key = model_key.lower()
    if "large" in key or "384" in key:
        return 16
    return 32

def get_hf_preprocessing_params(model_name, default_size=224):
    processor = AutoImageProcessor.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    input_size = default_size
    if hasattr(config, "image_size"):
        input_size = config.image_size
    elif hasattr(processor, "crop_size") and isinstance(processor.crop_size, collections.abc.Mapping):
        input_size = processor.crop_size.get("height", input_size)
    elif hasattr(processor, "size"):
        size_config = processor.size
        if isinstance(size_config, (int, float)):
            input_size = int(size_config)
        elif isinstance(size_config, collections.abc.Mapping):
            input_size = size_config.get("height", size_config.get("shortest_edge", input_size))
    print(f"[preprocessing] {model_name}: input_size={input_size}")
    mean = getattr(processor, "image_mean", IMAGENET_MEAN)
    std = getattr(processor, "image_std", IMAGENET_STD)
    return int(input_size), mean, std

class EndoscopyFrozenDataset(Dataset):
    def __init__(self, df, data_dir, label_column, input_size, mean, std):
        self.filepaths = df['filename'].apply(lambda x: os.path.join(data_dir, x)).values
        self.labels = df[label_column].values.astype(np.int64)
        self.transform = T.Compose([
            T.Resize((int(input_size), int(input_size)), interpolation=T.InterpolationMode.LANCZOS),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    def __len__(self):
        return len(self.filepaths)
    def __getitem__(self, idx):
        image = Image.open(self.filepaths[idx]).convert("RGB")
        image = self.transform(image)
        label = self.labels[idx]
        return image, label

VIT_BACKBONES = {
    "google/vit-base-patch16-224": {"family": "ViT", "source": "huggingface", "pooling": "cls"},
    "google/vit-large-patch32-384": {"family": "ViT", "source": "huggingface", "pooling": "cls"},
    "facebook/deit-base-distilled-patch16-224": {"family": "ViT", "source": "huggingface", "pooling": "cls"},
    "facebook/deit-base-patch16-384": {"family": "ViT", "source": "huggingface", "pooling": "cls"},
    "facebook/vit-mae-large": {"family": "ViT", "source": "huggingface", "pooling": "cls"},
}

# NOTE ON RECONSTRUCTION
# -----------------------------------------------------------------------
# `build_torchvision_cnn_extractor` was referenced but not defined in the
# original Colab export used to prepare this release (it must have lived in
# a cell that was lost/edited out before the notebook was downloaded). The
# implementation below is a standard reconstruction: it loads the requested
# torchvision backbone with ImageNet_V1 weights (matching the GastroHUN
# benchmark protocol, see paper Section 3.1), strips the final classification
# layer, and applies global average pooling so every backbone returns a flat
# feature vector (2048-d for ResNet-50/101/152, 512-d for ResNet-18, and
# 25088-d for VGG-16/19 before pooling -> 512-d after adaptive avg pooling).
# Before trusting downstream numbers, re-run the feature-extraction stage and
# confirm the resulting embedding shapes/values reproduce your original
# `cnn_features/*` outputs.
_TORCHVISION_CTORS = {
    "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V1),
    "resnet101": (resnet101, ResNet101_Weights.IMAGENET1K_V1),
    "resnet152": (resnet152, ResNet152_Weights.IMAGENET1K_V1),
    "vgg16": (vgg16, VGG16_Weights.IMAGENET1K_V1),
    "vgg19": (vgg19, VGG19_Weights.IMAGENET1K_V1),
}


def build_torchvision_cnn_extractor(model_key: str) -> nn.Module:
    """Builds a frozen, pooled feature extractor for a torchvision CNN backbone."""
    if model_key not in _TORCHVISION_CTORS:
        raise ValueError(f"Unsupported torchvision backbone: {model_key}")
    ctor, weights = _TORCHVISION_CTORS[model_key]
    base_model = ctor(weights=weights)

    if model_key.startswith("resnet"):
        # Drop the final fully-connected layer; keep everything up to (and
        # including) the existing global average pooling layer.
        modules = list(base_model.children())[:-1]  # drops `fc`
        extractor = nn.Sequential(*modules, nn.Flatten(1))
    elif model_key.startswith("vgg"):
        # `features` -> conv stack, then adaptive average pooling to 1x1 to
        # obtain a compact, resolution-independent embedding (avoids the
        # 25,088-d flattened output of the raw 7x7 feature map).
        extractor = nn.Sequential(
            base_model.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
        )
    else:
        raise ValueError(f"Unsupported torchvision backbone: {model_key}")

    for p in extractor.parameters():
        p.requires_grad = False
    return extractor


def build_extractor_for_backbone(model_key, info):
    if info["source"] == "huggingface":
        input_size, mean, std = get_hf_preprocessing_params(model_key)
        model = AutoModel.from_pretrained(model_key)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        model.to(DEVICE)
        return model, input_size, mean, std, info["pooling"], "huggingface"
    elif info["source"] == "torchvision":
        extractor = build_torchvision_cnn_extractor(model_key)
        extractor.eval()
        extractor.to(DEVICE)
        return extractor, 224, IMAGENET_MEAN, IMAGENET_STD, None, "torchvision"
    else:
        raise ValueError(f"source unknown for {model_key}")

@torch.no_grad()
def run_extractor(model_or_extractor, source, pooling, dataloader, device=DEVICE):
    all_features, all_labels = [], []
    use_amp = device.type == "cuda"
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            if source == "huggingface":
                outputs = model_or_extractor(pixel_values=images)
                feats = outputs.last_hidden_state[:, 0, :] if pooling == "cls" else outputs.pooler_output
            else:
                feats = model_or_extractor(images)
                feats = torch.flatten(feats, 1)
        all_features.append(feats.float().cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)

def extract_features_for_backbone(model_key, info, train_df, val_df, test_df, num_workers=2):
    slug = model_key.replace("/", "_").replace("-", "_")
    out_dir = os.path.join(FEATURES_DIR, slug)

    # If the folder exists, assume extraction already ran (checks the 3 expected files)
    needed = [
        "train_features.npy", "validation_features.npy", "test_features.npy",
        "train_patients.npy", "validation_patients.npy", "test_patients.npy",
    ]
    if os.path.exists(out_dir):
        files_present = [os.path.exists(os.path.join(out_dir, f)) for f in needed]
        if all(files_present):
            print(f">>> SKIP: {model_key} already complete at {slug}.")
            return
        else:
            print(f">>> WARNING: folder {slug} exists but is incomplete. Re-extracting...")

    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*70}\nExtracting: {model_key}\n{'='*70}")
    model_or_extractor, input_size, mean, std, pooling, source = build_extractor_for_backbone(model_key, info)
    batch_size = get_batch_size(model_key)

    splits = [("train", train_df, LABEL_COLUMN_TRAINVAL),
              ("validation", val_df, LABEL_COLUMN_TRAINVAL),
              ("test", test_df, LABEL_COLUMN_TEST)]

    for split_name, df, label_col in splits:
        dataset = EndoscopyFrozenDataset(df, DATA_DIR, label_col, input_size, mean, std)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(DEVICE.type == "cuda"))
        features, labels = run_extractor(model_or_extractor, source, pooling, loader)
        np.save(os.path.join(out_dir, f"{split_name}_features.npy"), features)
        np.save(os.path.join(out_dir, f"{split_name}_labels.npy"), labels)
        np.save(
            os.path.join(out_dir, f"{split_name}_patients.npy"),
            df["num patient"].to_numpy(dtype=np.int64),
        )
        print(f"  {split_name} OK")

    del model_or_extractor
    if DEVICE.type == "cuda": torch.cuda.empty_cache()

for model_key, info in VIT_BACKBONES.items():
    extract_features_for_backbone(model_key, info, train_df, val_df, test_df)

import glob

# Look for all already-extracted feature/label files
existing_files = glob.glob(os.path.join(FEATURES_DIR, "**/*.npy"), recursive=True)

if not existing_files:
    print(f"No .npy files found at: {FEATURES_DIR}")
else:
    print(f"Files found at {FEATURES_DIR}:\n")
    # Group by folder (model) for readability
    found_models = {}
    for f in sorted(existing_files):
        model_name = os.path.basename(os.path.dirname(f))
        if model_name not in found_models:
            found_models[model_name] = []
        found_models[model_name].append(os.path.basename(f))

    for model, files in found_models.items():
        print(f"[ {model} ]")
        for file in files:
            print(f"  - {file}")
        print()

"""## CNN feature extraction

Same idea as ViT, for CNN backbones. No [CLS] token: embedding = global
average pooling over the last conv feature map.

- ConvNeXt-Tiny/Base/Large: HuggingFace `AutoModel`, `pooler_output`.
- ResNet-18/50/101/152, VGG-16/19: torchvision, ImageNet_V1 weights (same
  pretraining spec as the original GastroHUN benchmark).
"""

CNN_BACKBONES_HF = {
    "facebook/convnext-tiny-224": {"family": "CNN", "source": "huggingface", "pooling": "pooler"},
    "facebook/convnext-base-224": {"family": "CNN", "source": "huggingface", "pooling": "pooler"},
}

CNN_BACKBONES_TORCHVISION = {
    "resnet18":  {"family": "CNN", "source": "torchvision"},
    "resnet50":  {"family": "CNN", "source": "torchvision"},
    "resnet101": {"family": "CNN", "source": "torchvision"},
    "resnet152": {"family": "CNN", "source": "torchvision"},
    "vgg16":     {"family": "CNN", "source": "torchvision"},
    "vgg19":     {"family": "CNN", "source": "torchvision"},
}

CNN_BACKBONES = {**CNN_BACKBONES_HF, **CNN_BACKBONES_TORCHVISION}

for model_key, info in CNN_BACKBONES.items():
    slug = model_key.replace("/", "_").replace("-", "_")
    out_dir = os.path.join(FEATURES_DIR, slug)
    if os.path.exists(out_dir):
        print(f">>> SKIP: {model_key} already present at {slug}.")
        continue
    extract_features_for_backbone(model_key, info, train_df, val_df, test_df)

ALL_BACKBONES = {**VIT_BACKBONES, **CNN_BACKBONES}
print(f"\nTotal backbones in pipeline: {len(ALL_BACKBONES)}")

"""## SVM classifier

RBF-kernel SVM with `class_weight="balanced"` (handles class imbalance,
e.g. OTHERCLASS has 45 train samples vs. 262 for A1). Hyperparameters (C,
gamma) selected via nested cross-validation: outer loop gives an honest
F1-macro estimate on data the grid search never saw; inner loop selects
C/gamma per outer fold. Pure scikit-learn (no torch/tensorflow dependency).
"""

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import os

# Grid tuned for throughput on an A100 GPU
SVM_PARAM_GRID = {
    'svc__C': [1, 10, 100],
    'svc__gamma': ['scale'],
}

def make_svm_pipeline():
    # Fit scaling inside each training fold to prevent leakage.
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', class_weight='balanced', probability=False,
                    random_state=42, cache_size=8000)),
    ])

def nested_cv_f1_scores_with_checkpoint(X, y, groups, outer_cv, model_name, results_file):
    fold_scores = []
    if os.path.exists(results_file):
        try:
            df_tmp = pd.read_csv(results_file)
            if model_name in df_tmp.columns:
                fold_scores = df_tmp[model_name].dropna().tolist()
                print(f"      [INFO] Resuming {model_name} from fold {len(fold_scores)+1}")
        except: pass

    for i, (train_idx, val_idx) in enumerate(outer_cv.split(X, y, groups)):
        if i < len(fold_scores):
            continue

        X_train_out, X_val_out = X[train_idx], X[val_idx]
        y_train_out, y_val_out = y[train_idx], y[val_idx]
        groups_train_out = groups[train_idx]

        # Both CV levels preserve patient groups.
        inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
        grid = GridSearchCV(make_svm_pipeline(), SVM_PARAM_GRID, scoring='f1_macro', cv=inner_cv, n_jobs=-1, verbose=1)
        grid.fit(X_train_out, y_train_out, groups=groups_train_out)

        y_pred = grid.predict(X_val_out)
        score = f1_score(y_val_out, y_pred, average='macro')
        fold_scores.append(score)

        # Checkpoint por fold no disco
        current_results = {}
        if os.path.exists(results_file):
            current_results = pd.read_csv(results_file).to_dict(orient='list')

        max_len = 10
        scores_padded = fold_scores + [np.nan] * (max_len - len(fold_scores))
        current_results[model_name] = scores_padded
        pd.DataFrame(current_results).to_csv(results_file, index=False)

        print(f"      [Fold {i+1}/10] F1-macro: {score:.4f} | Params: {grid.best_params_}", flush=True)

    return np.array(fold_scores)

"""## Nested 10-fold CV across all backbones

Same `StratifiedGroupKFold(random_state=42)` object reused for all 13 backbones
(labels are identical across backbones, only features change) so the
Friedman test's paired-block comparison is valid.
"""

if 'ALL_BACKBONES' not in locals():
    vit_b = locals().get('VIT_BACKBONES', {})
    cnn_hf = locals().get('CNN_BACKBONES_HF', {})
    cnn_tv = locals().get('CNN_BACKBONES_TORCHVISION', {})
    ALL_BACKBONES = {**vit_b, **cnn_hf, **cnn_tv}

OUTER_CV = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
# Keep corrected patient-grouped scores separate from legacy ungrouped runs.
RESULTS_FILE = os.path.join(RESULTS_DIR, "cv_f1macro_per_fold_all_backbones_grouped.csv")

for model_key, info in ALL_BACKBONES.items():
    # 1. Standardize the folder name (slug) to look up cached features
    slug = model_key.replace("/", "_").replace("-", "_")
    feat_dir = os.path.join(FEATURES_DIR, slug)

    # 2. Check whether features exist in that folder
    required_files = [
        "train_features.npy", "train_labels.npy", "train_patients.npy",
        "validation_features.npy", "validation_labels.npy", "validation_patients.npy",
    ]
    if not all(os.path.exists(os.path.join(feat_dir, filename)) for filename in required_files):
        print(f">>> Skipping {model_key}: features or patient groups are incomplete.")
        continue

    # Check whether the model already has 10 folds recorded in the CSV
    if os.path.exists(RESULTS_FILE):
        df_check = pd.read_csv(RESULTS_FILE)
        # No CSV, os nomes estao como 'google/vit-base-patch16-224'
        if model_key in df_check.columns and df_check[model_key].notna().sum() == 10:
            print(f">>> Skipping {model_key}: already complete in CSV.")
            continue

    # Load features from the folder and run CV, saving under the CSV key
    X_train = np.load(os.path.join(feat_dir, "train_features.npy"))
    y_train = np.load(os.path.join(feat_dir, "train_labels.npy"))
    groups_train = np.load(os.path.join(feat_dir, "train_patients.npy"))
    X_val = np.load(os.path.join(feat_dir, "validation_features.npy"))
    y_val = np.load(os.path.join(feat_dir, "validation_labels.npy"))
    groups_val = np.load(os.path.join(feat_dir, "validation_patients.npy"))

    # The paper ranks models on the official train+validation pool. Groups
    # remain attached to each row so no patient can cross a CV boundary.
    X_rank = np.vstack([X_train, X_val])
    y_rank = np.hstack([y_train, y_val])
    groups_rank = np.hstack([groups_train, groups_val])

    print(f"\n>>> Starting nested CV for: {model_key}", flush=True)
    nested_cv_f1_scores_with_checkpoint(
        X_rank, y_rank, groups_rank, OUTER_CV, model_key, RESULTS_FILE
    )

print("\n[DONE] Processing complete (or all models already up to date).")

import pandas as pd
import os

# Path to the results file
results_path = os.path.join(RESULTS_DIR, "cv_f1macro_per_fold_all_backbones_grouped.csv")

if os.path.exists(results_path):
    df_check = pd.read_csv(results_path)
    print(f"Results found for {len(df_check.columns)} models.\n")
    print("Models saved in CSV:")
    for col in df_check.columns:
        print(f"  - {col}")

    print("\nBasic statistics (F1-macro):")
    print(df_check.describe())

    print("\nPrimeiras linhas dos scores por fold:")
    print(df_check.head())
else:
    print(f"File {results_path} does not exist yet or contains no data.")

"""## Friedman + Nemenyi statistical test

Friedman: rank-based non-parametric test for any significant difference
across the 13 backbones (10 folds as paired blocks). If H0 is rejected
(p < 0.05), Nemenyi post-hoc runs all pairwise comparisons with multiple-
comparison correction. Critical-difference (CD) diagram per Demsar (2006):
models plotted by mean rank, horizontal bars link models not significantly
different (rank gap < CD).
"""

from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

cv_results_df = pd.read_csv(os.path.join(RESULTS_DIR, "cv_f1macro_per_fold_all_backbones_grouped.csv"))

# --- Friedman Test ---
stat, p_value = friedmanchisquare(*[cv_results_df[col] for col in cv_results_df.columns])
print(f"Friedman chi2={stat:.3f}, p-value={p_value:.5f}")

# --- Post-hoc Nemenyi ---
nemenyi_pvalues = sp.posthoc_nemenyi_friedman(cv_results_df.values)
nemenyi_pvalues.columns = cv_results_df.columns
nemenyi_pvalues.index = cv_results_df.columns
nemenyi_pvalues.to_csv(os.path.join(RESULTS_DIR, "nemenyi_pvalues.csv"))

def compute_average_ranks(df):
    ranks = df.rank(axis=1, ascending=False)
    return ranks.mean(axis=0).sort_values()

avg_ranks = compute_average_ranks(cv_results_df)

# --- Critical Difference Plotting with English Mapping ---
def plot_critical_difference_diagram_article(avg_ranks, families, name_map, cd, output_path):
    k = len(avg_ranks)
    fig, ax = plt.subplots(figsize=(10, 0.5 * k + 2))

    family_colors = {"CNN": "#d62728", "ViT": "#1f77b4"}
    sorted_ranks = avg_ranks.sort_values()
    y_positions = np.arange(len(sorted_ranks), 0, -1)

    for y, (model_id, rank) in zip(y_positions, sorted_ranks.items()):
        family = families[model_id]
        article_name = name_map.get(model_id, model_id)
        ax.scatter(rank, y, color=family_colors.get(family, "gray"), s=80, zorder=3)
        ax.text(rank, y + 0.15, f"{article_name}\n(rank={rank:.2f})",
                ha='center', va='bottom', fontsize=9)

    ranks_values = sorted_ranks.values
    i = 0
    bar_y = -1.0
    while i < len(ranks_values):
        j = i
        while j + 1 < len(ranks_values) and (ranks_values[j + 1] - ranks_values[i]) <= cd:
            j += 1
        if j > i:
            ax.plot([ranks_values[i], ranks_values[j]], [bar_y, bar_y], color='black', lw=3)
            bar_y -= 0.4
        i += 1

    ax.set_xlabel(f"Average Rank (CD = {cd:.3f})")
    ax.set_yticks([])
    ax.set_title("Critical Difference Diagram - Nemenyi Post-hoc Test (alpha=0.05)")

    legend_handles = [plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=c, markersize=10, label=fam)
                       for fam, c in family_colors.items()]
    ax.legend(handles=legend_handles, loc='lower right', title="Architecture")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

# NOTE ON RECONSTRUCTION: like `build_torchvision_cnn_extractor` above,
# `critical_difference` was called but never defined in the original Colab
# export. This is the standard Nemenyi critical-difference formula (Demsar,
# 2006, Eq. 6): CD = q_alpha * sqrt(k(k+1) / (6N)), where k is the number of
# models compared, N is the number of folds/blocks, and q_alpha is the
# studentized-range critical value at alpha=0.05 for k models. Verify this
# reproduces the CD reported in the article before relying on it.
_NEMENYI_Q_ALPHA_005 = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
    9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354,
    15: 3.391, 16: 3.426,
}


def critical_difference(k: int, N: int, alpha: float = 0.05) -> float:
    """Nemenyi critical difference for comparing k models over N blocks (folds)."""
    if alpha != 0.05:
        raise ValueError("Only alpha=0.05 studentized-range table is bundled here.")
    if k not in _NEMENYI_Q_ALPHA_005:
        raise ValueError(f"No tabulated q_alpha for k={k} models; extend the table above.")
    q_alpha = _NEMENYI_Q_ALPHA_005[k]
    return q_alpha * np.sqrt((k * (k + 1)) / (6.0 * N))


# Calculate CD and plot
cd_value = critical_difference(k=len(avg_ranks), N=len(cv_results_df))
families = {k: v["family"] for k, v in ALL_BACKBONES.items()}
MODEL_NAME_MAP = {
    'facebook/vit-mae-large': 'ViT-MAE (Large)',
    'facebook/deit-base-distilled-patch16-224': 'DeiT-Distilled',
    'google/vit-large-patch32-384': 'ViT-Large',
    'facebook/deit-base-patch16-384': 'DeiT-Base (384)',
    'google/vit-base-patch16-224': 'ViT-Base',
    'resnet18': 'ResNet-18',
    'resnet50': 'ResNet-50',
    'resnet101': 'ResNet-101',
    'resnet152': 'ResNet-152',
    'vgg16': 'VGG-16',
    'vgg19': 'VGG-19',
    'facebook/convnext-base-224': 'ConvNext-Base',
    'facebook/convnext-tiny-224': 'ConvNext-Tiny'
}

plot_critical_difference_diagram_article(
    avg_ranks, families, MODEL_NAME_MAP,
    cd_value,
    output_path=os.path.join(RESULTS_DIR, "critical_difference_diagram.png")
)

"""## Final test evaluation: best ViT vs. best CNN

Picks the lowest-mean-rank model within each family, refits on train+val,
evaluates once on the held-out test set (Complete agreement).
"""

from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize

# Simplified grid for a faster final refit
SVM_FAST_GRID = {
    'svc__C': [1, 10, 100],
    'svc__gamma': ['scale', 0.1]
}

# Split original keys by family
vit_keys = [k for k, info in ALL_BACKBONES.items() if info["family"] == "ViT"]
cnn_keys = [k for k, info in ALL_BACKBONES.items() if info["family"] == "CNN"]

# Filtramos o ranking usando as chaves exatas
available_vit = avg_ranks[avg_ranks.index.isin(vit_keys)]
available_cnn = avg_ranks[avg_ranks.index.isin(cnn_keys)]

if not available_vit.empty and not available_cnn.empty:
    best_vit_key = available_vit.idxmin()
    best_cnn_key = available_cnn.idxmin()

    print(f"Best ViT (CV): {best_vit_key} (mean rank={avg_ranks[best_vit_key]:.2f})")
    print(f"Best CNN (CV): {best_cnn_key} (mean rank={avg_ranks[best_cnn_key]:.2f})")

    def final_fit_and_evaluate(model_key, results_dir=RESULTS_DIR):
        slug = model_key.replace("/", "_").replace("-", "_")
        feat_dir = os.path.join(FEATURES_DIR, slug)

        print(f"\n--- Training final refit: {model_key} ---")
        X_train = np.load(os.path.join(feat_dir, "train_features.npy"))
        y_train = np.load(os.path.join(feat_dir, "train_labels.npy"))
        X_val = np.load(os.path.join(feat_dir, "validation_features.npy"))
        y_val = np.load(os.path.join(feat_dir, "validation_labels.npy"))
        X_test = np.load(os.path.join(feat_dir, "test_features.npy"))
        y_test = np.load(os.path.join(feat_dir, "test_labels.npy"))
        groups_train = np.load(os.path.join(feat_dir, "train_patients.npy"))
        groups_val = np.load(os.path.join(feat_dir, "validation_patients.npy"))

        # Merge train + validation
        X_fit = np.vstack([X_train, X_val])
        y_fit = np.hstack([y_train, y_val])
        groups_fit = np.hstack([groups_train, groups_val])

        # Reduced inner CV (3 folds) for speed
        inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
        grid = GridSearchCV(make_svm_pipeline(), SVM_FAST_GRID, scoring='f1_macro', cv=inner_cv, n_jobs=-1, verbose=1)
        grid.fit(X_fit, y_fit, groups=groups_fit)

        y_pred = grid.best_estimator_.predict(X_test)
        y_scores = grid.best_estimator_.decision_function(X_test)

        acc = (y_pred == y_test).mean()
        f1_m = f1_score(y_test, y_pred, average='macro', zero_division=0)
        print(f"Test results: Acc={acc:.4f}, F1-Macro={f1_m:.4f}")

        # Save report
        report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
        pd.DataFrame(report).transpose().to_csv(os.path.join(results_dir, f"test_report_{slug}.csv"))

        return {"key": model_key, "y_test": y_test, "y_pred": y_pred,
                "y_scores": y_scores, "cm": confusion_matrix(y_test, y_pred)}

    result_vit = final_fit_and_evaluate(best_vit_key)
    result_cnn = final_fit_and_evaluate(best_cnn_key)
else:
    print("WARNING: check available models (best_vit_key/best_cnn_key not resolved).")

"""## Article tables and figures

Exports everything above as paper-ready tables (`article_tables/*.tex` +
`.csv`) and 300dpi figures (`article_figures/*.png`):
Table 1 class distribution; Table 2 SVM grid; Table 3 13-backbone ranking
(mean/std/CI95%); Figure 3 F1-macro boxplot; Figure 4 CD diagram; Table 4
clinical metrics (sensitivity/specificity/AUC) per class, best ViT and CNN;
Figure 5 ROC curves; Figure 6 confusion matrices; Table 5 final ViT-vs-CNN
test summary.
"""

import shutil

ARTICLE_TABLES_DIR = os.path.join(RESULTS_DIR, "article_tables")
ARTICLE_FIGURES_DIR = os.path.join(RESULTS_DIR, "article_figures")
os.makedirs(ARTICLE_TABLES_DIR, exist_ok=True)
os.makedirs(ARTICLE_FIGURES_DIR, exist_ok=True)


def df_to_latex_file(df, path, float_format="%.3f", index=False):
    """Saves a DataFrame as a LaTeX table body (for \\input inside a
    \\begin{table}...\\end{table} environment)."""
    latex_str = df.to_latex(index=index, float_format=float_format, escape=True)
    with open(path, "w") as f:
        f.write(latex_str)
    print(f"  LaTeX table: {path}")

# -----------------------------------------------------------------------
# TABLE 0 - Mapping: original name vs. paper name
# -----------------------------------------------------------------------
table0_mapping = pd.DataFrame([
    {"Original Name (Source)": k, "Article Name": v, "Family": families.get(k, "-")}
    for k, v in MODEL_NAME_MAP.items()
])

df_to_latex_file(table0_mapping, os.path.join(ARTICLE_TABLES_DIR, "table0_model_name_mapping.tex"))
table0_mapping.to_csv(os.path.join(ARTICLE_TABLES_DIR, "table0_model_name_mapping.csv"), index=False)

print("Table 0 (name mapping) created.")
print(table0_mapping)

# -----------------------------------------------------------------------
# TABLE 1 - Class distribution (train+val / test / total)
# -----------------------------------------------------------------------
table1 = pd.DataFrame({
    "Index": range(NUM_CLASSES),
    "Class": CLASS_NAMES,
    "Train+Val": dist["Train"] + dist["Validation"],
    "Test": dist["Test"],
})
table1["Total"] = table1["Train+Val"] + table1["Test"]
total_row = pd.DataFrame([{
    "Index": "-", "Class": "Total",
    "Train+Val": table1["Train+Val"].sum(),
    "Test": table1["Test"].sum(),
    "Total": table1["Total"].sum(),
}])
table1_full = pd.concat([table1, total_row], ignore_index=True)
table1_full.to_csv(os.path.join(ARTICLE_TABLES_DIR, "table1_class_distribution.csv"), index=False)
df_to_latex_file(table1_full, os.path.join(ARTICLE_TABLES_DIR, "table1_class_distribution.tex"),
                  float_format="%.0f")

# -----------------------------------------------------------------------
# TABLE 2 - SVM Hyperparameter search space
# -----------------------------------------------------------------------
table2 = pd.DataFrame([
    {"Stage": "Ranking (Nested CV, 13 backbones)", "Hyperparameter": "C",
     "Evaluated Values": str(SVM_PARAM_GRID['svc__C'])},
    {"Stage": "Ranking (Nested CV, 13 backbones)", "Hyperparameter": "gamma (RBF)",
     "Evaluated Values": str(SVM_PARAM_GRID['svc__gamma'])},
    {"Stage": "Final Adjustment (Best ViT and CNN)", "Hyperparameter": "C",
     "Evaluated Values": str(SVM_FAST_GRID['svc__C'])},
    {"Stage": "Final Adjustment (Best ViT and CNN)", "Hyperparameter": "gamma (RBF)",
     "Evaluated Values": str(SVM_FAST_GRID['svc__gamma'])},
])
table2.to_csv(os.path.join(ARTICLE_TABLES_DIR, "table2_hyperparameter_search_space.csv"), index=False)
df_to_latex_file(table2, os.path.join(ARTICLE_TABLES_DIR, "table2_hyperparameter_search_space.tex"))

# ------------------------------------------------="
# TABLE 3 - CV performance comparison (friendly model names)
# ------------------------------------------------
def compute_ci95(scores):
    mean = scores.mean()
    sem = scores.std(ddof=1) / np.sqrt(len(scores))
    return mean - 1.96 * sem, mean + 1.96 * sem

rows = []
for model in cv_results_df.columns:
    scores = cv_results_df[model].values
    ci_low, ci_high = compute_ci95(scores)
    rows.append({
        "Model": MODEL_NAME_MAP.get(model, model),
        "Family": families[model],
        "Rank (mean)": avg_ranks[model],
        "F1-score (Mean)": scores.mean(),
        "Std. Dev.": scores.std(ddof=1),
        "CI (95%)": f"[{ci_low:.3f}, {ci_high:.3f}]",
    })
table3 = pd.DataFrame(rows).sort_values("Rank (mean)").reset_index(drop=True)
table3.insert(0, "Rank", range(1, len(table3) + 1))
table3.drop(columns=["Rank (mean)"], inplace=True)

df_to_latex_file(table3, os.path.join(ARTICLE_TABLES_DIR, "table3_cv_performance.tex"))
table3.to_csv(os.path.join(ARTICLE_TABLES_DIR, "table3_cv_performance.csv"), index=False)
print(table3.head())

# ------------------------------------------------
# FIGURE 3 - Boxplot with English labels
# ------------------------------------------------
def plot_f1_boxplot_english(cv_results_df, families, name_map, output_path):
    order = cv_results_df.mean().sort_values(ascending=False).index
    clean_labels = [name_map.get(m, m) for m in order]
    palette = {"ViT": "#1f77b4", "CNN": "#d62728"}

    fig, ax = plt.subplots(figsize=(12, 6))
    box = ax.boxplot([cv_results_df[m].values for m in order],
                      tick_labels=clean_labels, patch_artist=True)

    for patch, model in zip(box['boxes'], order):
        patch.set_facecolor(palette[families[model]])

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_ylabel("F1-macro (10-fold CV)")
    ax.set_title("Performance Distribution by Architecture")

    # Legend in English
    legend_handles = [plt.Rectangle((0,0),1,1, color=c) for c in palette.values()]
    ax.legend(legend_handles, palette.keys(), loc="upper right", title="Architecture")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

plot_f1_boxplot_english(cv_results_df, families, MODEL_NAME_MAP,
                        os.path.join(ARTICLE_FIGURES_DIR, "figure3_f1_boxplot.png"))

# -----------------------------------------------------------------------
# FIGURE 4 - Critical difference diagram (Nemenyi), generated earlier;
# copy into the article figures folder with a standardized name.
# -----------------------------------------------------------------------
shutil.copy(
    os.path.join(RESULTS_DIR, "critical_difference_diagram.png"),
    os.path.join(ARTICLE_FIGURES_DIR, "figure4_critical_difference_diagram.png"),
)

# -----------------------------------------------------------------------
# TABLE 4 - Clinical metrics in English
# -----------------------------------------------------------------------
def save_clinical_table_latex(df, filename):
    """Write a clinical-metrics table into the article tables directory."""
    df_to_latex_file(df, os.path.join(ARTICLE_TABLES_DIR, filename))


def plot_roc_curves(y_true, y_scores, title, output_path):
    """Plot one-vs-rest ROC curves from multiclass decision-function scores."""
    classes = np.arange(len(CLASS_NAMES))
    y_true_bin = label_binarize(y_true, classes=classes)

    fig, ax = plt.subplots(figsize=(10, 8))
    plotted = 0
    for index, class_name in enumerate(CLASS_NAMES):
        if y_true_bin[:, index].min() == y_true_bin[:, index].max():
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, index], y_scores[:, index])
        ax.plot(fpr, tpr, linewidth=1.3, label=class_name)
        plotted += 1

    ax.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1)
    ax.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title=title,
        xlim=(0, 1),
        ylim=(0, 1),
    )
    if plotted:
        ax.legend(title="Class", ncol=2, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def results_to_clinical_df_english(result_dict):
    report = classification_report(result_dict['y_test'],
                                   result_dict['y_pred'],
                                   target_names=CLASS_NAMES,
                                   output_dict=True,
                                   zero_division=0)
    df = pd.DataFrame(report).transpose()
    # English column renaming
    df.columns = ["Precision", "Recall (Sensitivity)", "F1-Score", "Support"]
    return df

clin_vit = results_to_clinical_df_english(result_vit)
clin_cnn = results_to_clinical_df_english(result_cnn)

save_clinical_table_latex(clin_vit, "table4_clinical_metrics_vit.tex")
save_clinical_table_latex(clin_cnn, "table4_clinical_metrics_cnn.tex")

# ------------------------------------------------
# FIGURE 5 - ROC curves with friendly titles
# ------------------------------------------------
# Uses MODEL_NAME_MAP for figure titles
plot_roc_curves(result_vit["y_test"], result_vit["y_scores"], MODEL_NAME_MAP.get(best_vit_key, best_vit_key),
                 os.path.join(ARTICLE_FIGURES_DIR, "figure5a_roc_vit.png"))
plot_roc_curves(result_cnn["y_test"], result_cnn["y_scores"], MODEL_NAME_MAP.get(best_cnn_key, best_cnn_key),
                 os.path.join(ARTICLE_FIGURES_DIR, "figure5b_roc_cnn.png"))

import seaborn as sns

def plot_confusion_matrix_article(cm, title, output_path, model_slug):
    # 1. Save visual plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(title)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

    # 2. Save numerical CSV for detailed analysis of confusions
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    csv_path = os.path.join(ARTICLE_TABLES_DIR, f"confusion_matrix_data_{model_slug}.csv")
    cm_df.to_csv(csv_path)
    print(f"  Figure saved: {output_path}")
    print(f"  Numerical CSV saved: {csv_path}")

# -----------------------------------------------------------------------
# FIGURE 6 - Confusion Matrices & Numerical CSVs (English Labels)
# -----------------------------------------------------------------------
vit_slug = best_vit_key.replace("/", "_").replace("-", "_")
cnn_slug = best_cnn_key.replace("/", "_").replace("-", "_")

vit_article_name = MODEL_NAME_MAP.get(best_vit_key, best_vit_key)
cnn_article_name = MODEL_NAME_MAP.get(best_cnn_key, best_cnn_key)

plot_confusion_matrix_article(result_vit["cm"], f"Confusion Matrix - {vit_article_name}",
                               os.path.join(ARTICLE_FIGURES_DIR, "figure6a_confusion_matrix_vit.png"),
                               vit_slug)

plot_confusion_matrix_article(result_cnn["cm"], f"Confusion Matrix - {cnn_article_name}",
                               os.path.join(ARTICLE_FIGURES_DIR, "figure6b_confusion_matrix_cnn.png"),
                               cnn_slug)

# -----------------------------------------------------------------------
# TABLE 5 - Final summary: best ViT vs. best CNN on test
# -----------------------------------------------------------------------
# Criando o DataFrame 'summary' que estava faltando
summary_data = []
for res in [result_vit, result_cnn]:
    y_test = res["y_test"]
    y_pred = res["y_pred"]

    acc = (y_pred == y_test).mean()
    f1_m = f1_score(y_test, y_pred, average='macro', zero_division=0)

    summary_data.append({
        "Family": families[res["key"]],
        "Model": res["key"],
        "Test Accuracy": acc,
        "Test F1-Macro": f1_m
    })

summary = pd.DataFrame(summary_data)

# Export files
df_to_latex_file(summary, os.path.join(ARTICLE_TABLES_DIR, "table5_final_summary_vit_vs_cnn.tex"),
                  float_format="%.4f")
summary.to_csv(os.path.join(ARTICLE_TABLES_DIR, "table5_final_summary_vit_vs_cnn.csv"), index=False)

print("\nTable 5 (Final Summary):")
print(summary.to_string(index=False))

print(f"\nTables (.csv + .tex) at: {ARTICLE_TABLES_DIR}")
print(f"Figures (.png, 300dpi) at: {ARTICLE_FIGURES_DIR}")
print("\nPipeline complete.")



"""### Clinical metrics tables (final, friendly model names)"""

from sklearn.metrics import roc_auc_score

def calculate_clinical_metrics(y_true, y_scores, class_names, y_pred):
    """Computes sensitivity, specificity and AUC per class (one-vs-rest)."""
    metrics_list = []

    for i, class_name in enumerate(class_names):
        # One-vs-rest binarization
        y_true_bin = (y_true == i).astype(int)
        y_pred_bin = (y_pred == i).astype(int)

        tp = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
        tn = np.sum((y_true_bin == 0) & (y_pred_bin == 0))
        fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
        fn = np.sum((y_true_bin == 1) & (y_pred_bin == 0))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        try:
            auc = roc_auc_score(y_true_bin, y_scores[:, i])
        except:
            auc = np.nan

        metrics_list.append({
            "Class": class_name,
            "Sensitivity (Recall)": sensitivity,
            "Specificity": specificity,
            "AUC": auc
        })

    df_metrics = pd.DataFrame(metrics_list)
    # Append macro-average row
    macro_row = pd.DataFrame([{
        "Class": "Macro Average",
        "Sensitivity (Recall)": df_metrics["Sensitivity (Recall)"].mean(),
        "Specificity": df_metrics["Specificity"].mean(),
        "AUC": df_metrics["AUC"].dropna().mean()
    }])
    return pd.concat([df_metrics, macro_row], ignore_index=True)

# Generate clinical metrics tables
clin_metrics_vit = calculate_clinical_metrics(
    result_vit["y_test"], result_vit["y_scores"], CLASS_NAMES, result_vit["y_pred"]
)
clin_metrics_cnn = calculate_clinical_metrics(
    result_cnn["y_test"], result_cnn["y_scores"], CLASS_NAMES, result_cnn["y_pred"]
)

# Export with the final paper-ready names
df_to_latex_file(clin_metrics_vit, os.path.join(ARTICLE_TABLES_DIR, "table4_clinical_metrics_vit.tex"), float_format="%.3f")
df_to_latex_file(clin_metrics_cnn, os.path.join(ARTICLE_TABLES_DIR, "table4_clinical_metrics_cnn.tex"), float_format="%.3f")

clin_metrics_vit.to_csv(os.path.join(ARTICLE_TABLES_DIR, "table4_clinical_metrics_vit.csv"), index=False)
clin_metrics_cnn.to_csv(os.path.join(ARTICLE_TABLES_DIR, "table4_clinical_metrics_cnn.csv"), index=False)

print("Table 4 updated with sensitivity, specificity and AUC.")
print(clin_metrics_vit.tail(1))



"""## Output-to-paper mapping

cv_f1macro_per_fold_all_backbones_grouped.csv -> Table 3 (mean/std/CI per backbone)
critical_difference_diagram.png       -> Figure 4
nemenyi_pvalues.csv                   -> pairwise significance for the text
test_report_<model>.csv (x2)          -> per-class test report, best ViT/CNN
clinical_metrics_best_vit/_cnn.csv    -> Table 4
cm_<model>.png (x2)                   -> Figure 6
final_summary_vit_vs_cnn.csv          -> Table 5 / results-section headline

Caveat: these frozen-feature + SVM numbers are NOT directly comparable to
the ~88% macro-F1 from full fine-tuning in the original GastroHUN
benchmark. The fair comparison here is CNN-frozen vs. ViT-frozen under one
identical protocol, not frozen vs. fine-tuned.
"""

print("\nPipeline complete. See output files in:", RESULTS_DIR)

import sys, torch, torchvision, transformers, sklearn, pandas
print(sys.version)
print(torch.__version__)
print(torchvision.__version__)
print(transformers.__version__)
print(sklearn.__version__)
print(pandas.__version__)
