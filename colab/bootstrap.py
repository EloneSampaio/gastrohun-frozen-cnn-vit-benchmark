import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

REMOTE_ROOT = "/content/gastrohun-benchmark"
for path in (
    f"{REMOTE_ROOT}/src",
    f"{REMOTE_ROOT}/configs",
    f"{REMOTE_ROOT}/outputs/frozen_features",
    f"{REMOTE_ROOT}/outputs/results",
    "/content/gastrohun-data/images",
    "/content/gastrohun-data/official_splits",
):
    os.makedirs(path, exist_ok=True)

# Keep the Colab CUDA stack intact; install only the package absent from the
# base runtime and record every relevant version below.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", "scikit-posthocs==0.17.1"],
    check=True,
)

packages = (
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "scikit-posthocs",
    "matplotlib",
    "seaborn",
    "Pillow",
    "torch",
    "torchvision",
    "transformers",
)
manifest = {
    "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    "python": sys.version,
    "platform": platform.platform(),
    "cwd": os.getcwd(),
    "packages": {},
    "paths": {
        "remote_root": REMOTE_ROOT,
        "data_dir": "/content/gastrohun-data/images",
        "split_csv": "/content/gastrohun-data/official_splits/image_classification.csv",
        "features_dir": f"{REMOTE_ROOT}/outputs/frozen_features",
        "results_dir": f"{REMOTE_ROOT}/outputs/results",
    },
}
for package in packages:
    try:
        manifest["packages"][package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        manifest["packages"][package] = None

try:
    import torch

    manifest["cuda"] = {
        "available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
except ImportError as exc:
    manifest["cuda"] = {"error": str(exc)}

manifest_path = f"{REMOTE_ROOT}/colab_runtime_manifest.json"
with open(manifest_path, "w", encoding="utf-8") as stream:
    json.dump(manifest, stream, indent=2)
print(json.dumps(manifest, indent=2))
