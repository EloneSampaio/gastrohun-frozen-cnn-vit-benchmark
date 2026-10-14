import os
import subprocess
import sys

for path in (
    "/content/gastrohun-benchmark/src",
    "/content/gastrohun-benchmark/configs",
    "/content/gastrohun-benchmark/outputs",
    "/content/gastrohun-data/images",
    "/content/gastrohun-data/official_splits",
):
    os.makedirs(path, exist_ok=True)

subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", "scikit-posthocs"],
    check=True,
)
print("scikit-posthocs installed")
