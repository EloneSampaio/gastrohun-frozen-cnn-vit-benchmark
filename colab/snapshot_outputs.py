import os
import tarfile

source = "/content/gastrohun-benchmark"
target = "/content/gastrohun_outputs.tar.gz"
members = (
    "outputs",
    "configs/model_checkpoints.yaml",
    "src/cnn_vs_vit_pipeline_pytorch.py",
    "colab_runtime_manifest.json",
)

with tarfile.open(target, "w:gz") as archive:
    for member in members:
        path = os.path.join(source, member)
        if os.path.exists(path):
            archive.add(path, arcname=member)
print(f"snapshot: {target}")
print(f"bytes: {os.path.getsize(target)}")
