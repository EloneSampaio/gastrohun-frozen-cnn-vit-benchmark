import os
import platform
import sys

print("python:", sys.version.split()[0])
print("platform:", platform.platform())
print("cwd:", os.getcwd())

try:
    import torch

    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    print("cuda_version:", torch.version.cuda)
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        print("gpu_memory_gb:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
except ImportError as exc:
    print("torch_import_error:", exc)

for module_name in (
    "torchvision",
    "transformers",
    "sklearn",
    "pandas",
    "scipy",
    "scikit_posthocs",
    "numpy",
    "matplotlib",
    "seaborn",
    "PIL",
):
    try:
        module = __import__(module_name)
        print(f"{module_name}:", getattr(module, "__version__", "available"))
    except ImportError as exc:
        print(f"{module_name}_import_error:", exc)
