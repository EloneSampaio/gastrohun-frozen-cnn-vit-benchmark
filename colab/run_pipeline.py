import os
import runpy

os.environ.update(
    {
        "GASTROHUN_DATA_DIR": "/content/gastrohun-data/images",
        "GASTROHUN_SPLIT_CSV": "/content/gastrohun-data/official_splits/image_classification.csv",
        "GASTROHUN_FEATURES_DIR": "/content/gastrohun-benchmark/outputs/frozen_features",
        "GASTROHUN_RESULTS_DIR": "/content/gastrohun-benchmark/outputs/results",
    }
)
os.chdir("/content/gastrohun-benchmark")
runpy.run_path("src/cnn_vs_vit_pipeline_pytorch.py", run_name="__main__")
