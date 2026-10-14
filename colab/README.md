# Colab reproduction workflow

The Colab VM is disposable. The benchmark configuration, bootstrap code, and
remote-path conventions are stored in this directory and in the repository
root. Do not store credentials or tokens in the project.

## Create the runtime

```bash
colab new -s gastrohun-a100 --gpu A100
```

If the account rejects A100, use the exact same workflow with `--gpu T4` and
record that change in the experiment log. The active runtime must be checked
before an expensive run:

```bash
colab exec -s gastrohun-a100 -f Monografia/benchmark/colab/bootstrap.py --timeout 300
```

## Sync the inputs

The official split CSV is versioned in the dissertation workspace. The image
archive is downloaded directly in Colab from the official Figshare record by
`tools/colab_download_images.py`; videos and labeled sequences are not needed.

```bash
colab upload -s gastrohun-a100 \
  Monografia/dissertacao/GastroVision_image_classification.csv \
  /content/gastrohun-data/official_splits/image_classification.csv

colab exec -s gastrohun-a100 \
  -f Monografia/benchmark/tools/colab_download_images.py \
  --timeout 3600
```

The downloader verifies the official archive size and MD5 before extracting
all 8,834 images into `/content/gastrohun-data/images`.

## Sync and run the code

Upload the canonical source and configuration, then execute the persisted
runner. The runner sets all paths explicitly and writes checkpoints below
`/content/gastrohun-benchmark/outputs`.

```bash
colab upload -s gastrohun-a100 \
  Monografia/benchmark/src/cnn_vs_vit_pipeline_pytorch.py \
  /content/gastrohun-benchmark/src/cnn_vs_vit_pipeline_pytorch.py
colab upload -s gastrohun-a100 \
  Monografia/benchmark/configs/model_checkpoints.yaml \
  /content/gastrohun-benchmark/configs/model_checkpoints.yaml

colab exec -s gastrohun-a100 \
  -f Monografia/benchmark/colab/run_pipeline.py \
  --timeout 14400
```

The pipeline is resumable. Existing complete feature directories and CV
checkpoints are skipped by the canonical script.

## Persist outputs locally

Before stopping or abandoning a runtime, make a compressed snapshot and
download it into the project. Outputs are not safe in `/content` after the VM
is deleted.

```bash
colab exec -s gastrohun-a100 \
  -f Monografia/benchmark/colab/snapshot_outputs.py \
  --timeout 3600
colab download -s gastrohun-a100 \
  /content/gastrohun_outputs.tar.gz \
  Monografia/benchmark/archives/gastrohun_outputs_a100.tar.gz
```

After downloading, extract the archive locally under
`Monografia/benchmark/outputs/` and keep the runtime manifest beside it.

## Provenance

- Dataset record: `10.6084/m9.figshare.27308133.v1`.
- Image archive file ID: `50010117`.
- Image archive MD5: `0d6b48a36a21388123cc5d0231dbd362`.
- Split CSV: `Monografia/dissertacao/GastroVision_image_classification.csv`.
- Seed and model protocol: `configs/model_checkpoints.yaml`.
- Canonical pipeline: `src/cnn_vs_vit_pipeline_pytorch.py`.
- No HF token, Google authorization code, or personal path belongs in this
  repository.
