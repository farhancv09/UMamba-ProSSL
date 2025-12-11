# UMamba-ProSSL: Self-Supervised Large-Scale Pretraining with Multi-Task UMamba Advances Prostate Cancer Detection in Biparametric MRI
UMamba-ProSSL achieves state-of-the-art performance on a large-scale benchmark by combining a UMamba-MTL backbone with MAE-based large-scale selfsupervised pretraining, thereby advancing prostate cancer (PCa) detection. 

## Framework
<object data="figures/overall_process.pdf" type="application/pdf" width="100%" height="600px">
  <a href="figures/overall_process.pdf">UMamba-ProSSL framework (PDF)</a>
</object>

This repository keeps the two main workflows separated:

- `Pretraining/` contains the nnssl-based pipeline used to craft the MG, VF, and MAE objectives on top of a UMamba trainer.
- `Fine-tuning/` mirrors the csPCa algorithms codebase and hosts the downstream multi-task trainer plus experiment configs.

Use this readme as the quick-start guide for running both stages; the sub-readmes (`Pretraining/nnssl/readme.md` and `Fine-tuning/UMamba-ProSSL/readme.md`) hold the detailed API documentation.

## Repository overview
- `Pretraining/dataset_json/pretrain.py` – helper to assemble a `pretrain_data.json` 
- `Pretraining/nnssl/` – full nnssl checkout with custom UMambaBot architectures and trainers.
- `Fine-tuning/UMamba-ProSSL/` – To fine-tune models for MG, VF and MAE models.


## Pretraining
Follow the `Pretraining/nnssl/readme.md` instructions first; the summary below highlights the minimum steps needed to reproduce the UMamba MG, VF, and MAE runs.

### 1. Environment and paths
```bash
cd Pretraining/nnssl
python -m venv .venv && source .venv/bin/activate  # or use conda/mamba
pip install -e .
export nnssl_raw=/path/to/raw/data
export nnssl_preprocessed=/path/for/preprocessed
export nnssl_results=/path/for/experiments
```
The three environment variables are mandatory because nnssl writes and resolves dataset paths through them.

### 2. Create `pretrain_data.json`
1. Inspect `documentation/dataset_format.md` inside the nnssl repo for the exact schema.
2. Either call `nnssl_convert_openmind` for OpenMind data or run the provided helper:
   ```bash
   cd Pretraining
   python dataset_json/pretrain.py --imagesTr /data/Dataset110_DLR/imagesTr \
     --collection_index 110 --collection_name DLR --use-env-var --out dataset_json/DLR/pretrain.json
   ```
3. Copy the resulting file into `$nnssl_raw/<dataset>/pretrain_data.json`. This metadata file is what `nnssl_plan_and_preprocess` consumes when fingerprinting.

### 3. Plan and preprocess
```bash
cd Pretraining/nnssl
nnssl_plan_and_preprocess -d Dataset110_DLR
```
Planning builds the resampling strategy; preprocessing writes the cropped, normalized bloscv2 volumes into `$nnssl_preprocessed`. Verify the generated plan name (for example `nnsslPlans`) because it is referenced when launching training.

### 4. Launch the custom UMamba SSL trainers
UMamba-ProSSL extends nnssl with trainers that pair the UMambaBot encoder/decoder with MG, VF, and MAE objectives:

| Objective | Trainer name | Notes |
|-----------|--------------|-------|
| Models Genesis (MG) | `ModelGenesisTrainerUMambaBot` | Introduces reconstruction losses for anatomical priors |
| Volume Fusion (VF) | `VolumeFusionTrainerUMambaBot` | Use when multi-contrast patches are available |
| Masked Autoencoder (MAE) | `BaseMAETrainerUMambaBot` | Default masked modeling pipeline |

Example launch commands (`ID` should point to the dataset folder created during preprocessing and `CONFIG` is one of the configs shipped in `nnsslPlans`):
```bash
nnssl_train Dataset110_DLR nnsslPlans \
  -tr ModelGenesisTrainerUMambaBot -p nnsslPlans 

nnssl_train Dataset110_DLR nnsslPlans \
  -tr VolumeFusionTrainerUMambaBot -p nnsslPlans 

nnssl_train Dataset110_DLR nnsslPlans \
  -tr BaseMAETrainerUMambaBot -p nnsslPlans 
```
Set `WANDB_API_KEY`, `logger.project`, and `logger.entity` inside the trainer configs if you want to preserve the scalars logged by nnssl. Each run drops checkpoints under `$nnssl_results/<dataset>/<trainer>/<timestamp>/fold_X/network_final.pth`.

### 5. Collect checkpoints
After training finishes, pick the desired checkpoint (e.g., `checkpoint_final.pth`) and move/copy it to a central location – you will feed this path into the fine-tuning YAML (`checkpoint:` field).

## Fine tuning
The downstream pipeline mirrors the `cspca_algos` repository. Refer back to `/cspca_algos/readme.md` if you need deeper explanations about json datalists, shared modules, or inference packaging.

### 1. Environment
```bash
cd Fine-tuning/UMamba-ProSSL
conda env create -f environment.yml
conda activate umamba_mtl
wandb login
```
The environment matches csPCa experiments, so shared modules load without edits.

### 2. Prepare the experiment config
Open `Fine-tuning/UMamba-ProSSL/experiments/picai/umamba_mtl/config_f0.yaml` and update:
- `checkpoint`: absolute path to the pretrained UMamba checkpoint you exported from the nnssl stage (MG/MAE need decoder surgery, VF loads as-is).
- `logger.project`, `logger.entity`, and `logger.dir`: your WandB identifiers and log storage path.
- `data.data_dir`: root directory that contains the PI-CAI/Prostate json datalists referenced in `json_list`.
- Any fold-specific overrides such as `json_list`, `batch_size`, or augmentation knobs.

### 3. Point the runner at the YAML
In `experiments/picai/umamba_mtl/fine_tune_f0.py` set:
```python
repo_root = "/cluster/home/syedfa/SSL_study/UMamba-ProSSL/Fine-tuning/UMamba-ProSSL"
config = load_config("experiments/picai/umamba_mtl/config_f0.yaml")
```
This ensures relative imports resolve correctly and the script actually loads your YAML file.

### 4. Launch fine tuning
```bash
cd Fine-tuning/UMamba-ProSSL
python experiments/picai/umamba_mtl/fine_tune_f0.py
```
The Lightning module will:
1. Initialize the UMamba MTL network from your YAML.
2. If `checkpoint` is set, load the encoder weights and reinitialize the decoder heads (needed when adapting MAE or MG checkpoints, skipped for VF).
3. Train/eval with the callbacks specified in `config_f0.yaml`.
Use `config.resume_ckpt=True` to continue a previous attempt or set `test_mode=True` to run evaluation-only sweeps.

### 5. Optional: docker packaging
When you're satisfied with validation metrics, reuse the instructions from `Fine-tuning/UMamba-ProSSL/readme.md` (Docker export scripts under `gc_algorithms/`) to create PI-CAI submissions. Remember to update the YAML paths before freezing the container so the runtime can locate the trained weights.

## Datasets
The project expects PI-CAI-derived data under `data/` (mirroring csPCa algos):
- PI-CAI dataset: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6624726.svg)](https://doi.org/10.5281/zenodo.6624726)
- PI-CAI labels: https://github.com/DIAGNijmegen/picai_labels
Populate `json_datalists/{dataset_id}` with fold definitions (examples provided in `Fine-tuning/UMamba-ProSSL/json_datalists`), then keep the same file structure when pointing `json_list` in the YAML.



## Citation
```
@inproceedings{
larsen2025prostate,
title={Prostate Cancer Detection in Bi-Parametric {MRI} using Zonal Anatomy-Guided U-Mamba with Multi-Task Learning},
author={Michael S. Larsen and Syed Farhan Abbas and Gabriel Kiss and Mattijs Elschot and Tone F. Bathen and Frank Lindseth},
booktitle={Submitted to Medical Imaging with Deep Learning},
year={2025},
url={https://openreview.net/forum?id=ZkmVQinyAE},
note={Accepted}
}
```
