# HS-WAE: A Pipeline for Early Insect Damage Detection in Hyperspectral Soybean Data by Enhanced Deep Autoencoder

[![IEEE](https://img.shields.io/badge/IEEE-00629B.svg?style=for-the-badge&logo=IEEE&logoColor=white)][ieee-paper]
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official repository of "[HS-WAE: A Pipeline for Early Insect Damage Detection in Hyperspectral Soybean Data by Enhanced Deep Autoencoder][ieee-paper]", accepted in **IEEE Transactions on Artificial Intelligence**. IEEE Xplore link coming soon.

## Abstract

*Early detection of pest-induced stress in soybean plants is challenging since spectral differences are subtle during the initial stages of infestation. This work proposes HS-WAE, a spectral deep learning pipeline combining spectral band grouping, discrete wavelet transform (DWT), a wavelet token attention module (WTAM), and an autoencoder-based classifier. DWT provides a multiscale spectral representation, while WTAM adaptively weights the resulting wavelet levels for each sample before latent-space compression and classification. The method was evaluated on a soybean dataset acquired over seven consecutive days after infestation, considering multiple spectral regions and repeated runs with statistical analysis. The results show that DWT is a key component, significantly outperforming the raw spectral input and enabling earlier discrimination between control and damaged plants. In the near-infrared (NIR) region, HS-WAE achieved 97.2% accuracy, 97.9% F1-score, and 100.0% AUROC on day 02, and perfect metrics on day 03, whereas the raw spectral input achieved its best NIR performance only later. The WTAM further refines the DWT representation through sample-specific wavelet-level weighting, with repeated-run latent-space and counterfactual analyses confirming its functional relevance, particularly in the more challenging multiclass setting. Experiments on the public HyperBlood dataset also show that the proposed pipeline generalizes to a different hyperspectral domain and multiclass setting. Overall, DWT-based spectral representations provide substantial advantages over raw spectra in informative spectral regions, and HS-WAE is effective for early pest-damage detection in soybean plants.*

<img src="hs-wae-pipeline.png" alt="HS-WAE pipeline: spectral preprocessing, DWT, wavelet token attention, and autoencoder-based classification." width="1000">

## Contents

1. [Dataset](#1-dataset)
2. [Setup](#2-setup)
3. [Usage](#3-usage)
4. [Evaluation](#4-evaluation)
5. [Citation](#citation)
6. [License](#license)

## 1. Dataset

**Soybean:** available upon request. Contact **Prof. Liang Zhao** at **[zhao@usp.br](mailto:zhao@usp.br)**.

**HyperBlood:** download from [Zenodo](https://zenodo.org/records/3984905); see the [original repository](https://github.com/iitis/HSI_blood_detection) and [preparation instructions](data/README.md).

Process the annotated hyperspectral images into **one mean hyperspectral curve per object** (connected component). Averaging its pixels at each wavelength produces one labeled spectrum per CSV row, matching the soybean pipeline's sample format.

Place the processed CSVs as follows:

```text
data/HS-Class/soybean_HSI.csv
data/HSI_Blood/HSI_Blood.csv
```

## 2. Setup

From the repository root, create and activate the Conda environment (Python 3.10):

```bash
conda env create -f environment.yml
conda activate hs-wae
python -m pip install --upgrade pip
```

Install **one** PyTorch build:

```bash
# NVIDIA GPU: CUDA 13.0 (compatible driver required)
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu130
```

```bash
# CPU only
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
```

Install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip check
```

See [environment notes](environment/README.md) for other PyTorch builds.

## 3. Usage

Use Bash (Git Bash on Windows) from the HS-WAE root. Native Python commands are in the [execution guide](RUNNING.md).

### 3.1. Training

Paper experiments: **25 repetitions, up to 300 epochs each**.

```bash
bash scripts/run_soybean.sh hswae --band_region nir --days 2 --devices 0
bash scripts/run_soybean.sh raw --band_region nir --days 2 --devices 0
bash scripts/run_hyperblood.sh hswae --band_region vis --devices 0
bash scripts/run_hyperblood.sh raw --band_region vis --devices 0
```

CPU execution:

```bash
python -m tools.configure --cpu --num-workers 0
CONFIG=configs/local/dwt_attn.yaml bash scripts/run_soybean.sh hswae --band_region nir --days 2
```

Use `--overwrite` to regenerate existing local configs. Ablations, controlled WTAM variants, and baselines are listed in the [execution guide](RUNNING.md#ablations-and-baselines).

### 3.2. Single run and checkpoint evaluation

```bash
cd HS-AE
python -m main --mode train --config configs/dwt_attn.yaml --band_region nir --days 2 --devices 0
python -m main --mode test --config configs/dwt_attn.yaml --band_region nir --days 2 --devices 0 --ckpt /path/to/model.ckpt
cd ..
```

### 3.3. TensorBoard

The standard training scripts log losses, metrics, and confusion matrices. From the HS-WAE root, in a second terminal with `hs-wae` activated:

```bash
tensorboard --logdir . --port 6006
```

Open **[http://localhost:6006](http://localhost:6006)**. Runs are stored under each project's `logs/` or `v0/logs/` directory. Controlled WTAM runs save result files without TensorBoard logging; use the [plotting commands](RUNNING.md#tensorboard-and-saved-plots).

## 4. Evaluation

Evaluate a selected checkpoint on the test split. From the repository root, this example uses soybean **Day 2** and the **NIR** spectral region:

```bash
cd HS-AE

python -m main --mode test \
  --config configs/dwt_attn.yaml \
  --ckpt "/path/to/selected_checkpoint.ckpt" \
  --days 2 \
  --band_region nir \
  --devices 0
```

Replace `--ckpt` with your checkpoint file. Use the configuration, day selection, spectral region, and data split corresponding to its training run.
Metrics are printed and saved as `test_metrics_<checkpoint_name>.txt` in the evaluation run's log directory.

## Citation

If this work supports your research, please cite the oficial paper:

```bibtex
@article{francaniHSWAE,
  author  = {Fran{\c{c}}ani, Andr{\'e} Oliveira and Zhao, Liang and
             Ferreira e Silva, Jo{\~a}o de Deus and Gao, Kaizhou and
             de Castro Jorge, L{\'u}cio Andr{\'e}},
  title   = {{HS-WAE}: A Pipeline for Early Insect Damage Detection in
             Hyperspectral Soybean Data by Enhanced Deep Autoencoder},
  journal = {IEEE Transactions on Artificial Intelligence},
  note    = {Accepted for publication}
}
```

Final publication metadata and DOI will be added when available.

## License

This project's code is released under the [MIT License](LICENSE). Third-party materials retain their original licenses; see the notices under [preprocessing/hyperblood](preprocessing/hyperblood).

## References

- [HyperBlood dataset](https://zenodo.org/records/3984905) and [HSI blood detection code](https://github.com/iitis/HSI_blood_detection).
- Dataset and loader notices are retained under [preprocessing/hyperblood](preprocessing/hyperblood).

<!-- Replace #citation below with the final IEEE Xplore article URL. -->
[ieee-paper]: #citation
