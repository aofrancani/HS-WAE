# Datasets

Place the processed CSVs at:

```text
data/HS-Class/soybean_HSI.csv
data/HSI_Blood/HSI_Blood.csv
```

Data files are ignored by Git. To use external paths, run `python -m tools.configure --soybean-root /path/to/HS-Class --blood-csv /path/to/HSI_Blood.csv` from the repository root.

## Soybean

Available upon request from **Prof. Liang Zhao**, **[zhao@usp.br](mailto:zhao@usp.br)**.

The CSV requires `ID`, `Date`, `GroundTruth`, `Label`, and integer-named wavelength columns. Labels are 0 (control) and 1 (damaged). Preserve the supplied dates and row order: the loader derives acquisition days before filtering and uses a fixed split. After filtering, the dataset contains 1,219 samples across seven days.

## HyperBlood

Download the cubes and annotations from [Zenodo](https://zenodo.org/records/3984905). See the [original repository](https://github.com/iitis/HSI_blood_detection) for dataset documentation. The downloaded root must contain `data/` (ENVI hyperspectral cubes: `.float`/`.hdr`) and `anno/` (pixel-level labels: `.npz`).

From the HS-WAE root:

```bash
python -m tools.prepare_hyperblood --dataset-root /path/to/HyperBlood \
  --out data/HSI_Blood/HSI_Blood.csv --min-pixels 60 --connectivity 2 --debug-dir ""
```

To obtain **one hyperspectral curve per object**, the script identifies 8-connected components within each annotated class and averages their pixel reflectance separately at each wavelength. Each CSV row contains one object's mean spectrum and class label, matching the soybean pipeline's one-spectrum-per-sample format. Components smaller than 60 pixels, background, and uncertain blood are excluded. After noisy-band removal, the processed dataset contains 360 samples, 113 wavelengths, 14 scenes, and seven classes. Rows also retain scene/blob metadata; spectral columns use `wl_###.#` names.

Optional scene illustration:

```bash
python -m tools.prepare_hyperblood --dataset-root /path/to/HyperBlood --task figure \
  --scenes 'B(1)' --outdir figures/hyperblood
```

Retained upstream materials include the dataset's [CC BY 4.0 notice](../preprocessing/hyperblood/licence.txt), [readme](../preprocessing/hyperblood/readme.txt), and the loader's Apache 2.0 header.
