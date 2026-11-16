# preprocessing

Windowing + labeling + cross-validation pipeline for HAR on the RealWorld (Sztyler)
dataset, using accelerometer and gyroscope only. Its output feeds both the
placement-rotation experiments in `../scripts/` and the baseline models in
`../models/`.

## Pipeline

```
config.py       constants: subjects, activities, positions, window size, CV setup
loading.py      reads acc/gyr CSVs directly out of the dataset's zip files
windowing.py    aligns acc+gyr to a common time grid, slices into fixed windows
dataset.py      orchestrates loading+windowing across all subjects/activities, forearm only
cv_split.py     partitions subjects into 5 folds of 3 held-out subjects each
build_dataset.py  CLI entry point, run this to produce the dataset
```

Run:
```bash
python build_dataset.py
```
Writes `processed/windows.npz` (arrays `X` [n_windows, 150, 6], `activity`, `subject`,
`position`, `start_time_s`. Channel order is `[acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z]`)
and `processed/folds.json` (5 folds, 3 held-out test subjects each, remaining 12 train).

Data root defaults to `../../../../data/sztyler` relative to this folder, override with
`--data-root` or the `SZTYLER_DATA_ROOT` env var.
