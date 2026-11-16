# models

HAR classifiers used across the experiments, all trained on 6-channel
(acc_x,y,z, gyr_x,y,z) 150-frame windows at 50 Hz, forearm position, 8
activity classes.

- `common.py` — shared training/eval utilities used by all deep models:
  fold loading, an `AdamW` + cosine-LR training loop with checkpointing,
  evaluation, and 5-fold cross-validation orchestration (`run_cv`).
- `model_lightgbm.py` — LightGBM on handcrafted per-window features
  (statistical, frequency-domain, and cross-axis descriptors). No
  checkpointing. Feature extraction is cached instead (`processed/`,
  gitignored).
- `model_deepconvlstm.py`, `model_tinyhar.py`, `model_cnnhar.py` — deep
  models for this project's 6-channel, 8-class, 50 Hz setup. CNN-HAR is implemented but
  not used in the paper's headline results (see `experiments/scripts/baseline.py`).

`checkpoints/` and `processed/` are gitignored and get populated by running
the scripts in `../scripts/`.
