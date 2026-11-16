import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get(
    "SZTYLER_DATA_ROOT",
    Path(__file__).resolve().parents[4] / "data" / "sztyler",
))

SUBJECTS = [f"subject_{i}" for i in range(1, 16)]
ACTIVITIES = [
    "climbing_down",
    "climbing_up",
    "jumping",
    "lying",
    "running",
    "sitting",
    "standing",
    "walking",
]
SENSOR_LOCATIONS = ["chest", "forearm", "head", "shin", "thigh", "upperarm", "waist"]
SENSORS = ["acc", "gyr"]

WINDOW_SECONDS = 3.0
OVERLAP = 0.5
TARGET_HZ = 50
WINDOW_SAMPLES = int(round(WINDOW_SECONDS * TARGET_HZ))

N_FOLDS = 5
SUBJECTS_PER_FOLD = 3
RANDOM_SEED = 42
