"""Reads RealWorld (Sztyler) acc/gyr CSVs out of the dataset's zip files,
handling split-recording and zip-of-zips packaging quirks."""

import io
import re
import zipfile
from pathlib import Path

import pandas as pd

_PART_RE_TEMPLATE = r"^{sensor}_{act}(_(\d+))?_(csv|sqlite)\.zip$"


def _activity_key(activity: str) -> str:
    return activity.replace("_", "")


def _find_candidate_zips(activity_dir: Path, sensor: str, activity: str) -> list[Path]:
    """Sibling files for one (sensor, activity), ordered by part number.

    Usually a single `{sensor}_{act}_csv.zip`. Some subjects additionally ship
    `{sensor}_{act}_2_csv.zip`, `_3_csv.zip`, ... as separate sibling files
    (observed for subject_4's gyroscope). A missing part number means part 1.
    Some subjects only ship a `_sqlite.zip` instead of `_csv.zip`. Despite the
    name it contains the same flat CSV layout, so it is matched identically.
    """
    pattern = re.compile(
        _PART_RE_TEMPLATE.format(sensor=re.escape(sensor), act=re.escape(_activity_key(activity))),
        re.IGNORECASE,
    )
    matches = []
    for f in activity_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            part_num = int(m.group(2)) if m.group(2) else 1
            matches.append((part_num, f))
    matches.sort(key=lambda x: x[0])
    return [f for _, f in matches]


def _find_csv_member(namelist: list[str], position: str) -> str | None:
    for name in namelist:
        if name == "readMe":
            continue
        if position.lower() in name.lower():
            return name
    return None


def _read_csv_member(zf: zipfile.ZipFile, member: str) -> pd.DataFrame:
    with zf.open(member) as fh:
        df = pd.read_csv(fh, index_col=False)
    df = df.rename(columns={"attr_time": "time_ms", "attr_x": "x", "attr_y": "y", "attr_z": "z"})
    return df[["time_ms", "x", "y", "z"]].sort_values("time_ms").reset_index(drop=True)


def load_sensor_parts(subject_dir: Path, activity: str, sensor: str, position: str) -> list[pd.DataFrame]:
    """Load one (subject, activity, sensor, position) as a list of chronologically
    ordered DataFrames ("parts"). Almost always a single-element list. A few
    subjects (4, 7, 14) have climbing_down/climbing_up recordings split into
    multiple time-disjoint parts, packaged either as sibling zip files or as
    zip-of-zips nesting. Both are resolved here into a flat list of parts.
    Callers must NOT concatenate parts into one continuous timeline -- there
    are multi-minute gaps between them (see experiments/README.md).
    """
    activity_dir = subject_dir / activity
    candidate_zips = _find_candidate_zips(activity_dir, sensor, activity)
    parts: list[pd.DataFrame] = []

    for zip_path in candidate_zips:
        with zipfile.ZipFile(zip_path) as zf:
            namelist = zf.namelist()
            inner_zip_names = sorted(n for n in namelist if n.lower().endswith(".zip"))
            if inner_zip_names:
                for inner_name in inner_zip_names:
                    inner_bytes = zf.read(inner_name)
                    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner_zf:
                        member = _find_csv_member(inner_zf.namelist(), position)
                        if member is not None:
                            parts.append(_read_csv_member(inner_zf, member))
            else:
                member = _find_csv_member(namelist, position)
                if member is not None:
                    parts.append(_read_csv_member(zf, member))

    return parts
