# SMPLy Reproducible

> **Note:** An update of this repository is coming soon.

## Motivation

A persistent challenge in wearable sensing research is that sensor placement is rarely reproducible. Studies typically describe placement in natural language ("IMU attached to the dorsal side of the right wrist") — a description that is too ambiguous to reconstruct precisely across participants, experimenters, or labs. Even small differences in placement change the orientation of the sensor's local coordinate frame relative to the body, which directly affects accelerometer and gyroscope readings and makes cross-study comparisons unreliable.

**SMPLy Reproducible** addresses this by using the [SMPL](https://smpl.is.tue.mpg.de/) parametric body model as a shared anatomical reference. Every sensor placement is stored as an exact vertex index on the SMPL mesh — a single integer that unambiguously encodes both position and surface orientation on a standardised 3-D body. A YAML configuration file captures the full placement (position, sensor type, in-plane rotation, body shape parameters), making a setup shareable, version-controllable, and recreatable by anyone who has the SMPL model.

The toolkit also computes the **relative rotation** between any two placements of the same sensor type. This rotation matrix can be applied directly to raw IMU measurements (accelerometer and gyroscope) to express data from one mounting position as if the sensor had been placed at a reference position — enabling post-hoc alignment of data collected with slightly different placements.

---

## Setup

Everything in this repo (including `RealWorld_experiments/`) is developed against the
`smplyreproducible` conda environment:

```bash
conda create -n smplyreproducible python=3.10
conda activate smplyreproducible
pip install -r requirements.txt
```

The SMPL model files (`smpl/*.pkl`) are **not included** — download them from the [SMPL project page](https://smpl.is.tue.mpg.de/) and place them in the `smpl/` directory.

---

## Repository structure

```
SmplyReproducible/
├── smpl_loader.py                 # shared utilities: load SMPL model, apply betas, surface/orientation helpers
├── pick_vertex.py                 # step 0 (optional): click the body to find vertex indices
├── place_sensors.py               # step 1: interactive click-to-place sensor tool
├── place_sensors_via_vertex.py    # step 1 (alt): place sensors at known vertex indices
├── visualize.py                   # step 2: static viewer for a saved sensor placement YAML
├── transform.py                   # step 3: relative rotation matrix between two sensors
├── accel_sim.py                   # standalone demo verifying transform.py's rotation math
├── sensors_placed_transform.yaml  # example sensor placement config (output of the placement tools)
├── requirements.txt               # Python dependencies
├── .gitignore                     # excludes smpl/*.pkl, __pycache__, venvs, etc.
├── smpl/                          # SMPL model .pkl files — not tracked, download separately (see Setup)
└── README.md
```

---

## Files and how they build on each other

```
smpl_loader.py               ← shared foundation, imported by everything else
    │
    ├── pick_vertex.py       ← step 0 (optional): find vertex indices by clicking
    │
    ├── place_sensors.py     ┐
    ├── place_sensors_via_vertex.py ┘  ← step 1: create a sensor placement YAML
    │
    ├── visualize.py         ← step 2: review the placement
    │
    └── transform.py         ← step 3: compute relative rotation between sensors

accel_sim.py                 ← standalone demo that verifies transform.py's math
```

### `smpl_loader.py`
Shared utilities imported by all other scripts. Loads SMPL `.pkl` files without requiring the `chumpy` dependency, applies shape parameters (betas) to deform the body mesh, and provides geometry helpers (`local_frame_from_normal`, `surface_pos_and_frame`) used to orient sensors on the body surface.

### `pick_vertex.py` — find vertex indices
Interactive 3-D viewer: click anywhere on the body and the nearest vertex index, 3-D position, and local coordinate frame are shown in the window and printed to the console. Use this to look up `vertex_idx` values before running `place_sensors_via_vertex.py`.

```bash
python pick_vertex.py
python pick_vertex.py --view front
```

### `place_sensors.py` — interactive click-to-place
Opens a 3-D body and lets you left-click to drop sensors directly on the surface. The vertex index of each placed sensor is printed to the console. Switch sensor type with `i/c/e/g` (IMU / Camera / EMG / PPG), adjust in-plane rotation with `h/j` (±10° around the surface normal), undo with `u`, and save to YAML with `s`.

```bash
python place_sensors.py                        # saves to sensors_placed.yaml
python place_sensors.py my_setup.yaml --view back
```

### `place_sensors_via_vertex.py` — place by vertex index
Alternative placement tool for when you already know the vertex IDs (e.g. from `pick_vertex.py` or a vertex mask from a model). Pass them on the command line; sensors appear immediately so you can fine-tune orientation and type before saving.

```bash
python place_sensors_via_vertex.py 1961 5424 876 4362 411 3021
python place_sensors_via_vertex.py 2206 2235 --output my_setup.yaml --view front
```

Both placement scripts write the same YAML format:

```yaml
smpl_path: smpl/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
betas: [0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0]   # SMPL shape parameters
sensors:
  - name: Imu_1
    type: imu
    vertex_idx: 2206
    position_3d: [0.72241, 0.25537, -0.05669]
    orientation:
      euler_xyz_deg: [0, 0, 30]
```

The `betas` field encodes the body shape used during placement. Including it in the config ensures that anyone reproducing the setup uses the same reference body.

#### How `orientation.euler_xyz_deg` is interpreted

`euler_xyz_deg` is **not** an absolute orientation — it's a twist applied *on top of* a frame the SMPL mesh already determines for you at `vertex_idx`. Two things compose to produce a sensor's final orientation:

1. **The SMPL-determined local frame (`R_local`).** `vertex_idx` fixes both the sensor's position *and* its surface normal — the direction pointing straight out of the skin at that point. `smpl_loader.local_frame_from_normal()` builds a right-handed frame from that normal alone: Z = the outward normal, X = the "most-upward" direction that still lies flat in the tangent plane, Y = Z × X. This is the frame you get with `euler_xyz_deg: [0, 0, 0]` — sensor lying flat on the skin, no rotation applied. Nothing about this frame is under your control except *where* you click.
2. **Your in-plane twist (`R_user`).** `euler_xyz_deg` is converted to a rotation matrix (`scipy.spatial.transform.Rotation.from_euler("xyz", ..., degrees=True)`) and composed as `R_local @ R_user`. In practice only the Z component is ever non-zero (that's what the `h`/`j` rotate keys in `place_sensors.py` control, ±10° per press) — a pure rotation about the surface normal. This spins the sensor's in-plane axes (X, Y) around however you like while leaving the Z axis — the surface normal itself — completely unchanged, so the sensor always stays flat against the skin at the exact point you picked.

In short: **where** you click determines the normal (and therefore two of the three orientation degrees of freedom for free); `euler_xyz_deg` only ever controls the remaining one — how the sensor is twisted around that normal. This is also precisely the $R_{A \leftarrow S}$ (sensor-to-segment orientation) used by `transform.py` to compute the relative rotation between two placements.

### `visualize.py` — review a placement
Renders the SMPL body with all sensors from a YAML file. Sensors can be specified by `vertex_idx`, `position_3d`, or a named SMPL joint (`location: right_wrist`). Default file: `sensors_placed.yaml`.

```bash
python visualize.py
python visualize.py my_setup.yaml --view front
```

Keyboard controls: `↑↓` opacity · `7/8` background · `k` shiny surface · `q` quit.

### `transform.py` — compute relative rotation
Given a sensor placement YAML, computes the rotation **R** between any two sensors such that:

```
x_in_A_frame = R @ x_in_B_frame
```

Apply **R** to raw IMU measurements from sensor B (accelerometer or gyroscope) to express them as if the sensor had been mounted at position A. Valid only when both sensors are on the **same rigid body segment** (no joint between them). Runs interactively (click sensor A then B in the viewer) or non-interactively:

```bash
python transform.py sensors_placed.yaml           # interactive
python transform.py sensors_placed.yaml 0 1       # sensor 1 → sensor 0
```

Outputs the rotation matrix, quaternion (both `w x y z` and `x y z w` conventions), Euler XYZ angles, and angular distance in degrees.

### `accel_sim.py` — verify the rotation math
Self-contained simulation (no YAML or SMPL model needed) that places two virtual sensors at known orientations, computes **R** the same way `transform.py` does, and confirms that `R @ a_B == a_A` for static, dynamic, and decomposed (gravity + linear) scenarios. Useful for building intuition about what the rotation actually does to sensor measurements.

```bash
python accel_sim.py
```

---

## Typical workflow

1. **Find vertex IDs** (optional): `python pick_vertex.py` → note the printed indices
2. **Place sensors**:
   - `python place_sensors.py my_setup.yaml` — click on the body to place
   - `python place_sensors_via_vertex.py 1961 5424 ... --output my_setup.yaml` — place at known indices
3. **Review**: `python visualize.py my_setup.yaml`
4. **Get relative rotation**: `python transform.py my_setup.yaml` → copy **R** into your processing pipeline
