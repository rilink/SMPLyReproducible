# SmplyReproducible

A toolkit for placing virtual wearable sensors on the [SMPL](https://smpl.is.tue.mpg.de/) body model, computing relative rotations between sensor placements, and verifying the rotation math with an accelerometer simulation.

## Setup

```bash
pip install -r requirements.txt
```

The SMPL model files (`smpl/*.pkl`) are **not included** — download them from the [SMPL project page](https://smpl.is.tue.mpg.de/) and place them in the `smpl/` directory.

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
         └── accel_sim.py   ← standalone demo that verifies transform.py's math
```

### `smpl_loader.py`
Shared utilities imported by all other scripts. Loads SMPL `.pkl` files without requiring the `chumpy` dependency, applies shape parameters (betas) to deform the body mesh, and provides geometry helpers (`local_frame_from_normal`, `surface_pos_and_frame`) used to orient sensors on the body surface.

### `pick_vertex.py` — find vertex indices
Interactive 3-D viewer: click anywhere on the body and the nearest vertex index and its position are printed to the console. Use this to look up `vertex_idx` values before running `place_sensors_via_vertex.py`.

```bash
python pick_vertex.py
python pick_vertex.py --view front
```

### `place_sensors.py` — interactive click-to-place
Opens a 3-D body and lets you left-click to drop sensors directly on the surface. Switch sensor type with `i/c/e/g` (IMU / Camera / EMG / PPG), adjust in-plane rotation with `h/j`, undo with `u`, and save to YAML with `s`.

```bash
python place_sensors.py my_setup.yaml
```

### `place_sensors_via_vertex.py` — place by vertex index
Alternative placement tool for when you already know the vertex IDs (e.g. from `pick_vertex.py`). Pass them on the command line; sensors appear immediately so you can fine-tune orientation before saving.

```bash
python place_sensors_via_vertex.py 2206 2235 --output my_setup.yaml
```

Both placement scripts write the same YAML format:

```yaml
smpl_path: smpl/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
sensors:
  - name: Imu_1
    type: imu
    vertex_idx: 2206
    position_3d: [0.72241, 0.25537, -0.05669]
    orientation:
      euler_xyz_deg: [0, 0, 30]
```

`sensors_placed_transform.yaml` is an example output with two IMUs on the right wrist.

### `visualize.py` — review a placement
Renders the SMPL body with all sensors from a YAML file. Sensors can be specified by `vertex_idx`, `position_3d`, or a named SMPL joint (`location: right_wrist`).

```bash
python visualize.py sensors_placed_transform.yaml
python visualize.py sensors_placed_transform.yaml --view front
```

### `transform.py` — compute relative rotation
Given a sensor placement YAML, computes the rotation **R** between any two sensors such that:

```
x_in_A_frame = R @ x_in_B_frame
```

Apply **R** to raw measurements from sensor B to express them as if the sensor had been mounted at position A. Runs interactively (click sensor A then B in the viewer) or non-interactively:

```bash
python transform.py sensors_placed_transform.yaml          # interactive
python transform.py sensors_placed_transform.yaml 0 1      # sensor 1 → sensor 0
```

Outputs the rotation matrix, quaternion (both conventions), Euler angles, and angular distance.

### `accel_sim.py` — verify the rotation math
Self-contained simulation (no YAML needed) that places two virtual sensors at known orientations on a wrist, computes **R** the same way `transform.py` does, and confirms that `R @ a_B == a_A` for static, dynamic, and decomposed (gravity + linear) scenarios. Run it to sanity-check that the frame convention is correct.

```bash
python accel_sim.py
```

---

## Typical workflow

1. **Find vertex IDs** (optional): `pick_vertex.py` → note the printed indices
2. **Place sensors**: `place_sensors.py` (click on body) **or** `place_sensors_via_vertex.py <idx ...>` → saves `sensors_placed.yaml`
3. **Review**: `visualize.py sensors_placed.yaml`
4. **Get relative rotation**: `transform.py sensors_placed.yaml` → copy **R** into your data processing pipeline
