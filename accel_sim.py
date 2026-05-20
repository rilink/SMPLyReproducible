"""
Simulates accelerometer readings for two sensors (A and B) mounted at different
orientations on the same rigid body segment, and shows why applying the relative
rotation R from transform.py correctly maps B's measurements into A's frame —
including the gravity component.

Run:
    python accel_sim.py
"""

import numpy as np
from scipy.spatial.transform import Rotation

# ── world frame convention ────────────────────────────────────────────────────
# Z points up, so gravity points down:
g_world = np.array([0.0, 0.0, -9.81])   # m/s²

# ── what an accelerometer actually measures ───────────────────────────────────
#
# An accelerometer measures SPECIFIC FORCE — the non-gravitational force per
# unit mass acting on it, expressed in the sensor's own frame.
#
# When the body has true acceleration a_body (world frame):
#
#   a_measured = R_world_to_sensor @ (a_body - g_world)
#
# Static (a_body = 0):  a_measured = R_world_to_sensor @ (-g_world)
#                                   = R_world_to_sensor @ [0, 0, +9.81]
# → sensor "feels" reaction against gravity along its own axes.

def simulate_accel(R_world_to_sensor, a_body_world):
    """Return what a sensor measures given its mounting rotation and body accel."""
    return R_world_to_sensor @ (a_body_world - g_world)


# ── define two sensors on the same wrist ─────────────────────────────────────
#
# Sensor A: lying flat — Z axis pointing straight up (aligned with world Z).
#   R_world_to_A = identity
R_world_to_A = np.eye(3)

# Sensor B: same wrist, but rotated 45° around Y and 20° around Z relative to A.
#   This mimics a sensor strapped slightly twisted on the wrist.
R_world_to_B = Rotation.from_euler("yz", [45, 20], degrees=True).as_matrix()

# ── relative rotation (same formula as transform.py) ─────────────────────────
#
# R_local in transform.py has COLUMNS = sensor axes in world coords, so:
#   R_local      = R_sensor_to_world   (takes sensor vectors → world)
#   R_local.T    = R_world_to_sensor   (takes world vectors → sensor)
#
# Here we already have R_world_to_sensor, so the surface-frame matrices are:
R_A = R_world_to_A.T   # columns = A's axes in world  (= identity here)
R_B = R_world_to_B.T   # columns = B's axes in world

# Relative rotation: transforms vectors FROM B's frame INTO A's frame
R_rel = R_A.T @ R_B    # same line as in transform.py


# ── scenario 1: body is STATIC (standing still) ───────────────────────────────
print("=" * 60)
print("SCENARIO 1 — static body (no motion)")
print("=" * 60)

a_body = np.zeros(3)

a_A = simulate_accel(R_world_to_A, a_body)
a_B = simulate_accel(R_world_to_B, a_body)

print(f"\nGravity in world frame :  {g_world}")
print(f"\nSensor A measures       :  {np.round(a_A, 4)}")
print(f"Sensor B measures       :  {np.round(a_B, 4)}")
print()
print("Both contain gravity — just expressed in different sensor axes.")
print("A is flat so it sees all gravity on Z: [0, 0, +9.81]")
print("B is tilted so gravity spreads across axes.")

a_B_in_A = R_rel @ a_B
print(f"\nR_rel @ B's measurement :  {np.round(a_B_in_A, 4)}")
print(f"A's measurement         :  {np.round(a_A, 4)}")
print(f"Match: {np.allclose(a_B_in_A, a_A)}")
print()
print("Gravity got 'rotated' in the sense that its components changed,")
print("but it still represents the same physical downward force —")
print("just now described in A's coordinate system.")


# ── scenario 2: body is MOVING ────────────────────────────────────────────────
print()
print("=" * 60)
print("SCENARIO 2 — arm moving (a_body = [2, 1, 3] m/s² in world)")
print("=" * 60)

a_body = np.array([2.0, 1.0, 3.0])

a_A = simulate_accel(R_world_to_A, a_body)
a_B = simulate_accel(R_world_to_B, a_body)

print(f"\nSensor A measures       :  {np.round(a_A, 4)}  (linear accel + gravity)")
print(f"Sensor B measures       :  {np.round(a_B, 4)}  (linear accel + gravity)")

a_B_in_A = R_rel @ a_B
print(f"\nR_rel @ B's measurement :  {np.round(a_B_in_A, 4)}")
print(f"A's measurement         :  {np.round(a_A, 4)}")
print(f"Match: {np.allclose(a_B_in_A, a_A)}")


# ── scenario 3: show the gravity and linear parts separately ──────────────────
print()
print("=" * 60)
print("SCENARIO 3 — decompose into gravity and linear parts")
print("=" * 60)

g_in_A = R_world_to_A @ (-g_world)      # gravity component in A's frame
g_in_B = R_world_to_B @ (-g_world)      # gravity component in B's frame
lin_in_A = R_world_to_A @ a_body        # linear part in A's frame
lin_in_B = R_world_to_B @ a_body        # linear part in B's frame

print(f"\nGravity in A's frame    :  {np.round(g_in_A, 4)}")
print(f"Gravity in B's frame    :  {np.round(g_in_B, 4)}")
print(f"\nR_rel rotates B gravity :  {np.round(R_rel @ g_in_B, 4)}")
print(f"Should equal A gravity  :  {np.round(g_in_A, 4)}")
print(f"Match: {np.allclose(R_rel @ g_in_B, g_in_A)}")

print(f"\nLinear accel in A       :  {np.round(lin_in_A, 4)}")
print(f"Linear accel in B       :  {np.round(lin_in_B, 4)}")
print(f"\nR_rel rotates B linear  :  {np.round(R_rel @ lin_in_B, 4)}")
print(f"Should equal A linear   :  {np.round(lin_in_A, 4)}")
print(f"Match: {np.allclose(R_rel @ lin_in_B, lin_in_A)}")

print()
print("Both gravity and linear components transform correctly.")
print("R distributes over addition: R@(g+lin) = R@g + R@lin.")
print("No need to remove gravity before applying R.")
