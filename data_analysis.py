import math
import numpy as np
import re

def angle_diff(a, b):
    d = a - b
    return (d + math.pi) % (2 * math.pi) - math.pi

def rmse(x):
    return math.sqrt(np.mean(np.square(x)))

pf_pos_err, pf_ang_err = [], []
ekf_pos_err, ekf_ang_err = [], []
pf_times, ekf_times = [], []

true_pose = pf_pose = ekf_pose = None

time_re = re.compile(r"PF time = ([\d.]+) ms, EKF time = ([\d.]+) ms")
pose_re = re.compile(r"\(([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\)")

# with open("gpu_comparison.out", "r", encoding="utf-8", errors="ignore") as f:
with open("gpu_comparison.out", "r", encoding="utf-16") as f:
    # for i, line in enumerate(f):
    #     if i < 20:
    #         print(repr(line))

    for line in f:

        # --- timing ---
        if "PF time" in line:
            m = time_re.search(line)
            if m:
                pf_times.append(float(m.group(1)))
                ekf_times.append(float(m.group(2)))

        # --- poses ---
        elif line.strip().startswith("True"):
            true_pose = tuple(map(float, pose_re.search(line).groups()))

        # elif line.strip().startswith("PF"):
        #     pf_pose = tuple(map(float, pose_re.search(line).groups()))

        # elif line.strip().startswith("EKF"):
        #     ekf_pose = tuple(map(float, pose_re.search(line).groups()))
        elif line.strip().startswith("PF"):
            m = pose_re.search(line)
            if m:
                pf_pose = tuple(map(float, m.groups()))

        elif line.strip().startswith("EKF"):
            m = pose_re.search(line)
            if m:
                ekf_pose = tuple(map(float, m.groups()))


        # --- compute once all are available ---
        if true_pose and pf_pose and ekf_pose:
            tx, ty, tt = true_pose
            px, py, pt = pf_pose
            ex, ey, et = ekf_pose

            pf_pos_err.append(math.hypot(px - tx, py - ty))
            ekf_pos_err.append(math.hypot(ex - tx, ey - ty))

            pf_ang_err.append(abs(angle_diff(pt, tt)))
            ekf_ang_err.append(abs(angle_diff(et, tt)))

            true_pose = pf_pose = ekf_pose = None

# ---------- SAFETY CHECK ----------
print(f"\nParsed steps: {len(pf_pos_err)}")

assert len(pf_pos_err) > 0, "No steps parsed — check input file name/path"

# ---------- RESULTS ----------
print("\n=== Test 1: Baseline Performance Comparison ===\n")

print("Particle Filter:")
print(f"  Position RMSE     : {rmse(pf_pos_err):.3f} m")
print(f"  Orientation RMSE  : {rmse(pf_ang_err):.3f} rad")
print(f"  Avg time / step   : {np.mean(pf_times):.2f} ms\n")

print("Extended Kalman Filter:")
print(f"  Position RMSE     : {rmse(ekf_pos_err):.3f} m")
print(f"  Orientation RMSE  : {rmse(ekf_ang_err):.3f} rad")
print(f"  Avg time / step   : {np.mean(ekf_times):.2f} ms")
