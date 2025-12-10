import os
import sys
import numpy as np
import pybullet as p
import time
from math import cos, sin

from lidar_utils import create_lidar_scan, visualize_lidar
from particle_model import ParticleFilter, build_occupancy_grid
from utils import load_env
from pybullet_tools.utils import set_base_values, get_link_pose, link_from_name

# Force display for Windows
if sys.platform == 'win32':
    os.environ['DISPLAY'] = ':0'


def main():
    print("Connecting to PyBullet GUI...")
    client = p.connect(p.GUI)
    if not hasattr(sys.modules["pybullet_tools.utils"], "CLIENTS"):
        sys.modules["pybullet_tools.utils"].CLIENTS = {}
    sys.modules["pybullet_tools.utils"].CLIENTS[client] = True

    # Load environment
    robots, obstacles = load_env("8path_env.json")
    pr2 = robots.get("pr2", list(robots.values())[0])
    link_name = "head_tilt_link"

    # Define map bounds and resolution for occupancy grid
    xmin, xmax, ymin, ymax = -5.0, 15.0, -5.0, 15.0  # adjust to your map
    resolution = 0.2

    print("Building occupancy grid...")
    occ, xs, ys = build_occupancy_grid(
        xmin, xmax, ymin, ymax, resolution
    )
    print(f"Occupancy grid shape: {occ.shape}")

    # Initialize particle filter
    pf = ParticleFilter(
        occ=occ,
        xs=xs,
        ys=ys,
        n_particles=500,
        lidar_max_range=10.0,
        lidar_min_range=0.1,
        z_lidar=0.5,
        scan_subsample=8
    )

    print("Particle filter initialized!")

    # Simple simulation loop
    step = 0
    x, y, theta = 0.0, 0.0, 0.0  # initial dummy pose
    while True:
        time.sleep(0.05)
        step += 1

        # -----------------------------
        # 1. Move robot (dummy motion)
        # -----------------------------
        dx, dy, dtheta = 0.01, 0.0, 0.02
        x += dx
        theta += dtheta
        set_base_values(pr2, (x, y, theta))

        # Update particles with motion + noise
        noise = np.random.normal(0, 0.01, size=3)
        pf.motion_update((dx + noise[0], dy + noise[1], dtheta + noise[2]))

        # -----------------------------
        # 2. LiDAR scan
        # -----------------------------
        ranges, angles, _ = create_lidar_scan(pr2, link_name)
        visualize_lidar(ranges, angles, pr2, link_name)

        # -----------------------------
        # 3. Measurement update
        # -----------------------------
        pf.measurement_update(ranges, angles)

        # -----------------------------
        # 4. Resample if necessary
        # -----------------------------
        ess = pf.effective_sample_size()
        if ess < 0.5 * pf.n_particles:
            pf.resample()

        # -----------------------------
        # 5. Estimate pose and visualize
        # -----------------------------
        est = pf.estimate()
        est_start = (est[0], est[1], pf.z_lidar)
        est_end = (est[0] + cos(est[2]) * 0.3, est[1] + sin(est[2]) * 0.3, pf.z_lidar)
        p.addUserDebugLine(est_start, est_end, [1, 0, 0], lineWidth=3, lifeTime=0.1)
        pf.draw_particles(life_time=0.1)

        # Optional: print step info
        if step % 20 == 0:
            print(f"Step {step}: Estimated pose = {est}")


if __name__ == "__main__":
    main()
