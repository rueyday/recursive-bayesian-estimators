import os
import sys
import time
import numpy as np
import pybullet as p
from math import cos, sin

from lidar_utils import create_lidar_scan
from particle_model import ParticleFilter, build_occupancy_grid
from kalman_model import ExtendedKalmanFilter
from utils import load_env
from pybullet_tools.utils import get_link_pose, link_from_name

# Force display on Windows
if sys.platform == "win32":
    os.environ["DISPLAY"] = ':0'

# ============================================================
# EXPERIMENT CONFIG
# ============================================================

waypoints = [
    (-2.5,  2.2),
    ( 0.0,  2.2),
    ( 0.0, -2.2),
    ( 2.5, -2.2),
    ( 2.5, -6.0),
    (-2.5, -6.0),
    (-2.5, -2.2),
    ( 0.0, -2.2),
    ( 0.0,  2.2),
    ( 2.5,  2.2),
    ( 2.5,  6.0),
    (-2.5,  6.0)
]

linear_speed = 0.05
angular_speed = 0.15
wp_threshold = 0.2
NUM_STEPS = 1200

# ============================================================
# HELPERS
# ============================================================

def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def set_base_link_pose(robot, x_link, y_link, theta):
    _, _, z = p.getBasePositionAndOrientation(robot)[0]
    p.resetBasePositionAndOrientation(
        robot, [0, 0, z], p.getQuaternionFromEuler([0, 0, theta])
    )

    link_pose = get_link_pose(robot, link_from_name(robot, "base_link"))
    dx, dy = link_pose[0][0], link_pose[0][1]

    p.resetBasePositionAndOrientation(
        robot, [x_link - dx, y_link - dy, z],
        p.getQuaternionFromEuler([0, 0, theta])
    )

# ============================================================
# MAIN
# ============================================================

def main():

    client = p.connect(p.GUI)
    sys.modules["pybullet_tools.utils"].CLIENTS = {client: True}

    # --------------------------------------------------------
    # LOAD ENVIRONMENT
    # --------------------------------------------------------
    robots, _ = load_env("figure8_env.json")
    robot = robots.get("pr2", list(robots.values())[0])
    link_name = "base_link"

    # --------------------------------------------------------
    # OCCUPANCY GRID
    # --------------------------------------------------------
    occ, xs, ys = build_occupancy_grid(
        xmin=-6.0, xmax=6.0,
        ymin=-6.0, ymax=6.0,
        resolution=0.2
    )

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------
    pf = ParticleFilter(
        occ=occ, xs=xs, ys=ys,
        n_particles=500,
        lidar_max_range=10.0,
        lidar_min_range=0.1,
        z_lidar=0.5,
        scan_subsample=8
    )

    ekf = ExtendedKalmanFilter(
        initial_pose=(-4.0, 5.0, np.pi/2),
        motion_noise=(0.02, 0.02, 0.01),
        lidar_max_range=10.0,
        lidar_min_range=0.1,
        z_lidar=0.5,
        sigma_range=0.2,
        scan_subsample=8
    )

    ekf.P = 0.5 * np.eye(3)

    # PF initialization (same belief)
    pf.particles[:, 0] = -4.0 + np.random.normal(0, 0.5, pf.n_particles)
    pf.particles[:, 1] =  5.0 + np.random.normal(0, 0.5, pf.n_particles)
    pf.particles[:, 2] = np.pi/2 + np.random.normal(0, 0.3, pf.n_particles)
    pf.weights[:] = 1.0 / pf.n_particles

    # --------------------------------------------------------
    # DATA LOGS
    # --------------------------------------------------------
    true_poses = []
    pf_estimates = []
    ekf_estimates = []
    pf_times = []
    ekf_times = []

    # --------------------------------------------------------
    # INITIAL TRUE STATE
    # --------------------------------------------------------
    # (-2.5,  6.0),
    x, y, theta = -2.5, 6.0, np.pi / 2
    set_base_link_pose(robot, x, y, theta)
    current_wp = 0

    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------
    print("Running baseline comparison...")
    print(f"Total steps: {NUM_STEPS}")
    print("Progress: ", end='', flush=True)
    
    for step in range(NUM_STEPS):
        time.sleep(0.03)
        
        # Progress indicator
        if step % 100 == 0:
            print(f"{step}...", end='', flush=True)

        x_prev, y_prev, theta_prev = x, y, theta

        # ---- MOTION CONTROL ----
        wp_x, wp_y = waypoints[current_wp]
        dx, dy = wp_x - x, wp_y - y
        target_theta = np.arctan2(dy, dx)
        dtheta = wrap_angle(target_theta - theta)

        if abs(dtheta) < 0.3:
            dist = np.hypot(dx, dy)
            step_dist = min(dist, linear_speed)
            x += step_dist * cos(theta)
            y += step_dist * sin(theta)

        theta += np.clip(dtheta, -angular_speed, angular_speed)
        theta = wrap_angle(theta)

        set_base_link_pose(robot, x, y, theta)

        if np.hypot(dx, dy) < wp_threshold:
            current_wp = (current_wp + 1) % len(waypoints)

        # ---- TRUE ODOMETRY ----
        dx = x - x_prev
        dy = y - y_prev
        dtheta = wrap_angle(theta - theta_prev)

        # ---- MOTION UPDATE ----
        pf.motion_update((dx, dy, dtheta))
        ekf.motion_update((dx, dy, dtheta))

        # ---- MEASUREMENT UPDATE ----
        ranges, angles, _ = create_lidar_scan(robot, link_name)

        t0 = time.time()
        pf.measurement_update(ranges, angles)
        pf_times.append(time.time() - t0)

        t0 = time.time()
        ekf.measurement_update(ranges, angles)
        ekf_times.append(time.time() - t0)

        if pf.effective_sample_size() < 0.5 * pf.n_particles:
            pf.resample()

        # ---- ESTIMATES ----
        pf_est = pf.estimate()
        ekf_est = ekf.estimate()

        # ---- GROUND TRUTH ----
        pose = get_link_pose(robot, link_from_name(robot, link_name))
        true_x, true_y = pose[0][0], pose[0][1]
        true_theta = p.getEulerFromQuaternion(pose[1])[2]

        # ---- LOG DATA ----
        true_poses.append([true_x, true_y, true_theta])
        pf_estimates.append(pf_est)
        ekf_estimates.append(ekf_est)

    print("Done!")
    
    # --------------------------------------------------------
    # SAVE DATA
    # --------------------------------------------------------
    np.savez(
        "baseline_filter_data.npz",
        true_poses=np.array(true_poses),
        pf_estimates=np.array(pf_estimates),
        ekf_estimates=np.array(ekf_estimates),
        pf_times=np.array(pf_times),
        ekf_times=np.array(ekf_times)
    )

    print("Data saved to baseline_filter_data.npz")
    
    # Disconnect PyBullet and clean up
    print("Closing PyBullet simulation...")
    try:
        p.disconnect(client)
    except:
        pass
    
    # Clean up CLIENTS dictionary
    try:
        if hasattr(sys.modules["pybullet_tools.utils"], "CLIENTS"):
            if client in sys.modules["pybullet_tools.utils"].CLIENTS:
                del sys.modules["pybullet_tools.utils"].CLIENTS[client]
    except:
        pass
    
    return client


if __name__ == "__main__":
    main()