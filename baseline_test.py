# baseline_calculator_v2.py - CORRECTED VERSION
import os
import sys
import time
import numpy as np
import pybullet as p
from math import cos, sin, atan2, hypot

# =================================================================
# IMPORTANT: ASSUMED EXTERNAL IMPORTS
# =================================================================

try:
    from lidar_utils import create_lidar_scan
    from particle_model import ParticleFilter, build_occupancy_grid
    from kalman_model import ExtendedKalmanFilter
    from utils import load_env
    from pybullet_tools.utils import get_link_pose, link_from_name
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import custom modules: {e}")
    sys.exit(1)

# Force display on Windows
if sys.platform == "win32":
    os.environ["DISPLAY"] = ':0'

waypoints = [
    (-2.5,  2.2), ( 0.0,  2.2), ( 0.0, -2.2), ( 2.5, -2.2), ( 2.5, -6.0),
    (-2.5, -6.0), (-2.5, -2.2), ( 0.0, -2.2), ( 0.0,  2.2), ( 2.5,  2.2),
    ( 2.5,  6.0), (-2.5,  6.0)
]

linear_speed = 0.05
angular_speed = 0.15
wp_threshold = 0.2
NUM_STEPS = 900

# ============================================================
# HELPERS
# ============================================================

def wrap_angle(a):
    """Wraps an angle to the range [-pi, pi]"""
    return np.fmod(a + np.pi, 2 * np.pi) - np.pi

def set_base_link_pose(robot_id, x_link, y_link, theta):
    """Sets the robot's pose in PyBullet using the body ID."""
    try:
        _, _, z = p.getBasePositionAndOrientation(robot_id)[0]
        p.resetBasePositionAndOrientation(
            robot_id, [0, 0, z], p.getQuaternionFromEuler([0, 0, theta])
        )
        
        link_pose = get_link_pose(robot_id, link_from_name(robot_id, "base_link"))
        dx, dy = link_pose[0][0], link_pose[0][1]

        p.resetBasePositionAndOrientation(
            robot_id, [x_link - dx, y_link - dy, z],
            p.getQuaternionFromEuler([0, 0, theta])
        )
    except Exception as e:
        pass

def calculate_rmse(true_poses, estimates):
    """Calculates Position and Orientation RMSE."""
    true_poses = np.array(true_poses)
    estimates = np.array(estimates)
    
    if len(true_poses) == 0:
        return 0.0, 0.0
    
    pos_error = true_poses[:, :2] - estimates[:, :2]
    pos_mse = np.mean(np.sum(pos_error**2, axis=1))
    pos_rmse = np.sqrt(pos_mse)
    
    theta_error = wrap_angle(true_poses[:, 2] - estimates[:, 2])
    theta_mse = np.mean(theta_error**2)
    theta_rmse = np.sqrt(theta_mse)
    
    return pos_rmse, theta_rmse

# ============================================================
# MAIN SIMULATION AND CALCULATION
# ============================================================

def main():

    client = None
    try:
        client = p.connect(p.GUI)
        if hasattr(sys.modules["pybullet_tools.utils"], "CLIENTS"):
             sys.modules["pybullet_tools.utils"].CLIENTS = {client: True}
    except Exception as e:
        print(f"Could not connect to PyBullet: {e}")
        return
    
    # --------------------------------------------------------
    # LOAD ENVIRONMENT & INIT ROBOT
    # --------------------------------------------------------
    robot_id = None
    try:
        robots_dict, _ = load_env("figure8_env.json")
        robot_id = robots_dict.get("pr2", list(robots_dict.values())[0])
        link_name = "base_link"
        if not isinstance(robot_id, int):
             print(f"Warning: Expected robot ID to be an integer, got {type(robot_id)}")
    except Exception as e:
        print(f"Error loading environment: {e}")
        return

    # --------------------------------------------------------
    # OCCUPANCY GRID & FILTERS INIT
    # --------------------------------------------------------
    occ, xs, ys = build_occupancy_grid(xmin=-8.0, xmax=8.0, ymin=-8.0, ymax=8.0, resolution=0.2)

    # CORRECTED: Increased motion noise and more particles
    pf = ParticleFilter(
        occ=occ, xs=xs, ys=ys, 
        n_particles=1000,  # Increased from 500
        lidar_max_range=10.0,
        lidar_min_range=0.1, 
        z_lidar=0.5, 
        scan_subsample=4,  # Reduced from 8 for better measurement
        motion_noise=(0.1, 0.1, 0.1),  # Increased from (0.02, 0.02, 0.01)
        sigma_range=0.3  # Increased from 0.2
    )

    ekf = ExtendedKalmanFilter(
        initial_pose=(-2.5, 6.0, np.pi/2), 
        motion_noise=(0.05, 0.05, 0.03),  # Increased
        lidar_max_range=10.0, 
        lidar_min_range=0.1, 
        z_lidar=0.5,
        sigma_range=0.3,  # Increased
        scan_subsample=4  # Reduced from 8
    )

    # CORRECTED: Larger initial uncertainty
    ekf.P = 1.0 * np.eye(3)  # Increased from 0.5
    pf.particles[:, 0] = -2.5 + np.random.normal(0, 0.5, pf.n_particles)  # Increased from 0.1
    pf.particles[:, 1] = 6.0 + np.random.normal(0, 0.5, pf.n_particles)  # Increased from 0.1
    pf.particles[:, 2] = np.pi/2 + np.random.normal(0, 0.5, pf.n_particles)  # Increased from 0.3
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
    x, y, theta = -2.5, 6.0, np.pi / 2
    set_base_link_pose(robot_id, x, y, theta)
    current_wp = len(waypoints) - 1 

    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------
    print("Running baseline comparison simulation...")
    print(f"Total steps: {NUM_STEPS}")
    print("Progress: ", end='', flush=True)
    
    for step in range(NUM_STEPS):
        time.sleep(0.03)

        if (step + 1) % 100 == 0 or step == NUM_STEPS - 1:
            print(f"{step+1}...", end='', flush=True)

        x_prev, y_prev, theta_prev = x, y, theta

        # ---- MOTION CONTROL ----
        wp_x, wp_y = waypoints[current_wp]
        dx_to_wp, dy_to_wp = wp_x - x, wp_y - y
        target_theta = atan2(dy_to_wp, dx_to_wp)
        dtheta_err = wrap_angle(target_theta - theta)

        if abs(dtheta_err) < 0.3:
            dist = hypot(dx_to_wp, dy_to_wp)
            step_dist = min(dist, linear_speed)
            x += step_dist * cos(theta)
            y += step_dist * sin(theta)

        theta += np.clip(dtheta_err, -angular_speed, angular_speed)
        theta = wrap_angle(theta)

        set_base_link_pose(robot_id, x, y, theta)

        if hypot(dx_to_wp, dy_to_wp) < wp_threshold:
            current_wp = (current_wp + 1) % len(waypoints)

        # ---- CORRECTED ODOMETRY CALCULATION ----
        # Calculate displacement in global frame
        dx_global = x - x_prev
        dy_global = y - y_prev
        delta_theta = wrap_angle(theta - theta_prev)
        
        # CRITICAL FIX: Transform to robot's LOCAL frame using PREVIOUS orientation
        # This gives us the motion in the robot's coordinate system
        cos_prev = cos(theta_prev)
        sin_prev = sin(theta_prev)
        
        # Inverse rotation: R^T * [dx_global, dy_global]
        dx_local = cos_prev * dx_global + sin_prev * dy_global
        dy_local = -sin_prev * dx_global + cos_prev * dy_global
        
        # Now odometry is correctly in the robot's local frame
        odom = (dx_local, dy_local, delta_theta)

        # ---- MOTION/MEASUREMENT UPDATE ----
        pf.motion_update(odom)
        ekf.motion_update(odom)

        ranges, angles, _ = create_lidar_scan(robot_id, link_name)

        t0 = time.time()
        pf.measurement_update(ranges, angles)
        pf_times.append(time.time() - t0)

        t0 = time.time()
        ekf.measurement_update(ranges, angles)
        ekf_times.append(time.time() - t0)

        if pf.effective_sample_size() < 0.3 * pf.n_particles:  # Changed from 0.5
            pf.resample()
        
        pf_est = pf.estimate()
        ekf_est = ekf.estimate()

        pf_pos_err = hypot(x - pf_est[0], y - pf_est[1])
        ekf_pos_err = hypot(x - ekf_est[0], y - ekf_est[1])
        
        if (step + 1) % 20 == 0:
            print(f"\nStep {step+1}:")
            print(f"  True Pose: ({x:.2f}, {y:.2f}, {theta:.2f})")
            print(f"  PF  Estimate: ({pf_est[0]:.2f}, {pf_est[1]:.2f}, {pf_est[2]:.2f}) | Error: {pf_pos_err:.3f} m")
            print(f"  EKF Estimate: ({ekf_est[0]:.2f}, {ekf_est[1]:.2f}, {ekf_est[2]:.2f}) | Error: {ekf_pos_err:.3f} m")

        # ---- LOG DATA ----
        true_poses.append([x, y, theta])
        pf_estimates.append(pf.estimate())
        ekf_estimates.append(ekf.estimate())

    print("\nSimulation complete!")
    
    # --------------------------------------------------------
    # FINAL CALCULATION AND OUTPUT
    # --------------------------------------------------------
    pf_pos_rmse, pf_theta_rmse = calculate_rmse(true_poses, pf_estimates)
    ekf_pos_rmse, ekf_theta_rmse = calculate_rmse(true_poses, ekf_estimates)
    
    avg_pf_time_ms = np.mean(pf_times) * 1000 if pf_times else 0.0
    avg_ekf_time_ms = np.mean(ekf_times) * 1000 if ekf_times else 0.0

    print("\n" + "="*50)
    print("BASELINE PERFORMANCE COMPARISON RESULTS")
    print("="*50)

    print(f"\nTable 1: Baseline Performance Comparison ({NUM_STEPS} Steps)")
    print("{:<25}{:<15}{:<20}{}".format(
        "Filter", "Position RMSE", "Orientation RMSE", "Comp. Time/Step"
    ))
    print("-" * 75)
    print("{:<25}{:<15.3f}{:<20.3f}{:.2f} ms".format(
        "Particle Filter", pf_pos_rmse, pf_theta_rmse, avg_pf_time_ms
    ))
    print("{:<25}{:<15.3f}{:<20.3f}{:.2f} ms".format(
        "Extended Kalman Filter", ekf_pos_rmse, ekf_theta_rmse, avg_ekf_time_ms
    ))
    print("="*50)

    # --------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------
    try:
        if client:
            p.disconnect(client)
    except:
        pass

if __name__ == "__main__":
    main()