# baseline_calculator_v2.py
import os
import sys
import time
import numpy as np
import pybullet as p
from math import cos, sin, atan2, hypot

# =================================================================
# IMPORTANT: ASSUMED EXTERNAL IMPORTS
# (Keep these comments to remind user of required custom files)
# ...
# =================================================================

try:
    from lidar_utils import create_lidar_scan
    from particle_model import ParticleFilter, build_occupancy_grid
    from kalman_model import ExtendedKalmanFilter
    from utils import load_env
    from pybullet_tools.utils import get_link_pose, link_from_name
except ImportError as e:
    # Use placeholder/mock imports for structure only
    # (The user will replace this with their actual working environment)
    print(f"CRITICAL ERROR: Failed to import custom modules. Please ensure your environment is set up correctly. Missing module or function: {e}")
    class DummyFilter:
        def __init__(self, **kwargs): self.n_particles = 500
        def motion_update(self, odom): pass
        def measurement_update(self, ranges, angles): pass
        def estimate(self): return np.array([0.0, 0.0, 0.0])
        def effective_sample_size(self): return 100
        def resample(self): pass
        def __setattr__(self, name, value):
            if name == 'P': self._P = value
            else: super().__setattr__(name, value)
        
    ParticleFilter = DummyFilter
    ExtendedKalmanFilter = DummyFilter
    
    def build_occupancy_grid(xmin, xmax, ymin, ymax, resolution): return np.zeros((1,1)), np.array([0]), np.array([0])
    def load_env(env_file): return {"pr2": 0}, None # Mock returns a dictionary with ID 0
    def create_lidar_scan(robot, link_name): return [], [], None
    def get_link_pose(robot, link): return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    def link_from_name(robot, name): return 0
    
    class PyBulletMock:
        def connect(self, mode): return 1
        def getBasePositionAndOrientation(self, body): return [0, 0, 0], [0, 0, 0, 1]
        def resetBasePositionAndOrientation(self, body, pos, orn): pass
        def getQuaternionFromEuler(self, euler): return [0, 0, 0, 1]
        def getEulerFromQuaternion(self, quat): return [0, 0, 0]
        def disconnect(self, client): pass
    p = PyBulletMock()
    
    class PybulletToolsMock:
        CLIENTS = {}
    sys.modules["pybullet_tools.utils"] = PybulletToolsMock()
# =================================================================


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

# MODIFIED: set_base_link_pose now takes the robot ID (int) instead of a wrapper object
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
        # print(f"Warning in set_base_link_pose: {e}")
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
        # Ensure PyBullet client is registered with pybullet_tools.utils
        if hasattr(sys.modules["pybullet_tools.utils"], "CLIENTS"):
             sys.modules["pybullet_tools.utils"].CLIENTS = {client: True}
    except Exception as e:
        print(f"Could not connect to PyBullet. Simulation may fail: {e}")
        client = 100 
    
    # --------------------------------------------------------
    # LOAD ENVIRONMENT & INIT ROBOT
    # --------------------------------------------------------
    robot_id = None
    try:
        robots_dict, _ = load_env("figure8_env.json")
        # Assume the value is the PyBullet body ID (an int)
        robot_id = robots_dict.get("pr2", list(robots_dict.values())[0])
        link_name = "base_link"
        if not isinstance(robot_id, int):
             print(f"Warning: Expected robot ID to be an integer (PyBullet body ID), but got {type(robot_id)}")
    except Exception as e:
        print(f"Error loading environment: {e}. Cannot proceed.")
        return

    # --------------------------------------------------------
    # OCCUPANCY GRID & FILTERS INIT
    # --------------------------------------------------------
    occ, xs, ys = build_occupancy_grid(xmin=-6.0, xmax=6.0, ymin=-6.0, ymax=6.0, resolution=0.2)

    # Filter initialization assumes success
    pf = ParticleFilter(
        occ=occ, xs=xs, ys=ys, n_particles=500, lidar_max_range=10.0,
        lidar_min_range=0.1, z_lidar=0.5, scan_subsample=8
    )

    ekf = ExtendedKalmanFilter(
        initial_pose=(-4.0, 5.0, np.pi/2), motion_noise=(0.02, 0.02, 0.01),
        lidar_max_range=10.0, lidar_min_range=0.1, z_lidar=0.5,
        sigma_range=0.2, scan_subsample=8
    )

    # Initial Belief Setup
    ekf.P = 0.5 * np.eye(3)
    pf.particles[:, 0] = -4.0 + np.random.normal(0, 0.5, pf.n_particles)
    pf.particles[:, 1] = 5.0 + np.random.normal(0, 0.5, pf.n_particles)
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

        # ---- TRUE ODOMETRY ----
        delta_x = x - x_prev
        delta_y = y - y_prev
        delta_theta = wrap_angle(theta - theta_prev)

        # ---- MOTION/MEASUREMENT UPDATE ----
        pf.motion_update((delta_x, delta_y, delta_theta))
        ekf.motion_update((delta_x, delta_y, delta_theta))

        ranges, angles, _ = create_lidar_scan(robot_id, link_name)

        t0 = time.time()
        pf.measurement_update(ranges, angles)
        pf_times.append(time.time() - t0)

        t0 = time.time()
        ekf.measurement_update(ranges, angles)
        ekf_times.append(time.time() - t0)

        if pf.effective_sample_size() < 0.5 * pf.n_particles:
            pf.resample()

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

    # Print Table 1
    print("Table 1: Baseline Performance Comparison (1200 Steps)")
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