import os
import sys
import time
import numpy as np
import pybullet as p
from math import cos, sin, atan2, hypot

try:
    from utils import load_env, build_occupancy_grid, visualize_lidar, create_lidar_scan
    from pybullet_tools.utils import get_link_pose, link_from_name
    from filters.pf_model import ParticleFilter
    from filters.ekf_model import ExtendedKalmanFilter
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

NUM_STEPS = 800
linear_speed = 0.15
angular_speed = 0.1
wp_threshold = 0.2

def wrap_angle(a):
    """Wraps an angle to the range [-pi, pi]"""
    return np.fmod(a + np.pi, 2 * np.pi) - np.pi

def set_base_link_pose(robot_id, x_link, y_link, theta):
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

# MAIN SIMULATION AND CALCULATION
def main():
    client = None
    try:
        client = p.connect(p.GUI)
        if hasattr(sys.modules["pybullet_tools.utils"], "CLIENTS"):
             sys.modules["pybullet_tools.utils"].CLIENTS = {client: True}
    except Exception as e:
        print(f"Could not connect to PyBullet: {e}")
        return
    
    # LOAD ENVIRONMENT & INIT ROBOT
    robot_id = None
    try:
        robots_dict, _ = load_env("environment/figure8_env.json")
        robot_id = robots_dict.get("pr2", list(robots_dict.values())[0])
        link_name = "base_link"
        if not isinstance(robot_id, int):
             print(f"Warning: Expected robot ID to be an integer, got {type(robot_id)}")
    except Exception as e:
        print(f"Error loading environment: {e}")
        return

    # OCCUPANCY GRID
    occ, xs, ys = build_occupancy_grid(xmin=-8.0, xmax=8.0, ymin=-8.0, ymax=8.0, resolution=0.2)
    true_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.04, rgbaColor=(0, 1, 0, 1))
    pf_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.04, rgbaColor=(0, 0, 1, 1))
    
    # READ BACK ACTUAL POSE
    link_pose = get_link_pose(robot_id, link_from_name(robot_id, link_name))
    x, y = link_pose[0][0], link_pose[0][1]
    quat = link_pose[1]
    euler = p.getEulerFromQuaternion(quat)
    theta = euler[2]
    
    # Particle filter init
    pf = ParticleFilter(
        occ=occ, xs=xs, ys=ys, 
        n_particles=1000,
        lidar_max_range=10.0,
        lidar_min_range=0.1, 
        z_lidar=0.5, 
        scan_subsample=4,
        motion_noise=(0.1, 0.1, 0.1),
        sigma_range=0.3
    )
    
    pf.particles[:, 0] = x + np.random.normal(0, 0.5, pf.n_particles)
    pf.particles[:, 1] = y + np.random.normal(0, 0.5, pf.n_particles)
    pf.particles[:, 2] = theta + np.random.normal(0, 0.5, pf.n_particles)
    pf.weights[:] = 1.0 / pf.n_particles

    # EKF init
    # ekf = ExtendedKalmanFilter(
    #     initial_pose=(x, y, theta),  # Use ACTUAL pose!
    #     motion_noise=(0.1, 0.1, 0.1),
    #     lidar_max_range=10.0, 
    #     lidar_min_range=0.1, 
    #     z_lidar=0.5,
    #     sigma_range=0.3,
    #     scan_subsample=4
    # )
    # ekf.P = 1.0 * np.eye(3)

    # DATA LOGS
    true_poses = []
    pf_estimates = []
    # ekf_estimates = []
    pf_times = []
    # ekf_times = []
    current_wp = len(waypoints) - 1 
    
    # MAIN LOOP
    print("\nRunning baseline comparison simulation...")
    print(f"Total steps: {NUM_STEPS}")
    
    for step in range(NUM_STEPS):
        
        if ((step + 1) % (NUM_STEPS//10)) == 0 or step == NUM_STEPS - 1:
            print(f"step: {step + 1}")
        time.sleep(0.1)
        x_prev, y_prev, theta_prev = x, y, theta
        
        # MOTION CONTROL
        wp_x, wp_y = waypoints[current_wp]
        dx_to_wp, dy_to_wp = wp_x - x, wp_y - y
        target_theta = atan2(dy_to_wp, dx_to_wp)

        dtheta_err = wrap_angle(target_theta - theta)
        if abs(dtheta_err) < 0.3:
            dist = hypot(dx_to_wp, dy_to_wp)
            step_dist = min(dist, linear_speed)
            x += step_dist * cos(theta)
            y += step_dist * sin(theta)
        
        theta = wrap_angle(theta + np.clip(dtheta_err, -angular_speed, angular_speed))
        set_base_link_pose(robot_id, x, y, theta + np.pi/2)
        if hypot(dx_to_wp, dy_to_wp) < wp_threshold:
            current_wp = (current_wp + 1) % len(waypoints)
        
        # ODOMETRY CALCULATION
        # Calculate displacement in GLOBAL frame
        dx_global = x - x_prev
        dy_global = y - y_prev
        delta_theta = wrap_angle(theta - theta_prev)
        # For particle filter: convert to local frame using PREVIOUS orientation
        cos_prev = cos(theta_prev)
        sin_prev = sin(theta_prev)
        dx_local = cos_prev * dx_global + sin_prev * dy_global
        dy_local = -sin_prev * dx_global + cos_prev * dy_global
        odom_pf = (dx_local, dy_local, delta_theta)
    #     # For EKF: use global frame directly
    #     odom_ekf = (dx_global, dy_global, delta_theta)
        
        # MOTION/MEASUREMENT UPDATE
        pf.motion_update(odom_pf)
    #     ekf.motion_update(odom_ekf)
        
        ranges, angles, _ = create_lidar_scan(robot_id, link_name)

        t0 = time.time()
        pf.measurement_update(ranges, angles)
        pf_times.append(time.time() - t0)
    #     t0 = time.time()
    #     ekf.measurement_update(ranges, angles)
    #     ekf_times.append(time.time() - t0)
        if pf.effective_sample_size() < 0.3 * pf.n_particles:
            pf.resample()
        pf_est = pf.estimate()
    #     ekf_est = ekf.estimate()
    #     ekf_pos_err = hypot(x - ekf_est[0], y - ekf_est[1])

        # LOG DATA
        true_poses.append([x, y, theta])
        pf_estimates.append(pf.estimate())
    #     ekf_estimates.append(ekf.estimate())
        
        visualize_lidar(ranges, angles, robot_id, link_name) 
        p.createMultiBody(basePosition=[pf_est[0], pf_est[1], 0], baseCollisionShapeIndex=-1, baseVisualShapeIndex=pf_id)
        p.createMultiBody(basePosition=[x, y, 0], baseCollisionShapeIndex=-1, baseVisualShapeIndex=true_id)

    print("\nSimulation complete!")
    
    # FINAL CALCULATION
    print("="*60)
    print("BASELINE PERFORMANCE COMPARISON RESULTS")
    print("="*60)
    pf_pos_rmse, pf_theta_rmse = calculate_rmse(true_poses, pf_estimates)
    # ekf_pos_rmse, ekf_theta_rmse = calculate_rmse(true_poses, ekf_estimates)
    avg_pf_time_ms = np.mean(pf_times) * 1000 if pf_times else 0.0
    # avg_ekf_time_ms = np.mean(ekf_times) * 1000 if ekf_times else 0.0
    
    print(f"\nBaseline Performance Comparison ({NUM_STEPS} Steps)")
    print("{:<25}{:<20}{:<20}{}".format(
        "Filter", "Position RMSE", "Rotation RMSE", "Comp. Time/Step"
    ))
    print("-" * 75)
    print("{:<25}{:<20.3f}{:.2f} ms".format(
        "Particle Filter", pf_pos_rmse, pf_theta_rmse, avg_pf_time_ms
    ))
    # print("{:<25}{:<20.3f}{:.2f} ms".format(
    #     "Extended Kalman Filter", ekf_pos_rmse, avg_ekf_time_ms
    # ))
    print("="*60)
    print("ALL BASELINE COMPARISON COMPLETED (PRESS ANY KEY TO EXIT)")
    input("="*60)
    
    try:
        if client:
            p.disconnect(client)
    except:
        pass

if __name__ == "__main__":
    main()