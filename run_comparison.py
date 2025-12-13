import os
import sys
import numpy as np
import pybullet as p
import time
from math import cos, sin

from lidar_utils import create_lidar_scan, visualize_lidar
from particle_model import ParticleFilter, build_occupancy_grid
from kalman_model import ExtendedKalmanFilter
from utils import load_env
from pybullet_tools.utils import set_base_values, get_link_pose, link_from_name, get_base_values

# Force display for Windows
if sys.platform == 'win32':
    os.environ['DISPLAY'] = ':0'

# Waypoint list
waypoints = [(-4.0, 9.0), (4.0, 9.0), (4.0, 2.0), (-4.0, 2.0)]

linear_speed = 0.05      # m per update step
angular_speed = 0.15     # rad per update step
wp_threshold = 0.2       # how close counts as "reached"

def set_base_link_pose(robot, x_link, y_link, theta_link):
    """
    Set robot base so that base_link ends up at the desired (x, y, theta).
    """
    _, _, z_base = p.getBasePositionAndOrientation(robot)[0]
    p.resetBasePositionAndOrientation(
        robot,
        [0, 0, z_base],
        p.getQuaternionFromEuler([0, 0, theta_link])
    )
    
    link_pose = get_link_pose(robot, link_from_name(robot, "base_link"))
    link_pos_at_origin = link_pose[0]
    
    offset_x = link_pos_at_origin[0]
    offset_y = link_pos_at_origin[1]
    offset_z = link_pos_at_origin[2]
    
    x_base = x_link - offset_x
    y_base = y_link - offset_y
    z_base_desired = z_base
    
    p.resetBasePositionAndOrientation(
        robot,
        [x_base, y_base, z_base_desired],
        p.getQuaternionFromEuler([0, 0, theta_link])
    )

def main():
    print("=" * 60)
    print("PARTICLE FILTER vs EXTENDED KALMAN FILTER COMPARISON")
    print("=" * 60)
    print("\nConnecting to PyBullet GUI...")
    client = p.connect(p.GUI)
    if not hasattr(sys.modules["pybullet_tools.utils"], "CLIENTS"):
        sys.modules["pybullet_tools.utils"].CLIENTS = {}
    sys.modules["pybullet_tools.utils"].CLIENTS[client] = True

    # Load environment
    robots, obstacles = load_env("8path_env.json")
    pr2 = robots.get("pr2", list(robots.values())[0])
    link_name = "base_link"

    # Define map bounds and resolution for occupancy grid (for PF)
    xmin, xmax, ymin, ymax = -10.0, 10.0, -10.0, 10.0
    resolution = 0.2

    print("\nBuilding occupancy grid for Particle Filter...")
    occ, xs, ys = build_occupancy_grid(xmin, xmax, ymin, ymax, resolution)
    print(f"Occupancy grid shape: {occ.shape}")

    # Initialize Particle Filter
    print("\nInitializing Particle Filter...")
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

    # Initialize Extended Kalman Filter
    print("Initializing Extended Kalman Filter...")
    ekf = ExtendedKalmanFilter(
        initial_pose=(-4.0, 5.0, np.pi/2),
        motion_noise=(0.02, 0.02, 0.01),
        lidar_max_range=10.0,
        lidar_min_range=0.1,
        z_lidar=0.5,
        sigma_range=0.2,
        scan_subsample=8
    )

    # Initial pose
    x, y, theta = -4.0, 5.0, np.pi/2
    set_base_link_pose(pr2, x, y, theta)
    
    # Initialize PF particles around starting pose
    pf.particles[:, 0] = x + np.random.normal(0, 0.5, pf.n_particles)
    pf.particles[:, 1] = y + np.random.normal(0, 0.5, pf.n_particles)
    pf.particles[:, 2] = theta + np.random.normal(0, 0.3, pf.n_particles)
    pf.weights = np.ones(pf.n_particles) / pf.n_particles

    print("\n" + "=" * 60)
    print("Both filters initialized! Starting comparison...")
    print("Colors: RED = PF estimate, BLUE = EKF estimate, GREEN = PF particles")
    print("=" * 60 + "\n")

    step_num = 0
    current_wp = 0

    # Error tracking
    pf_pos_errors = []
    pf_theta_errors = []
    ekf_pos_errors = []
    ekf_theta_errors = []

    while True:
        time.sleep(0.05)
        step_num += 1

        # Store previous pose
        x_prev, y_prev, theta_prev = x, y, theta

        # Move toward current waypoint
        wp_x, wp_y = waypoints[current_wp]
        dx_to_goal = wp_x - x
        dy_to_goal = wp_y - y
        target_theta = np.arctan2(dy_to_goal, dx_to_goal)
        dtheta = (target_theta - theta + np.pi) % (2*np.pi) - np.pi

        # Execute motion
        if abs(dtheta) < 0.3:
            dist = np.sqrt(dx_to_goal**2 + dy_to_goal**2)
            step = min(dist, linear_speed)
            x += step * np.cos(theta)
            y += step * np.sin(theta)

        theta += np.clip(dtheta, -angular_speed, +angular_speed)
        theta = (theta + np.pi) % (2*np.pi) - np.pi
        
        # Update robot
        set_base_link_pose(pr2, x, y, theta)

        # Check waypoint
        if np.hypot(wp_x - x, wp_y - y) < wp_threshold:
            current_wp = (current_wp + 1) % len(waypoints)
            print(f"\n>>> Reached waypoint → moving to WP {current_wp}: {waypoints[current_wp]} <<<\n")

        # Compute odometry
        actual_dx = x - x_prev
        actual_dy = y - y_prev
        actual_dtheta = theta - theta_prev
        actual_dtheta = (actual_dtheta + np.pi) % (2*np.pi) - np.pi
        
        # Update both filters - Motion model
        pf.motion_update((actual_dx, actual_dy, actual_dtheta))
        ekf.motion_update((actual_dx, actual_dy, actual_dtheta))

        # LiDAR scan
        ranges, angles, _ = create_lidar_scan(pr2, link_name)
        visualize_lidar(ranges, angles, pr2, link_name)
        
        # Update both filters - Measurement model
        pf.measurement_update(ranges, angles)
        ekf.measurement_update(ranges, angles)

        # Resample PF if necessary
        ess = pf.effective_sample_size()
        if ess < 0.5 * pf.n_particles:
            pf.resample()

        # Get estimates
        pf_est = pf.estimate()
        ekf_est = ekf.estimate()

        # Visualize
        # PF: Red arrow + green particles
        pf_start = (pf_est[0], pf_est[1], pf.z_lidar)
        pf_end = (pf_est[0] + cos(pf_est[2]) * 0.3, pf_est[1] + sin(pf_est[2]) * 0.3, pf.z_lidar)
        p.addUserDebugLine(pf_start, pf_end, [1, 0, 0], lineWidth=3, lifeTime=0.1)
        pf.draw_particles(life_time=0.1)
        
        # EKF: Blue arrow + uncertainty ellipse
        ekf.draw_estimate(color=[0, 0, 1], life_time=0.1)

        # Get true pose
        link_pose = get_link_pose(pr2, link_from_name(pr2, link_name))
        true_x, true_y = link_pose[0][0], link_pose[0][1]
        true_theta = p.getEulerFromQuaternion(link_pose[1])[2]
        
        # Calculate errors
        pf_pos_error = np.sqrt((pf_est[0] - true_x)**2 + (pf_est[1] - true_y)**2)
        pf_theta_error = abs((pf_est[2] - true_theta + np.pi) % (2*np.pi) - np.pi)
        ekf_pos_error = np.sqrt((ekf_est[0] - true_x)**2 + (ekf_est[1] - true_y)**2)
        ekf_theta_error = abs((ekf_est[2] - true_theta + np.pi) % (2*np.pi) - np.pi)
        
        pf_pos_errors.append(pf_pos_error)
        pf_theta_errors.append(pf_theta_error)
        ekf_pos_errors.append(ekf_pos_error)
        ekf_theta_errors.append(ekf_theta_error)
        
        # Get EKF covariance
        P = ekf.get_covariance()
        ekf_pos_uncertainty = np.sqrt(P[0,0] + P[1,1])
        
        print(f"Step {step_num}:")
        print(f"  True  = ({true_x:.2f}, {true_y:.2f}, {true_theta:.2f})")
        print(f"  PF    = ({pf_est[0]:.2f}, {pf_est[1]:.2f}, {pf_est[2]:.2f}) | Error: {pf_pos_error:.3f}m, {np.degrees(pf_theta_error):.1f}°")
        print(f"  EKF   = ({ekf_est[0]:.2f}, {ekf_est[1]:.2f}, {ekf_est[2]:.2f}) | Error: {ekf_pos_error:.3f}m, {np.degrees(ekf_theta_error):.1f}°")
        print(f"  ESS   = {ess:.0f} | EKF Unc = {ekf_pos_uncertainty:.3f}m")
        
        # Comparative statistics every 50 steps
        if step_num % 50 == 0:
            print("\n" + "=" * 60)
            print(f"STATISTICS - Last 50 steps")
            print("=" * 60)
            
            pf_avg_pos = np.mean(pf_pos_errors[-50:])
            pf_avg_theta = np.mean(pf_theta_errors[-50:])
            ekf_avg_pos = np.mean(ekf_pos_errors[-50:])
            ekf_avg_theta = np.mean(ekf_theta_errors[-50:])
            
            print(f"Particle Filter:")
            print(f"  Avg Position Error: {pf_avg_pos:.3f}m")
            print(f"  Avg Theta Error: {np.degrees(pf_avg_theta):.1f}°")
            
            print(f"\nExtended Kalman Filter:")
            print(f"  Avg Position Error: {ekf_avg_pos:.3f}m")
            print(f"  Avg Theta Error: {np.degrees(ekf_avg_theta):.1f}°")
            
            print(f"\nComparison:")
            if ekf_avg_pos < pf_avg_pos:
                improvement = ((pf_avg_pos - ekf_avg_pos) / pf_avg_pos) * 100
                print(f"  EKF is {improvement:.1f}% more accurate in position")
            else:
                improvement = ((ekf_avg_pos - pf_avg_pos) / ekf_avg_pos) * 100
                print(f"  PF is {improvement:.1f}% more accurate in position")
            
            print("=" * 60 + "\n")


if __name__ == "__main__":
    main()