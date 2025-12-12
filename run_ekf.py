import os
import sys
import numpy as np
import pybullet as p
import time
from math import cos, sin

from lidar_utils import create_lidar_scan, visualize_lidar
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
    
    Args:
        robot: PyBullet body ID
        x_link, y_link: desired base_link position
        theta_link: desired base_link orientation (yaw)
    """
    # First, put robot at origin with desired orientation
    _, _, z_base = p.getBasePositionAndOrientation(robot)[0]
    p.resetBasePositionAndOrientation(
        robot,
        [0, 0, z_base],
        p.getQuaternionFromEuler([0, 0, theta_link])
    )
    
    # Now measure where base_link ended up
    link_pose = get_link_pose(robot, link_from_name(robot, "base_link"))
    link_pos_at_origin = link_pose[0]
    
    # The offset from base to base_link (in world frame, at this orientation)
    offset_x = link_pos_at_origin[0]
    offset_y = link_pos_at_origin[1]
    offset_z = link_pos_at_origin[2]
    
    # To get base_link to (x_link, y_link), we need to put base at:
    x_base = x_link - offset_x
    y_base = y_link - offset_y
    z_base_desired = z_base  # Keep same height
    
    # Set the base to the corrected position
    p.resetBasePositionAndOrientation(
        robot,
        [x_base, y_base, z_base_desired],
        p.getQuaternionFromEuler([0, 0, theta_link])
    )

def main():
    print("Connecting to PyBullet GUI...")
    client = p.connect(p.GUI)
    if not hasattr(sys.modules["pybullet_tools.utils"], "CLIENTS"):
        sys.modules["pybullet_tools.utils"].CLIENTS = {}
    sys.modules["pybullet_tools.utils"].CLIENTS[client] = True

    # Load environment
    robots, obstacles = load_env("8path_env.json")
    pr2 = robots.get("pr2", list(robots.values())[0])
    link_name = "base_link"

    # Initialize Extended Kalman Filter
    ekf = ExtendedKalmanFilter(
        initial_pose=(-4.0, 5.0, np.pi/2),
        motion_noise=(0.02, 0.02, 0.01),
        lidar_max_range=10.0,
        lidar_min_range=0.1,
        z_lidar=0.5,
        sigma_range=0.2,
        scan_subsample=8
    )

    print("Extended Kalman Filter initialized!")

    # Initial pose (in base_link frame)
    x, y, theta = -4.0, 5.0, np.pi/2
    set_base_link_pose(pr2, x, y, theta)

    step_num = 0
    current_wp = 0

    # For tracking errors
    position_errors = []
    theta_errors = []

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
        
        # Update robot using base_link pose
        set_base_link_pose(pr2, x, y, theta)

        # Check waypoint
        if np.hypot(wp_x - x, wp_y - y) < wp_threshold:
            current_wp = (current_wp + 1) % len(waypoints)
            print(f"Reached waypoint → moving to WP {current_wp}: {waypoints[current_wp]}")

        # Compute actual odometry
        actual_dx = x - x_prev
        actual_dy = y - y_prev
        actual_dtheta = theta - theta_prev
        actual_dtheta = (actual_dtheta + np.pi) % (2*np.pi) - np.pi
        
        # EKF Prediction step
        ekf.motion_update((actual_dx, actual_dy, actual_dtheta))

        # LiDAR scan and measurement update
        ranges, angles, _ = create_lidar_scan(pr2, link_name)
        visualize_lidar(ranges, angles, pr2, link_name)
        
        # EKF Correction step
        ekf.measurement_update(ranges, angles)

        # Get estimate and visualize
        est = ekf.estimate()
        ekf.draw_estimate(color=[0, 0, 1], life_time=0.1)  # Blue for EKF

        # Get true pose from base_link
        link_pose = get_link_pose(pr2, link_from_name(pr2, link_name))
        true_x, true_y = link_pose[0][0], link_pose[0][1]
        true_theta = p.getEulerFromQuaternion(link_pose[1])[2]
        
        # Calculate errors
        pos_error = np.sqrt((est[0] - true_x)**2 + (est[1] - true_y)**2)
        theta_error = abs((est[2] - true_theta + np.pi) % (2*np.pi) - np.pi)
        position_errors.append(pos_error)
        theta_errors.append(theta_error)
        
        # Get covariance for uncertainty reporting
        P = ekf.get_covariance()
        pos_uncertainty = np.sqrt(P[0,0] + P[1,1])
        theta_uncertainty = np.sqrt(P[2,2])
        
        print(f"Step {step_num}:")
        print(f"  Est=({est[0]:.2f}, {est[1]:.2f}, {est[2]:.2f})")
        print(f"  True=({true_x:.2f}, {true_y:.2f}, {true_theta:.2f})")
        print(f"  Pos Error={pos_error:.3f}m, Theta Error={np.degrees(theta_error):.1f}°")
        print(f"  Uncertainty: Pos={pos_uncertainty:.3f}m, Theta={np.degrees(theta_uncertainty):.1f}°")
        
        # Print statistics every 50 steps
        if step_num % 50 == 0:
            avg_pos_error = np.mean(position_errors[-50:])
            avg_theta_error = np.mean(theta_errors[-50:])
            print(f"\n=== Statistics (last 50 steps) ===")
            print(f"  Avg Position Error: {avg_pos_error:.3f}m")
            print(f"  Avg Theta Error: {np.degrees(avg_theta_error):.1f}°")
            print("===================================\n")


if __name__ == "__main__":
    main()