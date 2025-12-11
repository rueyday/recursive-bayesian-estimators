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

# Waypoint list
waypoints = [(-4.0, 0.0), (-12.0, 0.0), (-12.0, 7.0), (-4.0, 7.0)]

linear_speed = 0.05      # m per update step
angular_speed = 0.15     # rad per update step
wp_threshold = 0.2       # how close counts as "reached"

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

    # Define map bounds and resolution for occupancy grid
    xmin, xmax, ymin, ymax = -15.0, 0.0, 0.0, 15.0  # adjust to your map
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
    step_num = 0
    x, y, theta = -4.0, 5.0, np.pi/2  # initial dummy pose
    # After setting initial pose
    pf.particles[:, 0] = x + np.random.normal(0, 0.5, pf.n_particles)
    pf.particles[:, 1] = y + np.random.normal(0, 0.5, pf.n_particles)
    pf.particles[:, 2] = theta + np.random.normal(0, 0.3, pf.n_particles)
    current_wp = 0

    while True:
        time.sleep(0.05)
        step_num += 1

        # Store previous pose
        x_prev, y_prev, theta_prev = x, y, theta

        # ---- Motion Control ----
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
        
        set_base_values(pr2, (x, y, theta))

        # Check waypoint
        if np.hypot(wp_x - x, wp_y - y) < wp_threshold:
            current_wp = (current_wp + 1) % len(waypoints)
            print(f"Reached waypoint → moving to WP {current_wp}")

        # ---- Particle Filter Update ----
        # Compute actual odometry (what the robot actually did)
        actual_dx = x - x_prev
        actual_dy = y - y_prev
        actual_dtheta = theta - theta_prev
        actual_dtheta = (actual_dtheta + np.pi) % (2*np.pi) - np.pi
        
        # Update particles with actual motion
        pf.motion_update((actual_dx, actual_dy, actual_dtheta))

        # LiDAR and measurement update
        ranges, angles, _ = create_lidar_scan(pr2, link_name)
        visualize_lidar(ranges, angles, pr2, link_name)
        pf.measurement_update(ranges, angles)

        # Resample if needed
        ess = pf.effective_sample_size()
        if ess < 0.5 * pf.n_particles:
            pf.resample()

        # Visualize
        est = pf.estimate()
        est_start = (est[0], est[1], pf.z_lidar)
        est_end = (est[0] + cos(est[2]) * 0.3, est[1] + sin(est[2]) * 0.3, pf.z_lidar)
        p.addUserDebugLine(est_start, est_end, [1, 0, 0], lineWidth=3, lifeTime=0.1)
        pf.draw_particles(life_time=0.1)

        true_pose = get_link_pose(pr2, link_from_name(pr2, "base_link"))
        print(f"Step {step_num}: Est={est}, True={true_pose[0][:2]}, theta_diff={abs(est[2]-true_pose[1][2]):.3f}")


if __name__ == "__main__":
    main()
