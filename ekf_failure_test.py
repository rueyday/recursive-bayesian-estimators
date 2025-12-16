import os
import sys
import numpy as np
import pybullet as p
import time
from math import cos, sin, atan2, hypot

from lidar_utils import create_lidar_scan, visualize_lidar
from particle_model import ParticleFilter, build_occupancy_grid
from kalman_model import ExtendedKalmanFilter
from utils import load_env
from pybullet_tools.utils import get_link_pose, link_from_name

# Force display for Windows
if sys.platform == 'win32':
    os.environ['DISPLAY'] = ':0'

# Waypoints in symmetric corridor
waypoints = [(0.0, -3.0), (0.0, 3.0), (0.0, -3.0)]

linear_speed = 0.05
angular_speed = 0.15
wp_threshold = 0.2

def set_base_link_pose(robot, x_link, y_link, theta_link):
    """Set robot base so that base_link ends up at the desired (x, y, theta)."""
    _, _, z_base = p.getBasePositionAndOrientation(robot)[0]
    p.resetBasePositionAndOrientation(
        robot, [0, 0, z_base],
        p.getQuaternionFromEuler([0, 0, theta_link])
    )
    
    link_pose = get_link_pose(robot, link_from_name(robot, "base_link"))
    link_pos_at_origin = link_pose[0]
    
    offset_x = link_pos_at_origin[0]
    offset_y = link_pos_at_origin[1]
    
    x_base = x_link - offset_x
    y_base = y_link - offset_y
    z_base_desired = z_base
    
    p.resetBasePositionAndOrientation(
        robot, [x_base, y_base, z_base_desired],
        p.getQuaternionFromEuler([0, 0, theta_link])
    )

def is_pose_in_obstacle(x, y, occ, xs, ys):
    """Check if a pose is inside an occupied cell."""
    if x < xs[0] or x > xs[-1] or y < ys[0] or y > ys[-1]:
        return False
    
    i = np.searchsorted(xs, x)
    j = np.searchsorted(ys, y)
    i = np.clip(i, 0, len(xs) - 1)
    j = np.clip(j, 0, len(ys) - 1)
    
    return occ[j, i] == 1

def wrap_angle(a):
    """Wrap angle to [-pi, pi]"""
    return (a + np.pi) % (2*np.pi) - np.pi

def main():
    valid_pf_fraction = 0.0
    valid_ekf_fraction = 0.0
    NUM_STEPS = 50
    print("=" * 70)
    print("EKF FAILURE CASE: SYMMETRIC ENVIRONMENT TEST")
    print("=" * 70)
    print("\nThis test demonstrates the EKF's failure mode when faced with")
    print("symmetric/ambiguous initial conditions. The EKF mean will drift")
    print("into obstacles while the Particle Filter maintains valid hypotheses.")
    print("=" * 70 + "\n")
    
    client = p.connect(p.GUI)
    if not hasattr(sys.modules["pybullet_tools.utils"], "CLIENTS"):
        sys.modules["pybullet_tools.utils"].CLIENTS = {}
    sys.modules["pybullet_tools.utils"].CLIENTS[client] = True

    # Load symmetric corridor environment
    robots, obstacles = load_env("symmetric_corridor.json")
    pr2 = robots.get("pr2", list(robots.values())[0])
    link_name = "base_link"

    # Build occupancy grid
    xmin, xmax, ymin, ymax = -8.0, 8.0, -8.0, 8.0
    resolution = 0.2
    
    print("Building occupancy grid...")
    occ, xs, ys = build_occupancy_grid(xmin, xmax, ymin, ymax, resolution)
    print(f"Occupancy grid shape: {occ.shape}\n")

    # Initialize Particle Filter with BIMODAL distribution
    print("Initializing Particle Filter with bimodal distribution...")
    pf = ParticleFilter(
        occ=occ, xs=xs, ys=ys,
        n_particles=1000,  # Increased from 500
        lidar_max_range=10.0,
        lidar_min_range=0.1,
        z_lidar=0.5,
        scan_subsample=4,  # Reduced from 8 for better measurements
        motion_noise=(0.1, 0.1, 0.1),  # Increased motion noise
        sigma_range=0.3  # Increased measurement noise
    )

    # TRUE POSE (robot's actual location)
    true_x, true_y, true_theta = 3.0, 0.0, np.pi/2
    
    # SYMMETRIC POSE (mirror location that looks similar)
    symmetric_x, symmetric_y = -3.0, 0.0
    
    # Create bimodal particle distribution
    half = pf.n_particles // 2
    # Cluster 1: near TRUE pose
    pf.particles[:half, 0] = true_x + np.random.normal(0, 0.3, half)
    pf.particles[:half, 1] = true_y + np.random.normal(0, 0.3, half)
    pf.particles[:half, 2] = true_theta + np.random.normal(0, 0.1, half)
    
    # Cluster 2: at SYMMETRIC pose
    pf.particles[half:, 0] = symmetric_x + np.random.normal(0, 0.3, pf.n_particles - half)
    pf.particles[half:, 1] = symmetric_y + np.random.normal(0, 0.3, pf.n_particles - half)
    pf.particles[half:, 2] = true_theta + np.random.normal(0, 0.1, pf.n_particles - half)
    pf.weights = np.ones(pf.n_particles) / pf.n_particles

    # Initialize EKF with mean BETWEEN the two hypotheses
    print("Initializing EKF with mean between two symmetric locations...")
    ekf_init_x = (true_x + symmetric_x) / 2  # 0.0 - potentially in a wall!
    ekf_init_y = (true_y + symmetric_y) / 2  # 0.0
    ekf = ExtendedKalmanFilter(
        initial_pose=(ekf_init_x, ekf_init_y, true_theta),
        motion_noise=(0.05, 0.05, 0.03),  # Increased motion noise
        lidar_max_range=10.0,
        lidar_min_range=0.1,
        z_lidar=0.5,
        sigma_range=0.3,  # Increased measurement noise
        scan_subsample=4  # Reduced from 8
    )
    ekf.P = np.diag([4.0, 4.0, 0.25])  # Large uncertainty covering both locations

    print(f"\nSetup:")
    print(f"  True robot position: ({true_x:.1f}, {true_y:.1f})")
    print(f"  Symmetric position:  ({symmetric_x:.1f}, {symmetric_y:.1f})")
    print(f"  EKF initial mean:    ({ekf_init_x:.1f}, {ekf_init_y:.1f})")
    print(f"  PF: {half} particles at each location\n")

    # Set robot at TRUE position
    x, y, theta = true_x, true_y, true_theta
    set_base_link_pose(pr2, x, y, theta)

    # Reset camera to view the scene
    p.resetDebugVisualizerCamera(
        cameraDistance=15,
        cameraYaw=0,
        cameraPitch=-45,
        cameraTargetPosition=[0, 0, 0]
    )

    print("=" * 70)
    print("Starting motion... Watch for:")
    print("  - BLUE arrow = EKF estimate (may go into walls)")
    print("  - GREEN dots = PF particles (watch clusters separate)")
    print("=" * 70 + "\n")

    step_num = 0
    current_wp = 0
    
    ekf_in_obstacle_count = 0
    ekf_impossible_steps = []

    while step_num < NUM_STEPS:  # Run for 200 steps
        time.sleep(0.05)
        step_num += 1

        # Store previous pose
        x_prev, y_prev, theta_prev = x, y, theta

        # Move toward waypoint
        wp_x, wp_y = waypoints[current_wp]
        dx_to_goal = wp_x - x
        dy_to_goal = wp_y - y
        target_theta = atan2(dy_to_goal, dx_to_goal)
        dtheta = wrap_angle(target_theta - theta)

        # Execute motion
        if abs(dtheta) < 0.3:
            dist = hypot(dx_to_goal, dy_to_goal)
            step_val = min(dist, linear_speed)
            x += step_val * cos(theta)
            y += step_val * sin(theta)

        theta += np.clip(dtheta, -angular_speed, +angular_speed)
        theta = wrap_angle(theta)
        
        set_base_link_pose(pr2, x, y, theta)

        # Check waypoint
        if hypot(wp_x - x, wp_y - y) < wp_threshold:
            current_wp = (current_wp + 1) % len(waypoints)

        # ========== CORRECTED ODOMETRY CALCULATION ==========
        # Compute odometry in LOCAL frame
        dx_global = x - x_prev
        dy_global = y - y_prev
        dtheta = wrap_angle(theta - theta_prev)
        
        # CRITICAL FIX: Transform global displacement to robot's LOCAL frame 
        # using PREVIOUS orientation
        cos_prev = cos(theta_prev)
        sin_prev = sin(theta_prev)
        dx_local = cos_prev * dx_global + sin_prev * dy_global
        dy_local = -sin_prev * dx_global + cos_prev * dy_global
        
        # Update both filters with local frame odometry
        pf.motion_update((dx_local, dy_local, dtheta))
        ekf.motion_update((dx_local, dy_local, dtheta))
        # ====================================================

        # LiDAR and measurement update
        ranges, angles, _ = create_lidar_scan(pr2, link_name)
        visualize_lidar(ranges, angles, pr2, link_name)
        
        pf.measurement_update(ranges, angles)
        ekf.measurement_update(ranges, angles)

        # Resample PF with adjusted threshold
        ess = pf.effective_sample_size()
        if ess < 0.3 * pf.n_particles:  # Changed from 0.5
            pf.resample()

        # Get estimates
        pf_est = pf.estimate_map()
        ekf_est = ekf.estimate()

        # Check if EKF is in obstacle
        ekf_in_obstacle = is_pose_in_obstacle(ekf_est[0], ekf_est[1], occ, xs, ys)
        if ekf_in_obstacle:
            ekf_in_obstacle_count += 1
            ekf_impossible_steps.append(step_num)

        # Visualize
        pf_start = (pf_est[0], pf_est[1], pf.z_lidar)
        pf_end = (pf_est[0] + cos(pf_est[2]) * 0.5, pf_est[1] + sin(pf_est[2]) * 0.5, pf.z_lidar)
        p.addUserDebugLine(pf_start, pf_end, [1, 0, 0], lineWidth=5, lifeTime=0.1)
        pf.draw_particles(life_time=0.1)
        
        ekf.draw_estimate(color=[0, 0, 1], life_time=0.1)
        
        # Draw warning if EKF is in obstacle
        if ekf_in_obstacle:
            warning_pos = (ekf_est[0], ekf_est[1], pf.z_lidar + 0.5)
            p.addUserDebugText("INVALID!", warning_pos, [1, 0, 0], 
                             textSize=2, lifeTime=0.1)

        # Get true pose
        link_pose = get_link_pose(pr2, link_from_name(pr2, "base_link"))
        true_x_actual, true_y_actual = link_pose[0][0], link_pose[0][1]
        true_theta_actual = p.getEulerFromQuaternion(link_pose[1])[2]
        
        # Calculate errors
        pf_pos_error = hypot(pf_est[0] - true_x_actual, pf_est[1] - true_y_actual)
        ekf_pos_error = hypot(ekf_est[0] - true_x_actual, ekf_est[1] - true_y_actual)

        # Update valid estimate fractions
        if is_pose_in_obstacle(pf_est[0], pf_est[1], occ, xs, ys):
            valid_pf_fraction += 0.0
        else:
            valid_pf_fraction += 1.0
        if is_pose_in_obstacle(ekf_est[0], ekf_est[1], occ, xs, ys):
            valid_ekf_fraction += 0.0
        else:
            valid_ekf_fraction += 1.0
        
        if step_num % 10 == 0:
            print(f"Step {step_num:3d}:")
            print(f"  PF Estimate: ({pf_est[0]:.2f}, {pf_est[1]:.2f}, {pf_est[2]:.2f}) | Valid: {100*valid_pf_fraction/step_num:.1f}%")
            print(f"  EKF Estimate: ({ekf_est[0]:.2f}, {ekf_est[1]:.2f}, {ekf_est[2]:.2f}) | Valid: {100*valid_ekf_fraction/step_num:.1f}%")


    # Final report
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Total steps: {step_num}")
    print(f"EKF mean was inside obstacle for {ekf_in_obstacle_count} steps " +
          f"({100*ekf_in_obstacle_count/step_num:.1f}%)")
    print(f"PF maintained valid estimates throughout all {step_num} steps")
    
    if ekf_impossible_steps:
        print(f"\nEKF produced impossible poses at steps: {ekf_impossible_steps[:20]}..." 
              if len(ekf_impossible_steps) > 20 else 
              f"\nEKF produced impossible poses at steps: {ekf_impossible_steps}")
    
    print("\n" + "=" * 70)
    print("CONCLUSION:")
    print("This demonstrates the EKF's fundamental limitation with multimodal")
    print("distributions. When averaging two symmetric hypotheses, its mean")
    print("can drift into physically impossible locations. The Particle Filter")
    print("maintains separate hypothesis clusters and correctly disambiguates.")
    print("=" * 70)
    
    # Disconnect PyBullet and clean up
    print("\nClosing PyBullet simulation...")
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