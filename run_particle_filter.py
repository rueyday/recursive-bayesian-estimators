from motion_model import RobotModel
from particle_model import ParticleFilter
from lidar_utils import create_lidar_scan, visualize_lidar, visualize_lidar_pointcloud
import os
import sys

# Force display for Windows
if sys.platform == 'win32':
    os.environ['DISPLAY'] = ':0'

import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
import pybullet_tools.utils
from pybullet_tools.utils import disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS, get_base_pose
import pybullet as p
import time

waypoints = [
    (2.0, 0.0),
    (2.0, 2.0),
    (0.0, 2.0),
    (0.0, 0.0)
]

def compute_control(est, goal):
    dx = goal[0] - est["x"]
    dy = goal[1] - est["y"]

    goal_theta = np.arctan2(dy, dx)
    theta_err = goal_theta - est["theta"]
    theta_err = (theta_err + np.pi) % (2*np.pi) - np.pi

    v = 0.5
    omega = 2.0 * theta_err

    omega = np.clip(omega, -2.0, 2.0)
    dist = np.hypot(dx, dy)
    v = min(0.5, dist)

    return v, omega


def main(screenshot=False):
    print("Connecting to PyBullet GUI...")
    
    # Manual connection that properly registers with CLIENTS
    client = p.connect(p.GUI)
    print(f"Connected! Physics client ID: {client}")
    
    # Manually register the client in the CLIENTS dictionary
    # This is what the connect() utility normally does
    # CLIENTS[client_id] should store the rendering state (True/False)
    if not hasattr(pybullet_tools.utils, 'CLIENTS'):
        pybullet_tools.utils.CLIENTS = {}
    
    pybullet_tools.utils.CLIENTS[client] = True  # True means rendering is enabled
    
    print(f"Registered client. CLIENTS dictionary: {pybullet_tools.utils.CLIENTS}")
    
    robots, obstacles = load_env('8path_env.json')
    
    print("Available robots:", list(robots.keys()))
    
    if 'pr2' in robots:
        pr2 = robots['pr2']
    elif 'robot' in robots:
        pr2 = robots['robot']
    else:
        pr2 = list(robots.values())[0]
    
    print(f"Using robot ID: {pr2}")

    robot_model = RobotModel()

    # Initial guess (rough, not perfect)
    init_state = {"x": 0.0, "y": 0.0, "theta": 0.0}

    pf = ParticleFilter(
        robot_model=robot_model,
        num_particles=500,
        init_state=init_state
    )

    dt = 0.1
    truth = {"x": -4.0, "y": 5.0, "theta": np.pi/2}
    
    wait_if_gui('Environment loaded. Press to start.')
    
    print("Starting lidar visualization. Press Ctrl+C to stop.")
    try:
        while True:

            # -------------------------
            # 1. APPLY CONTROL (TRUTH)
            # -------------------------
            v, omega = compute_control(truth, waypoints[0])

            truth["x"] += v * np.cos(truth["theta"]) * dt
            truth["y"] += v * np.sin(truth["theta"]) * dt
            truth["theta"] += omega * dt
            truth["theta"] = (truth["theta"] + np.pi) % (2*np.pi) - np.pi
            p.resetBasePositionAndOrientation(
                pr2,
                (truth["x"], truth["y"], 0.0),
                p.getQuaternionFromEuler((0, 0, truth["theta"]))
            )


            # -------------------------
            # 2. PARTICLE PREDICTION
            # -------------------------
            pf.predict(v, omega, dt)

            # -------------------------
            # 3. SENSOR UPDATE
            # -------------------------
            ranges, angles, _ = create_lidar_scan(
                pr2, 'head_tilt_link', num_rays=360, max_range=10.0
            )

            pf.update(ranges)
            pf.resample()

            # -------------------------
            # 4. ESTIMATE STATE
            # -------------------------
            est = pf.estimate()
            print(f"PF estimate: x={est['x']:.2f}, y={est['y']:.2f}")

            # -------------------------
            # 5. VISUALIZATION
            # -------------------------
            draw_sphere_marker((est['x'], est['y'], 0.1), radius=0.1, color=(0,1,0))

            visualize_lidar_pointcloud(ranges, angles, pr2, 'head_tilt_link')

            print(f"TRUTH: {truth}")
            print(f"EST: {pf.estimate()}")


            time.sleep(dt)

            
    except KeyboardInterrupt:
        print("\nStopping lidar visualization.")
    
    wait_if_gui('Done. Press to exit.')
    disconnect()


if __name__ == '__main__':
    main(screenshot=False)