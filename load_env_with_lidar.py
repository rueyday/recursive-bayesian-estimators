import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import pybullet as p
import time

def create_lidar_scan(robot, link_name, num_rays=360, max_range=10.0, min_range=0.1):
    """
    Simulate a 2D lidar scan using raycast
    
    Args:
        robot: PyBullet robot body ID
        link_name: Name of link to attach lidar to (e.g., 'head_tilt_link')
        num_rays: Number of rays in the scan
        max_range: Maximum detection range in meters
        min_range: Minimum detection range in meters
    
    Returns:
        ranges: Array of range measurements
        angles: Array of angles for each ray
        hit_positions: 3D positions of ray hits
    """
    link_id = link_from_name(robot, link_name)
    link_pose = get_link_pose(robot, link_id)
    position, orientation = link_pose
    
    rot_matrix = p.getMatrixFromQuaternion(orientation)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)
    
    ranges = []
    angles = []
    hit_positions = []
    
    # Create rays in a circle
    for i in range(num_rays):
        angle = 2 * np.pi * i / num_rays
        angles.append(angle)
        
        local_direction = np.array([np.cos(angle), np.sin(angle), 0])
        world_direction = rot_matrix @ local_direction
        
        ray_from = position
        ray_to = tuple(np.array(position) + world_direction * max_range)
        
        # Perform raycast
        result = p.rayTest(ray_from, ray_to)
        
        if result[0][0] != -1:  # Hit something
            hit_fraction = result[0][2]
            hit_pos = result[0][3]
            distance = hit_fraction * max_range
            
            if distance >= min_range:
                ranges.append(distance)
                hit_positions.append(hit_pos)
            else:
                ranges.append(max_range)  # Out of range
                hit_positions.append(None)
        else:
            ranges.append(max_range)
            hit_positions.append(None)
    
    return np.array(ranges), np.array(angles), hit_positions


def visualize_lidar(ranges, angles, robot, link_name, color=[1, 0, 0]):
    """
    Visualize lidar scan as debug lines and points
    
    Args:
        ranges: Array of range measurements
        angles: Array of angles
        robot: PyBullet robot body ID
        link_name: Name of link lidar is attached to
        color: RGB color for visualization
    """
    link_id = link_from_name(robot, link_name)
    link_pose = get_link_pose(robot, link_id)
    position, orientation = link_pose
    
    rot_matrix = p.getMatrixFromQuaternion(orientation)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)
    
    # Draw rays
    for i, (distance, angle) in enumerate(zip(ranges, angles)):
        # Direction in local frame
        local_direction = np.array([np.cos(angle), np.sin(angle), 0])
        
        # Transform to world frame
        world_direction = rot_matrix @ local_direction
        
        # End point
        end_point = np.array(position) + world_direction * distance
        
        # Draw line
        p.addUserDebugLine(position, end_point, color, lineWidth=1, lifeTime=0.1)
        
        # Draw hit point if within range
        if distance < 10.0:  # Assuming max_range is 10
            p.addUserDebugPoints([end_point], [color], pointSize=3, lifeTime=0.1)


def visualize_lidar_pointcloud(ranges, angles, robot, link_name):
    """
    Visualize lidar as a point cloud
    """
    link_id = link_from_name(robot, link_name)
    link_pose = get_link_pose(robot, link_id)
    position, orientation = link_pose
    
    rot_matrix = p.getMatrixFromQuaternion(orientation)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)
    
    points = []
    colors = []
    
    for distance, angle in zip(ranges, angles):
        if distance < 10.0:  # Filter out max range readings
            local_direction = np.array([np.cos(angle), np.sin(angle), 0])
            world_direction = rot_matrix @ local_direction
            point = np.array(position) + world_direction * distance
            points.append(point)
            
            # Color based on distance (close = red, far = blue)
            color_val = distance / 10.0
            colors.append([1 - color_val, 0, color_val])
    
    if points:
        p.addUserDebugPoints(points, colors, pointSize=5, lifeTime=0.1)


def main(screenshot=False):
    connect(use_gui=True)
    
    robots, obstacles = load_env('8path_env.json')
    
    print("Available robots:", list(robots.keys()))
    
    if 'pr2' in robots:
        pr2 = robots['pr2']
    elif 'robot' in robots:
        pr2 = robots['robot']
    else:
        pr2 = list(robots.values())[0]
    
    print(f"Using robot ID: {pr2}")
    
    wait_if_gui('Environment loaded. Press to start lidar scan.')
    
    print("Starting lidar visualization. Press Ctrl+C to stop.")
    try:
        while True:
            # Perform lidar scan from head
            ranges, angles, hit_positions = create_lidar_scan(
                pr2, 
                'head_tilt_link',
                num_rays=360,
                max_range=10.0
            )
            
            # Visualize the scan
            visualize_lidar_pointcloud(ranges, angles, pr2, 'head_tilt_link')
            # Or use line visualization:
            # visualize_lidar(ranges, angles, pr2, 'head_tilt_link', color=[1, 0, 0])
            
            valid_ranges = ranges[ranges < 10.0]
            if len(valid_ranges) > 0:
                print(f"Min distance: {valid_ranges.min():.2f}m, "
                      f"Max distance: {valid_ranges.max():.2f}m, "
                      f"Avg distance: {valid_ranges.mean():.2f}m")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping lidar visualization.")
    
    wait_if_gui('Done. Press to exit.')
    disconnect()


if __name__ == '__main__':
    main(screenshot=False)