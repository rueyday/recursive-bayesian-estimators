import numpy as np
import json
import pybullet as p
from pybullet_tools.parse_json import parse_robot, parse_body
from pybullet_tools.utils import disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
# from pybullet_tools.utils import wait_for_duration, get_collision_fn
# from pybullet_tools.pr2_utils import get_disabled_collisions

def load_env(env_file):
    # load robot and obstacles defined in a json file
    with open(env_file, 'r') as f:
        env_json = json.loads(f.read())
    robots = {robot['name']: parse_robot(robot) for robot in env_json['robots']}
    bodies = {body['name']: parse_body(body) for body in env_json['bodies']}
    return robots, bodies

# Occupancy grid builder
def build_occupancy_grid(xmin, xmax, ymin, ymax, resolution,
                         z_top=3.0, z_bottom=-1.0,
                         height_threshold=0.1):
    """
        occ: 2D numpy array (ny, nx). 1=occupied, 0=free
        xs, ys: 1D arrays of cell centers (x, y)
    """
    xs = np.arange(xmin + resolution/2.0, xmax, resolution)
    ys = np.arange(ymin + resolution/2.0, ymax, resolution)
    nx = xs.size
    ny = ys.size

    origins = []
    ends = []
    for y in ys:
        for x in xs:
            origins.append((x, y, z_top))
            ends.append((x, y, z_bottom))

    # batched ray tests
    results = p.rayTestBatch(origins, ends)
    occ = np.zeros((ny, nx), dtype=np.uint8)
    idx = 0
    for j in range(ny):
        for i in range(nx):
            res = results[idx]
            hit_obj = res[0]
            if hit_obj != -1:
                # fraction < 1.0 indicates hit along the ray segment
                hit_fraction = res[2]
                # compute z of hit: z_top + fraction*(z_bottom - z_top)
                z_hit = z_top + hit_fraction * (z_bottom - z_top)
                # if we care about small obstacles threshold, check difference
                if (z_hit - z_bottom) > height_threshold or (z_top - z_hit) > height_threshold:
                    occ[j, i] = 1
            idx += 1
    return occ, xs, ys

def create_lidar_scan(robot, link_name, num_rays=360, max_range=10.0, min_range=0.1):
    """
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

def visualize_lidar(ranges, angles, robot, link_name, color=(1, 0, 0), max_point_dist=10.0, life_time=0.5, step=16):
    link_id = link_from_name(robot, link_name)
    position, orientation = get_link_pose(robot, link_id)
    position = np.asarray(position)

    rot_matrix = np.asarray(
        p.getMatrixFromQuaternion(orientation)
    ).reshape(3, 3)

    # Slice the arrays to only process every N-th line
    ranges = np.asarray(ranges)[::step]
    angles = np.asarray(angles)[::step]

    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    local_dirs = np.column_stack((cos_a, sin_a, np.zeros_like(cos_a)))
    world_dirs = local_dirs @ rot_matrix.T
    end_points = position + world_dirs * ranges[:, None]
    
    for end in end_points:
        p.addUserDebugLine(position, end, color, lineWidth=0.1, lifeTime=life_time)