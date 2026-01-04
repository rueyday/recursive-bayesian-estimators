import numpy as np
import json
import pybullet as p
from pybullet_tools.parse_json import parse_robot, parse_body
# from pybullet_tools.utils import set_joint_positions, \
#     wait_if_gui, wait_for_duration, get_collision_fn
# from pybullet_tools.pr2_utils import get_disabled_collisions

def load_env(env_file):
    # load robot and obstacles defined in a json file
    with open(env_file, 'r') as f:
        env_json = json.loads(f.read())
    robots = {robot['name']: parse_robot(robot) for robot in env_json['robots']}
    bodies = {body['name']: parse_body(body) for body in env_json['bodies']}
    return robots, bodies

# ---------- Occupancy grid builder ----------
def build_occupancy_grid(xmin, xmax, ymin, ymax, resolution,
                         z_top=3.0, z_bottom=-1.0,
                         height_threshold=0.1):
    """
    Build a 2D occupancy grid by casting vertical rays from z_top to z_bottom
    at every grid cell center. If a hit exists within the vertical segment,
    mark cell occupied.
    Returns:
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

# def get_collision_fn_PR2(robot, joints, obstacles):
#     # check robot collision with environment
#     disabled_collisions = get_disabled_collisions(robot)
#     return get_collision_fn(robot, joints, obstacles=obstacles, attachments=[], \
#         self_collisions=True, disabled_collisions=disabled_collisions)

# def execute_trajectory(robot, joints, path, sleep=None):
#     # Move the robot according to a given path
#     if path is None:
#         print('Path is empty')
#         return
#     print('Executing trajectory')
#     for bq in path:
#         set_joint_positions(robot, joints, bq)
#         if sleep is None:
#             wait_if_gui('Continue?')
#         else:
#             wait_for_duration(sleep)
#     print('Finished')

# def draw_sphere_marker(position, radius, color):
#    vs_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
#    marker_id = p.createMultiBody(basePosition=position, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)
#    return marker_id


# def draw_line(start, end, width, color):
#     line_id = p.addUserDebugLine(start, end, color, width)
#     return line_id