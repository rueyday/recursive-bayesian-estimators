# motion_sensor.py
import numpy as np
import math
import csv
import time

# ---------- Utility functions ----------
def wrap_to_pi(angle):
    """Wrap angle to [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def pose_to_array(pose):
    """pose = (x, y, theta) or quaternion pose; ensure numpy array [x,y,theta]."""
    return np.array(pose, dtype=float).reshape(3)

# ---------- Motion model (unicycle) ----------
def propagate_unicycle(state, control, dt):
    """
    Deterministic propagation of unicycle model.
    state: [x, y, theta]
    control: [v, omega] (linear velocity, angular velocity)
    dt: time step
    returns next_state (wrapped theta)
    """
    x, y, th = state
    v, w = control

    # simple forward Euler discrete model
    x_next = x + v * math.cos(th) * dt
    y_next = y + v * math.sin(th) * dt
    th_next = wrap_to_pi(th + w * dt)

    return np.array([x_next, y_next, th_next], dtype=float)

# ---------- Noisy controls (odometry) ----------
def sample_noisy_control(u_cmd, control_std):
    """
    Additive Gaussian noise in control space.
    u_cmd: [v_cmd, omega_cmd]
    control_std: [sigma_v, sigma_omega]
    returns noisy_control
    """
    v_cmd, w_cmd = u_cmd
    sigma_v, sigma_w = control_std
    v_noisy = v_cmd + np.random.normal(0.0, sigma_v)
    w_noisy = w_cmd + np.random.normal(0.0, sigma_w)
    return np.array([v_noisy, w_noisy], dtype=float)

# ---------- Process noise covariance in state space ----------
def control_noise_to_state_cov(state, control, dt, control_cov):
    """
    Map control (v,w) noise covariance into state covariance Q using linearization:
    x_{t+1} = f(x_t, u_t) + L * noise_u
    L = df/du
    control_cov: 2x2 covariance in (v,w)
    returns Q (3x3)
    """
    x, y, th = state
    v, w = control
    # L (df/du) shape 3x2
    L = np.zeros((3, 2))
    L[0, 0] = math.cos(th) * dt        # dx/dv
    L[1, 0] = math.sin(th) * dt        # dy/dv
    L[2, 1] = dt                       # dtheta/domega

    Q = L @ control_cov @ L.T
    return Q

# ---------- Jacobians for EKF ----------
def jacobian_F(state, control, dt):
    """
    Jacobian of f wrt state x: F = df/dx (3x3)
    Based on unicycle model used in propagate_unicycle.
    """
    x, y, th = state
    v, w = control
    F = np.eye(3)
    F[0, 2] = -v * math.sin(th) * dt   # dx/dtheta
    F[1, 2] =  v * math.cos(th) * dt   # dy/dtheta
    return F

def jacobian_L(state, control, dt):
    """
    Jacobian of f wrt control (v,w) used to map control noise into state (same as L above).
    Returns 3x2 matrix.
    """
    x, y, th = state
    L = np.zeros((3, 2))
    L[0, 0] = math.cos(th) * dt
    L[1, 0] = math.sin(th) * dt
    L[2, 1] = dt
    return L

# ---------- Measurement model (direct noisy pose sensor) ----------
def measure_pose_true(pybullet_robot, get_link_pose_fn, link_name=None):
    """
    Read ground-truth pose from PyBullet.
    get_link_pose_fn should be a function that returns (pos, quat) given (body, link_id)
    If link_name is None, assume base link pose is available via get_base_pose()
    Return: [x, y, theta]  (theta is yaw)
    """
    # This function left intentionally abstract: I'll show a concrete example later.
    raise NotImplementedError("Use measure_pose_noisy or pass a concrete get_link_pose function.")

def sample_noisy_pose(true_pose, meas_std):
    """
    Additive Gaussian noise on pose measurements.
    true_pose: [x, y, theta]
    meas_std: [sigma_x, sigma_y, sigma_theta]
    returns noisy measurement z = [x_m, y_m, theta_m]
    """
    sx, sy, sth = meas_std
    x, y, th = true_pose
    xm = x + np.random.normal(0.0, sx)
    ym = y + np.random.normal(0.0, sy)
    thm = wrap_to_pi(th + np.random.normal(0.0, sth))
    return np.array([xm, ym, thm], dtype=float)

def measurement_covariance(meas_std):
    """3x3 diagonal measurement covariance R."""
    sx, sy, sth = meas_std
    return np.diag([sx**2, sy**2, sth**2])

# ---------- Likelihood for particle filter (Gaussian) ----------
def measurement_likelihood(z_meas, predicted_pose, R):
    """
    Compute probability (unnormalized) of measurement z_meas given predicted_pose
    under Gaussian R. Use 3D gaussian on [x,y,theta] (theta error wrapped).
    Returns likelihood value (scalar).
    """
    # error vector (wrap theta)
    dx = z_meas[0] - predicted_pose[0]
    dy = z_meas[1] - predicted_pose[1]
    dth = wrap_to_pi(z_meas[2] - predicted_pose[2])
    err = np.array([dx, dy, dth])
    # Mahalanobis
    try:
        invR = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        invR = np.linalg.pinv(R)
    exponent = -0.5 * (err.T @ invR @ err)
    denom = np.sqrt(((2 * np.pi) ** 3) * np.linalg.det(R) + 1e-12)
    return math.exp(exponent) / (denom + 1e-12)

# ---------- Example trajectory generator ----------
def generate_waypoint_path(waypoints, speed_lin=0.5, speed_ang=0.5, dt=0.1):
    """
    Create a simple sequence of controls (v, w) that attempts to drive the robot
    through the provided waypoints (list of [x,y]) using a simple proportional controller.
    This produces a list of (v,omega,dt) commands.
    """
    controls = []
    # simple controller gains
    k_rho = 1.0
    k_alpha = 2.0
    for wp in waypoints:
        # We'll not simulate here; this returns a generator-like list of controls for higher-level loop
        controls.append(('goto', wp, speed_lin, speed_ang))
    return controls

# ---------- High-level simulation runner (records logs) ----------
def run_simulation_loop(pybullet_robot_id, get_base_pose_fn, cmd_generator,
                        dt=0.1, total_time=20.0,
                        control_noise_std=(0.02, 0.01),
                        meas_noise_std=(0.05, 0.05, 0.02),
                        control_apply_fn=None,  # function(body, v, w, dt) -> applies command in PyBullet
                        record_file=None):
    """
    cmd_generator: a generator or list of (v_cmd, w_cmd) or higher-level commands that you interpret
    get_base_pose_fn: function that returns true [x,y,theta] for robot in PyBullet
    control_apply_fn: function that will actually move the PyBullet robot given (v,w,dt)
    Returns logs: dict with arrays for gt, odom_controls, meas
    """
    steps = int(total_time / dt)
    t = 0.0

    # Covariances
    control_cov = np.diag([control_noise_std[0]**2, control_noise_std[1]**2])
    R = measurement_covariance(meas_noise_std)

    # initialize state from ground truth
    gt_pose = get_base_pose_fn(pybullet_robot_id)
    est_state = gt_pose.copy()

    logs = {
        't': [],
        'gt': [],            # ground truth poses
        'u_cmd': [],        # commanded controls (v_cmd, w_cmd)
        'u_odom': [],       # noisy odometry controls used for propagation
        'z': [],            # noisy pose measurements
    }

    cmd_iter = iter(cmd_generator) if not hasattr(cmd_generator, '__call__') else cmd_generator

    for step in range(steps):
        # --- get a command ---
        try:
            # if a generator returns tuples (v,w)
            cmd = next(cmd_iter) if hasattr(cmd_iter, '__next__') else cmd_iter()
        except StopIteration:
            # no more commands — stop issuing motion
            v_cmd, w_cmd = 0.0, 0.0
        else:
            # Interpret possible higher-level commands
            if isinstance(cmd, tuple) and len(cmd) == 2:
                v_cmd, w_cmd = cmd
            elif isinstance(cmd, tuple) and cmd[0] == 'goto':
                # simple goto: returns a small velocity command computed from current gt
                _, wp, max_v, max_w = cmd
                # compute vector in world frame
                gt_pose = get_base_pose_fn(pybullet_robot_id)
                dx = wp[0] - gt_pose[0]
                dy = wp[1] - gt_pose[1]
                rho = math.hypot(dx, dy)
                desired_theta = math.atan2(dy, dx)
                alpha = wrap_to_pi(desired_theta - gt_pose[2])
                # proportional law (saturate)
                v_cmd = max_v * (1.0 if rho > 0.1 else 0.0) * min(1.0, rho / 1.0)
                w_cmd = max_w * (alpha / (abs(alpha) + 1e-6))
            else:
                # fallback zero velocity
                v_cmd, w_cmd = 0.0, 0.0

        # Apply noisy control sampling to simulate odometry
        u_odom = sample_noisy_control((v_cmd, w_cmd), control_noise_std)

        # propagate internal estimate (optional)
        est_state = propagate_unicycle(est_state, u_odom, dt)

        # apply to pybullet simulation if function provided
        if control_apply_fn is not None:
            control_apply_fn(pybullet_robot_id, v_cmd, w_cmd, dt)  # use commanded or odom depending on desired simulate design

        # advance simulation externally (the caller should step PyBullet)
        # get ground truth from pybullet
        gt_pose = get_base_pose_fn(pybullet_robot_id)

        # generate noisy pose measurement
        z = sample_noisy_pose(gt_pose, meas_noise_std)

        logs['t'].append(t)
        logs['gt'].append(gt_pose.copy())
        logs['u_cmd'].append(np.array([v_cmd, w_cmd]))
        logs['u_odom'].append(u_odom.copy())
        logs['z'].append(z.copy())

        t += dt

    # convert lists to arrays
    for k in ['t', 'gt', 'u_cmd', 'u_odom', 'z']:
        logs[k] = np.array(logs[k])

    return logs, control_cov, R
