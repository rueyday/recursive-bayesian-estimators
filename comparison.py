import os
import sys
import time
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from math import cos, sin, atan2, hypot

# Import custom tools
try:
    import pybullet_tools.utils as pyb_utils
    from utils import load_env, build_occupancy_grid, create_lidar_scan, visualize_lidar
    from filters.pf_model import ParticleFilter
    from filters.ekf_model import ExtendedKalmanFilter
    from filters.ukf_model import UnscentedKalmanFilter
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import custom modules: {e}")
    sys.exit(1)

# Waypoints
waypoints = [
    (-2.5,  2.2), ( 0.0,  2.2), ( 0.0, -2.2), ( 2.5, -2.2), ( 2.5, -6.0),
    (-2.5, -6.0), (-2.5, -2.2), ( 0.0, -2.2), ( 0.0,  2.2), ( 2.5,  2.2),
    ( 2.5,  6.0), (-2.5,  6.0)
]

NUM_STEPS = 800
linear_speed = 0.05
angular_speed = 0.1
wp_threshold = 0.2
WINDOW = 5
PLOT_LOOKBACK = 80 

# NEW: Teleportation detection threshold
TELEPORT_THRESHOLD = 1.0  # meters

def smooth(data, window):
    if len(data) < window: return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

def wrap_angle(a):
    return np.fmod(a + np.pi, 2 * np.pi) - np.pi

def set_base_link_pose(robot_id, x_link, y_link, theta):
    _, _, z = p.getBasePositionAndOrientation(robot_id)[0]
    p.resetBasePositionAndOrientation(robot_id, [0, 0, z], p.getQuaternionFromEuler([0, 0, theta]))
    link_idx = pyb_utils.link_from_name(robot_id, "base_link")
    link_pose = pyb_utils.get_link_pose(robot_id, link_idx)
    dx, dy = link_pose[0][0], link_pose[0][1]
    p.resetBasePositionAndOrientation(robot_id, [x_link - dx, y_link - dy, z], p.getQuaternionFromEuler([0, 0, theta]))

def main():
    # --- 1. SETUP PYBULLET ---
    client = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    if not hasattr(pyb_utils, 'CLIENTS'): pyb_utils.CLIENTS = {}
    pyb_utils.CLIENTS[client] = True 

    # --- 2. SETUP REAL-TIME PLOTS ---
    plt.ion()
    fig, (ax_err, ax_time) = plt.subplots(2, 1, figsize=(10, 8))
    fig.canvas.manager.set_window_title('Filter Performance Comparison')
    
    line_err_pf,  = ax_err.plot([], [], 'b-', label='PF Error', alpha=0.7)
    line_err_ekf, = ax_err.plot([], [], 'y-', label='EKF Error')
    line_err_ukf, = ax_err.plot([], [], 'c-', label='UKF Error')
    ax_err.set_ylabel("Error (m)"); ax_err.legend(); ax_err.grid(True)

    line_t_pf,  = ax_time.plot([], [], 'b-', label='PF Time', alpha=0.7)
    line_t_ekf, = ax_time.plot([], [], 'y-', label='EKF Time')
    line_t_ukf, = ax_time.plot([], [], 'c-', label='UKF Time')
    ax_time.set_yscale('log')
    ax_time.set_ylabel("Latency (ms) [Log Scale]")
    ax_time.set_xlabel("Steps"); ax_time.legend(); ax_time.grid(True, which="both", ls="-", alpha=0.2)

    # --- 3. LOAD ENVIRONMENT ---
    robots_dict, _ = load_env("environment/figure8_env.json")
    robot_id = robots_dict.get("pr2", list(robots_dict.values())[0])
    link_name = "base_link"
    occ, xs, ys = build_occupancy_grid(xmin=-8.0, xmax=8.0, ymin=-8.0, ymax=8.0, resolution=0.2)
    
    l_idx = pyb_utils.link_from_name(robot_id, link_name)
    l_pose = pyb_utils.get_link_pose(robot_id, l_idx)
    rx, ry = l_pose[0][0], l_pose[0][1]
    rt = p.getEulerFromQuaternion(l_pose[1])[2]

    # Enhanced Configuration
    cfg = dict(lidar_max_range=6.0, lidar_min_range=0.1, z_lidar=0.5, 
               scan_subsample=12, motion_noise=(0.5, 0.5, 0.8), sigma_range=0.3)
    
    # INCREASED FROM 1000 TO 3000 PARTICLES
    pf = ParticleFilter(occ=occ, xs=xs, ys=ys, n_particles=3000, 
                    initial_pose=(rx, ry, rt), **cfg)
    ekf = ExtendedKalmanFilter(initial_pose=(rx, ry, rt), **cfg)
    # Adding UKF scaling parameters to keep sigma points tight
    ukf = UnscentedKalmanFilter(initial_pose=(rx, ry, rt), alpha=0.001, kappa=0, **cfg)

    # Data Storage
    s_idx, e_p, e_e, e_u, t_p, t_e, t_u = [], [], [], [], [], [], []
    current_wp = 0

    # --- 4. MAIN LOOP ---
    for step in range(NUM_STEPS):
        keys = p.getKeyboardEvents()
        
        # NEW: Track if teleportation occurs
        teleported = False
        
        # Challenge the filters with periodic jumps OR manual spacebar
        if (step > 0 and step % 160 == 0) or (32 in keys and keys[32] & p.KEY_WAS_TRIGGERED):
            current_wp = (current_wp + 3) % len(waypoints)
            rx, ry = waypoints[current_wp]
            teleported = True
            print(f"\n{'='*60}")
            print(f"TELEPORT EVENT at step {step} to waypoint {current_wp}: ({rx:.2f}, {ry:.2f})")
            print(f"{'='*60}\n")

        rx_old, ry_old, rt_old = rx, ry, rt
        tx, ty = waypoints[current_wp]
        dist = hypot(tx - rx, ty - ry)
        target_t = atan2(ty - ry, tx - rx)
        dt_err = wrap_angle(target_t - rt)
        
        if abs(dt_err) < 0.5:
            move = min(dist, linear_speed)
            rx += move * cos(rt)
            ry += move * sin(rt)
        
        rt = wrap_angle(rt + np.clip(dt_err, -angular_speed, angular_speed))
        set_base_link_pose(robot_id, rx, ry, rt + np.pi/2)
        if dist < wp_threshold:
            current_wp = (current_wp + 1) % len(waypoints)

        dx_g, dy_g = rx - rx_old, ry - ry_old
        odom = (cos(rt_old)*dx_g + sin(rt_old)*dy_g, -sin(rt_old)*dx_g + cos(rt_old)*dy_g, wrap_angle(rt - rt_old))
        
        # NEW: Detect large motion (teleportation)
        odom_magnitude = hypot(dx_g, dy_g)
        if odom_magnitude > TELEPORT_THRESHOLD and not teleported:
            # This catches any unexpected large motion
            teleported = True
            print(f"\n*** UNEXPECTED LARGE MOTION DETECTED at step {step}: {odom_magnitude:.2f}m ***\n")

        ranges, angles, _ = create_lidar_scan(robot_id, link_name)
        visualize_lidar(ranges, angles, robot_id, link_name)

        # NEW: Handle teleportation for particle filter
        if teleported:
            # Strategy 1: Reinitialize around new position with large uncertainty
            print(f"  >> Reinitializing Particle Filter around ({rx:.2f}, {ry:.2f}, {rt:.2f})")
            pf.handle_teleportation(rx, ry, rt, uncertainty_radius=1.0)
            
            # Strategy 2: Just do measurement update (no motion update)
            # The measurement will pull particles toward the correct location
            t0 = time.time()
            pf.measurement_update(ranges, angles)
            pf.resample()
            mp = max((time.time() - t0) * 1000, 0.001)
            est_pf = pf.estimate()
            ep = hypot(rx - est_pf[0], ry - est_pf[1])
        else:
            # Normal update for PF
            def run_f(f, is_p=False):
                t0 = time.time()
                f.motion_update(odom)
                f.measurement_update(ranges, angles)
                if is_p: f.resample()
                ms = max((time.time() - t0) * 1000, 0.001)
                est = f.estimate()
                return ms, hypot(rx - est[0], ry - est[1])
            
            mp, ep = run_f(pf, True)
        
        # EKF and UKF handle teleportation naturally through their measurement update
        def run_f_gaussian(f):
            t0 = time.time()
            f.motion_update(odom)
            f.measurement_update(ranges, angles)
            ms = max((time.time() - t0) * 1000, 0.001)
            est = f.estimate()
            return ms, hypot(rx - est[0], ry - est[1])
        
        me, ee = run_f_gaussian(ekf)
        mu, eu = run_f_gaussian(ukf)

        s_idx.append(step); e_p.append(ep); e_e.append(ee); e_u.append(eu)
        t_p.append(mp); t_e.append(me); t_u.append(mu)
        
        print(f"Step: {step} | Truth: ({rx:.2f}, {ry:.2f})")
        print(f"  PF Est: ({pf.estimate()[0]:.2f}, {pf.estimate()[1]:.2f}) Error: {ep:.2f}m")
        print(f"  UKF Est: ({ukf.estimate()[0]:.2f}, {ukf.estimate()[1]:.2f}) Error: {eu:.2f}m")
        print(f"  EKF Est: ({ekf.estimate()[0]:.2f}, {ekf.estimate()[1]:.2f}) Error: {ee:.2f}m")
        
        if step % 5 == 0 and step >= WINDOW:
            sm_steps = np.array(s_idx[WINDOW-1:])
            sm_ep, sm_ee, sm_eu = smooth(e_p, WINDOW), smooth(e_e, WINDOW), smooth(e_u, WINDOW)
            sm_tp, sm_te, sm_tu = smooth(t_p, WINDOW), smooth(t_e, WINDOW), smooth(t_u, WINDOW)

            line_err_pf.set_data(sm_steps, sm_ep); line_err_ekf.set_data(sm_steps, sm_ee); line_err_ukf.set_data(sm_steps, sm_eu)
            line_t_pf.set_data(sm_steps, sm_tp); line_t_ekf.set_data(sm_steps, sm_te); line_t_ukf.set_data(sm_steps, sm_tu)

            # --- IMPROVED Y-AXIS AUTO-SCALING ---
            # Define window slice for Y-limit calculation
            view_slice = slice(-max(1, PLOT_LOOKBACK // 5), None)
            
            max_err = max(np.max(sm_ep[view_slice]), np.max(sm_ee[view_slice]), np.max(sm_eu[view_slice]), 0.2)
            ax_err.set_ylim(0, max_err / 0.8)

            t_all = np.concatenate([sm_tp[view_slice], sm_te[view_slice], sm_tu[view_slice]])
            ax_time.set_ylim(np.min(t_all) * 0.5, np.max(t_all) / 0.7)

            for ax in [ax_err, ax_time]:
                ax.set_xlim(max(0, step - PLOT_LOOKBACK), max(PLOT_LOOKBACK, step))
            
            fig.canvas.flush_events()
            plt.pause(0.001) 

    # --- 5. FINAL PERFORMANCE SUMMARY ---
    print("\n" + "="*65)
    print(f"{'Filter Type':<15} | {'RMSE (m)':<12} | {'Avg Comp Time (ms)':<18}")
    print("-" * 65)
    for name, errors, times in [("Particle", e_p, t_p), ("EKF", e_e, t_e), ("UKF", e_u, t_u)]:
        rmse = np.sqrt(np.mean(np.array(errors)**2))
        avg_t = np.mean(times)
        print(f"{name:<15} | {rmse:<12.4f} | {avg_t:<18.2f}")
    print("="*65)
    plt.ioff(); plt.show()

if __name__ == "__main__":
    main()