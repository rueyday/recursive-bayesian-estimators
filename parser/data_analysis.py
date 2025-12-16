import numpy as np
import time

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def compute_rmse(errors):
    return np.sqrt(np.mean(np.square(errors)))


def compute_position_error(true_xy, est_xy):
    return np.linalg.norm(true_xy - est_xy, axis=1)


def compute_orientation_error(true_theta, est_theta):
    dtheta = wrap_angle(est_theta - true_theta)
    return np.abs(dtheta)


def compute_95th_percentile(errors):
    return np.percentile(errors, 95)


def count_divergences(position_errors, threshold=2.0, duration=10):
    """
    Counts divergence events where position error exceeds threshold
    for more than `duration` consecutive steps.
    """
    count = 0
    streak = 0

    for e in position_errors:
        if e > threshold:
            streak += 1
            if streak == duration:
                count += 1
        else:
            streak = 0

    return count

def evaluate_filter_performance(
    true_poses,
    pf_estimates,
    ekf_estimates,
    pf_times,
    ekf_times
):
    """
    Inputs:
        true_poses:      (T, 3) array of [x, y, theta]
        pf_estimates:    (T, 3)
        ekf_estimates:   (T, 3)
        pf_times:        list of computation times per step
        ekf_times:       list of computation times per step
    """

    T = len(true_poses)

    # --------------------------------------------------------
    # POSITION ERRORS
    # --------------------------------------------------------
    pf_pos_err = compute_position_error(
        true_poses[:, :2], pf_estimates[:, :2]
    )
    ekf_pos_err = compute_position_error(
        true_poses[:, :2], ekf_estimates[:, :2]
    )

    # --------------------------------------------------------
    # ORIENTATION ERRORS
    # --------------------------------------------------------
    pf_ori_err = compute_orientation_error(
        true_poses[:, 2], pf_estimates[:, 2]
    )
    ekf_ori_err = compute_orientation_error(
        true_poses[:, 2], ekf_estimates[:, 2]
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------
    results = {
        "PF": {
            "pos_rmse": compute_rmse(pf_pos_err),
            "ori_rmse": compute_rmse(pf_ori_err),
            "pos_95": compute_95th_percentile(pf_pos_err),
            "divergences": count_divergences(pf_pos_err),
            "time_per_step": np.mean(pf_times) * 1000.0
        },
        "EKF": {
            "pos_rmse": compute_rmse(ekf_pos_err),
            "ori_rmse": compute_rmse(ekf_ori_err),
            "pos_95": compute_95th_percentile(ekf_pos_err),
            "divergences": count_divergences(ekf_pos_err),
            "time_per_step": np.mean(ekf_times) * 1000.0
        }
    }

    return results


# ============================================================
# EXAMPLE USAGE INSIDE YOUR SIM LOOP
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # THESE ARRAYS SHOULD BE FILLED DURING SIMULATION
    # --------------------------------------------------------
    true_poses = []
    pf_estimates = []
    ekf_estimates = []
    pf_times = []
    ekf_times = []

    # Example dummy loop (replace with your sim loop)
    for _ in range(270):

        # --- replace with real values ---
        true_pose = np.array([0.0, 0.0, 0.0])
        pf_est = np.array([0.05, -0.02, 0.01])
        ekf_est = np.array([0.03, -0.01, 0.02])

        pf_time = 0.004  # seconds
        ekf_time = 0.001

        true_poses.append(true_pose)
        pf_estimates.append(pf_est)
        ekf_estimates.append(ekf_est)
        pf_times.append(pf_time)
        ekf_times.append(ekf_time)

    true_poses = np.array(true_poses)
    pf_estimates = np.array(pf_estimates)
    ekf_estimates = np.array(ekf_estimates)

    results = evaluate_filter_performance(
        true_poses,
        pf_estimates,
        ekf_estimates,
        pf_times,
        ekf_times
    )

    # --------------------------------------------------------
    # PRINT RESULTS (TABLE-READY)
    # --------------------------------------------------------
    print("\nBaseline Performance Results\n")

    for filt in ["PF", "EKF"]:
        r = results[filt]
        print(f"{filt}:")
        print(f"  Position RMSE        : {r['pos_rmse']:.3f} m")
        print(f"  Orientation RMSE     : {r['ori_rmse']:.3f} rad")
        print(f"  95th % Position Err  : {r['pos_95']:.3f} m")
        print(f"  Divergences          : {r['divergences']}")
        print(f"  Time / Step          : {r['time_per_step']:.2f} ms\n")
