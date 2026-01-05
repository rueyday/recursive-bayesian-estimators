import numpy as np
import pybullet as p
import math

class UnscentedKalmanFilter:
    def __init__(self, initial_pose=(0, 0, 0), motion_noise=(0.02, 0.02, 0.01),
                 lidar_max_range=10.0, lidar_min_range=0.1, z_lidar=0.5,
                 sigma_range=0.2, scan_subsample=4, 
                 alpha=0.1, beta=2.0, kappa=0.0):
        
        self.state = np.array(initial_pose, dtype=float)
        
        # Slightly increased initial uncertainty (between PF and EKF)
        self.P = np.eye(3) * 0.30  # Was 0.25, now 0.30
        
        # Increased process noise slightly more than EKF
        self.Q = np.diag([n**2 * 1.3 for n in motion_noise])  # 1.3x multiplier
        
        # Slightly underestimate measurement accuracy
        self.R_val = sigma_range**2 * 1.15  # Between PF (1.0x) and EKF (1.2x)
        
        self.lidar_max_range = lidar_max_range
        self.lidar_min_range = lidar_min_range
        self.z_lidar = z_lidar
        self.scan_subsample = scan_subsample
        self.angles = np.linspace(0, 2*np.pi, 360, endpoint=False)

        # UKF Parameters - using provided values
        self.n = 3
        self.alpha = alpha
        self.kappa = kappa
        self.beta = beta
        self.lambda_ = self.alpha**2 * (self.n + self.kappa) - self.n
        
        # Precompute weights
        self.w_m = np.full(2 * self.n + 1, 1.0 / (2 * (self.n + self.lambda_)))
        self.w_c = np.copy(self.w_m)
        self.w_m[0] = self.lambda_ / (self.n + self.lambda_)
        self.w_c[0] = self.w_m[0] + (1 - self.alpha**2 + self.beta)

    def _get_sigma_points(self):
        try:
            res = np.linalg.cholesky((self.n + self.lambda_) * self.P)
        except np.linalg.LinAlgError:
            res = np.sqrt(np.maximum(0, (self.n + self.lambda_) * self.P))
            
        sigmas = np.zeros((2 * self.n + 1, self.n))
        sigmas[0] = self.state
        for k in range(self.n):
            sigmas[k + 1] = self.state + res[:, k]
            sigmas[k + 1 + self.n] = self.state - res[:, k]
        return sigmas

    def motion_update(self, odom):
        dx_l, dy_l, dtheta = odom
        sigmas = self._get_sigma_points()
        new_sigmas = np.zeros_like(sigmas)

        # Predict sigma points through non-linear motion model
        for i in range(len(sigmas)):
            x, y, theta = sigmas[i]
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            new_sigmas[i, 0] = x + cos_t * dx_l - sin_t * dy_l
            new_sigmas[i, 1] = y + sin_t * dx_l + cos_t * dy_l
            new_sigmas[i, 2] = (theta + dtheta + np.pi) % (2*np.pi) - np.pi

        # Recombine sigmas into predicted state and covariance
        self.state = np.dot(self.w_m, new_sigmas)
        self.state[2] = math.atan2(np.dot(self.w_m, np.sin(new_sigmas[:,2])), 
                                   np.dot(self.w_m, np.cos(new_sigmas[:,2])))
        
        self.P = self.Q.copy()
        for i in range(len(sigmas)):
            diff = new_sigmas[i] - self.state
            diff[2] = (diff[2] + np.pi) % (2*np.pi) - np.pi
            self.P += self.w_c[i] * np.outer(diff, diff)

    def measurement_update(self, real_ranges, real_angles=None):
        angles_to_use = real_angles if real_angles is not None else self.angles
        ranges_sub = real_ranges[::self.scan_subsample]
        angles_sub = angles_to_use[::self.scan_subsample]
        num_rays = len(ranges_sub)

        sigmas = self._get_sigma_points()
        sigmas_h = np.zeros((len(sigmas), num_rays))

        # Transform sigmas through measurement model
        for i in range(len(sigmas)):
            sigmas_h[i] = self._simulate(sigmas[i], angles_sub)

        # Mean predicted measurement
        z_pred = np.dot(self.w_m, sigmas_h)

        # Measurement covariance S and Cross-covariance Pxz
        S = np.eye(num_rays) * self.R_val
        Pxz = np.zeros((self.n, num_rays))
        
        for i in range(len(sigmas)):
            z_diff = sigmas_h[i] - z_pred
            S += self.w_c[i] * np.outer(z_diff, z_diff)
            
            x_diff = sigmas[i] - self.state
            x_diff[2] = (x_diff[2] + np.pi) % (2*np.pi) - np.pi
            Pxz += self.w_c[i] * np.outer(x_diff, z_diff)

        # Kalman Gain
        try:
            K = Pxz @ np.linalg.inv(S)
            innovation = ranges_sub - z_pred
            self.state += K @ innovation
            self.state[2] = (self.state[2] + np.pi) % (2*np.pi) - np.pi
            self.P -= K @ S @ K.T
        except np.linalg.LinAlgError:
            pass

    def _simulate(self, pose, angles):
        origins = [(pose[0], pose[1], self.z_lidar)] * len(angles)
        ends = [(pose[0] + math.cos(pose[2]+a)*self.lidar_max_range, 
                 pose[1] + math.sin(pose[2]+a)*self.lidar_max_range, self.z_lidar) for a in angles]
        res = p.rayTestBatch(origins, ends)
        return np.array([r[2]*self.lidar_max_range if r[0] != -1 else self.lidar_max_range for r in res])

    def estimate(self):
        return tuple(self.state)