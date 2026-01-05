import numpy as np
import pybullet as p
import math

class ExtendedKalmanFilter:
    def __init__(self, initial_pose=(0, 0, 0), motion_noise=(0.02, 0.02, 0.01),
                 lidar_max_range=10.0, lidar_min_range=0.1, z_lidar=0.5,
                 sigma_range=0.2, scan_subsample=4):
        self.state = np.array(initial_pose, dtype=float)
        
        # Slightly increased initial uncertainty
        self.P = np.eye(3) * 0.35  # Was 0.25, now 0.35
        
        self.motion_noise = motion_noise
        
        # Increased process noise to make EKF less confident
        self.Q = np.diag([n**2 * 1.5 for n in motion_noise])  # 1.5x multiplier
        
        # Slightly underestimate measurement accuracy
        self.R_val = sigma_range**2 * 1.2  # Was 1.0, now 1.2x
        
        self.lidar_max_range = lidar_max_range
        self.lidar_min_range = lidar_min_range
        self.z_lidar = z_lidar
        self.scan_subsample = scan_subsample
        self.angles = np.linspace(0, 2*np.pi, 360, endpoint=False)

    def motion_update(self, odom):
        dx_l, dy_l, dtheta = odom
        x, y, theta = self.state
        
        # State Update
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        self.state[0] += cos_t * dx_l - sin_t * dy_l
        self.state[1] += sin_t * dx_l + cos_t * dy_l
        self.state[2] = (theta + dtheta + np.pi) % (2*np.pi) - np.pi

        # Jacobian F of motion model
        F = np.array([
            [1, 0, -sin_t * dx_l - cos_t * dy_l],
            [0, 1,  cos_t * dx_l - sin_t * dy_l],
            [0, 0, 1]
        ])
        self.P = F @ self.P @ F.T + self.Q

    def measurement_update(self, real_ranges, real_angles=None):
        angles_to_use = real_angles if real_angles is not None else self.angles
        
        # Subsample to match the PF's information density
        angles_sub = angles_to_use[::self.scan_subsample]
        ranges_sub = real_ranges[::self.scan_subsample]
        
        # 1. Simulation from current EKF estimate
        expected = self._simulate(self.state, angles_sub)
        
        # 2. Innovation (Residual)
        innovation = ranges_sub - expected
        
        # 3. Jacobian H (Measurement Model)
        H = self._jacobian_h(self.state, angles_sub, expected)
        
        # 4. Kalman Gain Calculation
        R = np.eye(len(ranges_sub)) * self.R_val
        S = H @ self.P @ H.T + R
        
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
            
            # 5. State Correction
            self.state += K @ innovation
            self.state[2] = (self.state[2] + np.pi) % (2*np.pi) - np.pi
            
            # 6. Covariance Update (Joseph form for stability)
            I_KH = np.eye(3) - K @ H
            self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
            
        except np.linalg.LinAlgError:
            pass

    def _simulate(self, pose, angles):
        origins = [(pose[0], pose[1], self.z_lidar)] * len(angles)
        ends = [(pose[0] + math.cos(pose[2]+a)*self.lidar_max_range, 
                 pose[1] + math.sin(pose[2]+a)*self.lidar_max_range, self.z_lidar) for a in angles]
        res = p.rayTestBatch(origins, ends)
        return np.array([r[2]*self.lidar_max_range if r[0] != -1 else self.lidar_max_range for r in res])

    def _jacobian_h(self, pose, angles, expected):
        H = np.zeros((len(angles), 3))
        eps = 1e-3
        for i in range(3):
            p_plus = pose.copy(); p_plus[i] += eps
            r_plus = self._simulate(p_plus, angles)
            H[:, i] = (r_plus - expected) / eps
        return H

    def estimate(self): 
        return tuple(self.state)