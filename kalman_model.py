import numpy as np
import pybullet as p
import math

class ExtendedKalmanFilter:
    def __init__(self, initial_pose=(0, 0, 0), 
                 motion_noise=(0.02, 0.02, 0.01),
                 lidar_max_range=10.0,
                 lidar_min_range=0.1,
                 z_lidar=0.5,
                 sigma_range=0.2,
                 angles_full=None,
                 scan_subsample=4):
        
        self.state = np.array(initial_pose, dtype=float)
        self.P = np.eye(3) * 0.5
        
        self.motion_noise = motion_noise
        sx, sy, st = motion_noise
        self.Q = np.diag([sx**2, sy**2, st**2])
        
        self.sigma_range = sigma_range
        self.R_single = sigma_range**2
        
        self.lidar_max_range = lidar_max_range
        self.lidar_min_range = lidar_min_range
        self.z_lidar = z_lidar
        self.scan_subsample = max(1, int(scan_subsample))
        
        if angles_full is None:
            self.angles = np.linspace(0, 2*np.pi, 360, endpoint=False)
        else:
            self.angles = np.array(angles_full)
        
        self.angles_sub = self.angles[::self.scan_subsample]
    
    def motion_update(self, odom):
        dx_global, dy_global, dtheta = odom
        x, y, theta = self.state
        
        x_new = x + dx_global
        y_new = y + dy_global
        theta_new = (theta + dtheta + np.pi) % (2*np.pi) - np.pi
        
        F = np.eye(3)
        
        self.state = np.array([x_new, y_new, theta_new])
        self.P = F @ self.P @ F.T + self.Q
    
    def measurement_update(self, real_ranges, real_angles=None, z_lidar=None):
        if real_angles is None:
            real_angles = self.angles
        if z_lidar is None:
            z_lidar = self.z_lidar
        
        # Subsample measurements aggressively for speed and stability
        subsample = self.scan_subsample * 2
        real_ranges_sub = real_ranges[::subsample]
        angles_sub = self.angles[::subsample]
        
        # Filter valid measurements
        valid_mask = (real_ranges_sub >= self.lidar_min_range) & \
                     (real_ranges_sub < self.lidar_max_range)
        
        if np.sum(valid_mask) < 5:
            return
        
        valid_ranges = real_ranges_sub[valid_mask]
        valid_angles = angles_sub[valid_mask]
        n_valid = len(valid_ranges)
        
        if n_valid > 20:
            indices = np.linspace(0, n_valid-1, 20, dtype=int)
            valid_ranges = valid_ranges[indices]
            valid_angles = valid_angles[indices]
            n_valid = 20
        
        expected_ranges = self._simulate_lidar(self.state, valid_angles, z_lidar)
        
        innovation = valid_ranges - expected_ranges
        
        # AGGRESSIVE outlier rejection - only keep very good matches
        median_innovation = np.median(np.abs(innovation))
        outlier_threshold = max(0.5, 3.0 * median_innovation)
        outlier_mask = np.abs(innovation) < outlier_threshold
        
        if np.sum(outlier_mask) < 5:
            return  # Need at least 5 good matches
        
        # Keep only inliers
        innovation = innovation[outlier_mask]
        valid_angles = valid_angles[outlier_mask]
        n_valid = len(innovation)
        
        # Recompute expected for inliers only
        expected_ranges = self._simulate_lidar(self.state, valid_angles, z_lidar)
        innovation = valid_ranges[outlier_mask] - expected_ranges
        
        # Compute Jacobian using ANALYTIC derivatives (more stable than numerical)
        H = self._compute_measurement_jacobian_analytic(self.state, valid_angles, 
                                                         expected_ranges, z_lidar)
        
        # Check for degenerate Jacobian
        if np.max(np.abs(H)) < 1e-6:
            return  # Jacobian too small, skip update
        
        # Measurement noise covariance
        R = np.eye(n_valid) * self.R_single * 4.0  # Increase measurement noise for conservatism
        
        # Innovation covariance
        S = H @ self.P @ H.T + R
        S = (S + S.T) / 2  # Ensure symmetry
        S += np.eye(n_valid) * 1e-4  # Regularization
        
        # Compute Kalman gain
        try:
            S_inv = np.linalg.inv(S)
            K = self.P @ H.T @ S_inv
        except np.linalg.LinAlgError:
            return  # Skip if singular
        
        # Apply correction with VERY conservative limit
        state_correction = K @ innovation
        
        # Limit correction to 2cm position, 0.05 rad orientation
        max_pos_correction = 0.02
        max_angle_correction = 0.05
        
        pos_correction_mag = np.sqrt(state_correction[0]**2 + state_correction[1]**2)
        if pos_correction_mag > max_pos_correction:
            scale = max_pos_correction / pos_correction_mag
            state_correction[0] *= scale
            state_correction[1] *= scale
        
        if abs(state_correction[2]) > max_angle_correction:
            state_correction[2] = np.sign(state_correction[2]) * max_angle_correction
        
        # Update state
        self.state = self.state + state_correction
        self.state[2] = (self.state[2] + np.pi) % (2*np.pi) - np.pi
        
        # Update covariance (Joseph form)
        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = (self.P + self.P.T) / 2
        self.P += np.eye(3) * 1e-8
    
    def _compute_measurement_jacobian_analytic(self, pose, angles, expected_ranges, z):
        n = len(angles)
        H = np.zeros((n, 3))
        
        x, y, theta = pose
        
        for i, alpha in enumerate(angles):
            angle_global = theta + alpha
            
            # Approximate Jacobian (assumes obstacle is far enough)
            # This is a simplified model but more stable than numerical diff
            cos_a = np.cos(angle_global)
            sin_a = np.sin(angle_global)
            
            # Position derivatives (negative because moving towards obstacle decreases range)
            H[i, 0] = -cos_a  # dr/dx
            H[i, 1] = -sin_a  # dr/dy
            
            # Orientation derivative (depends on where obstacle is)
            # Use small numerical perturbation for theta only
            eps = 0.01
            pose_plus = np.array([x, y, theta + eps])
            pose_minus = np.array([x, y, theta - eps])
            
            r_plus = self._simulate_lidar(pose_plus, [alpha], z)[0]
            r_minus = self._simulate_lidar(pose_minus, [alpha], z)[0]
            
            H[i, 2] = (r_plus - r_minus) / (2 * eps)
        
        # Handle numerical issues
        H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
        
        return H
    
    def _simulate_lidar(self, pose, angles, z):
        """
        Simulate LiDAR scan from given pose.
        """
        x, y, theta = pose
        origins = []
        ends = []
        
        for a in angles:
            ang = theta + a
            ox, oy, oz = x, y, z
            ex = ox + math.cos(ang) * self.lidar_max_range
            ey = oy + math.sin(ang) * self.lidar_max_range
            ez = oz
            origins.append((ox, oy, oz))
            ends.append((ex, ey, ez))
        
        results = p.rayTestBatch(origins, ends)
        ranges = np.full(len(angles), self.lidar_max_range, dtype=float)
        
        for i, res in enumerate(results):
            if res[0] != -1:
                ranges[i] = res[2] * self.lidar_max_range
        
        return ranges
    
    def _compute_measurement_jacobian(self, pose, angles, z):
        """
        Legacy numerical Jacobian (kept for compatibility).
        """
        n = len(angles)
        H = np.zeros((n, 3))
        epsilon = np.array([1e-3, 1e-3, 1e-3])
        
        for i in range(3):
            pose_forward = pose.copy()
            pose_forward[i] += epsilon[i]
            h_forward = self._simulate_lidar(pose_forward, angles, z)
            
            pose_backward = pose.copy()
            pose_backward[i] -= epsilon[i]
            h_backward = self._simulate_lidar(pose_backward, angles, z)
            
            H[:, i] = (h_forward - h_backward) / (2 * epsilon[i])
        
        H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
        return H
    
    def estimate(self):
        return tuple(self.state)
    
    def get_covariance(self):
        return self.P.copy()
    
    def draw_estimate(self, color=[0, 0, 1], life_time=0.1):
        x, y, theta = self.state
        start = (x, y, self.z_lidar)
        end = (x + math.cos(theta) * 0.3, y + math.sin(theta) * 0.3, self.z_lidar)
        p.addUserDebugLine(start, end, color, lineWidth=3, lifeTime=life_time)
        
        P_pos = self.P[0:2, 0:2]
        eigenvalues, eigenvectors = np.linalg.eig(P_pos)
        n_points = 20
        theta_ellipse = np.linspace(0, 2*np.pi, n_points)
        scale = 2.0
        sqrt_eig = np.sqrt(np.abs(eigenvalues)) * scale
        
        ellipse_points = []
        for t in theta_ellipse:
            point_local = np.array([sqrt_eig[0] * np.cos(t), sqrt_eig[1] * np.sin(t)])
            point_rotated = eigenvectors @ point_local
            point_world = [x + point_rotated[0], y + point_rotated[1], self.z_lidar]
            ellipse_points.append(point_world)
        
        for i in range(len(ellipse_points)):
            p1 = ellipse_points[i]
            p2 = ellipse_points[(i + 1) % len(ellipse_points)]
            p.addUserDebugLine(p1, p2, color, lineWidth=1, lifeTime=life_time)