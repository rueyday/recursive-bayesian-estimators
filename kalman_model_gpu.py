import numpy as np
import pybullet as p
import math
from pybullet_tools.utils import link_from_name, get_link_pose

class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for robot localization with 2D LiDAR (NumPy only).
    State: [x, y, theta]
    """
    def __init__(self, initial_pose=(0, 0, 0), 
                 motion_noise=(0.02, 0.02, 0.01),
                 lidar_max_range=10.0,
                 lidar_min_range=0.1,
                 z_lidar=0.5,
                 sigma_range=0.2,
                 angles_full=None,
                 scan_subsample=4):
        
        # State estimate [x, y, theta]
        self.state = np.array(initial_pose, dtype=float)
        
        # State covariance matrix (3x3)
        self.P = np.eye(3) * 0.5  # Initial uncertainty
        
        # Process noise covariance (motion model)
        self.motion_noise = motion_noise
        sx, sy, st = motion_noise
        self.Q = np.diag([sx**2, sy**2, st**2])
        
        # Measurement noise
        self.sigma_range = sigma_range
        self.R_single = sigma_range**2  # Variance for single range measurement
        
        # LiDAR parameters
        self.lidar_max_range = lidar_max_range
        self.lidar_min_range = lidar_min_range
        self.z_lidar = z_lidar
        self.scan_subsample = max(1, int(scan_subsample))
        
        # LiDAR angles
        if angles_full is None:
            self.angles = np.linspace(0, 2*np.pi, 360, endpoint=False)
        else:
            self.angles = np.array(angles_full)
        
        # Subsampled angles for measurement update
        self.angles_sub = self.angles[::self.scan_subsample]
    
    def motion_update(self, odom):
        """Prediction step: update state estimate based on odometry."""
        dx, dy, dtheta = odom
        x, y, theta = self.state
        
        # Motion model: transform local frame motion to global frame
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        # Predicted state
        x_new = x + cos_t * dx - sin_t * dy
        y_new = y + sin_t * dx + cos_t * dy
        theta_new = theta + dtheta
        
        # Normalize angle
        theta_new = (theta_new + np.pi) % (2*np.pi) - np.pi
        
        # Jacobian of motion model with respect to state
        F = np.array([
            [1, 0, -sin_t * dx - cos_t * dy],
            [0, 1,  cos_t * dx - sin_t * dy],
            [0, 0,  1]
        ])
        
        # Update state
        self.state = np.array([x_new, y_new, theta_new])
        
        # Update covariance: P = F * P * F^T + Q
        self.P = F @ self.P @ F.T + self.Q
    
    def measurement_update(self, real_ranges, real_angles=None, z_lidar=None):
        """Correction step: update state estimate based on LiDAR measurements."""
        if real_angles is None:
            real_angles = self.angles
        if z_lidar is None:
            z_lidar = self.z_lidar
        
        # Subsample measurements
        real_ranges_sub = real_ranges[::self.scan_subsample]
        
        # Filter out invalid measurements
        valid_mask = (real_ranges_sub >= self.lidar_min_range) & (real_ranges_sub < self.lidar_max_range)
        if not np.any(valid_mask):
            return  # No valid measurements
        
        valid_ranges = real_ranges_sub[valid_mask]
        valid_angles = self.angles_sub[valid_mask]
        n_valid = len(valid_ranges)
        
        # Get expected measurements from current state
        expected_ranges = self._simulate_lidar(self.state, valid_angles, z_lidar)
        
        # Innovation (measurement residual)
        innovation = valid_ranges - expected_ranges
        
        # Compute measurement Jacobian H (n_valid x 3)
        H = self._compute_measurement_jacobian(self.state, valid_angles, z_lidar)
        
        # Measurement noise covariance (n_valid x n_valid diagonal)
        R = np.eye(n_valid) * self.R_single
        
        # Innovation covariance: S = H * P * H^T + R
        S = H @ self.P @ H.T + R
        
        # Kalman gain: K = P * H^T * S^-1
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        
        # Update state: x = x + K * innovation
        state_correction = K @ innovation
        self.state = self.state + state_correction
        
        # Normalize angle
        self.state[2] = (self.state[2] + np.pi) % (2*np.pi) - np.pi
        
        # Update covariance: P = (I - K*H) * P
        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ self.P
        
        # Ensure P remains symmetric and positive definite
        self.P = (self.P + self.P.T) / 2
    
    def _simulate_lidar(self, pose, angles, z):
        """Simulate LiDAR scan from given pose (PyBullet call)."""
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
        
        # Batch raycast
        results = p.rayTestBatch(origins, ends)
        ranges = np.full(len(angles), self.lidar_max_range, dtype=float)
        
        for i, res in enumerate(results):
            hit_obj = res[0]
            if hit_obj != -1:
                hit_fraction = res[2]
                ranges[i] = hit_fraction * self.lidar_max_range
        
        return ranges
    
    def _compute_measurement_jacobian(self, pose, angles, z):
        """Compute Jacobian of measurement function h(x) with respect to state (Numerical)."""
        n = len(angles)
        H = np.zeros((n, 3))
        epsilon = 1e-4
        
        # Numerical gradient (requires 3 extra rayTestBatch calls)
        h_current = self._simulate_lidar(pose, angles, z)
        
        for i in range(3):
            pose_perturbed = pose.copy()
            pose_perturbed[i] += epsilon
            h_perturbed = self._simulate_lidar(pose_perturbed, angles, z)
            H[:, i] = (h_perturbed - h_current) / epsilon
        
        return H
    
    def estimate(self):
        """Return current state estimate."""
        return tuple(self.state)
    
    def get_covariance(self):
        """Return current covariance matrix."""
        return self.P.copy()
    
    def draw_estimate(self, color=[0, 0, 1], life_time=0.1):
        """Visualize state estimate and uncertainty ellipse in PyBullet."""
        x, y, theta = self.state
        
        # Draw orientation arrow
        start = (x, y, self.z_lidar)
        end = (x + math.cos(theta) * 0.3, y + math.sin(theta) * 0.3, self.z_lidar)
        p.addUserDebugLine(start, end, color, lineWidth=3, lifeTime=life_time)
        
        # Draw uncertainty ellipse (using eigenvalues of position covariance)
        P_pos = self.P[0:2, 0:2]
        eigenvalues, eigenvectors = np.linalg.eig(P_pos)
        
        # Draw ellipse points (2-sigma confidence region)
        n_points = 20
        theta_ellipse = np.linspace(0, 2*np.pi, n_points)
        
        # Scale by 2-sigma (95% confidence)
        scale = 2.0
        sqrt_eig = np.sqrt(np.abs(eigenvalues)) * scale
        
        # Generate ellipse points
        ellipse_points = []
        for t in theta_ellipse:
            point_local = np.array([sqrt_eig[0] * np.cos(t), sqrt_eig[1] * np.sin(t)])
            point_rotated = eigenvectors @ point_local
            point_world = [x + point_rotated[0], y + point_rotated[1], self.z_lidar]
            ellipse_points.append(point_world)
        
        # Draw ellipse as connected line segments
        for i in range(len(ellipse_points)):
            p1 = ellipse_points[i]
            p2 = ellipse_points[(i + 1) % len(ellipse_points)]
            p.addUserDebugLine(p1, p2, color, lineWidth=1, lifeTime=life_time)