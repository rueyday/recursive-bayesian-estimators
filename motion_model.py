import numpy as np

class RobotModel:
    def __init__(self, init_state=None, motion_noise=True):
        """
        Minimal robot model with unicycle motion and sensor interface.
        init_state: dict with keys x, y, theta
        motion_noise: whether motion model injects noise
        """
        if init_state is None:
            init_state = {"x": 0.0, "y": 0.0, "theta": 0.0}

        self.state = init_state.copy()
        self.motion_noise = motion_noise

        # Noise parameters — tune these later
        self.v_noise_std = 0.02        # m/s
        self.omega_noise_std = 0.01    # rad/s
        self.sensor_noise_std = 0.05   # m (example — tune later)

    # -------------------------
    # MOTION MODEL
    # -------------------------
    def motion_update(self, v, omega, dt):
        """
        Apply unicycle motion model.
        Suitable for EKF or particle filter prediction step.
        """
        x, y, theta = self.state["x"], self.state["y"], self.state["theta"]

        # Add noise if enabled
        if self.motion_noise:
            v += np.random.normal(0, self.v_noise_std)
            omega += np.random.normal(0, self.omega_noise_std)

        # Unicycle model update
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += omega * dt

        # Normalize angle to [-pi, pi]
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        # Save and return
        self.state = {"x": x, "y": y, "theta": theta}
        return self.state

    # -------------------------
    # SENSOR MODEL (ABSTRACT)
    # ------------------------

    def expected_lidar_from_map(px, py, yaw, distance_field, max_range=10.0, num_rays=360, resolution=0.05, grid_size=200):
        width = grid_size
        height = grid_size
        angles = np.linspace(0, 2*np.pi, num_rays, endpoint=False)
        ranges = np.zeros(num_rays)

        for i, a in enumerate(angles):
            angle = yaw + a
            
            # March along the ray at coarse steps (fast)
            for d in np.linspace(0, max_range, 200):
                wx = px + d * np.cos(angle)
                wy = py + d * np.sin(angle)

                gx = int(wx / resolution)
                gy = int(wy / resolution)

                # Out of bounds
                if gx < 0 or gy < 0 or gx >= width or gy >= height:
                    break

                if distance_field[gy, gx] < 0.01:  # close to wall
                    ranges[i] = d
                    break
            else:
                ranges[i] = max_range

        return ranges
    
    def scan_likelihood(real_scan, exp_scan, sigma=0.1):
        diff = real_scan - exp_scan
        return np.exp(-(diff*diff).sum() / (2*sigma*sigma))

    
    def sensor_update(self, lidar_ranges, distance_field, max_range=10.0, num_rays=360, resolution=0.5, grid_size=20):
        """
        Particle Filter sensor likelihood.
        Uses simple statistics of the LiDAR scan.
        Returns p(z | x).
        """

        px = self.state["x"]
        py = self.state["y"]
        yaw = self.state["theta"]

        exp_ranges = RobotModel.expected_lidar_from_map(
            px, py, yaw, distance_field,
            max_range=max_range,
            num_rays=num_rays,
            resolution=resolution,
            grid_size=grid_size
        )

        likelihood = RobotModel.scan_likelihood(lidar_ranges, exp_ranges, sigma=self.sensor_noise_std)
        return likelihood
        
    def propagate_state(self, state, v, omega, dt):
        """
        Stateless motion model.
        Used by particle filter and EKF prediction step.
        """
        x, y, theta = state["x"], state["y"], state["theta"]

        v += np.random.normal(0, self.v_noise_std)
        omega += np.random.normal(0, self.omega_noise_std)

        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += omega * dt

        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        return {"x": x, "y": y, "theta": theta}


    # -------------------------
    # STATE ACCESS HELPERS
    # -------------------------
    def get_state(self):
        return self.state.copy()

    def set_state(self, x, y, theta):
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        self.state = {"x": x, "y": y, "theta": theta}
