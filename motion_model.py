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
