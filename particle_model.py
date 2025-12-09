import numpy as np

class ParticleFilter:
    def __init__(self, robot_model, num_particles, init_state):
        self.robot = robot_model
        self.num_particles = num_particles

        self.particles = []
        self.weights = np.ones(num_particles) / num_particles

        for _ in range(num_particles):
            self.particles.append({
                "x": init_state["x"] + np.random.normal(0, 0.2),
                "y": init_state["y"] + np.random.normal(0, 0.2),
                "theta": init_state["theta"] + np.random.normal(0, 0.1)
            })

    def predict(self, v, omega, dt):
        for i, p in enumerate(self.particles):
            self.particles[i] = self.robot.propagate_state(
                p, v, omega, dt
            )

    def update(self, lidar_ranges):
        for i in range(self.num_particles):
            # Call RobotModel sensor model
            likelihood = self.robot.sensor_update(lidar_ranges)
            self.weights[i] = likelihood

        self.weights += 1e-300
        self.weights /= np.sum(self.weights)


    def resample(self):
        indices = np.random.choice(
            range(self.num_particles),
            size=self.num_particles,
            p=self.weights
        )

        self.particles = [self.particles[i].copy() for i in indices]
        self.weights.fill(1.0 / self.num_particles)

    def estimate(self):
        xs = np.array([p["x"] for p in self.particles])
        ys = np.array([p["y"] for p in self.particles])
        thetas = np.array([p["theta"] for p in self.particles])

        return {
            "x": np.average(xs, weights=self.weights),
            "y": np.average(ys, weights=self.weights),
            "theta": np.arctan2(
                np.average(np.sin(thetas), weights=self.weights),
                np.average(np.cos(thetas), weights=self.weights),
            )
        }

