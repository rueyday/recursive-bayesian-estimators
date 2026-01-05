import numpy as np
import pybullet as p
import math

def sample_free_cells(occ, xs, ys, n_samples):
    ny, nx = occ.shape
    free_indices = np.argwhere(occ == 0)
    if free_indices.size == 0:
        raise RuntimeError("No free cells in occupancy grid.")
    picks = np.random.choice(free_indices.shape[0], size=n_samples, replace=True)
    coords = free_indices[picks]
    ys_idx = coords[:,0]; xs_idx = coords[:,1]
    
    x_samples = xs[xs_idx] + (np.random.rand(n_samples)-0.5) * (xs[1]-xs[0])
    y_samples = ys[ys_idx] + (np.random.rand(n_samples)-0.5) * (ys[1]-ys[0])
    thetas = np.random.uniform(-np.pi, np.pi, size=n_samples)
    return np.stack([x_samples, y_samples, thetas], axis=1)

# ---------- Particle Filter ----------
class ParticleFilter:
    """
    PRODUCTION PARTICLE FILTER - Enhanced with teleportation handling
    
    Key improvements:
    1. Correct motion model (theta += dtheta, not theta *= dtheta)
    2. Minimal resampling jitter (0.005 not 0.05)
    3. Robust likelihood (capped at 5 sigma)
    4. Adaptive exploration based on weight concentration
    5. NEW: Teleportation detection and recovery
    6. NEW: Aggressive recovery when completely lost
    """
    def __init__(self, occ, xs, ys, n_particles=500,
                 lidar_max_range=10.0, lidar_min_range=0.1,
                 z_lidar=0.5, sigma_range=0.2,
                 motion_noise=(0.02, 0.02, 0.01),
                 scan_subsample=4, 
                 initial_pose=None):
        self.occ, self.xs, self.ys = occ, xs, ys
        self.n_particles = n_particles
        self.lidar_max_range = lidar_max_range
        self.lidar_min_range = lidar_min_range
        self.z_lidar = z_lidar
        self.sigma_range = sigma_range
        self.motion_noise = motion_noise
        self.scan_subsample = max(1, int(scan_subsample))
        self.angles = np.linspace(0, 2*np.pi, 360, endpoint=False)
        
        # Initialize particles
        if initial_pose is not None:
            # Start concentrated around initial pose
            self.particles = np.array(initial_pose) + np.random.normal(0, 0.01, (n_particles, 3))
            self.particles[:, 2] = (self.particles[:, 2] + np.pi) % (2*np.pi) - np.pi
        else:
            self.particles = sample_free_cells(self.occ, self.xs, self.ys, self.n_particles)
            
        self.weights = np.ones(self.n_particles) / self.n_particles
        
        # NEW: Track filter confidence
        self.recovery_mode = False
        self.steps_since_teleport = 0

    def handle_teleportation(self, new_x, new_y, new_theta, uncertainty_radius=1.0):
        """
        Handle sudden teleportation by reinitializing particles around new position.
        
        Args:
            new_x, new_y, new_theta: New robot pose
            uncertainty_radius: Spatial uncertainty in meters (default 1.0m)
        """
        print(f"    [PF] Teleportation handler: Reinitializing {self.n_particles} particles")
        
        # Create particles around new position with specified uncertainty
        # 80% concentrated, 20% exploring
        n_concentrated = int(0.8 * self.n_particles)
        n_explore = self.n_particles - n_concentrated
        
        # Concentrated particles around new pose
        concentrated = np.random.normal(
            loc=[new_x, new_y, new_theta],
            scale=[uncertainty_radius, uncertainty_radius, np.pi/4],
            size=(n_concentrated, 3)
        )
        concentrated[:, 2] = (concentrated[:, 2] + np.pi) % (2*np.pi) - np.pi
        
        # Exploration particles in free space
        explore = sample_free_cells(self.occ, self.xs, self.ys, n_explore)
        
        self.particles = np.vstack([concentrated, explore])
        self.weights = np.ones(self.n_particles) / self.n_particles
        
        # Mark as in recovery mode
        self.recovery_mode = True
        self.steps_since_teleport = 0

    def motion_update(self, odom):
        dx_l, dy_l, dtheta = odom
        sx, sy, st = self.motion_noise
        theta_i = self.particles[:, 2]
        
        # Add motion noise (increased during recovery)
        noise_scale = 2.0 if self.recovery_mode else 1.0
        noisy_dx = dx_l + np.random.normal(0, sx * noise_scale, self.n_particles)
        noisy_dy = dy_l + np.random.normal(0, sy * noise_scale, self.n_particles)
        noisy_dt = dtheta + np.random.normal(0, st * noise_scale, self.n_particles)
        
        # Apply motion model
        cos_t, sin_t = np.cos(theta_i), np.sin(theta_i)
        self.particles[:, 0] += cos_t * noisy_dx - sin_t * noisy_dy
        self.particles[:, 1] += sin_t * noisy_dx + cos_t * noisy_dy
        self.particles[:, 2] = (theta_i + noisy_dt + np.pi) % (2*np.pi) - np.pi  # CRITICAL: + not *

    def measurement_update(self, real_ranges, real_angles=None):
        angles_to_use = real_angles if real_angles is not None else self.angles
        real_ranges_sub = real_ranges[::self.scan_subsample]
        angles_sub = angles_to_use[::self.scan_subsample]
        
        num_rays = len(angles_sub)
        
        # Prepare raycast batch
        origins = np.repeat(self.particles[:, :2], num_rays, axis=0)
        origins = np.hstack([origins, np.full((len(origins), 1), self.z_lidar)])
        
        particle_thetas = np.repeat(self.particles[:, 2], num_rays)
        ray_angles = particle_thetas + np.tile(angles_sub, self.n_particles)
        
        ends_x = origins[:, 0] + np.cos(ray_angles) * self.lidar_max_range
        ends_y = origins[:, 1] + np.sin(ray_angles) * self.lidar_max_range
        ends = np.stack([ends_x, ends_y, np.full_like(ends_x, self.z_lidar)], axis=1)

        # Batch Raycast
        batch_size = 4000
        results = []
        for i in range(0, len(origins), batch_size):
            results.extend(p.rayTestBatch(origins[i:i+batch_size].tolist(), 
                                         ends[i:i+batch_size].tolist()))

        sim_ranges = np.array([r[2] * self.lidar_max_range if r[0] != -1 
                               else self.lidar_max_range for r in results])
        sim_ranges = sim_ranges.reshape(self.n_particles, -1)

        # Compute likelihood with robust loss
        rr = np.clip(real_ranges_sub, self.lidar_min_range, self.lidar_max_range)
        sq_err = (sim_ranges - rr[np.newaxis, :])**2
        normalized_err = sq_err / (self.sigma_range**2)
        
        # Cap errors at 5 sigma (robust to outliers)
        normalized_err = np.minimum(normalized_err, 25.0)
        
        log_like = -0.5 * np.sum(normalized_err, axis=1)
        
        # Update weights with numerical stability
        max_log = np.max(log_like)
        self.weights *= np.exp(log_like - max_log)
        
        w_sum = np.sum(self.weights)
        if w_sum < 1e-100:
            self.weights.fill(1.0 / self.n_particles)
        else:
            self.weights /= w_sum
        
        # Update recovery status
        if self.recovery_mode:
            self.steps_since_teleport += 1
            if self.steps_since_teleport > 10:  # Exit recovery after 10 steps
                self.recovery_mode = False
                print("    [PF] Exiting recovery mode")

    def resample(self):
        neff = 1.0 / np.sum(self.weights**2)
        max_weight = np.max(self.weights)
        
        # Detect if filter is completely lost
        # If weights are extremely uniform, all particles are equally bad
        weight_entropy = -np.sum(self.weights * np.log(self.weights + 1e-100))
        max_entropy = np.log(self.n_particles)
        normalized_entropy = weight_entropy / max_entropy
        
        # If entropy is very high (>0.95), weights are nearly uniform = lost
        if normalized_entropy > 0.95 and not self.recovery_mode:
            print(f"    [PF] WARNING: Filter appears lost (entropy={normalized_entropy:.3f})")
            print(f"    [PF] Max weight={max_weight:.6f}, Neff={neff:.1f}")
            # Add aggressive exploration
            n_explore = int(0.5 * self.n_particles)
            explore = sample_free_cells(self.occ, self.xs, self.ys, n_explore)
            
            # Keep best 50% of current particles
            n_keep = self.n_particles - n_explore
            best_indices = np.argsort(self.weights)[-n_keep:]
            self.particles = np.vstack([self.particles[best_indices], explore])
            self.weights = np.ones(self.n_particles) / self.n_particles
            return
        
        # Resample when effective sample size drops
        if neff < self.n_particles / 2.0:
            # Adaptive exploration based on weight concentration and recovery mode
            if self.recovery_mode:
                # More aggressive during recovery
                n_explore = int(0.3 * self.n_particles)
            elif max_weight > 0.5:
                n_explore = int(0.15 * self.n_particles)
            elif max_weight > 0.2:
                n_explore = int(0.05 * self.n_particles)
            else:
                n_explore = int(0.02 * self.n_particles)
            
            n_resample = self.n_particles - n_explore
            
            # Resample from weighted distribution
            indices = np.random.choice(self.n_particles, size=n_resample, p=self.weights)
            resampled = self.particles[indices].copy()
            
            # Add MINIMAL jitter to prevent depletion (not 0.05!)
            # Increased jitter during recovery
            jitter_std = 0.02 if self.recovery_mode else 0.005
            resampled += np.random.normal(0, jitter_std, resampled.shape)
            resampled[:, 2] = (resampled[:, 2] + np.pi) % (2*np.pi) - np.pi
            
            # Add exploration particles
            if n_explore > 0:
                explore = sample_free_cells(self.occ, self.xs, self.ys, n_explore)
                self.particles = np.vstack([resampled, explore])
            else:
                self.particles = resampled
            
            # Reset weights uniformly
            self.weights.fill(1.0 / self.n_particles)

    def estimate(self):
        x_m = np.sum(self.particles[:,0] * self.weights)
        y_m = np.sum(self.particles[:,1] * self.weights)
        c_m = np.sum(np.cos(self.particles[:,2]) * self.weights)
        s_m = np.sum(np.sin(self.particles[:,2]) * self.weights)
        return (x_m, y_m, math.atan2(s_m, c_m))