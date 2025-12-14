import numpy as np
import cupy as cp # <--- ADDED CUPY
import pybullet as p
from pybullet_tools.utils import link_from_name, get_link_pose
import math
import time

# NOTE: This file assumes the caller (file1.py) is running on the CPU and
# only transfers data to the GPU when necessary for large array operations.

# ---------- Occupancy grid builder (CPU-bound map generation) ----------
def build_occupancy_grid(xmin, xmax, ymin, ymax, resolution,
                         z_top=3.0, z_bottom=-1.0,
                         height_threshold=0.1):
    """
    Build a 2D occupancy grid (CPU-based using numpy).
    """
    xs = np.arange(xmin + resolution/2.0, xmax, resolution)
    ys = np.arange(ymin + resolution/2.0, ymax, resolution)
    nx = xs.size
    ny = ys.size

    origins = []
    ends = []
    for y in ys:
        for x in xs:
            origins.append((x, y, z_top))
            ends.append((x, y, z_bottom))

    # batched ray tests (CPU/PyBullet)
    results = p.rayTestBatch(origins, ends)
    occ = np.zeros((ny, nx), dtype=np.uint8)
    idx = 0
    for j in range(ny):
        for i in range(nx):
            res = results[idx]
            hit_obj = res[0]
            if hit_obj != -1:
                hit_fraction = res[2]
                z_hit = z_top + hit_fraction * (z_bottom - z_top)
                if (z_hit - z_bottom) > height_threshold or (z_top - z_hit) > height_threshold:
                    occ[j, i] = 1
            idx += 1
    return occ, xs, ys

# ---------- Utility helpers (Adapted to use CuPy for random sampling) ----------
def pose_to_index(x, y, xs, ys):
    """Return (i_x, i_y) grid cell indices for pose (x,y)."""
    # Keep on CPU as it is only used once during map setup
    i = np.searchsorted(xs, x) 
    j = np.searchsorted(ys, y)
    i = np.clip(i, 0, xs.size-1)
    j = np.clip(j, 0, ys.size-1)
    return i, j

def sample_free_cells(occ, xs, ys, n_samples):
    """Return random (x,y) samples uniformly from free cells (CuPy)."""
    ny, nx = occ.shape
    # Transfer occ to GPU for argwhere if it's large, but here we keep it simple (using numpy)
    free_indices = np.argwhere(occ == 0)
    if free_indices.size == 0:
        raise RuntimeError("No free cells in occupancy grid to sample particles.")
    
    # Use CuPy for large scale random number generation
    cp.random.seed(int(time.time() * 1000))
    picks = cp.random.choice(len(free_indices), size=n_samples, replace=True)
    coords = free_indices[cp.asnumpy(picks)]  # Switch back to numpy for index access
    
    ys_idx = coords[:,0]; xs_idx = coords[:,1]
    
    # Generate random offsets on GPU
    rand_x = cp.random.rand(n_samples)
    rand_y = cp.random.rand(n_samples)
    thetas = cp.random.uniform(-cp.pi, cp.pi, size=n_samples)
    
    # Calculate positions on CPU/NumPy
    if xs.size > 1:
        x_samples = xs[xs_idx] + (cp.asnumpy(rand_x)-0.5) * 0.9 * (xs[1]-xs[0])
        y_samples = ys[ys_idx] + (cp.asnumpy(rand_y)-0.5) * 0.9 * (ys[1]-ys[0])
    else:
        x_samples = xs[xs_idx]
        y_samples = ys[ys_idx]

    # Combine and transfer final particles and weights to GPU
    particles_cpu = np.stack([x_samples, y_samples, cp.asnumpy(thetas)], axis=1)
    return cp.asarray(particles_cpu) # Return CuPy array

# ---------- Particle filter core (Heavy CuPy usage) ----------
class ParticleFilter:
    def __init__(self, occ, xs, ys, n_particles=500,
                 lidar_max_range=10.0, lidar_min_range=0.1,
                 z_lidar=0.5,
                 sigma_range=0.2,
                 motion_noise=(0.02, 0.02, 0.01),
                 angles_full=None,
                 scan_subsample=4,
                 random_seed=None):

        if random_seed is not None:
            cp.random.seed(random_seed) # Use CuPy seed
        self.occ = occ
        self.xs = xs
        self.ys = ys
        self.nx = xs.size
        self.ny = ys.size
        self.n_particles = n_particles
        self.lidar_max_range = lidar_max_range
        self.lidar_min_range = lidar_min_range
        self.z_lidar = z_lidar
        self.sigma_range = sigma_range
        self.motion_noise = motion_noise
        self.scan_subsample = max(1, int(scan_subsample))

        if angles_full is None:
            angles = np.linspace(0, 2*np.pi, 360, endpoint=False)
        else:
            angles = np.array(angles_full)
            
        # Store angles as CuPy arrays on GPU
        self.angles = cp.asarray(angles) 
        self.angles_sub = self.angles[::self.scan_subsample]

        # Initialize particles uniformly over free cells (CuPy array)
        self.particles = sample_free_cells(self.occ, self.xs, self.ys, self.n_particles)
        self.weights = cp.ones(self.n_particles, dtype=cp.float32) / self.n_particles

    def initialize_pose(self, x, y, theta):
        """Helper to initialize particles around a starting pose (on GPU)."""
        # Create noise and apply to particles on GPU
        noise_x = cp.random.normal(0, 0.5, self.n_particles)
        noise_y = cp.random.normal(0, 0.5, self.n_particles)
        noise_theta = cp.random.normal(0, 0.3, self.n_particles)
        
        self.particles[:, 0] = x + noise_x
        self.particles[:, 1] = y + noise_y
        self.particles[:, 2] = theta + noise_theta
        self.weights = cp.ones(self.n_particles, dtype=cp.float32) / self.n_particles
        # Normalize angles
        self.particles[:, 2] = (self.particles[:, 2] + cp.pi) % (2*cp.pi) - cp.pi


    def motion_update(self, odom):
        """odom: (dx, dy, dtheta) - ALL MATRIX OPS ON GPU"""
        dx, dy, dtheta = odom
        sx, sy, st = self.motion_noise
        
        # 1. Generate noise on GPU (massive speedup)
        noise_dx = cp.random.normal(0, sx, self.n_particles)
        noise_dy = cp.random.normal(0, sy, self.n_particles)
        noise_dt = cp.random.normal(0, st, self.n_particles)
        
        # 2. Add noise to odometry
        noisy_dx = dx + noise_dx
        noisy_dy = dy + dy + noise_dy # Bug fix: dy + dy should be dx, dy is already calculated
        noisy_dy = dy + noise_dy # Corrected: simply use dy
        noisy_dt = dtheta + noise_dt
        
        # 3. Use CuPy vectorized trigonometry (massive speedup)
        theta = self.particles[:, 2]
        cos_t = cp.cos(theta)
        sin_t = cp.sin(theta)
        
        # 4. Update position in global frame (vectorized on GPU)
        self.particles[:, 0] += cos_t * noisy_dx - sin_t * noisy_dy
        self.particles[:, 1] += sin_t * noisy_dx + cos_t * noisy_dy
        self.particles[:, 2] += noisy_dt
        
        # 5. Normalize angles on GPU
        self.particles[:, 2] = (self.particles[:, 2] + cp.pi) % (2*cp.pi) - cp.pi

    def measurement_update(self, real_ranges, real_angles=None, z_lidar=None, weight_clamp=(1e-300, 1e300)):
        """
        Calculates likelihoods on GPU, but performs batched raycasting on CPU/PyBullet.
        """
        if real_angles is None:
            real_angles = self.angles
        if z_lidar is None:
            z_lidar = self.z_lidar

        # Subsample real ranges (on CPU)
        real_ranges_sub = real_ranges[::self.scan_subsample]

        # --- PyBullet Raycasting (CPU-bound I/O) ---
        # Get particle poses from GPU to CPU for raycasting batch
        particles_cpu = cp.asnumpy(self.particles)
        angles_sub_cpu = cp.asnumpy(self.angles_sub)
        n_rays_sub = len(angles_sub_cpu)

        # 1. GENERATE ALL ORIGINS/ENDS (Still a big list)
        origins = []
        ends = []
        for p_idx in range(self.n_particles):
            x, y, theta = particles_cpu[p_idx]
            for a in angles_sub_cpu:
                ang = theta + a
                ox, oy, oz = x, y, z_lidar
                ex = ox + math.cos(ang) * self.lidar_max_range
                ey = oy + math.sin(ang) * self.lidar_max_range
                ez = oz
                origins.append((ox, oy, oz))
                ends.append((ex, ey, ez))
        
        # 2. BATCH THE PYBULLET CALLS (FIX)
        # Use a safe maximum batch size (e.g., 10,000 to be safe, though 16384 is often the max)
        MAX_RAYCAST_BATCH = 10000 
        all_results = []
        
        for i in range(0, len(origins), MAX_RAYCAST_BATCH):
            batch_origins = origins[i:i + MAX_RAYCAST_BATCH]
            batch_ends = ends[i:i + MAX_RAYCAST_BATCH]
            # PyBullet call
            all_results.extend(p.rayTestBatch(batch_origins, batch_ends)) 

        # 3. PARSE RESULTS
        sim_ranges_cpu = np.empty((self.n_particles, n_rays_sub), dtype=float)
        idx = 0
        for p_idx in range(self.n_particles):
            for r_i in range(n_rays_sub):
                res = all_results[idx]
                hit_obj = res[0]
                if hit_obj != -1:
                    hit_fraction = res[2]
                    sim_ranges_cpu[p_idx, r_i] = hit_fraction * self.lidar_max_range
                else:
                    sim_ranges_cpu[p_idx, r_i] = self.lidar_max_range
                idx += 1
        
        # --- Likelihood Calculation (GPU-Vectorized) ---
        # (Remaining code unchanged, still runs on GPU)
        sim_ranges = cp.asarray(sim_ranges_cpu)
        real_ranges_sub = cp.asarray(real_ranges_sub)

        var = self.sigma_range**2
        rr = real_ranges_sub.copy()
        rr[rr < self.lidar_min_range] = self.lidar_max_range

        # Calculate squared errors (GPU)
        sq_err = (sim_ranges - rr[cp.newaxis, :])**2
        
        # Log likelihood (GPU)
        log_like = -0.5 * cp.sum(sq_err / var, axis=1)
        
        # Convert to weights (GPU)
        max_log = cp.max(log_like)
        weights_unnorm = cp.exp(log_like - max_log)
        weights_unnorm = cp.clip(weights_unnorm, weight_clamp[0], weight_clamp[1])
        
        # Multiply into existing weights (GPU)
        self.weights *= weights_unnorm
        
        # Normalize (GPU)
        wsum = cp.sum(self.weights)
        if wsum == 0 or not cp.isfinite(wsum):
            self.weights = cp.ones(self.n_particles, dtype=cp.float32) / self.n_particles
        else:
            self.weights /= wsum

    def effective_sample_size(self):
        # Calculate ESS on GPU and return CPU scalar
        return cp.asnumpy(1.0 / cp.sum(self.weights**2))

    def resample(self):
        """Systematic resampling (Efficient CPU-based implementation is often fine, but we adapt for GPU arrays)"""
        n = self.n_particles
        
        # Transfer weights to CPU for systematic resampling logic
        weights_cpu = cp.asnumpy(self.weights)
        positions = (np.arange(n) + np.random.rand()) / n
        cumulative = np.cumsum(weights_cpu)
        
        # Perform resampling indices calculation on CPU
        new_indices = np.zeros(n, dtype=int)
        i = 0
        for j in range(n):
            while positions[j] > cumulative[i]:
                i += 1
            new_indices[j] = i
            
        # Select new particles on GPU using CuPy array indexing
        new_particles = self.particles[cp.asarray(new_indices)]
        self.particles = new_particles
        self.weights.fill(1.0 / n)

    def estimate(self):
        """
        Weighted mean pose. Returns (x_mean, y_mean, theta_mean) as CPU tuple.
        """
        # Transfer particles and weights to CPU for final weighted mean (or do entirely on GPU)
        # Doing it on GPU for max speed, then transferring the result
        x_mean = cp.sum(self.particles[:,0] * self.weights)
        y_mean = cp.sum(self.particles[:,1] * self.weights)
        
        # Circular mean on GPU
        cos_mean = cp.sum(cp.cos(self.particles[:,2]) * self.weights)
        sin_mean = cp.sum(cp.sin(self.particles[:,2]) * self.weights)
        theta_mean = cp.arctan2(sin_mean, cos_mean)
        
        # Return as numpy array/tuple
        result_cpu = cp.asnumpy(cp.array([x_mean, y_mean, theta_mean]))
        return tuple(result_cpu)

    def draw_particles(self, scale=0.05, color=[0,1,0], life_time=0.1):
        """
        Visualize particles. Must transfer GPU particles to CPU.
        """
        particles_cpu = cp.asnumpy(self.particles)
        p.addUserDebugPoints(particles_cpu[:,0:3], [color]*self.n_particles, pointSize=3, lifeTime=life_time)