import numpy as np
import pybullet as p
from pybullet_tools.utils import link_from_name, get_link_pose
import math
import time

# ---------- Occupancy grid builder ----------
def build_occupancy_grid(xmin, xmax, ymin, ymax, resolution,
                         z_top=3.0, z_bottom=-1.0,
                         height_threshold=0.1):
    """
    Build a 2D occupancy grid by casting vertical rays from z_top to z_bottom
    at every grid cell center. If a hit exists within the vertical segment,
    mark cell occupied.
    Returns:
        occ: 2D numpy array (ny, nx). 1=occupied, 0=free
        xs, ys: 1D arrays of cell centers (x, y)
    Notes:
        - Uses p.rayTestBatch for speed.
        - resolution in meters per cell.
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

    # batched ray tests
    results = p.rayTestBatch(origins, ends)
    occ = np.zeros((ny, nx), dtype=np.uint8)
    idx = 0
    for j in range(ny):
        for i in range(nx):
            res = results[idx]
            hit_obj = res[0]
            if hit_obj != -1:
                # fraction < 1.0 indicates hit along the ray segment
                hit_fraction = res[2]
                # compute z of hit: z_top + fraction*(z_bottom - z_top)
                z_hit = z_top + hit_fraction * (z_bottom - z_top)
                # if we care about small obstacles threshold, check difference
                if (z_hit - z_bottom) > height_threshold or (z_top - z_hit) > height_threshold:
                    occ[j, i] = 1
            idx += 1
    return occ, xs, ys

# ---------- Utility helpers ----------
def pose_to_index(x, y, xs, ys):
    """Return (i_x, i_y) grid cell indices for pose (x,y)."""
    i = np.searchsorted(xs, x)  # gives insertion index
    j = np.searchsorted(ys, y)
    # convert to 0-based cell index where xs[i] is center — adjust boundaries
    i = np.clip(i, 0, xs.size-1)
    j = np.clip(j, 0, ys.size-1)
    return i, j

def sample_free_cells(occ, xs, ys, n_samples):
    """Return random (x,y) samples uniformly from free cells."""
    ny, nx = occ.shape
    free_indices = np.argwhere(occ == 0)
    if free_indices.size == 0:
        raise RuntimeError("No free cells in occupancy grid to sample particles.")
    picks = np.random.choice(len(free_indices), size=n_samples, replace=True)
    coords = free_indices[picks]  # array of [row, col]
    ys_idx = coords[:,0]; xs_idx = coords[:,1]
    x_samples = xs[xs_idx] + (np.random.rand(n_samples)-0.5) * 0.9 * (xs[1]-xs[0]) if xs.size>1 else xs[xs_idx]
    y_samples = ys[ys_idx] + (np.random.rand(n_samples)-0.5) * 0.9 * (ys[1]-ys[0]) if ys.size>1 else ys[ys_idx]
    thetas = np.random.uniform(-np.pi, np.pi, size=n_samples)
    return np.stack([x_samples, y_samples, thetas], axis=1)

# ---------- Lidar simulation from an arbitrary pose ----------
def simulate_lidar_from_pose(pose, angles, max_range=10.0, z=0.5, batch=True):
    """
    Simulate a 2D lidar scan from a pose = (x, y, theta).
    Returns:
        ranges: numpy array same length as angles (float)
    Notes:
        - We create ray origins at (x,y,z) and ray ends at origin + dir*max_range.
        - Uses p.rayTestBatch for speed.
    """
    x, y, theta = pose
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    origins = []
    ends = []
    for a in angles:
        ang = theta + a
        dx = math.cos(ang)
        dy = math.sin(ang)
        origin = (x, y, z)
        ends.append((x + dx * max_range, y + dy * max_range, z))
        origins.append(origin)
    # batched query
    results = p.rayTestBatch(origins, ends)
    ranges = np.full(len(angles), max_range, dtype=float)
    for i, res in enumerate(results):
        hit_obj = res[0]
        if hit_obj != -1:
            hit_fraction = res[2]
            # fraction is fraction along segment origin->end, multiply by max_range
            ranges[i] = hit_fraction * max_range
    return ranges

# ---------- Particle filter core ----------
class ParticleFilter:
    def __init__(self, occ, xs, ys, n_particles=500,
                 lidar_max_range=10.0, lidar_min_range=0.1,
                 z_lidar=0.5,
                 sigma_range=0.2,
                 motion_noise=(0.02, 0.02, 0.01),
                 angles_full=None,
                 scan_subsample=4,
                 random_seed=None):
        """
        occ, xs, ys: occupancy grid and cell centers from build_occupancy_grid
        n_particles: number of particles
        sigma_range: std dev of measurement error (meters)
        motion_noise: (sx, sy, stheta) standard deviations for motion model
        angles_full: precomputed angles of lidar rays (if None, use 360)
        scan_subsample: only compare every Nth ray to speed up likelihood
        """
        if random_seed is not None:
            np.random.seed(random_seed)
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

        # default angles
        if angles_full is None:
            self.angles = np.linspace(0, 2*np.pi, 360, endpoint=False)
        else:
            self.angles = np.array(angles_full)
        # subsampled angles for likelihood evaluation
        self.angles_sub = self.angles[::self.scan_subsample]

        # initialize particles uniformly over free cells
        self.particles = sample_free_cells(self.occ, self.xs, self.ys, self.n_particles)
        self.weights = np.ones(self.n_particles) / self.n_particles

    def motion_update(self, odom):
        """
        odom: (dx_local, dy_local, dtheta)
              dx_local: Forward distance traveled (dist in baseline_test)
              dy_local: Side slip (set to 0.0 in baseline_test)
              dtheta: Change in heading
        
        CRITICAL FIX: The key issue was the order of operations. We need to:
        1. Get the current orientation BEFORE rotation
        2. Use that orientation to transform the local motion to global
        3. THEN apply the rotation update
        
        This ensures the displacement is applied in the correct direction.
        """
        dx_local, dy_local, dtheta = odom
        sx, sy, st = self.motion_noise
        
        # Get current particle orientations BEFORE rotation update
        theta_i = self.particles[:, 2]
        
        # Add motion noise to the control inputs
        noisy_dx = dx_local + np.random.normal(0, sx, self.n_particles)
        noisy_dy = dy_local + np.random.normal(0, sy, self.n_particles)
        noisy_dt = dtheta + np.random.normal(0, st, self.n_particles)
        
        # Transform local motion to global frame using CURRENT orientation
        # This is the rotation matrix: [cos(theta) -sin(theta); sin(theta) cos(theta)]
        cos_t = np.cos(theta_i)
        sin_t = np.sin(theta_i)
        
        # Apply rotation matrix to transform local displacement to global
        delta_x_global = cos_t * noisy_dx - sin_t * noisy_dy
        delta_y_global = sin_t * noisy_dx + cos_t * noisy_dy
        
        # Update particles: position first, then orientation
        self.particles[:, 0] += delta_x_global
        self.particles[:, 1] += delta_y_global
        self.particles[:, 2] += noisy_dt
            
        # Normalize angles to [-pi, pi]
        self.particles[:, 2] = (self.particles[:, 2] + np.pi) % (2*np.pi) - np.pi

    def measurement_update(self, real_ranges, real_angles=None, z_lidar=None, weight_clamp=(1e-300, 1e300)):
        """
        real_ranges: numpy array of real lidar ranges (full scan)
        real_angles: angles corresponding to real_ranges; if None, assumes same as self.angles
        z_lidar: lidar height to simulate from. If None, use self.z_lidar
        Updates self.weights based on comparing simulated scans to real_ranges.
        """
        if real_angles is None:
            real_angles = self.angles
        if z_lidar is None:
            z_lidar = self.z_lidar

        # Subsample real ranges to match angles_sub
        real_ranges_sub = real_ranges[::self.scan_subsample]

        # For each particle, simulate at its (x,y,theta) for angles_sub
        sim_ranges_all = []
        # To speed up, we will batch rays across particles:
        # Build lists of origins and ends for *all* particle rays, then call rayTestBatch once.
        origins = []
        ends = []
        particle_ray_counts = []  # number of rays per particle (same for all here)
        for p_idx in range(self.n_particles):
            x, y, theta = self.particles[p_idx]
            for a in self.angles_sub:
                ang = theta + a
                ox, oy, oz = x, y, z_lidar
                ex = ox + math.cos(ang) * self.lidar_max_range
                ey = oy + math.sin(ang) * self.lidar_max_range
                ez = oz
                origins.append((ox, oy, oz))
                ends.append((ex, ey, ez))
            particle_ray_counts.append(len(self.angles_sub))
        # Raycast batch (this could be big; watch memory)
        max_batch_size = 500  # safe value for your PyBullet
        results = []
        for start_idx in range(0, len(origins), max_batch_size):
            batch_origins = origins[start_idx:start_idx+max_batch_size]
            batch_ends = ends[start_idx:start_idx+max_batch_size]
            batch_results = p.rayTestBatch(batch_origins, batch_ends)
            results.extend(batch_results)

        # parse results into sim_ranges array per particle
        sim_ranges = np.empty((self.n_particles, len(self.angles_sub)), dtype=float)
        idx = 0
        for p_idx in range(self.n_particles):
            for r_i in range(len(self.angles_sub)):
                res = results[idx]
                hit_obj = res[0]
                if hit_obj != -1:
                    hit_fraction = res[2]
                    sim_ranges[p_idx, r_i] = hit_fraction * self.lidar_max_range
                else:
                    sim_ranges[p_idx, r_i] = self.lidar_max_range
                idx += 1

        # Compute weights using Gaussian likelihood (independent rays)
        # To avoid numerical underflow, compute log-likelihoods then exponentiate.
        var = self.sigma_range**2
        # clip real ranges: anything below min => treat as max_range
        rr = real_ranges_sub.copy()
        rr[rr < self.lidar_min_range] = self.lidar_max_range

        # calculate squared errors
        # shape (n_particles, n_rays)
        sq_err = (sim_ranges - rr[np.newaxis, :])**2
        # log likelihood for Gaussian with sigma: -0.5 * sum(err/var) + const
        log_like = -0.5 * np.sum(sq_err / var, axis=1)
        # convert to weights
        # subtraction of max for stability
        max_log = np.max(log_like)
        weights_unnorm = np.exp(log_like - max_log)
        weights_unnorm = np.clip(weights_unnorm, weight_clamp[0], weight_clamp[1])
        # multiply into existing weights (sequential measurement filtering)
        self.weights *= weights_unnorm
        # normalize
        wsum = np.sum(self.weights)
        if wsum == 0 or not np.isfinite(wsum):
            # reinitialize uniform if degeneracy
            self.weights = np.ones(self.n_particles) / self.n_particles
        else:
            self.weights /= wsum

    def effective_sample_size(self):
        return 1.0 / np.sum(self.weights**2)

    def resample(self):
        """
        Systematic resampling
        """
        n = self.n_particles
        positions = (np.arange(n) + np.random.rand()) / n
        cumulative = np.cumsum(self.weights)
        new_particles = np.zeros_like(self.particles)
        i = 0
        for j in range(n):
            while positions[j] > cumulative[i]:
                i += 1
            new_particles[j] = self.particles[i]
        self.particles = new_particles
        self.weights.fill(1.0 / n)

    def estimate(self):
        """
        Weighted mean pose (for angles we compute using circular mean).
        Returns (x_mean, y_mean, theta_mean)
        """
        x_mean = np.sum(self.particles[:,0] * self.weights)
        y_mean = np.sum(self.particles[:,1] * self.weights)
        # circular mean
        cos_mean = np.sum(np.cos(self.particles[:,2]) * self.weights)
        sin_mean = np.sum(np.sin(self.particles[:,2]) * self.weights)
        theta_mean = math.atan2(sin_mean, cos_mean)
        return (x_mean, y_mean, theta_mean)

    def estimate_map(self):
        """Maximum a posteriori estimate"""
        idx = np.argmax(self.weights)
        return self.particles[idx]


    def draw_particles(self, z=0.1, scale=0.05, color=[0,1,0], life_time=0.1):
        """
        Visualize particles as points at fixed height z.
        """
        pts = np.zeros((self.n_particles, 3))
        pts[:, 0] = self.particles[:, 0]  # x
        pts[:, 1] = self.particles[:, 1]  # y
        pts[:, 2] = z                     # FIXED positive z

        p.addUserDebugPoints(
            pts,
            [color] * self.n_particles,
            pointSize=3,
            lifeTime=life_time
        )
