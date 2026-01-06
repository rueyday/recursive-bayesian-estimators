# Recursive Bayesian Estimators: PF, EKF, and UKF

A comparative benchmark of Recursive Bayesian Estimation algorithms (Particle Filter, Extended Kalman Filter, and Unscented Kalman Filter) implemented for robot localization in a PyBullet physics environment.

[Watch the full demo video on YouTube](https://www.youtube.com/watch?v=6H18YxdOMcI)
![Simulation Demo](demo.gif)

## Overview

This project evaluates the performance of three localization filters on a PR2 mobile robot following a figure-8 trajectory. The system utilizes a 2D LiDAR sensor model and a noisy odometry motion model. A unique feature of this benchmark is the Teleportation (Kidnapped Robot) Challenge, where the robot is suddenly moved to a new location to test the filters' ability to recover from high-global-uncertainty events.

## Project Structure
```
├── comparison.py           # Main entry point: runs simulation and real-time plotting
├── demo.gif                # Simulation preview
├── requirements.txt        # Python dependencies
├── filters/
│   ├── ekf_model.py        # Extended Kalman Filter with Numerical Jacobians
│   ├── pf_model.py         # Particle Filter with Adaptive Exploration & Recovery
│   └── ukf_model.py        # Unscented Kalman Filter using Sigma Point Transforms
├── models/                 # URDF and Mesh files for the PR2 Robot
├── environment/            # JSON environment configuration (figure-8 path)
├── pybullet_tools/         # PyBullet utility functions and JSON parsers
└── utils.py                # LiDAR and occupancy grid utilities
```
## Quick Start

### Installation
1. Clone the repository
```bash
git clone https://github.com/rueyday/recursive-bayesian-estimators.git
cd recursive-bayesian-estimators
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
### Running the Comparison 
Launch the simulation and performance visualizer
```bash
# Spacebar manually triggers a teleportation event to test filter recovery
python comparison.py
```

## Algorithm Implementation Details

### Extended Kalman Filter (EKF)
The EKF handles non-linearity by linearizing the motion and measurement models using Numerical Jacobians. 
+ Stability: Implements the Joseph form covariance update for improved numerical robustness.
+ Model: Uses a batch-raycasting approach to calculate expected LiDAR measurements.

### Unscented Kalman Filter (UKF)
The UKF bypasses linearization by using the Unscented Transform. It propagates a set of deterministically chosen Sigma Points through the non-linear functions. Instead of approximating a non-linear function, it approximates the probability distribution.
+ Scaling: Uses $\alpha$, $\beta$, and $\kappa$ parameters to control the spread and weighting of the $2n + 1$ sigma points.
  + $\alpha$ (Primary Scaling): Determines the spread of the sigma points around the mean.
  + $\kappa$ (Secondary Scaling): It ensures the covariance matrix remains positive semi-definite.
  + $\beta$ (Distribution Prior): Used to incorporate prior knowledge of the distribution.
+ Accuracy: Typically provides better approximation for the mean and covariance than EKF in highly non-linear scenarios.

### Particle Filter (PF)
A non-parametric filter that represents the posterior distribution using 3,000 particles.
+ Adaptive Exploration: Includes a recovery mode that injects random particles from free space when the filter detects high entropy.
+ Resampling: Uses low-variance resampling with minimal jitter to maintain diversity while preventing particle depletion.

## LiDAR Sensor & Likelihood Model
The robot observes the environment through a 2D LiDAR scan. The measurement model $z_t$ maps the robot's state to a set of ranges. 

### Measurement Likelihood
For the Particle Filter, the weight of each particle is updated based on the likelihood of the observed LiDAR ranges $z_{real}$ given the simulated ranges $z_{sim}$ from the particle's hypothetical pose. Assume a Gaussian noise model for each beam. 

To handle outliers and non-Gaussian artifacts (like dynamic obstacles not in the map), the implementation includes:
+ Beam Subsampling: Reducing $M$ to improve computational frequency.
+ Error Capping: Innovation is capped at $5\sigma$ to prevent a single noisy beam from collapsing the particle distribution.
+ Numerical Jacobians (EKF): Since the map is discrete, the measurement Jacobian $H$ is computed via central difference. 

## Performance Benchmark
The simulation tracks the Euclidean distance between the filter estimate and the ground truth ($RMSE$) and the computational time required for each update ($Latency$).

| Filter Type | RMSE (m) | Avg Comp Time (ms) |
|:------------|---------:|-------------------:|
| Particle    | 0.6809   | 583.83             |
| EKF         | 6.2625   | 1.42               |
| UKF         | 6.0814   | 2.22               |

**Note:** While the Particle Filter is the most computationally expensive(utilizes 3,000 particles and adaptive injection), it is the only filter capable of recovering from the kidnapped robot scenario without manual re-initialization, where EKF and UKF diverge due to their unimodal Gaussian assumption.
