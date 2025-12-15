# Particle Filter vs EKF Localization Demo

## Overview
This demo showcases the advantages of Particle Filters over Extended Kalman Filters (EKF) for robot localization in challenging scenarios.

## Installation & Running

### Quick Start
```bash
./install.sh
python3 demo.py
```

### Expected Runtime
**Total: 5-8 minutes**
- Demo 1 (EKF Failure Case): 2-3 minutes
- Demo 2 (Baseline Comparison): 2-3 minutes

## Demo Components

### Demo 1: EKF Failure in Symmetric Environment
- **Problem**: Symmetric corridor with ambiguous initial conditions
- **Setup**: Robot could be at two mirror locations
- **Result**: 
  - EKF mean averages the two hypotheses → drifts into walls
  - Particle Filter maintains both hypotheses → correctly disambiguates

**Watch for:**
- RED arrow = Particle Filter estimate (tracks correctly)
- BLUE arrow = EKF estimate (may go into walls)
- GREEN dots = Particle clusters (watch them separate)
- "INVALID!" warnings when EKF is in obstacle

### Demo 2: Baseline Comparison on Figure-8 Trajectory
- **Problem**: Complex figure-8 navigation
- **Setup**: Both filters start with same initial belief
- **Result**: 
  - Quantitative comparison of accuracy and computation time
  - Generates plots and statistics

## Output Files
- `baseline_filter_data.npz` - Raw experimental data
- `localization_comparison.png` - Visualization plots showing:
  - Trajectory comparison
  - Position errors over time
  - Error distributions
  - Computational performance

## Requirements
- Python 3.x
- pybullet (pre-installed)
- numpy (pre-installed)
- matplotlib (pre-installed)
- scipy (installed by install.sh)

## Key Takeaways

### Particle Filter Advantages:
1. **Handles Multimodal Distributions**: Can maintain multiple hypotheses simultaneously
2. **No Gaussian Assumption**: Works with arbitrary probability distributions
3. **Robust to Ambiguity**: Correctly disambiguates in symmetric environments

### EKF Limitations:
1. **Unimodal Only**: Assumes single Gaussian distribution
2. **Mean Can Be Invalid**: Averaging hypotheses can create impossible poses
3. **Fails in Ambiguous Scenarios**: Cannot represent multiple equally-likely locations

### Trade-offs:
- PF: Higher computational cost, better accuracy in complex scenarios
- EKF: Lower computational cost, sufficient for well-localized robots

## Troubleshooting

If the demo doesn't run:
1. Ensure all required files are present
2. Check that pybullet is properly installed
3. Make sure you have a display (or use virtual display on headless systems)
4. Run `./install.sh` again to reinstall dependencies

## Files Structure
```
.
├── install.sh                    # Installation script
├── demo.py                       # Main demo orchestrator
├── ekf_failure_test.py          # Demo 1 implementation
├── baseline_test.py             # Demo 2 implementation
├── particle_model.py            # Particle Filter implementation
├── kalman_model.py              # EKF implementation
├── lidar_utils.py               # LiDAR simulation utilities
├── utils.py                     # Environment loading utilities
├── pybullet_tools/              # PyBullet helper functions            
├── symmetric_corridor.json      # Environment JSON files
└── figure8_env.json
```