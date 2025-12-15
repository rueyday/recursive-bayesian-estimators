#!/bin/bash

# Install script for Particle Filter vs EKF Localization Demo
# Assumes pybullet, Python 3, numpy, and matplotlib are already installed

echo "=========================================="
echo "Installing Required Packages"
echo "=========================================="

# Install scipy (for particle filter resampling)
echo "Installing scipy..."
pip install scipy

# Install any missing pybullet dependencies
echo "Checking pybullet installation..."
pip install --upgrade pybullet

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo "Run the demo with: python3 demo.py"
echo ""