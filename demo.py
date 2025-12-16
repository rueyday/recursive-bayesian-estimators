#!/usr/bin/env python3
import os
import sys
import numpy as np
from pathlib import Path

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

def print_banner(text):
    """Print a formatted banner"""
    width = 70
    print("\n" + "=" * width)
    print(text.center(width))
    print("=" * width + "\n")

def run_ekf_failure_demo():
    """Run the EKF failure case demonstration"""
    print_banner("DEMO 2: EKF FAILURE IN SYMMETRIC ENVIRONMENT")
    print("This demo shows how EKF fails when faced with symmetric/ambiguous")
    print("initial conditions. The EKF mean will drift into obstacles while")
    print("the Particle Filter maintains valid hypotheses.")
    print("\nExpected runtime: 15 minutes")
    print("=" * 70 + "\n")
    
    # Import and run the EKF failure test
    try:
        # Need to import pybullet to disconnect after
        import pybullet as p
        import ekf_failure_test
        
        # Run the test (it will disconnect internally)
        client_id = ekf_failure_test.main()
        
        # Give PyBullet a moment to fully disconnect
        import time
        time.sleep(1)
        
        # Ensure no connections are left open
        try:
            p.disconnect()
        except:
            pass
        
        return True
    except Exception as e:
        print(f"Error running EKF failure demo: {e}")
        import traceback
        traceback.print_exc()
        # Try to clean up any connections
        try:
            import pybullet as p
            p.disconnect()
        except:
            pass
        return False

def run_baseline_comparison():
    """Run the baseline comparison demonstration"""
    print_banner("DEMO 1: BASELINE COMPARISON ON FIGURE-8 TRAJECTORY")
    print("This demo compares Particle Filter and EKF performance on a complex")
    print("figure-8 trajectory, measuring accuracy and computational efficiency.")
    print("\nExpected runtime: 8 minutes")
    print("=" * 70 + "\n")
    
    # Import and run baseline test
    try:
        import pybullet as p
        import baseline_test
        
        # Run the test (it will disconnect internally)
        client_id = baseline_test.main()
        
        # Give PyBullet a moment to fully disconnect
        import time
        time.sleep(1)
        
        # Ensure no connections are left open
        try:
            p.disconnect()
        except:
            pass
        
        return True
    except Exception as e:
        print(f"Error running baseline comparison: {e}")
        import traceback
        traceback.print_exc()
        # Try to clean up any connections
        try:
            import pybullet as p
            p.disconnect()
        except:
            pass
        return False

def main():
    print_banner("PARTICLE FILTER vs EKF LOCALIZATION DEMO")
    print("This demo will run two experiments: Total run time: 15-25 min")
    print("  1. EKF Failure Case        Time: 10-15 min")
    print("  2. Baseline Comparison     Time: 5-10 min")

    success1 = run_baseline_comparison()

    if success1:
        print("\n" + "=" * 70)
        print("Demo 1 complete")
    
    success2 = run_ekf_failure_demo()
    
    if success2:
        print("\n" + "=" * 70)
        print("Demo 2 complete")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)