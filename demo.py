#!/usr/bin/env python3
"""
Particle Filter vs Extended Kalman Filter Localization Demo

This demo showcases the advantages of Particle Filters over EKF for robot localization
in challenging scenarios with multimodal distributions and symmetric environments.

Two demonstrations are included:
1. EKF Failure Case: Shows EKF failing with symmetric/ambiguous environments
2. Baseline Comparison: Compares both filters on a figure-8 trajectory

Expected Runtime: 5-8 minutes total
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Suppress pygame welcome message if present
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

def print_banner(text):
    """Print a formatted banner"""
    width = 70
    print("\n" + "=" * width)
    print(text.center(width))
    print("=" * width + "\n")

def run_ekf_failure_demo():
    """Run the EKF failure case demonstration"""
    print_banner("DEMO 1: EKF FAILURE IN SYMMETRIC ENVIRONMENT")
    print("This demo shows how EKF fails when faced with symmetric/ambiguous")
    print("initial conditions. The EKF mean will drift into obstacles while")
    print("the Particle Filter maintains valid hypotheses.")
    print("\nExpected runtime: 2-3 minutes")
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
    print_banner("DEMO 2: BASELINE COMPARISON ON FIGURE-8 TRAJECTORY")
    print("This demo compares Particle Filter and EKF performance on a complex")
    print("figure-8 trajectory, measuring accuracy and computational efficiency.")
    print("\nExpected runtime: 2-3 minutes")
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

def analyze_and_visualize_results():
    """Analyze results and create visualization plots"""
    print_banner("ANALYZING RESULTS")
    
    # Check if baseline data exists
    if not Path("baseline_filter_data.npz").exists():
        print("Warning: baseline_filter_data.npz not found. Skipping analysis.")
        return
    
    # Load data
    data = np.load("baseline_filter_data.npz")
    true_poses = data['true_poses']
    pf_estimates = data['pf_estimates']
    ekf_estimates = data['ekf_estimates']
    pf_times = data['pf_times']
    ekf_times = data['ekf_times']
    
    # Calculate errors
    pf_errors = np.sqrt(
        (pf_estimates[:, 0] - true_poses[:, 0])**2 +
        (pf_estimates[:, 1] - true_poses[:, 1])**2
    )
    ekf_errors = np.sqrt(
        (ekf_estimates[:, 0] - true_poses[:, 0])**2 +
        (ekf_estimates[:, 1] - true_poses[:, 1])**2
    )
    
    # Print statistics
    print("\nPOSITION ERROR STATISTICS:")
    print("-" * 70)
    print(f"{'Metric':<30} {'Particle Filter':>20} {'EKF':>20}")
    print("-" * 70)
    print(f"{'Mean Error (m)':<30} {np.mean(pf_errors):>20.4f} {np.mean(ekf_errors):>20.4f}")
    print(f"{'Median Error (m)':<30} {np.median(pf_errors):>20.4f} {np.median(ekf_errors):>20.4f}")
    print(f"{'Max Error (m)':<30} {np.max(pf_errors):>20.4f} {np.max(ekf_errors):>20.4f}")
    print(f"{'Std Dev (m)':<30} {np.std(pf_errors):>20.4f} {np.std(ekf_errors):>20.4f}")
    print("-" * 70)
    
    print("\nCOMPUTATIONAL PERFORMANCE:")
    print("-" * 70)
    print(f"{'Metric':<30} {'Particle Filter':>20} {'EKF':>20}")
    print("-" * 70)
    print(f"{'Mean Update Time (ms)':<30} {np.mean(pf_times)*1000:>20.2f} {np.mean(ekf_times)*1000:>20.2f}")
    print(f"{'Total Time (s)':<30} {np.sum(pf_times):>20.2f} {np.sum(ekf_times):>20.2f}")
    print("-" * 70)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Trajectories
    ax = axes[0, 0]
    ax.plot(true_poses[:, 0], true_poses[:, 1], 'k-', linewidth=2, label='Ground Truth', alpha=0.7)
    ax.plot(pf_estimates[:, 0], pf_estimates[:, 1], 'r-', linewidth=1, label='Particle Filter', alpha=0.6)
    ax.plot(ekf_estimates[:, 0], ekf_estimates[:, 1], 'b-', linewidth=1, label='EKF', alpha=0.6)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Trajectory Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Plot 2: Position Errors Over Time
    ax = axes[0, 1]
    steps = np.arange(len(pf_errors))
    ax.plot(steps, pf_errors, 'r-', linewidth=1, label='Particle Filter', alpha=0.7)
    ax.plot(steps, ekf_errors, 'b-', linewidth=1, label='EKF', alpha=0.7)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Position Error (m)')
    ax.set_title('Position Error Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Error Distribution
    ax = axes[1, 0]
    bins = np.linspace(0, max(np.max(pf_errors), np.max(ekf_errors)), 50)
    ax.hist(pf_errors, bins=bins, alpha=0.6, label='Particle Filter', color='red', edgecolor='black')
    ax.hist(ekf_errors, bins=bins, alpha=0.6, label='EKF', color='blue', edgecolor='black')
    ax.set_xlabel('Position Error (m)')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Computation Time Comparison
    ax = axes[1, 1]
    time_windows = 50
    pf_rolling = [np.mean(pf_times[max(0, i-time_windows):i+1])*1000 
                  for i in range(len(pf_times))]
    ekf_rolling = [np.mean(ekf_times[max(0, i-time_windows):i+1])*1000 
                   for i in range(len(ekf_times))]
    ax.plot(steps, pf_rolling, 'r-', linewidth=1, label='Particle Filter', alpha=0.7)
    ax.plot(steps, ekf_rolling, 'b-', linewidth=1, label='EKF', alpha=0.7)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Update Time (ms, rolling avg)')
    ax.set_title('Computational Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('localization_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: localization_comparison.png")
    
    # Show plot briefly
    plt.show(block=False)
    plt.pause(3)
    plt.close()

def print_summary():
    """Print final summary"""
    print_banner("DEMO COMPLETE - SUMMARY")
    print("This demonstration showed two key advantages of Particle Filters:")
    print()
    print("1. MULTIMODAL DISTRIBUTIONS:")
    print("   - EKF assumes unimodal Gaussian distributions")
    print("   - When faced with symmetric/ambiguous scenarios, EKF's mean")
    print("     can drift into physically impossible locations")
    print("   - Particle Filter maintains multiple hypotheses and correctly")
    print("     disambiguates as new measurements arrive")
    print()
    print("2. ROBUST PERFORMANCE:")
    print("   - Both filters perform well in standard scenarios")
    print("   - Particle Filter provides more robust estimates in complex")
    print("     environments with ambiguities")
    print("   - Trade-off: PF has higher computational cost but better")
    print("     handling of non-Gaussian, multimodal distributions")
    print()
    print("=" * 70)
    print("Output files generated:")
    print("  - baseline_filter_data.npz (raw data)")
    print("  - localization_comparison.png (visualization)")
    print("=" * 70 + "\n")

def main():
    """Main demo function"""
    print_banner("PARTICLE FILTER vs EKF LOCALIZATION DEMO")
    print("Expected Total Runtime: 15-20 minutes")
    print()
    print("This demo will run two experiments:")
    print("  1. EKF Failure Case")
    print("  2. Baseline Comparison")
    
    # Run Demo 1: EKF Failure
    success1 = run_ekf_failure_demo()
    
    if success1:
        print("\n" + "=" * 70)
        print("Demo 1 complete. Continuing to Demo 2...")
    
    # Run Demo 2: Baseline Comparison
    success2 = run_baseline_comparison()
    
    if success2:
        print("\n" + "=" * 70)
        print("Demo 2 complete. Analyzing results...")
        
        # Analyze and visualize
        try:
            analyze_and_visualize_results()
        except Exception as e:
            print(f"Warning: Could not generate analysis plots: {e}")
    
    print_summary()
    
    if success1 and success2:
        print("All demos completed successfully!")
    else:
        print("Some demos encountered issues. Check the output above.")

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