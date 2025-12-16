#!/usr/bin/env python3
import os
import sys
import numpy as np
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
    print_banner("DEMO 2: EKF FAILURE IN SYMMETRIC ENVIRONMENT")
    print("This demo shows how EKF fails when faced with symmetric/ambiguous")
    print("initial conditions. The EKF mean will drift into obstacles while")
    print("the Particle Filter maintains valid hypotheses.")
    print("\nExpected runtime: 10 minutes")
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
    print("\nExpected runtime: 10 minutes")
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
    """Analyze results and print statistics"""
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
    
    # Calculate position errors
    pf_pos_errors = np.sqrt(
        (pf_estimates[:, 0] - true_poses[:, 0])**2 +
        (pf_estimates[:, 1] - true_poses[:, 1])**2
    )
    ekf_pos_errors = np.sqrt(
        (ekf_estimates[:, 0] - true_poses[:, 0])**2 +
        (ekf_estimates[:, 1] - true_poses[:, 1])**2
    )
    
    # Calculate orientation errors
    pf_ori_errors = np.abs(pf_estimates[:, 2] - true_poses[:, 2])
    pf_ori_errors = np.minimum(pf_ori_errors, 2*np.pi - pf_ori_errors)  # Wrap to [-pi, pi]
    
    ekf_ori_errors = np.abs(ekf_estimates[:, 2] - true_poses[:, 2])
    ekf_ori_errors = np.minimum(ekf_ori_errors, 2*np.pi - ekf_ori_errors)
    
    # Calculate RMSE
    pf_pos_rmse = np.sqrt(np.mean(pf_pos_errors**2))
    ekf_pos_rmse = np.sqrt(np.mean(ekf_pos_errors**2))
    pf_ori_rmse = np.sqrt(np.mean(pf_ori_errors**2))
    ekf_ori_rmse = np.sqrt(np.mean(ekf_ori_errors**2))
    
    # Print Test 1 Results
    print("\n" + "=" * 70)
    print("TEST 1: BASELINE PERFORMANCE COMPARISON")
    print("=" * 70)
    print("\nExperimental Setup:")
    print("  - Trajectory: Figure-eight path with 1200 time steps")
    print("  - Initial pose: (-2.5, 6.0, π/2)")
    print("  - Motion noise: σ_x = σ_y = 0.02 m, σ_θ = 0.01 rad")
    print("  - Measurement noise: σ_r = 0.2 m")
    print("  - PF: 500 particles, initial σ = 0.5 m (position), 0.3 rad (orientation)")
    print("  - EKF: Initial covariance P = 0.5I")
    
    print("\n" + "-" * 70)
    print("Table 1: Baseline Performance Comparison")
    print("-" * 70)
    print(f"{'Filter':<25} {'Position RMSE':<18} {'Orientation RMSE':<20} {'Computation Time':<15}")
    print("-" * 70)
    print(f"{'Particle Filter':<25} {pf_pos_rmse:>10.3f} m      {pf_ori_rmse:>10.3f} rad        {np.mean(pf_times)*1000:>8.2f} ms")
    print(f"{'Extended Kalman Filter':<25} {ekf_pos_rmse:>10.3f} m      {ekf_ori_rmse:>10.3f} rad        {np.mean(ekf_times)*1000:>8.2f} ms")
    print("-" * 70)
    
    print("\nKey Findings:")
    if ekf_pos_rmse < pf_pos_rmse:
        print(f"  ✓ EKF achieved {((pf_pos_rmse - ekf_pos_rmse)/pf_pos_rmse*100):.1f}% lower position RMSE")
    else:
        print(f"  ✓ PF achieved {((ekf_pos_rmse - pf_pos_rmse)/ekf_pos_rmse*100):.1f}% lower position RMSE")
    
    if pf_ori_rmse < ekf_ori_rmse:
        print(f"  ✓ PF achieved {((ekf_ori_rmse - pf_ori_rmse)/ekf_ori_rmse*100):.1f}% lower orientation RMSE")
    else:
        print(f"  ✓ EKF achieved {((pf_ori_rmse - ekf_ori_rmse)/pf_ori_rmse*100):.1f}% lower orientation RMSE")
    
    speedup = np.mean(pf_times) / np.mean(ekf_times)
    print(f"  ✓ EKF is {speedup:.1f}x faster than PF")
    
    print("\nInterpretation:")
    print("  Under nominal conditions with a good initial estimate, the EKF performs")
    print("  efficiently with competitive accuracy. The PF shows strength in orientation")
    print("  estimation during high-curvature segments, demonstrating its ability to")
    print("  handle nonlinear belief distributions at the cost of computation.")
    
    # Save summary to file
    with open("test1_results.txt", "w") as f:
        f.write("TEST 1: BASELINE PERFORMANCE COMPARISON\n")
        f.write("=" * 70 + "\n\n")
        f.write("Table 1: Baseline Performance Comparison\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Filter':<25} {'Position RMSE':<18} {'Orientation RMSE':<20} {'Computation Time':<15}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Particle Filter':<25} {pf_pos_rmse:>10.3f} m      {pf_ori_rmse:>10.3f} rad        {np.mean(pf_times)*1000:>8.2f} ms\n")
        f.write(f"{'Extended Kalman Filter':<25} {ekf_pos_rmse:>10.3f} m      {ekf_ori_rmse:>10.3f} rad        {np.mean(ekf_times)*1000:>8.2f} ms\n")
        f.write("-" * 70 + "\n")
    
    print("\n✓ Results saved to test1_results.txt")

def print_summary():
    """Print final summary"""
    print_banner("DEMO COMPLETE - SUMMARY")
    print("This demonstration showed two key advantages of Particle Filters:")
    print()
    print("1. MULTIMODAL DISTRIBUTIONS (Test 2):")
    print("   - EKF assumes unimodal Gaussian distributions")
    print("   - When faced with symmetric/ambiguous scenarios, EKF's mean")
    print("     can drift into physically impossible locations")
    print("   - Particle Filter maintains multiple hypotheses and correctly")
    print("     disambiguates as new measurements arrive")
    print()
    print("2. EFFICIENCY VS ROBUSTNESS TRADE-OFF (Test 1):")
    print("   - EKF performs well under nominal conditions with lower cost")
    print("   - Particle Filter provides more robust estimates in complex")
    print("     environments with ambiguities")
    print("   - Trade-off: PF has higher computational cost but better")
    print("     handling of non-Gaussian, multimodal distributions")
    print()
    print("=" * 70)
    print("CRITICAL INSIGHT:")
    print("The choice between Kalman and particle filtering depends not on")
    print("computational resources alone, but on whether the posterior can")
    print("be well-approximated as unimodal Gaussian:")
    print("  • Unimodal case (good initial estimate) → EKF excels")
    print("  • Multimodal case (global localization, symmetry) → PF necessary")
    print("=" * 70)
    print()
    print("Output files generated:")
    print("  - test1_results.txt (Baseline performance data)")
    print("  - test2_results.txt (Failure case data)")
    print("  - baseline_filter_data.npz (Raw experimental data)")
    print("=" * 70 + "\n")

def main():
    print_banner("PARTICLE FILTER vs EKF LOCALIZATION DEMO")
    print("This demo will run two experiments:")
    print("  1. EKF Failure Case")
    print("  2. Baseline Comparison")

    success1 = run_baseline_comparison()

    if success1:
        print("\n" + "=" * 70)
        print("Demo 1 complete")
    
    success2 = run_ekf_failure_demo()
    
    if success2:
        print("\n" + "=" * 70)
        print("Demo 2 complete")
        
        # try:
        #     analyze_and_visualize_results()
        # except Exception as e:
        #     print(f"Warning: Could not generate analysis: {e}")
    
    print_summary()
    
    # if success1 and success2:
    #     print("All demos completed successfully!")
    # else:
    #     print("Some demos encountered issues.")

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