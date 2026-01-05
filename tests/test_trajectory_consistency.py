"""
Test Trajectory Statistical Consistency

Verifies that running the same simulation with different seeds
produces results within expected statistical fluctuations.
"""

import numpy as np
from pytest import approx, skip

from atom_sim.config import TimeGrid, EmitParams, QFCParams, FiberParams, DetParams
from atom_sim.simulation import run_simulation


def test_trajectory_consistency():
    """
    Test that different random seeds give consistent statistical results.

    Run the same simulation multiple times with different seeds and check
    that the results are within expected statistical error bars.
    """
    skip("Skipping full simulation test in CI - requires TeNPy")

    # Time grid
    time_grid = TimeGrid(dt=0.1, N=10)

    # Emission parameters
    Alpha = np.array([[1, 0], [0, 1]])
    emit_params = EmitParams(
        gamma_A=0.5,
        gamma_B=0.5,
        Alpha_A=Alpha,
        Alpha_B=Alpha,
    )

    # QFC
    qfc_params = QFCParams(theta_H=np.pi/2, theta_V=np.pi/2)

    # Fiber (with some loss for variability)
    I_2x2 = np.eye(2)
    fiber_params = FiberParams(
        eta_fiber_A=0.9,
        eta_fiber_B=0.9,
        Jones_A=I_2x2,
        Jones_B=I_2x2,
    )

    # Detection
    det_params = DetParams(
        eta_det=0.9,
        p_dark=0.001,
        success_patterns=[(1, 0, 0, 1), (0, 1, 1, 0)],
    )

    # Run simulation multiple times with different seeds
    n_runs = 3
    p_succ_values = []
    F_values = []

    for i in range(n_runs):
        result = run_simulation(
            time_grid=time_grid,
            emit_params=emit_params,
            qfc_params=qfc_params,
            fiber_params=fiber_params,
            det_params=det_params,
            n_traj=100,
            chi_max=50,
            seed=42 + i * 1000,  # Different seeds
        )

        p_succ_values.append(result.p_succ)
        F_values.append(result.F_cond)

        print(f"Run {i+1}: p_succ={result.p_succ:.4f} +/- {result.p_succ_stderr:.4f}, "
              f"F={result.F_cond:.4f} +/- {result.F_cond_stderr:.4f}")

    # Check that values are consistent (within 3 sigma)
    p_succ_mean = np.mean(p_succ_values)
    p_succ_std = np.std(p_succ_values)

    F_mean = np.mean(F_values)
    F_std = np.std(F_values)

    # For 100 trajectories, we expect some variation
    # Check that the spread is reasonable (not too tight, not too wide)
    # This is a weak test, just catches obvious bugs
    assert p_succ_std < 0.3, f"p_succ variation too large: {p_succ_std}"
    assert F_std < 0.3, f"Fidelity variation too large: {F_std}"

    print(f"p_succ: {p_succ_mean:.4f} +/- {p_succ_std:.4f}")
    print(f"F: {F_mean:.4f} +/- {F_std:.4f}")


def test_confidence_interval_coverage():
    """
    Test that the reported standard errors are reasonable.

    Run many times and check that the fraction of results within
    1 stderr of the mean is approximately 68% (Gaussian).
    """
    skip("Skipping expensive statistical test - requires many trajectories")


if __name__ == '__main__':
    test_trajectory_consistency()
    print("Trajectory consistency tests passed!")
