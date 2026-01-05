"""
Test HOM Mismatch

Verifies that the success probability decreases with temporal/polarization mismatch.
"""

import numpy as np
from pytest import approx, skip

from atom_sim.config import TimeGrid, EmitParams, QFCParams, FiberParams, DetParams
from atom_sim.simulation import run_simulation


def test_hom_mismatch_decreases_success():
    """
    Test that p_succ decreases as mismatch increases.

    Vary the relative delay (delta_bins) between arms and verify
    that the success probability decreases.
    """
    skip("Skipping full simulation test in CI - requires TeNPy")

    # Time grid
    time_grid = TimeGrid(dt=0.1, N=10)

    # Emission parameters (same for both arms)
    Alpha = np.array([[1, 0], [0, 1]])
    emit_params = EmitParams(
        gamma_A=0.5,
        gamma_B=0.5,
        Alpha_A=Alpha,
        Alpha_B=Alpha,
    )

    # QFC and fiber (no loss)
    I_2x2 = np.eye(2)
    qfc_params = QFCParams(theta_H=np.pi/2, theta_V=np.pi/2)
    det_params = DetParams(
        eta_det=1.0,
        p_dark=0.0,
        success_patterns=[(1, 0, 0, 1), (0, 1, 1, 0)],
    )

    # Test different mismatch values (delta_bins)
    delta_bins_values = [0, 1, 2]
    p_succ_values = []

    for delta_bins in delta_bins_values:
        fiber_params = FiberParams(
            eta_fiber_A=1.0,
            eta_fiber_B=1.0,
            Jones_A=I_2x2,
            Jones_B=I_2x2,
            delta_bins=delta_bins,
        )

        result = run_simulation(
            time_grid=time_grid,
            emit_params=emit_params,
            qfc_params=qfc_params,
            fiber_params=fiber_params,
            det_params=det_params,
            n_traj=100,
            chi_max=50,
        )

        p_succ_values.append(result.p_succ)
        print(f"delta_bins={delta_bins}: p_succ={result.p_succ:.4f}")

    # Check that p_succ decreases with delta_bins
    # Note: due to statistical noise, this might not be strictly monotonic
    # for small sample sizes, but the trend should hold
    assert p_succ_values[0] >= p_succ_values[-1], \
        f"p_succ should decrease with mismatch: {p_succ_values}"


def test_polarization_mismatch_decreases_fidelity():
    """
    Test that fidelity decreases with polarization mismatch.

    Vary the relative Jones rotation between arms and verify
    that the conditional fidelity decreases.
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

    # QFC (perfect)
    qfc_params = QFCParams(theta_H=np.pi/2, theta_V=np.pi/2)
    det_params = DetParams(
        eta_det=1.0,
        p_dark=0.0,
        success_patterns=[(1, 0, 0, 1), (0, 1, 1, 0)],
    )

    # Test different polarization rotations
    angles = [0, np.pi/8, np.pi/4]  # 0, 22.5, 45 degrees
    F_values = []

    for angle in angles:
        # Rotate Jones matrix for arm B
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        J_rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        I_2x2 = np.eye(2)
        fiber_params = FiberParams(
            eta_fiber_A=1.0,
            eta_fiber_B=1.0,
            Jones_A=I_2x2,
            Jones_B=J_rot,
        )

        result = run_simulation(
            time_grid=time_grid,
            emit_params=emit_params,
            qfc_params=qfc_params,
            fiber_params=fiber_params,
            det_params=det_params,
            n_traj=100,
            chi_max=50,
        )

        F_values.append(result.F_cond)
        print(f"angle={angle:.3f}: F={result.F_cond:.4f}")

    # Check that fidelity decreases with rotation angle
    # F should be highest at angle=0, lowest at angle=pi/4
    assert F_values[0] >= F_values[-1], \
        f"Fidelity should decrease with polarization mismatch: {F_values}"


if __name__ == '__main__':
    test_hom_mismatch_decreases_success()
    test_polarization_mismatch_decreases_fidelity()
    print("HOM mismatch tests passed!")
