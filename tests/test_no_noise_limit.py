"""
Test No-Noise Limit

Verifies that in the ideal case (no noise, perfect alignment),
the conditional fidelity F -> 1.
"""

import numpy as np
from pytest import approx, skip

from atom_sim.config import TimeGrid, EmitParams, QFCParams, FiberParams, DetParams
from atom_sim.simulation import run_simulation


def test_no_noise_fidelity():
    """
    Test that F -> 1 in the no-noise limit.

    In the ideal case:
    - No loss (eta = 1)
    - No dark counts (p_dark = 0)
    - Perfect detection (eta_det = 1)
    - Aligned Jones matrices
    - Orthogonal Alpha matrices

    The output should be a high-fidelity Bell state.
    """
    # This is a full integration test - may be slow
    skip("Skipping full simulation test in CI - requires TeNPy")

    # Time grid
    time_grid = TimeGrid(dt=0.1, N=10)

    # Emission: orthogonal polarization mapping
    Alpha = np.array([[1, 0], [0, 1]])  # σ+ -> H, σ- -> V (orthogonal)
    emit_params = EmitParams(
        gamma_A=0.5,
        gamma_B=0.5,
        Alpha_A=Alpha,
        Alpha_B=Alpha,
    )

    # QFC: perfect conversion
    qfc_params = QFCParams(theta_H=np.pi/2, theta_V=np.pi/2)

    # Fiber: no loss, identical Jones matrices
    I_2x2 = np.eye(2)
    fiber_params = FiberParams(
        eta_fiber_A=1.0,
        eta_fiber_B=1.0,
        Jones_A=I_2x2,
        Jones_B=I_2x2,
    )

    # Detection: perfect, no dark counts
    det_params = DetParams(
        eta_det=1.0,
        p_dark=0.0,
        success_patterns=[(1, 0, 0, 1), (0, 1, 1, 0)],  # Partial BSM
    )

    # Run simulation
    result = run_simulation(
        time_grid=time_grid,
        emit_params=emit_params,
        qfc_params=qfc_params,
        fiber_params=fiber_params,
        det_params=det_params,
        n_traj=100,
        chi_max=50,
    )

    # Check that fidelity is high (close to 1)
    print(f"p_succ = {result.p_succ:.4f} +/- {result.p_succ_stderr:.4f}")
    print(f"F_cond = {result.F_cond:.4f} +/- {result.F_cond_stderr:.4f}")

    # In the ideal case, F should be close to 1
    # Allow some tolerance for numerical errors and finite statistics
    assert result.F_cond > 0.9, f"Fidelity too low: {result.F_cond}"


if __name__ == '__main__':
    test_no_noise_fidelity()
    print("No-noise limit test passed!")
