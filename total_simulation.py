"""
Total Simulation: Dual-Atom Emission -> Time-Bin Wave Packets

This is the first stage of the total simulation:
- Two atoms (A and B) in excited state
- Emission to time bins (780nm only, no QFC yet)
- Final state ready for next gate (BSM at station)

Chain structure (dual-atom layout):
    atomA, atomB, A1, B1, A2, B2, ..., AN, BN

After SWAP conveyor belt:
    A1, B1, A2, B2, ..., AN, BN, atomA, atomB

This allows A_n and B_n to be adjacent for BSM operations.
"""

import sys
from pathlib import Path
import numpy as np
from typing import Optional, Tuple

# Add project root to path (for running as standalone script)
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from atom_sim.config import TimeGrid, EmitParams
from atom_sim.simulation import run_emission_only, EmissionResult
from atom_sim.visualization import plot_emission_with_atomic_evolution


# Backward compatibility: alias for EmissionResult
DualAtomEmissionResult = EmissionResult


def run_dual_atom_emission(
    n_bins: int = 200,
    dt_ns: float = 0.2,
    chi_max: int = 50,
    Alpha_A: Optional[np.ndarray] = None,
    Alpha_B: Optional[np.ndarray] = None,
    gamma_peak_A: float = 0.2,
    gamma_peak_B: float = 0.2,
    t0_A: Optional[float] = None,
    t0_B: Optional[float] = None,
    sigma: float = 12.0,
    verbose: bool = True,
) -> EmissionResult:
    """
    Run dual-atom emission simulation using SWAP conveyor belt protocol.

    This function creates the necessary parameters and calls the simulation layer.

    Parameters
    ----------
    n_bins : int
        Number of time bins
    dt_ns : float
        Time step in nanoseconds
    chi_max : int
        Maximum bond dimension for MPS
    Alpha_A : np.ndarray, optional
        2x2 polarization matrix for atom A
    Alpha_B : np.ndarray, optional
        2x2 polarization matrix for atom B
    gamma_peak_A : float
        Peak emission rate for atom A
    gamma_peak_B : float
        Peak emission rate for atom B
    t0_A : float, optional
        Peak time for atom A (ns)
    t0_B : float, optional
        Peak time for atom B (ns)
    sigma : float
        Width parameter for Gaussian emission profile (ns)
    verbose : bool
        Whether to print progress information

    Returns
    -------
    EmissionResult
        Container with simulation results
    """
    # Create time grid
    time_grid = TimeGrid(dt=dt_ns * 1e-9, N=n_bins)  # Convert ns to seconds
    t = time_grid.t

    # Set default peak times to center of time window
    if t0_A is None:
        t0_A = n_bins * dt_ns / 2
    if t0_B is None:
        t0_B = n_bins * dt_ns / 2

    # Create Gaussian emission rate functions
    def gamma_A_func(t_sec):
        t_ns = t_sec * 1e9  # Convert to ns for calculation
        return gamma_peak_A * np.exp(-0.5 * ((t_ns - t0_A) / sigma) ** 2)

    def gamma_B_func(t_sec):
        t_ns = t_sec * 1e9
        return gamma_peak_B * np.exp(-0.5 * ((t_ns - t0_B) / sigma) ** 2)

    # Set default Alpha matrices (identity = no polarization mixing)
    if Alpha_A is None:
        Alpha_A = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    if Alpha_B is None:
        Alpha_B = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)

    # Create emission parameters
    emit_params = EmitParams(
        gamma_A=gamma_A_func,
        gamma_B=gamma_B_func,
        Alpha_A=Alpha_A,
        Alpha_B=Alpha_B,
    )

    # Run simulation using the simulation layer
    result = run_emission_only(
        time_grid=time_grid,
        emit_params=emit_params,
        chi_max=chi_max,
        verbose=verbose,
    )

    return result


def main():
    """Main function for testing the dual-atom emission simulation."""
    print("Running dual-atom emission simulation...")

    # Run simulation with default parameters
    # Use n_bins=200 for production, n_bins=50 for quick testing
    result = run_dual_atom_emission(
        n_bins=200,
        dt_ns=0.2,
        chi_max=50,
        gamma_peak_A=0.2,
        gamma_peak_B=0.2,
        sigma=12.0,
        verbose=True,
    )

    # Visualize results
    print("\nGenerating visualization...")
    plot_emission_with_atomic_evolution(result, save_path="dual_atom_emission.png")

    # Show that we can access bins for next stage
    print("\n" + "=" * 70)
    print("Ready for next stage (BSM)")
    print("=" * 70)
    print(f"Bin indices for A_5, B_5: {result.get_bin_indices(4)}")
    print(f"Atom site indices: A={result.get_atom_site_indices()[0]}, B={result.get_atom_site_indices()[1]}")

    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    main()
