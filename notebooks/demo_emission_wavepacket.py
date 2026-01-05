"""
Demo: Atom Emission → Time-Bin Wave Packet Simulation

This demo simulates the complete process:
1. Atoms start in excited state |e>
2. For each time bin n:
   - Emission gate couples atoms to 780nm photons
   - QFC converts 780nm → 1517nm
3. Result: Time-bin wave packet with specific shape

Physics:
---------
Atom (3-level):
  |e>: excited state (5P_{3/2}, F'=0, m_F=0)
  |0>: ground state (5S_{1/2}, F=1, m_F=+1)
  |1>: ground state (5S_{1/2}, F=1, m_F=-1)

Selection rules:
  |e> → |0>: sigma+ photon (H polarization)
  |e> → |1>: sigma- photon (V polarization)

780nm subspace (3D): |vac>, |H>, |V>
1517nm subspace (6D): |vac>, |H>, |V>, |2H>, |2V>, |HV>
Bin space: 780(3D) × 1517(6D) = 18D
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atom_sim.core.mps import MPSState
from atom_sim.config import TimeGrid, EmitParams, QFCParams
from atom_sim.physics.gates import emission_gate, qfc_gate
from atom_sim.visualization import (
    plot_intensity_envelope,
    plot_single_photon_prob,
    plot_wavepacket,
    extract_wavepacket,
)


def gaussian_pulse(t: np.ndarray, t0: float, sigma: float) -> np.ndarray:
    """
    Gaussian emission rate profile.

    Parameters
    ----------
    t : np.ndarray
        Time points
    t0 : float
        Peak time
    sigma : float
        Width (standard deviation)

    Returns
    -------
    np.ndarray
        Gamma values at each time point
    """
    return np.exp(-0.5 * ((t - t0) / sigma) ** 2)


def simulate_emission_wavepacket(
    n_bins: int = 200,
    dt: float = 0.1,
    t0: float = 10.0,
    sigma: float = 3.0,
    theta_H: float = np.pi / 4,  # QFC angle for H
    theta_V: float = np.pi / 4,  # QFC angle for V
    alpha_H_plus: float = 1.0 / np.sqrt(2),
    alpha_H_minus: float = 1.0 / np.sqrt(2),
    alpha_V_plus: float = 1.0 / np.sqrt(2),
    alpha_V_minus: float = -1.0 / np.sqrt(2),
    chi_max: int = 50,
):
    """
    Simulate emission to time-bin wave packet.

    Chain layout: atom_A - atom_B - A1 - B1 - A2 - B2 - ... - AN - BN
    For simplicity, we simulate just one arm (A) with one atom.

    Parameters
    ----------
    n_bins : int
        Number of time bins
    dt : float
        Time bin width
    t0 : float
        Peak time of emission pulse
    sigma : float
        Pulse width
    theta_H, theta_V : float
        QFC conversion angles
    alpha_* : float
        Polarization mapping coefficients
    chi_max : int
        Maximum bond dimension

    Returns
    -------
    MPSState
        Final MPS state after all emissions
    """
    print("=" * 70)
    print("Simulating Atom Emission → Time-Bin Wave Packet")
    print("=" * 70)

    # Create time grid
    time_grid = TimeGrid(dt=dt, N=n_bins)
    t = time_grid.t

    # Create Gaussian pulse profile
    gamma_values = gaussian_pulse(t, t0, sigma)

    print(f"\nTime grid:")
    print(f"  N_bins = {n_bins}")
    print(f"  dt = {dt}")
    print(f"  Total time = {n_bins * dt}")
    print(f"  t range: [{t[0]:.1f}, {t[-1]:.1f}]")

    print(f"\nEmission pulse profile:")
    print(f"  Peak at t0 = {t0}")
    print(f"  Width sigma = {sigma}")
    print(f"  Peak gamma = {gamma_values.max():.4f}")

    # Setup polarization matrix
    Alpha = np.array([
        [alpha_H_plus, alpha_H_minus],
        [alpha_V_plus, alpha_V_minus]
    ], dtype=complex)

    print(f"\nPolarization mapping (Alpha matrix):")
    print(f"  [[{alpha_H_plus:.3f}, {alpha_H_minus:.3f}],")
    print(f"   [{alpha_V_plus:.3f}, {alpha_V_minus:.3f}]]")

    # Create emission parameters
    emit_params = EmitParams(
        gamma_A=gamma_values,  # Time-dependent
        Alpha_A=Alpha,
    )

    # QFC parameters
    qfc_params = QFCParams(theta_H=theta_H, theta_V=theta_V)

    # ========================================================================
    # Initialize MPS: atom(3D) - A1(18D) - A2(18D) - ... - AN(18D)
    # ========================================================================

    print(f"\nInitializing MPS...")
    print(f"  Chain layout: atom - A1 - A2 - ... - AN")
    print(f"  Atom dimension: 3D (|0>, |1>, |e>)")
    print(f"  Bin dimension: 18D (780 x 1517)")

    local_dims = [3] + [18] * n_bins

    # Initial state: atom in excited state |e> (index 2)
    # All bins in vacuum (index 0 in each 18D bin space)
    init_state = [2] + [0] * n_bins

    mps = MPSState(local_dims=local_dims, init_state=init_state, max_bond=chi_max)

    print(f"  MPS created: L={mps.L}, d={mps.d[:5]}...")
    print(f"  Initial chi = {mps.get_bond_dimensions()[:5]}...")

    # ========================================================================
    # Process each bin: emission + QFC
    # ========================================================================

    print(f"\nProcessing {n_bins} time bins...")

    # Pre-compute QFC gate (same for all bins)
    U_qfc = qfc_gate(theta_H=theta_H, theta_V=theta_V)
    print(f"  QFC gate computed (sin^2 theta = {np.sin(theta_H)**2:.3f} conversion)")

    for n in range(n_bins):
        # Site 0 is atom, site 1+n is bin A_n
        tn = t[n]
        gamma_n = gamma_values[n]

        # Skip bins with very small emission rate
        if gamma_n < 1e-6:
            continue

        # Create emission gate for this time step
        U_emit = emission_gate(
            gamma=gamma_n,
            dt=dt,
            Alpha=Alpha,
            which_atom='A'
        )

        # Apply emission gate (atom to bin n)
        # Note: emission_gate returns 27D (atom x 780), need to embed to 18D bin
        # For simplicity, we use the 27D gate but only apply to atom + 780 part
        mps.apply_bond_op(0, U_emit)

        # Apply QFC (780 → 1517 conversion within bin)
        mps.apply_one_site_gate(1 + n, U_qfc)

        if (n + 1) % 50 == 0 or n == 0 or n == n_bins - 1:
            chi = mps.get_bond_dimensions()
            print(f"  Bin {n+1:3d}/{n_bins}: gamma={gamma_n:.4f}, chi={chi[1]}")

    print(f"\nEmission complete!")
    print(f"Final bond dimensions: {mps.get_bond_dimensions()}")
    print(f"Final state norm: {mps.norm():.6f}")

    return mps, time_grid


def visualize_results(mps: MPSState, time_grid: TimeGrid):
    """
    Visualize the wave packet using the visualization module.

    Parameters
    ----------
    mps : MPSState
        Final MPS state
    time_grid : TimeGrid
        Time grid
    """
    print("\n" + "=" * 70)
    print("Visualizing Wave Packet")
    print("=" * 70)

    n_bins = time_grid.N

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ========================================================================
    # Plot 1: Single-photon probability (normalized)
    # ========================================================================

    ax = axes[0, 0]
    data_A, _ = extract_single_photon_prob(mps, n_bins, polarized=False)

    # Normalize to show shape
    data_A_norm = data_A / (data_A.sum() + 1e-15)

    x = time_grid.t
    ax.plot(x, data_A_norm, '-', color='tab:blue', linewidth=2, label='Single-photon prob')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Probability (normalized)')
    ax.set_title('Single-Photon Wave Packet (normalized)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ========================================================================
    # Plot 2: Intensity envelope
    # ========================================================================

    ax = axes[0, 1]
    data_A, _ = extract_intensity_envelope(mps, n_bins, polarized=False)

    ax.plot(x, data_A, '-', color='tab:orange', linewidth=2, label='<N>')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Intensity / <N>')
    ax.set_title('Intensity Envelope')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ========================================================================
    # Plot 3: Polarization-resolved wave packet
    # ========================================================================

    ax = axes[1, 0]
    data_A, _ = extract_single_photon_prob(mps, n_bins, polarized=True)

    # Normalize
    total = data_A.sum(axis=0, keepdims=True) + 1e-15
    data_A_norm = data_A / total

    ax.plot(x, data_A_norm[:, 0], '--', label='H pol', color='tab:blue', alpha=0.7)
    ax.plot(x, data_A_norm[:, 1], '-', label='V pol', color='tab:red', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Probability')
    ax.set_title('Polarization-Resolved Wave Packet')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ========================================================================
    # Plot 4: Cumulative probability
    # ========================================================================

    ax = axes[1, 1]
    cumulative = np.cumsum(data_A_norm)
    ax.plot(x, cumulative, '-', color='tab:green', linewidth=2)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50%')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('wavepacket_results.png', dpi=150)
    print("\nPlot saved to: wavepacket_results.png")

    return fig


def main():
    """Run the complete emission → wave packet simulation."""

    # ========================================================================
    # Simulation Parameters
    # ========================================================================

    print("\n" + "=" * 70)
    print("DEMO: Atom Emission → Time-Bin Wave Packet")
    print("=" * 70)

    # Time parameters
    n_bins = 200
    dt = 0.1
    t0 = 10.0  # Peak at middle of simulation
    sigma = 3.0  # Pulse width

    # QFC: 50% conversion (sin^2 theta = 0.5)
    theta_H = np.arcsin(np.sqrt(0.5)) / 2  # Gives sin^2(theta) = 0.5
    theta_V = theta_H

    # Polarization: circular (sigma+/- orthogonal)
    # This creates entanglement between atom state and polarization
    alpha_H_plus = 1.0 / np.sqrt(2)
    alpha_H_minus = 1.0 / np.sqrt(2)
    alpha_V_plus = 1j / np.sqrt(2)  # +90 deg phase
    alpha_V_minus = -1j / np.sqrt(2)

    print(f"\nSimulation parameters:")
    print(f"  n_bins = {n_bins}")
    print(f"  dt = {dt}")
    print(f"  Pulse: Gaussian with t0={t0}, sigma={sigma}")
    print(f"  QFC: theta_H={theta_H:.3f}, theta_V={theta_V:.3f}")
    print(f"  Polarization: Circular (sigma+/-)")

    # ========================================================================
    # Run Simulation
    # ========================================================================

    mps, time_grid = simulate_emission_wavepacket(
        n_bins=n_bins,
        dt=dt,
        t0=t0,
        sigma=sigma,
        theta_H=theta_H,
        theta_V=theta_V,
        alpha_H_plus=alpha_H_plus,
        alpha_H_minus=alpha_H_minus,
        alpha_V_plus=alpha_V_plus,
        alpha_V_minus=alpha_V_minus,
        chi_max=50,
    )

    # ========================================================================
    # Visualize Results
    # ========================================================================

    fig = visualize_results(mps, time_grid)

    # ========================================================================
    # Summary Statistics
    # ========================================================================

    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)

    # Extract wave packet data
    data_A, _ = extract_single_photon_prob(mps, n_bins, polarized=False)

    total_prob = data_A.sum()
    print(f"\nTotal single-photon probability: {total_prob:.6f}")
    print(f"  (Should be close to expected emission probability)")

    # Expected emission probability
    theta_peak = np.sqrt(dt * 1.0)  # Peak theta with gamma=1
    expected_p = np.sin(theta_peak) ** 2
    print(f"\nExpected single-bin emission probability (peak): {expected_p:.6f}")

    # Peak bin
    peak_idx = np.argmax(data_A)
    peak_prob = data_A[peak_idx]
    peak_time = time_grid.t[peak_idx]
    print(f"\nPeak bin: {peak_idx + 1}")
    print(f"Peak time: {peak_time:.2f} s")
    print(f"Peak probability: {peak_prob:.6f}")

    # Width (FWHM approximation)
    threshold = peak_prob / 2
    above_threshold = data_A > threshold
    if np.any(above_threshold):
        fwhm = (above_threshold.sum()) * dt
        print(f"FWHM (approx): {fwhm:.2f} s")

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)

    plt.show()


if __name__ == "__main__":
    main()
