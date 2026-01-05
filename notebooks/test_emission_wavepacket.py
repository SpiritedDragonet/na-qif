"""
Test: Atom Emission → Time-Bin Wave Packet (NO QFC, just atomic emission)

Debug test to verify emission only creates photons in 780nm subspace.
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atom_sim.core.mps import MPSState
from atom_sim.config import TimeGrid, EmitParams, QFCParams
from atom_sim.physics.gates import emission_gate, qfc_gate
from atom_sim.visualization import telecom_ops_bin18


def test_emission_wavepacket():
    """
    Test emission WITHOUT QFC to verify physics is correct.

    The emission should ONLY create photons in 780nm subspace,
    with ZERO leakage to 1517nm subspace (no fiber yet!).
    """
    print("=" * 70)
    print("Test: Atom Emission → 780nm Wave Packet (NO QFC)")
    print("=" * 70)

    # Parameters
    n_bins = 200  # More bins for high resolution
    dt_ns = 0.2  # High time resolution (0.2ns per bin)
    chi_max = 50  # Higher bond dimension for more accurate simulation

    print(f"\nParameters:")
    print(f"  n_bins = {n_bins}")
    print(f"  dt = {dt_ns} ns (high resolution)")
    print(f"  Total time = {n_bins * dt_ns:.1f} ns")
    print(f"  chi_max = {chi_max}")

    # Time grid (in nanoseconds)
    time_grid = TimeGrid(dt=dt_ns, N=n_bins)
    t = time_grid.t  # Time in ns

    # Gaussian emission rate profile (width ~28ns FWHM)
    t0 = n_bins * dt_ns / 2  # Peak at center (20ns)
    sigma = 12.0  # Width parameter: FWHM ≈ 2.35*sigma ≈ 28ns
    gamma_peak = 0.2  # Peak emission rate (single-step prob ≈ 4%)

    # Create time-dependent emission rate
    gamma_values = gamma_peak * np.exp(-0.5 * ((t - t0) / sigma) ** 2)

    print(f"\nEmission parameters:")
    print(f"  Gaussian pulse: t0={t0:.1f}ns, sigma={sigma:.1f}ns")
    print(f"  Peak gamma = {gamma_peak} (single-step prob ≈ {gamma_peak * dt_ns:.3f})")
    print(f"  Expected FWHM ≈ {2.35 * sigma:.1f} ns")

    # Print gamma values at some bins for verification
    peak_idx = np.argmin(np.abs(t - t0))
    print(f"  Gamma at peak (t={t[peak_idx]:.1f}ns): {gamma_values[peak_idx]:.3f}")

    # Simple H/V mapping (no entanglement for debugging)
    Alpha = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    print(f"  Alpha = simple (sigma+ -> H, sigma- -> V)")

    print(f"\nNOTE: NOT applying QFC - only testing atomic emission to 780nm")

    # ========================================================================
    # Initialize MPS
    # ========================================================================

    print(f"\nInitializing MPS...")
    local_dims = [3] + [18] * n_bins  # atom + n_bins
    init_state = [2] + [0] * n_bins  # |e> + vacuum

    mps = MPSState(local_dims=local_dims, init_state=init_state, max_bond=chi_max)

    print(f"  MPS: L={mps.L}, d={mps.d[:3]}... chi={mps.get_bond_dimensions()[:3]}...")

    # ========================================================================
    # Bin space structure explanation
    # ========================================================================

    print(f"\nBin space structure (18D = 780(3D) x 1517(6D)):")
    print(f"  780 basis: |vac>, |H>, |V>")
    print(f"  1517 basis: |vac>, |H>, |V>, |2H>, |2V>, |HV>")
    print(f"  Index formula: index = i_780 * 6 + i_1517")
    print(f"    780=vac (i_780=0): indices 0-5")
    print(f"    780=H   (i_780=1): indices 6-11")
    print(f"    780=V   (i_780=2): indices 12-17")

    # ========================================================================
    # Process bins WITHOUT SWAP and WITHOUT QFC
    # ========================================================================

    print(f"\nProcessing {n_bins} bins (accumulating at site 1, NO QFC)...")

    # Track evolution after each step
    evolution_H = []
    evolution_V = []
    evolution_total = []
    evolution_1517_H = []  # Track 1517 H to check for leakage
    evolution_1517_V = []  # Track 1517 V to check for leakage

    for n in range(n_bins):
        # Get emission rate for this bin (at this time step)
        gamma_n = float(gamma_values[n])

        if gamma_n >= 1e-6:
            # Emission gate for this time step
            # Acts on atom (site 0) and current bin (site 1)
            U_emit = emission_gate(
                gamma=gamma_n,
                dt=dt_ns,
                Alpha=Alpha,
                which_atom='A'
            )
            mps.apply_bond_op(0, U_emit)

        # NO QFC! Just track 780nm subspace
        rho_site1 = mps.get_reduced_density([1])

        # 780nm subspace: indices 0-5 (vac), 6-11 (H), 12-17 (V)
        p_780_H = rho_site1[6:12, 6:12].sum().real
        p_780_V = rho_site1[12:18, 12:18].sum().real

        # 1517nm subspace: we need to check if there's any probability there
        # 1517=H means checking specific indices across 780 subspaces
        # For 1517=H (i_1517=1): indices are 1, 7, 13
        # For 1517=V (i_1517=2): indices are 2, 8, 14
        p_1517_H = (
            rho_site1[1, 1].real +      # 780=vac, 1517=H
            rho_site1[7, 7].real +      # 780=H, 1517=H
            rho_site1[13, 13].real      # 780=V, 1517=H
        )
        p_1517_V = (
            rho_site1[2, 2].real +      # 780=vac, 1517=V
            rho_site1[8, 8].real +      # 780=H, 1517=V
            rho_site1[14, 14].real      # 780=V, 1517=V
        )

        evolution_H.append(p_780_H)
        evolution_V.append(p_780_V)
        evolution_total.append(p_780_H + p_780_V)
        evolution_1517_H.append(p_1517_H)
        evolution_1517_V.append(p_1517_V)

    print(f"  Complete!")
    print(f"  Final chi: {mps.get_bond_dimensions()}")
    print(f"  Norm: {mps.norm():.6f}")

    # Check for 1517 leakage
    max_1517_H = max(evolution_1517_H)
    max_1517_V = max(evolution_1517_V)
    print(f"\n1517 subspace leakage check:")
    print(f"  Max 1517-H probability: {max_1517_H:.6e}")
    print(f"  Max 1517-V probability: {max_1517_V:.6e}")
    if max_1517_H > 1e-10 or max_1517_V > 1e-10:
        print(f"  ERROR: Non-zero 1517 probability detected!")
    else:
        print(f"  OK: No leakage to 1517 subspace (as expected)")

    # ========================================================================
    # Extract and analyze wave packet
    # ========================================================================

    print("\n" + "=" * 70)
    print("Wave Packet Analysis (780nm subspace)")
    print("=" * 70)

    # Use evolution arrays for time-resolved data
    data_A_H = np.array(evolution_H)
    data_A_V = np.array(evolution_V)
    data_A_total = data_A_H + data_A_V

    total_prob = data_A_total[-1]  # Final value
    peak_idx = np.argmax(data_A_total)
    peak_time = t[peak_idx]
    peak_prob = data_A_total[peak_idx]

    print(f"\n780nm single-photon probability:")
    print(f"  Final total: {total_prob:.6f}")
    print(f"  Peak (incremental): {peak_prob:.6f} at bin {peak_idx + 1} (t={peak_time:.1f}ns)")

    # Print values around gamma peak
    gamma_peak_idx = np.argmax(gamma_values)
    print(f"\n  Around gamma peak (bin {gamma_peak_idx + 1}, t={t[gamma_peak_idx]:.1f}ns):")
    for i in range(max(0, gamma_peak_idx - 2), min(n_bins, gamma_peak_idx + 3)):
        print(f"    Bin {i + 1} (t={t[i]:.1f}ns): gamma={gamma_values[i]:.3f}, total={data_A_total[i]:.6f}")

    # Print detailed values around 20-25ns to see the curve
    print(f"\n  Detailed values (t=18ns to 25ns):")
    for i in range(n_bins):
        if 18 <= t[i] <= 25:
            print(f"    Bin {i + 1} (t={t[i]:.1f}ns): gamma={gamma_values[i]:.6f}, "
                  f"total={data_A_total[i]:.6f}, H={data_A_H[i]:.6f}, V={data_A_V[i]:.6f}")

    # ========================================================================
    # Visualize
    # ========================================================================

    print("\nPlotting results...")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Evolution of emission (cumulative)
    ax = axes[0]
    ax.plot(t, data_A_total, '-', linewidth=2, label='Total (H+V)')
    ax.plot(t, data_A_H, '--', linewidth=1.5, label='H pol', alpha=0.7)
    ax.plot(t, data_A_V, '--', linewidth=1.5, label='V pol', alpha=0.7)
    # Also show gamma profile scaled
    ax2 = ax.twinx()
    ax2.plot(t, gamma_values, ':', color='gray', alpha=0.5, label='Gamma profile')
    ax2.set_ylabel('Gamma (emission rate)')
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Cumulative probability')
    ax.set_title('780nm Wave Packet Evolution (Cumulative)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Incremental emission per bin
    ax = axes[1]
    # Compute incremental
    incremental = np.zeros_like(data_A_total)
    incremental[0] = data_A_total[0]
    for i in range(1, n_bins):
        incremental[i] = data_A_total[i] - data_A_total[i - 1]

    ax.bar(t - dt_ns/2, incremental, width=dt_ns, alpha=0.7, label='Incremental')
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Incremental probability')
    ax.set_title('Emission per Time Bin')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('test_emission_wavepacket.png', dpi=100)
    print("  Saved to: test_emission_wavepacket.png")

    # ========================================================================
    # Consistency checks
    # ========================================================================

    print("\n" + "=" * 70)
    print("Consistency Checks")
    print("=" * 70)

    # Check atom state
    rho_atom = mps.get_reduced_density([0])
    p_excited = rho_atom[2, 2].real
    p_g0 = rho_atom[0, 0].real
    p_g1 = rho_atom[1, 1].real

    print(f"\nAtomic state:")
    print(f"  P(|e>) = {p_excited:.6f}")
    print(f"  P(|0>) = {p_g0:.6f}")
    print(f"  P(|1>) = {p_g1:.6f}")
    print(f"  Total: {p_excited + p_g0 + p_g1:.6f}")

    # Expected total emission probability (sum over all bins)
    # For small angles: p_emit ~ gamma * dt, but this is only approximation
    # Since atom decays, actual probability saturates at 1
    print(f"\nWave packet statistics:")
    print(f"  Total emission probability: {total_prob:.3f}")
    print(f"  Peak bin: {peak_idx + 1} (t={t[peak_idx]:.1f}ns)")
    print(f"  Peak value: {peak_prob:.3f}")

    # Check if we have mainly single-photon or multi-photon
    if total_prob > 1.1:
        multi_photon = total_prob - 1.0
        print(f"  Multi-photon probability: ~{multi_photon:.3f}")
    elif total_prob < 0.9:
        print(f"  Warning: Emission probability is low")

    # Wave packet width (FWHM approximation)
    threshold = peak_prob / 2
    above_threshold = data_A_total > threshold
    if np.any(above_threshold):
        fwhm = above_threshold.sum() * dt_ns
        print(f"  Wave packet FWHM: ~{fwhm:.1f} ns")

    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)

    plt.show()


if __name__ == "__main__":
    test_emission_wavepacket()
