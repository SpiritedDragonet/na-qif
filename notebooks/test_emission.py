"""
Simple TEBD Test: Atom Emission (Using L0 Layer MPSState)

Physics:
---------
Atom (3-level):
  |e>: excited state
  |0>, |1>: ground states

Selection rules:
  |e> → |0>: sigma+ photon (H polarization)
  |e> |1>: sigma- photon (V polarization)

Purpose: Test MPSState.apply_two_site_gate() for TEBD evolution
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
# Note: __file__ is not available in some environments
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import L0 layer (core) MPS infrastructure
from atom_sim.core.mps import MPSState


def test_emission():
    """Run emission test."""

    # ========================================================================
    # Part 1: System Setup
    # ========================================================================

    print("=" * 70)
    print("Part 1: System Setup")
    print("=" * 70)

    print("\nAtom basis (3D): |0> [idx 0], |1> [idx 1], |e> [idx 2]")
    print("780nm photon basis (3D): |vac> [idx 0], |H> [idx 1], |V> [idx 2]")

    # ========================================================================
    # Part 2: Build Emission Gate (U_emit)
    # ========================================================================

    print("\n" + "=" * 70)
    print("Part 2: Build Emission Gate (U_emit)")
    print("=" * 70)

    # Gate parameters
    dt = 0.1
    g = 1.0  # coupling strength
    theta = np.sqrt(dt) * g

    # Dimensions
    d_atom = 3
    d_photon = 3
    d_combined = d_atom * d_photon

    print(f"\nGate construction: |e, vac> -> cos(theta)|e, vac> + sin(theta)/sqrt(2)(|0, H> + |1, V>)")
    print(f"  theta = sqrt(dt) * g = {theta:.4f}")

    # Build U_emit matrix
    U_emit = np.eye(d_combined, dtype=complex)

    # Index mapping: idx = atom_idx * d_photon + photon_idx
    e_vac_idx = 2 * d_photon + 0  # |e, vac>
    target_H_idx = 0 * d_photon + 1  # |0, H>
    target_V_idx = 1 * d_photon + 2  # |1, V>

    print(f"\nCombined space indices (9D):")
    print(f"|e, vac> -> idx {e_vac_idx}")
    print(f"|0, H>   -> idx {target_H_idx}")
    print(f"|1, V>   -> idx {target_V_idx}")

    # Set matrix elements
    U_emit[e_vac_idx, e_vac_idx] = np.cos(theta)
    U_emit[target_H_idx, e_vac_idx] = np.sin(theta) / np.sqrt(2)
    U_emit[target_V_idx, e_vac_idx] = np.sin(theta) / np.sqrt(2)

    # ========================================================================
    # Part 3: Create Initial MPS State
    # ========================================================================

    print("\n" + "=" * 70)
    print("Part 3: Create Initial MPS State")
    print("=" * 70)

    print("\nCreating MPS with local_dims = [3, 3]...")
    print("Initial state: atom in |e> (index 2), photon in |vac> (index 0)")

    mps = MPSState(local_dims=[d_atom, d_photon], init_state=[2, 0], max_bond=10)

    print(f"\nMPS created:")
    print(f"  L = {mps.L} sites")
    print(f"  d = {mps.d}")
    print(f"  chi = {mps.get_bond_dimensions()}")
    print(f"  norm = {mps.norm():.6f}")

    # Verify initial state
    rho_atom_init = mps.get_atom_state(system_site=0)
    print("\nAtomic density matrix (should show |e><e| at index 2):")
    print(rho_atom_init)

    # ========================================================================
    # Part 4: Apply Emission Gate
    # ========================================================================

    print("\n" + "=" * 70)
    print("Part 4: Apply Emission Gate via TEBD")
    print("=" * 70)

    print("\nApplying emission gate U^(emit) to sites (0, 1)...")

    mps.apply_two_site_gate(site_left=0, gate=U_emit, truncate=True)

    print("Emission gate applied!")
    print(f"Bond dimensions after gate: {mps.get_bond_dimensions()}")
    print(f"State norm: {mps.norm():.6f}")

    # ========================================================================
    # Part 5: Analyze Results
    # ========================================================================

    print("\n" + "=" * 70)
    print("Part 5: Analyze Results")
    print("=" * 70)

    rho_atom = mps.get_atom_state(system_site=0)
    print("\nAtomic reduced density matrix:")
    print(rho_atom)

    rho_photon = mps.get_reduced_density(sites=[1])
    print("\nPhoton reduced density matrix:")
    print(rho_photon)

    # Probabilities
    p_remain = rho_atom[2, 2].real
    p_H = rho_atom[0, 0].real
    p_V = rho_atom[1, 1].real
    p_emit = p_H + p_V

    print(f"\nProbabilities:")
    print(f"  P(atom in |e>)        = {p_remain:.6f}")
    print(f"  P(atom in |0>, H emit)  = {p_H:.6f}")
    print(f"  P(atom in |1>, V emit)  = {p_V:.6f}")
    print(f"  P(emission)            = {p_emit:.6f}")

    # Consistency check
    p_photon_H = rho_photon[1, 1].real
    p_photon_V = rho_photon[2, 2].real
    print(f"\nConsistency check:")
    print(f"  P(emit|atom)   = {p_emit:.6f}")
    print(f"  P(emit|photon) = {p_photon_H + p_photon_V:.6f}")
    print(f"  Difference = {abs(p_emit - (p_photon_H + p_photon_V)):.2e}")

    # ========================================================================
    # Part 6: Entanglement Analysis
    # ========================================================================

    print("\n" + "=" * 70)
    print("Part 6: Entanglement Analysis")
    print("=" * 70)

    chi = mps.get_bond_dimensions()
    print(f"\nBond dimensions: {chi}")
    print(f"Schmidt rank = {chi[0]}")
    print(f"\nExpected Schmidt rank = 3 (3 orthogonal terms in superposition)")

    # ========================================================================
    # Summary
    # ========================================================================

    print("\n" + "=" * 70)
    print("Summary: L0 Layer Functions Used")
    print("=" * 70)
    print("\n  1. MPSState.__init__()         - Create MPS")
    print("  2. MPSState.apply_two_site_gate() - Apply two-site unitary")
    print("  3. MPSState.get_atom_state()      - Get reduced density matrix")
    print("  4. MPSState.get_reduced_density() - Get any site's reduced state")
    print("  5. MPSState.norm()                - Get state norm")
    print("  6. MPSState.get_bond_dimensions() - Get entanglement info")

    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)


if __name__ == "__main__":
    test_emission()
