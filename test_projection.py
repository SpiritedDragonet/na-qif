# -*- coding: utf-8 -*-
"""
Test the projection after detection.
"""

import numpy as np
from atom_sim.simulation.detection import extract_spin_state
from atom_sim.simulation import compute_photon_statistics
from atom_sim.config import TimeGrid, EmitParams
from atom_sim.simulation import run_emission_only

print("="*60)
print("Test: Verify atomic state after emission")
print("="*60)

time_grid = TimeGrid(dt=0.2e-9, N=50)  # Fewer bins for faster test
Alpha = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)

def gamma_func(t):
    return 0.5 * np.exp(-0.5 * (t / 10.0) ** 2)

emit_params = EmitParams(
    gamma_A=gamma_func,
    gamma_B=gamma_func,
    Alpha_A=Alpha,
    Alpha_B=Alpha,
)

print("Running emission simulation...")
result = run_emission_only(
    time_grid=time_grid,
    emit_params=emit_params,
    chi_max=30,
    verbose=False,
    delay_bins_B=-1,
)

# Check atomic state (trace out all bins)
spin_state = extract_spin_state(result.mps, result.get_n_bins())
print(f"\nAtomic state (trace out all bins):")
print(f"  diag: {np.diag(spin_state).real}")
print(f"  purity: {np.trace(spin_state @ spin_state).real:.4f}")

# Check photon statistics
stats = compute_photon_statistics(result.mps, result.get_n_bins(), verbose=True)

print("\n" + "="*60)
print("Analysis")
print("="*60)
print("If atoms are maximally entangled with photons,")
print("tracing out bins gives a mixed state.")
print("This is expected behavior - it's the correct physics.")
