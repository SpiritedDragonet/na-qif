# -*- coding: utf-8 -*-
"""
Test to check the state of bins after detection.
"""

import numpy as np
from atom_sim.config import TimeGrid, EmitParams
from atom_sim.simulation import run_emission_only, apply_qfc, apply_780_filter, apply_bs
from atom_sim.simulation.detection import run_two_photon_detection, compute_photon_statistics
from atom_sim.core.mps import MPSState

print("="*60)
print("Test: Check bin states after detection")
print("="*60)

time_grid = TimeGrid(dt=0.2e-9, N=50)
Alpha = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)

def gamma_func(t):
    return 0.5 * np.exp(-0.5 * (t / 10.0) ** 2)

emit_params = EmitParams(
    gamma_A=gamma_func,
    gamma_B=gamma_func,
    Alpha_A=Alpha,
    Alpha_B=Alpha,
)

print("Running emission...")
result = run_emission_only(
    time_grid=time_grid,
    emit_params=emit_params,
    chi_max=30,
    verbose=False,
    delay_bins_B=-1,
)

print("Applying QFC...")
apply_qfc(result.mps, result.get_n_bins(), theta_H=np.pi/4, theta_V=np.pi/4, verbose=False)

print("Applying 780nm filter...")
apply_780_filter(result.mps, result.get_n_bins(), verbose=False)

print("Applying BS...")
apply_bs(result.mps, result.get_n_bins(), verbose=False)

# Check photon expectation BEFORE detection
print("\nBEFORE detection:")
stats_before = compute_photon_statistics(result.mps, result.get_n_bins(), verbose=True)

# Run detection
print("\nRunning detection...")
det_result = run_two_photon_detection(
    mps=result.mps,
    n_bins=result.get_n_bins(),
    eta_det=0.85,
    rng=np.random.default_rng(seed=42),
    verbose=False,
)

print(f"Detection result: success={det_result.success}, bell_state={det_result.bell_state}")
print(f"Clicks: {[(c.detector, c.bin_index) for c in det_result.clicks]}")

# Check photon expectation AFTER detection
print("\nAFTER detection:")
stats_after = compute_photon_statistics(result.mps, result.get_n_bins(), verbose=True)

# Check the state of the detected bin
detected_bin = det_result.clicks[0].bin_index if det_result.clicks else None

# Check the state of the detected bin
if detected_bin is not None:
    site_1 = 2 + 2 * detected_bin
    site_2 = 2 + 2 * detected_bin + 1

    rho_bin_1 = result.mps.get_reduced_density([site_1])
    rho_bin_2 = result.mps.get_reduced_density([site_2])

    print(f"\nDetected bin {detected_bin} state:")
    print(f"  Port 1 (site {site_1}) diagonal: {np.diag(rho_bin_1).real[:6]}")
    print(f"  Port 2 (site {site_2}) diagonal: {np.diag(rho_bin_2).real[:6]}")
