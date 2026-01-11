"""Test multiple detection trials to gather statistics."""

import numpy as np
from total_simulation import run_dual_atom_emission, run_detection_and_bsm
from atom_sim.simulation import (
    apply_qfc, apply_780_filter, apply_bs,
    compute_fidelity_with_bell, compute_photon_statistics,
)

n_trials = 30
success_count = 0
success_results = []
same_bin_count = 0

print("Testing detection with normalization before detection...")
print(f"Running {n_trials} trials with n_bins=50, sigma=3.0...\n")

for seed in range(n_trials):
    # Run emission - use smaller bins and tighter wavepacket for faster test
    result = run_dual_atom_emission(
        n_bins=50, dt_ns=0.2, chi_max=20,
        gamma_peak_A=0.3, gamma_peak_B=0.3, sigma=3.0,
        verbose=False
    )

    # Apply QFC
    apply_qfc(mps=result.mps, n_bins=result.get_n_bins(), theta_H=np.pi/4, theta_V=np.pi/4, verbose=False)

    # Apply 780nm filter
    apply_780_filter(mps=result.mps, n_bins=result.get_n_bins(), verbose=False)

    # Apply BS
    apply_bs(mps=result.mps, n_bins=result.get_n_bins(), verbose=False)

    # Compute photon statistics before normalization (only for first trial)
    if seed == 0:
        print("Photon statistics before normalization:")
        compute_photon_statistics(mps=result.mps, n_bins=result.get_n_bins(), verbose=True)

    # Normalize MPS to condition on two-photon arrival
    result.mps._mps.canonical_form_finite(renormalize=True)

    # Verify photon statistics after normalization (only for first trial)
    if seed == 0:
        print("\nPhoton statistics after normalization:")
        compute_photon_statistics(mps=result.mps, n_bins=result.get_n_bins(), verbose=True)
        print()

    # Run detection
    det_result = run_detection_and_bsm(
        result=result, eta_det=0.85, seed=seed, verbose=False
    )

    # Print outcome
    clicks = [(c.detector, c.bin_index) for c in det_result.clicks]
    status = 'SUCCESS' if det_result.success else 'FAIL'

    # Check if same-bin detection (even if not BSM success)
    if len(det_result.clicks) == 2:
        if det_result.clicks[0].bin_index == det_result.clicks[1].bin_index:
            same_bin_count += 1

    # Only print every 5 trials or if special events
    two_clicks = len(det_result.clicks) == 2
    same_bin = two_clicks and (det_result.clicks[0].bin_index == det_result.clicks[1].bin_index)
    if seed % 5 == 0 or det_result.success or same_bin:
        marker = " <-- same bin!" if same_bin and not det_result.success else ""
        print(f'Trial {seed:2d}: {status:7s} clicks={clicks}{marker}')

    if det_result.success:
        success_count += 1
        fid = compute_fidelity_with_bell(det_result.spin_state, det_result.bell_state)
        success_results.append((det_result.bell_state, fid))
        print(f'         Fidelity with |{det_result.bell_state}>: {fid:.4f}')

print(f'\nSummary: {success_count}/{n_trials} BSM successes ({100*success_count/n_trials:.1f}%)')
print(f'Same-bin detections: {same_bin_count}/{n_trials} ({100*same_bin_count/n_trials:.1f}%)')
if success_results:
    avg_fid = np.mean([r[1] for r in success_results])
    print(f'Average fidelity: {avg_fid:.4f}')
