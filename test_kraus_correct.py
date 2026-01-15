# -*- coding: utf-8 -*-
"""
Test Kraus operators with correct input states.
"""

import numpy as np
from atom_sim.simulation.detection import build_detection_kraus_18d

# Build Kraus operators
kraus_list, outcome_names = build_detection_kraus_18d(eta=0.85)

print("="*60)
print("Test 1: Verify completeness relation")
print("="*60)

sum_kdk = np.zeros((324, 324), dtype=complex)
for K in kraus_list:
    sum_kdk += K.conj().T @ K

print(f"Sum of K†K diagonal (first 10 elements): {np.diag(sum_kdk).real[:10]}")
print(f"Should be [1, 1, 1, ...] for vac, H, V states")

print("\n" + "="*60)
print("Test 2: H1 click on |H> state")
print("="*60)

# Find H1 outcome (H on port1, none on port2)
h1_idx = outcome_names.index("H1")
K_h1 = kraus_list[h1_idx]

# Input: |H>_port1 ⊗ |vac>_port2
psi_in = np.zeros(324, dtype=complex)
psi_in[1 * 18 + 0] = 1.0  # |H>_port1 (index 1) ⊗ |vac>_port2 (index 0)

psi_out = K_h1 @ psi_in
print(f"Input: |H>_port1 ⊗ |vac>_port2")
print(f"Output norm: {np.linalg.norm(psi_out):.4f}")
print(f"Output state index: {np.argmax(np.abs(psi_out))}")

# Interpret: should be |vac>_port1 ⊗ |vac>_port2 (index 0*18+0 = 0)
if psi_out[0] > 0:
    print(f"  Correct: |H> was absorbed -> |vac,vac>")

print("\n" + "="*60)
print("Test 3: V2 click on |V> state")
print("="*60)

# Find V2 outcome
v2_idx = outcome_names.index("V2")
K_v2 = kraus_list[v2_idx]

# Input: |vac>_port1 ⊗ |V>_port2
psi_in = np.zeros(324, dtype=complex)
psi_in[0 * 18 + 2] = 1.0  # |vac>_port1 (index 0) ⊗ |V>_port2 (index 2)

psi_out = K_v2 @ psi_in
print(f"Input: |vac>_port1 ⊗ |V>_port2")
print(f"Output norm: {np.linalg.norm(psi_out):.4f}")
print(f"Output state index: {np.argmax(np.abs(psi_out))}")

# Interpret: should be |vac>_port1 ⊗ |vac>_port2 (index 0*18+0 = 0)
if psi_out[0] > 0:
    print(f"  Correct: |V> was absorbed -> |vac,vac>")

print("\n" + "="*60)
print("Test 4: H1+V1 requires multi-photon or proper state")
print("="*60)
print("H1+V1 outcome means port1 detected both H and V")
print("This requires the input state to have both H and V components")
print("in port1, which is only possible with multi-photon states or specific superpositions")
print("\nFor our simulation, we typically have single photons,")
print("so H1+V1 would come from a state where port1 has |H> and |V>")
print("superposition (which requires more than one photon or specific entanglement)")
