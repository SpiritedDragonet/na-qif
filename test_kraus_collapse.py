# -*- coding: utf-8 -*-
"""
Check if Kraus operators correctly collapse bins to |vac>.
"""

import numpy as np
from atom_sim.simulation.detection import build_detection_kraus_18d

# Build Kraus operators
kraus_list, outcome_names = build_detection_kraus_18d(eta=0.85)

# Find the index of "H1+V1" outcome
h1v1_idx = outcome_names.index("H1+V1")
K_h1v1 = kraus_list[h1v1_idx]

print(f"Kraus operator for H1+V1 outcome:")
print(f"  Shape: {K_h1v1.shape}")

# Check if K†K = I (for all Kraus operators combined)
sum_kdk = np.zeros((324, 324), dtype=complex)
for K in kraus_list:
    sum_kdk += K.conj().T @ K

print(f"\nSum of K†K for all Kraus operators:")
print(f"  Should be identity matrix")
print(f"  Diagonal: {np.diag(sum_kdk).real[:10]}")

# Test the Kraus operator on a simple state
# Create a state with one H photon in port 1 and one V photon in port 1
# This is a two-mode state: |H>|V> in the same bin

# The 324D space is 18 x 18 = (port1) x (port2)
# Each port is 18D = 780(3) x 1517(6)

# Create a simple test: |H in port1, V in port1>
# This should trigger H1+V1 detection

# In 1517 subspace: |H>=index 1, |V>=index 2
# In 780 subspace: |vac>=index 0
# So |H> in 18D = |vac>_780 ⊗ |H>_1517 = index 0*6 + 1 = 1
# And |V> in 18D = |vac>_780 ⊗ |V>_1517 = index 0*6 + 2 = 2

# The joint state is |H>_port1 ⊗ |V>_port2
# In 324D: index = 1 * 18 + 2 = 20

psi_test = np.zeros(324, dtype=complex)
psi_test[1 * 18 + 2] = 1.0  # |H>_port1 ⊗ |V>_port2

print(f"\nTest state: |H>_port1 ⊗ |V>_port2")
print(f"  Nonzero index: {1*18+2}")

# Apply Kraus operator
psi_after = K_h1v1 @ psi_test

print(f"\nAfter applying K_H1V1:")
print(f"  Norm: {np.linalg.norm(psi_after)}")
print(f"  Nonzero elements (> 1e-10): {np.sum(np.abs(psi_after) > 1e-10)}")

# Check where the amplitude went
indices = np.where(np.abs(psi_after) > 1e-10)[0]
print(f"  Nonzero indices: {indices[:10]}")

# Interpret the result
for idx in indices[:5]:
    port1 = idx // 18
    port2 = idx % 18
    print(f"    idx {idx}: port1={port1}, port2={port2}")

# The Kraus operator should map |H>|V> to |vac>|vac> (both photons absorbed)
# |vac> in 18D = |vac>_780 ⊗ |vac>_1517 = index 0*6 + 0 = 0
# So |vac,vac> in 324D = 0 * 18 + 0 = 0

print(f"\nExpected: amplitude at index 0 (|vac,vac>)")
print(f"  psi_after[0] = {psi_after[0]}")
