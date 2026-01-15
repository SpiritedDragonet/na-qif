# -*- coding: utf-8 -*-
"""
Debug script to trace the emission process and understand the mixed state.
"""

import numpy as np
from atom_sim.config import TimeGrid, EmitParams
from atom_sim.simulation import run_emission_only
from atom_sim.simulation.detection import extract_spin_state, compute_fidelity_with_bell

# Test 1: Single atom emission
print("="*60)
print("Test 1: Single atom emission (A only)")
print("="*60)

time_grid = TimeGrid(dt=0.2e-9, N=100)
Alpha_A = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
Alpha_B = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)

# Create emission params with only A emitting
def gamma_A_only(t):
    return 0.5

def gamma_B_zero(t):
    return 0.0

emit_params_A_only = EmitParams(
    gamma_A=gamma_A_only,
    gamma_B=gamma_B_zero,
    Alpha_A=Alpha_A,
    Alpha_B=Alpha_B,
)

result_A_only = run_emission_only(
    time_grid=time_grid,
    emit_params=emit_params_A_only,
    chi_max=50,
    verbose=False,
    delay_bins_B=-1,  # B doesn't emit anyway, but delay needed
)

spin_A_only = extract_spin_state(result_A_only.mps, result_A_only.get_n_bins())
purity_A_only = np.trace(spin_A_only @ spin_A_only).real
print(f"Single atom A emission:")
print(f"  Qubit state diagonal (|00>, |01>, |10>, |11>): {np.diag(spin_A_only).real}")
print(f"  Qubit purity: {purity_A_only:.4f}")

# Check FULL atomic state including |e>
rho_full_A = result_A_only.mps.get_reduced_density([0, 1])
rho_full_A = rho_full_A.reshape(9, 9)
print(f"  Full atomic state diagonal (including |e>): {np.diag(rho_full_A).real}")
# Population in each state
pop_ee = rho_full_A[2*3 + 2, 2*3 + 2].real  # |e,e>
pop_e0 = rho_full_A[2*3 + 0, 2*3 + 0].real  # |e,0>
pop_e1 = rho_full_A[2*3 + 1, 2*3 + 1].real  # |e,1>
print(f"  Population in |e,e>: {pop_ee:.4f}")
print(f"  Population in |e,0>: {pop_e0:.4f}")
print(f"  Population in |e,1>: {pop_e1:.4f}")
print()

# Test 2: Single atom B emission
print("="*60)
print("Test 2: Single atom emission (B only)")
print("="*60)

def gamma_A_zero(t):
    return 0.0

def gamma_B_only(t):
    return 0.5

emit_params_B_only = EmitParams(
    gamma_A=gamma_A_zero,
    gamma_B=gamma_B_only,
    Alpha_A=Alpha_A,
    Alpha_B=Alpha_B,
)

result_B_only = run_emission_only(
    time_grid=time_grid,
    emit_params=emit_params_B_only,
    chi_max=50,
    verbose=False,
    delay_bins_B=-1,
)

spin_B_only = extract_spin_state(result_B_only.mps, result_B_only.get_n_bins())
purity_B_only = np.trace(spin_B_only @ spin_B_only).real
print(f"Single atom B emission:")
print(f"  Qubit state diagonal: {np.diag(spin_B_only).real}")
print(f"  Qubit purity: {purity_B_only:.4f}")

# Check FULL atomic state including |e>
rho_full_B = result_B_only.mps.get_reduced_density([0, 1])
rho_full_B = rho_full_B.reshape(9, 9)
print(f"  Full atomic state diagonal (including |e>): {np.diag(rho_full_B).real}")
print()

# Test 3: Dual atom emission
print("="*60)
print("Test 3: Dual atom emission (both emit)")
print("="*60)

def gamma_A_func(t):
    return 0.5 * np.exp(-0.5 * (t / 10.0) ** 2)

def gamma_B_func(t):
    return 0.5 * np.exp(-0.5 * (t / 10.0) ** 2)

emit_params_both = EmitParams(
    gamma_A=gamma_A_func,
    gamma_B=gamma_B_func,
    Alpha_A=Alpha_A,
    Alpha_B=Alpha_B,
)

result_both = run_emission_only(
    time_grid=time_grid,
    emit_params=emit_params_both,
    chi_max=50,
    verbose=False,
    delay_bins_B=-1,  # B delayed by 1 bin
)

spin_both = extract_spin_state(result_both.mps, result_both.get_n_bins())
purity_both = np.trace(spin_both @ spin_both).real
print(f"Dual atom emission (delay=-1):")
print(f"  Qubit state diagonal: {np.diag(spin_both).real}")
print(f"  Qubit purity: {purity_both:.4f}")
print(f"  Bell fidelities:")
for bell in ["Psi+", "Psi-", "Phi+", "Phi-"]:
    print(f"    F({bell}): {compute_fidelity_with_bell(spin_both, bell):.4f}")

# Check FULL atomic state including |e>
rho_full_both = result_both.mps.get_reduced_density([0, 1])
rho_full_both = rho_full_both.reshape(9, 9)
print(f"  Full atomic state diagonal (including |e>): {np.diag(rho_full_both).real}")
print()

# Test 4: Check if atoms are in the correct initial state
print("="*60)
print("Test 4: Check initial atomic state")
print("="*60)

from atom_sim.core.mps import MPSState

# Create a simple 2-atom system
local_dims = [3, 3]  # Two atoms
init_state = [2, 2]  # Both in |e> state
mps_test = MPSState(local_dims=local_dims, init_state=init_state, max_bond=10)

# Extract the state
rho_init = mps_test.get_reduced_density([0, 1])
rho_init_4d = rho_init.reshape(3, 3, 3, 3)

# Extract qubit subspace (|0>, |1> only, ignore |e>)
qubit_indices = [0, 1]
rho_qubit_init = np.zeros((2, 2), dtype=complex)
for i, qi in enumerate(qubit_indices):
    for j, qj in enumerate(qubit_indices):
        rho_qubit_init[i, j] = rho_init[qi, qj]

print(f"Initial two-atom state (|e>, |e>):")
print(f"  Qubit subspace (|0>, |1> only):")
print(f"    {rho_qubit_init}")
print(f"  Should be zero because atoms are in |e>")
print()

# Test 5: Check single emission gate effect
print("="*60)
print("Test 5: Single emission gate effect")
print("="*60)

from atom_sim.physics.gates import emission_gate

# Create initial state: |e>|vac>
local_dims_test = [3, 18]  # atom(3D) x bin(18D)
init_state_test = [2, 0]   # |e>|vac>
mps_single_emit = MPSState(local_dims=local_dims_test, init_state=init_state_test, max_bond=10)

# Apply emission gate
U_emit = emission_gate(
    gamma=0.5,
    dt=1.0,
    Alpha=Alpha_A,
    which_atom='A',
    bin_first=False
)

mps_single_emit.apply_bond_op(0, U_emit)

# Extract atomic state
rho_atom = mps_single_emit.get_reduced_density([0])
print(f"Atomic state after single emission gate:")
print(f"  rho_atom diagonal: {np.diag(rho_atom).real}")
print(f"  rho_atom full:")
print(f"    {rho_atom.real}")

# Extract bin state
rho_bin = mps_single_emit.get_reduced_density([1])
print(f"\nBin state after single emission gate:")
print(f"  rho_bin diagonal: {np.diag(rho_bin).real}")

# Check entanglement entropy
bond_dims = mps_single_emit._mps.chi
print(f"\nBond dimensions: {bond_dims}")
print(f"Max bond dim: {max(bond_dims)}")
print()

print("="*60)
print("Summary")
print("="*60)
print("The key question: After emission, are atoms entangled with photons?")
print("If yes -> tracing out photons gives mixed state")
print("If no -> tracing out photons gives pure state")
