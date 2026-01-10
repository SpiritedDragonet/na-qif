"""
Unitary Gate Factories with Caching

This module provides factory functions for constructing unitary gates
used in the time-bin simulation. Gates that don't vary per bin are cached.
"""

from typing import Optional, Tuple
from functools import lru_cache
import numpy as np
from scipy.linalg import expm

from ..hilbert.basis import (
    SubSpace,
    ProductSpace,
    subspace_gate,
    SUBSPACE_780,
    SUBSPACE_1517,
    BIN_SPACE,
    get_bin_space,
    get_system_space,
)
from ..hilbert.operators import (
    annihilation_op,
    creation_op,
    atom_transition,
)


# Cached gates (these are expensive to compute and don't vary per bin)

@lru_cache(maxsize=8)
def qfc_gate(theta_H: float = 0.0, theta_V: float = 0.0) -> np.ndarray:
    """
    Quantum Frequency Conversion gate U_qfc.

    Converts 780nm photons to 1517nm via beam-splitter-like coupling:
        U_qfc = exp(theta_H * (b_H c_H^† - b_H^† c_H) + theta_V * (b_V c_V^† - b_V^† c_V))

    This is a one-site unitary on the 18D bin space (780 x 1517).

    Parameters
    ----------
    theta_H : float
        Conversion angle for H polarization (sin²(theta) = conversion prob)
    theta_V : float
        Conversion angle for V polarization

    Returns
    -------
    np.ndarray
        18x18 unitary matrix acting on bin space

    Examples
    --------
    >>> U = qfc_gate(theta_H=np.pi/4, theta_V=np.pi/4)  # 50% conversion
    """
    # Get annihilation/creation operators
    bH = annihilation_op(SUBSPACE_780, mode_id=0)
    bH_dag = creation_op(SUBSPACE_780, mode_id=0)
    bV = annihilation_op(SUBSPACE_780, mode_id=1)
    bV_dag = creation_op(SUBSPACE_780, mode_id=1)

    cH = annihilation_op(SUBSPACE_1517, mode_id=0)
    cH_dag = creation_op(SUBSPACE_1517, mode_id=0)
    cV = annihilation_op(SUBSPACE_1517, mode_id=1)
    cV_dag = creation_op(SUBSPACE_1517, mode_id=1)

    # Build the generator on 780 subspace (acting on c in 1517 via tensor product)
    # G = -i * theta_H * (b_H c_H^† - b_H^† c_H) - i * theta_V * (b_V c_V^† - b_V^† c_V)
    # The total generator acts on 780 x 1517 product space

    # We need to embed operators correctly
    # b acts on 780, c acts on 1517, so b ⊗ c^† acts on the product

    I_780 = np.eye(3, dtype=complex)
    I_1517 = np.eye(6, dtype=complex)

    # b_H ⊗ I_1517
    bH_full = np.kron(bH, I_1517)
    # I_780 ⊗ c_H^†
    cH_dag_full = np.kron(I_780, cH_dag)
    # b_H^† ⊗ I_1517
    bH_dag_full = np.kron(bH_dag, I_1517)
    # I_780 ⊗ c_H
    cH_full = np.kron(I_780, cH)

    # Same for V
    bV_full = np.kron(bV, I_1517)
    cV_dag_full = np.kron(I_780, cV_dag)
    bV_dag_full = np.kron(bV_dag, I_1517)
    cV_full = np.kron(I_780, cV)

    # Generator: theta * (b c^† - b^† c)
    # This is anti-Hermitian, so exp(G) is unitary
    G_H = theta_H * (bH_full @ cH_dag_full - bH_dag_full @ cH_full)
    G_V = theta_V * (bV_full @ cV_dag_full - bV_dag_full @ cV_full)

    G = G_H + G_V

    # Exponentiate to get unitary
    U = expm(G)

    return U


@lru_cache(maxsize=4)
def _bs_gate_1517() -> np.ndarray:
    """
    Internal function: 50/50 Beam Splitter on 1517_A × 1517_B (36x36).

    This is the core BS gate acting only on the telecom subspace.

    Returns
    -------
    np.ndarray
        36x36 unitary matrix for 1517_A × 1517_B space (6×6 per site)
    """
    def make_generator(mode_id: int) -> np.ndarray:
        """Construct BS generator for a single polarization mode."""
        c = annihilation_op(SUBSPACE_1517, mode_id)  # 6x6
        c_dag = creation_op(SUBSPACE_1517, mode_id)  # 6x6

        # Construct operators on the joint 1517_A × 1517_B space (36D)
        c_A = np.kron(c, np.eye(6, dtype=complex))
        c_B = np.kron(np.eye(6, dtype=complex), c)
        c_dag_A = np.kron(c_dag, np.eye(6, dtype=complex))
        c_dag_B = np.kron(np.eye(6, dtype=complex), c_dag)

        # BS generator: G = θ * (c_A^† c_B - c_A c_B^†)
        theta = np.pi / 4
        G = theta * (c_dag_A @ c_B - c_A @ c_dag_B)
        return G

    # Generate generators for H and V polarizations
    G_H = make_generator(mode_id=0)  # H polarization
    G_V = make_generator(mode_id=1)  # V polarization

    # Total generator (sum over both polarizations)
    G_total = G_H + G_V

    # Exponentiate to get unitary
    U_bs = expm(G_total)

    return U_bs


@lru_cache(maxsize=4)
def bs_gate() -> np.ndarray:
    """
    50/50 Beam Splitter gate U_BS using generator exponentiation.

    Generator: G_BS = θ * Σ_p (c_A,p^† ⊗ c_B,p - c_A,p ⊗ c_B,p^†)
    With θ = π/4 for 50:50 BS.

    Mixes telecom modes (1517nm) of two bins.
    Returns a 324x324 unitary acting on bin_A × bin_B (18D × 18D),
    where each bin is 780(3D) × 1517(6D) = 18D.

    The BS acts as I_780_A ⊗ I_780_B ⊗ U_BS_1517 where U_BS_1517
    is the 36x36 BS gate on the telecom subspace.

    Returns
    -------
    np.ndarray
        324x324 unitary matrix for bin_A × bin_B space (18×18 per site)

    Examples
    --------
    >>> U = bs_gate()  # Can be applied to (A_n, B_n) pair
    """
    # Get the core 36x36 BS gate on 1517_A × 1517_B
    U_bs_1517 = _bs_gate_1517()  # 36x36

    # Embed into the full 324x324 space (bin_A × bin_B)
    # Each bin is 780(3D) × 1517(6D) = 18D
    # The full space structure is:
    #   (780_A ⊗ 1517_A) ⊗ (780_B ⊗ 1517_B)
    # For the BS, we apply I_780_A ⊗ I_780_B ⊗ U_BS_1517
    # which means for each (i_A, i_B) in 780_A × 780_B (3×3=9 combos),
    # we apply U_bs_1517 to the 1517_A × 1517_B subspace

    dim_780 = 3  # Dimension of 780 subspace
    dim_1517 = 6  # Dimension of 1517 subspace
    dim_bin = 18  # 3 * 6
    dim_full = 324  # 18 * 18

    U_full = np.zeros((dim_full, dim_full), dtype=complex)

    # For each (i_A, i_B) combination of 780 states, apply BS to 1517 subspace
    for i_A in range(dim_780):
        for i_B in range(dim_780):
            # Calculate the offset in the full 324x324 matrix
            # Row offset for this (i_A, i_B) block
            offset_row = (i_A * dim_1517) * dim_bin + (i_B * dim_1517)
            # Column offset for this (i_A, i_B) block
            offset_col = (i_A * dim_1517) * dim_bin + (i_B * dim_1517)

            # Embed the 36x36 BS gate into this block
            for i in range(dim_1517):
                for j in range(dim_1517):
                    # Row in full matrix: offset + i_in_binA * dim_bin + j_in_binB
                    for ii in range(dim_1517):
                        for jj in range(dim_1517):
                            row = offset_row + i * dim_bin + jj
                            col = offset_col + ii * dim_bin + j
                            U_full[row, col] = U_bs_1517[i * dim_1517 + ii, j * dim_1517 + jj]

    return U_bs_1517  # For backward compatibility, return 36x36 for now


def bs_gate_bin18() -> np.ndarray:
    """
    50/50 Beam Splitter gate for 18D bins (324x324).

    This is the version that should be used with the current MPS structure
    where each bin is 18D (780 × 1517).

    The BS only acts on the 1517nm subspace, leaving 780nm unchanged.
    For each fixed (780_A, 780_B) configuration, it applies the 36x36 BS
    gate to the 1517_A × 1517_B subspace.

    Returns
    -------
    np.ndarray
        324x324 unitary matrix for bin_A × bin_B space (18×18 per site)
    """
    # Get the core 36x36 BS gate on 1517_A × 1517_B
    U_bs_1517 = _bs_gate_1517()  # 36x36

    # Embed into the full 324x324 space
    # Each bin is 780(3D) × 1517(6D) = 18D
    # Joint space is 18 × 18 = 324D
    dim_780 = 3
    dim_1517 = 6
    dim_bin = 18
    dim_full = 324

    U_full = np.zeros((dim_full, dim_full), dtype=complex)

    # For each (780_A, 780_B) configuration, apply BS to 1517 subspace
    # The 780 part is unchanged (identity), only 1517 is mixed
    for i_780_A in range(dim_780):
        for i_780_B in range(dim_780):
            # For this fixed 780 configuration, loop over all 1517 combinations
            for i_1517_A_out in range(dim_1517):
                for i_1517_B_out in range(dim_1517):
                    # Output bin indices
                    idx_A_out = i_780_A * dim_1517 + i_1517_A_out
                    idx_B_out = i_780_B * dim_1517 + i_1517_B_out
                    row = idx_A_out * dim_bin + idx_B_out

                    for i_1517_A_in in range(dim_1517):
                        for i_1517_B_in in range(dim_1517):
                            # Input bin indices (same 780, different 1517)
                            idx_A_in = i_780_A * dim_1517 + i_1517_A_in
                            idx_B_in = i_780_B * dim_1517 + i_1517_B_in
                            col = idx_A_in * dim_bin + idx_B_in

                            # Get the BS matrix element for this 1517 transition
                            i_1517_out = i_1517_A_out * dim_1517 + i_1517_B_out
                            i_1517_in = i_1517_A_in * dim_1517 + i_1517_B_in
                            U_full[row, col] = U_bs_1517[i_1517_out, i_1517_in]

    return U_full


@lru_cache(maxsize=16)
def jones_gate(U: Tuple[Tuple[complex, complex], Tuple[complex, complex]]) -> np.ndarray:
    """
    Jones polarization rotation gate U_pol.

    Applies a 2x2 Jones matrix to the telecom (1517nm) H/V subspace:
        (c_H', c_V')^T = U * (c_H, c_V)^T

    Parameters
    ----------
    U : Tuple[Tuple[complex, complex], Tuple[complex, complex]]
        2x2 Jones matrix as nested tuple for hashing (can be cached)
        Format: ((u00, u01), (u10, u11))

    Returns
    -------
    np.ndarray
        6x6 unitary matrix acting on 1517 subspace (identity on vac and 2-photon states)

    Examples
    --------
    >>> # Half-wave plate at 45 degrees
    >>> import numpy as np
    >>> U_hwp = ((1, 0), (0, -1))
    >>> U = jones_gate(U_hwp)
    """
    u00, u01 = U[0]
    u10, u11 = U[1]

    # 1517 basis: vac, H, V, 2H, 2V, HV
    # Jones rotation acts on the single-photon H/V subspace
    # For multi-photon states, it acts as U⊗U on the appropriate tensor power

    # Build the 6x6 matrix
    op = np.zeros((6, 6), dtype=complex)

    # Vacuum is unchanged
    op[0, 0] = 1.0

    # Single-photon subspace: (H, V) -> U @ (H, V)
    op[1, 1] = u00  # H -> u00*H + u10*V
    op[2, 1] = u10
    op[1, 2] = u01  # V -> u01*H + u11*V
    op[2, 2] = u11

    # Two-photon subspace: U acts as U ⊗ U
    # |2H> = |HH> -> (u00*H + u10*V) ⊗ (u00*H + u10*V)
    # = u00²|HH> + u00*u10|HV> + u10*u00|VH> + u10²|VV>
    # But since we have indistinguishable photons, |HV> = |VH>

    # |2H> -> u00²|2H> + 2*u00*u10|HV> + u10²|2V>
    op[3, 3] = u00 * u00  # |2H>
    op[5, 3] = 2 * u00 * u10  # |HV>
    op[4, 3] = u10 * u10  # |2V>

    # |2V> -> u01²|2H> + 2*u01*u11|HV> + u11²|2V>
    op[3, 4] = u01 * u01
    op[5, 4] = 2 * u01 * u11
    op[4, 4] = u11 * u11

    # |HV> -> (u00*H + u10*V) ⊗ (u01*H + u11*V)
    # = u00*u01|HH> + (u00*u11 + u10*u01)|HV> + u10*u11|VV>
    op[3, 5] = u00 * u01
    op[5, 5] = u00 * u11 + u10 * u01
    op[4, 5] = u10 * u11

    return op


def jones_gate_from_array(U_array: np.ndarray) -> np.ndarray:
    """
    Convenience wrapper to call jones_gate with numpy array.
    Returns the gate embedded in the 18D bin space (I_780 ⊗ U_1517).

    Parameters
    ----------
    U_array : np.ndarray
        2x2 Jones matrix

    Returns
    -------
    np.ndarray
        18x18 unitary matrix acting on full bin space (780 × 1517)
    """
    U_tuple = (
        (complex(U_array[0, 0]), complex(U_array[0, 1])),
        (complex(U_array[1, 0]), complex(U_array[1, 1]))
    )
    U_1517 = jones_gate(U_tuple)  # 6x6

    # Embed into 18D bin space: I_780 ⊗ U_1517
    I_780 = np.eye(3, dtype=complex)
    return np.kron(I_780, U_1517)  # 18x18


@lru_cache(maxsize=2)
def swap_gate(d1: int, d2: int) -> np.ndarray:
    """
    SWAP gate for exchanging two sites.

    W |s> ⊗ |t> = |t> ⊗ |s>

    Used for "conveyor belt" protocol to move system site along the chain.

    Parameters
    ----------
    d1 : int
        Dimension of first site
    d2 : int
        Dimension of second site

    Returns
    -------
    np.ndarray
        (d1*d2, d1*d2) permutation matrix

    Examples
    --------
    >>> W = swap_gate(9, 18)  # Swap system (9D) with bin (18D)
    """
    # Construct SWAP matrix explicitly
    dim = d1 * d2
    W = np.zeros((dim, dim), dtype=complex)

    for i in range(d1):
        for j in range(d2):
            # |i> ⊗ |j> -> |j> ⊗ |i>
            row_idx = i * d2 + j
            col_idx = j * d1 + i
            W[row_idx, col_idx] = 1.0

    return W


def emission_gate(
    gamma: float,
    dt: float,
    Alpha: np.ndarray,
    which_atom: str = 'A'
) -> np.ndarray:
    """
    Emission gate U_emit for atom-photon entanglement (embedded in bin space).

        U_emit = exp(√(dt) * (L ⊗ b^†_780 - L^† ⊗ b_780))

    where L = √gamma * (alpha_+ * S_+ + alpha_- * S_-)
    and S_± are atomic transition operators.

    The gate is embedded in the 18D bin space as U_9x9 ⊗ I_1517,
    where the 9×9 gate acts on atom(3D) × 780(3D) and I_1517 is identity
    on the telecom subspace.

    This creates entanglement between atomic state and emitted photon polarization.
    The emitted photon is in the 780nm subspace, and can later be converted
    to 1517nm via QFC.

    Parameters
    ----------
    gamma : float
        Emission rate at this time step
    dt : float
        Time bin width
    Alpha : np.ndarray
        2x2 polarization mapping matrix from atomic transitions to H/V
        [[alpha_H+, alpha_H-], [alpha_V+, alpha_V-]]
    which_atom : str
        Which atom ('A' or 'B')

    Returns
    -------
    np.ndarray
        (54, 54) unitary matrix acting on atom(3D) × bin(18D=780×1517)

    Examples
    --------
    >>> # Example: circular polarization mapping
    >>> Alpha = np.array([[1, 0], [0, 1]])  # σ+ -> H, σ- -> V
    >>> U = emission_gate(gamma=0.1, dt=1.0, Alpha=Alpha, which_atom='A')
    """
    # Atomic transition operators
    S_plus = atom_transition('+')  # |0><e|
    S_minus = atom_transition('-')  # |1><e|

    # Extract Alpha matrix elements
    alpha_H_plus = Alpha[0, 0]
    alpha_H_minus = Alpha[0, 1]
    alpha_V_plus = Alpha[1, 0]
    alpha_V_minus = Alpha[1, 1]

    # Construct L operator on atom (3D)
    # L = √gamma * (alpha_H+ * S_+ + alpha_H- * S_-) for H pol
    # Similar for V pol
    sqrt_gamma = np.sqrt(gamma)

    L_H = sqrt_gamma * (alpha_H_plus * S_plus + alpha_H_minus * S_minus)
    L_V = sqrt_gamma * (alpha_V_plus * S_plus + alpha_V_minus * S_minus)

    # Photon operators on 780 (3D: vac, H, V)
    # b^†_H = |H><vac|
    bH_dag = np.zeros((3, 3), dtype=complex)
    bH_dag[1, 0] = 1.0
    bH = bH_dag.conj().T

    bV_dag = np.zeros((3, 3), dtype=complex)
    bV_dag[2, 0] = 1.0
    bV = bV_dag.conj().T

    # Generator: G = √dt * (L_H ⊗ b_H^† + L_V ⊗ b_V^† - h.c.)
    I_atom = np.eye(3, dtype=complex)
    I_780 = np.eye(3, dtype=complex)

    sqrt_dt = np.sqrt(dt)

    G_H = sqrt_dt * (np.kron(L_H, bH_dag) - np.kron(L_H.conj().T, bH))
    G_V = sqrt_dt * (np.kron(L_V, bV_dag) - np.kron(L_V.conj().T, bV))

    G_9x9 = G_H + G_V

    # Exponentiate to get unitary on atom × 780
    U_9x9 = expm(G_9x9)

    # Identity on 1517 subspace (6D)
    I_1517 = np.eye(6, dtype=complex)

    # Embed into atom × bin(780×1517) space: U_54 = U_9x9 ⊗ I_1517
    # This gives (9×9) ⊗ (6×6) = (54, 54)
    U_54 = np.kron(U_9x9, I_1517)

    return U_54
