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
        U_qfc = exp(-i * (theta_H * (b_H c_H^† + h.c.) + theta_V * (b_V c_V^† + h.c.)))

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

    # Generator: -i * theta * (b c^† - b^† c)
    G_H = -1j * theta_H * (bH_full @ cH_dag_full - bH_dag_full @ cH_full)
    G_V = -1j * theta_V * (bV_full @ cV_dag_full - bV_dag_full @ cV_full)

    G = G_H + G_V

    # Exponentiate to get unitary
    U = expm(G)

    return U


@lru_cache(maxsize=4)
def bs_gate() -> np.ndarray:
    """
    50/50 Beam Splitter gate U_BS.

    Mixes telecom modes (1517nm) of two bins:
        (d_1, d_2)^T = (1/sqrt(2)) * [[1, 1], [1, -1]] * (c_A, c_B)^T

    This is a two-site unitary acting on the 1517 subspace of both bins.

    Returns
    -------
    np.ndarray
        36x36 unitary matrix (6x6 per site, but only acts on telecom subspace)
        Actually returns (36, 36) for full two-bin telecom space

    Examples
    --------
    >>> U = bs_gate()  # Can be applied to (A_n, B_n) pair
    """
    # 50/50 beam splitter matrix
    BS = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    # We need to construct the BS operator on the 6D telecom space
    # The telecom basis is: vac, H, V, 2H, 2V, HV
    # BS only mixes H and V between the two input modes

    # For each polarization, we have a BS transformation
    # The full operator on two sites is constructed by:
    # U_BS = exp(theta * (a_A^† a_B - a_A a_B^†)) with theta = pi/4

    # For simplicity, construct directly in the number basis
    # The BS acts identically on H and V polarizations

    # Single-mode BS on one polarization:
    # |0>|0> -> |0>|0>
    # |1>|0> -> (|1>|0> + |0>|1>)/sqrt(2)
    # |0>|1> -> (|1>|0> - |0>|1>)/sqrt(2)
    # |1>|1> -> (|2>|0> - |0>|2>)/sqrt(2)  (bunching)

    # We need to construct this on the 6D telecom space for each site
    # Total dimension: 6 x 6 = 36

    I_780 = np.eye(3, dtype=complex)

    # Build the BS operator on the 1517 telecom subspace
    # We'll work in the joint Fock basis for the two modes

    # Basis for two modes with up to 2 photons total:
    # (n1, n2) where n1, n2 in {0,1,2} and n1+n2 <= 2
    basis_2mode = [
        (0, 0),  # vac,vac
        (1, 0),  # H,vac
        (0, 1),  # vac,H
        (2, 0),  # 2H,vac
        (0, 2),  # vac,2H
        (1, 1),  # H,H
    ]

    # BS transformation on this basis (for one polarization)
    dim_2mode = len(basis_2mode)
    BS_pol = np.zeros((dim_2mode, dim_2mode), dtype=complex)

    for i, (n1, n2) in enumerate(basis_2mode):
        # Apply BS: |n1, n2> -> sum_k binomial(n, k)^(1/2) * ... |n1-k, n2+k>
        # Actually easier: BS is generated by G = theta (a^†_1 a_2 - a_1 a^†_2)
        # For theta=pi/4, we have specific coefficients

        if n1 == 0 and n2 == 0:
            BS_pol[0, 0] = 1.0
        elif n1 == 1 and n2 == 0:
            BS_pol[0, 1] = 1.0 / np.sqrt(2)
            BS_pol[1, 1] = 1.0 / np.sqrt(2)
        elif n1 == 0 and n2 == 1:
            BS_pol[0, 2] = 1.0 / np.sqrt(2)
            BS_pol[1, 2] = -1.0 / np.sqrt(2)
        elif n1 == 2 and n2 == 0:
            BS_pol[3, 3] = 0.5  # |2,0> -> 0.5|2,0> + ...
            BS_pol[5, 3] = np.sqrt(0.5)  # -> sqrt(0.5)|1,1> + ...
        elif n1 == 0 and n2 == 2:
            BS_pol[4, 4] = 0.5  # |0,2> -> ...
            BS_pol[5, 4] = -np.sqrt(0.5)  # -> -sqrt(0.5)|1,1> + ...
        elif n1 == 1 and n2 == 1:
            BS_pol[3, 5] = 1.0 / np.sqrt(2)  # |1,1> -> sqrt(0.5)|2,0>
            BS_pol[4, 5] = -1.0 / np.sqrt(2)  # -> -sqrt(0.5)|0,2>

    # Now extend to two polarizations
    # The full 36D operator is BS_pol_H ⊗ BS_pol_V on the appropriate tensor factors
    # But we need to be careful about the tensor product structure

    # Simpler approach: construct the full BS on 1517 x 1517
    # by applying BS_pol to both H and V modes
    U_telecom = np.kron(BS_pol, BS_pol)

    # Now embed into full bin x bin space (18 x 18 = 324, but only acts on telecom)
    # Actually we want a (36, 36) matrix acting on telecom_A x telecom_B
    # where each telecom is 6D

    return U_telecom


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

    Parameters
    ----------
    U_array : np.ndarray
        2x2 Jones matrix

    Returns
    -------
    np.ndarray
        6x6 unitary matrix acting on 1517 subspace
    """
    U_tuple = (
        (complex(U_array[0, 0]), complex(U_array[0, 1])),
        (complex(U_array[1, 0]), complex(U_array[1, 1]))
    )
    return jones_gate(U_tuple)


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
