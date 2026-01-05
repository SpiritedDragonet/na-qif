"""
Basic Operator Factory for Fock Spaces and Atomic Transitions

This module provides functions to construct creation, annihilation,
number operators, and atomic transition operators.
"""

from typing import Tuple, Union
import numpy as np
from .basis import SubSpace, ProductSpace, SUBSPACE_780, SUBSPACE_1517, ATOM_3D


def annihilation_op(space: SubSpace, mode_id: int = 0) -> np.ndarray:
    """
    Construct annihilation operator a[i] on a Fock subspace.

    For a basis of occupancy tuples (n_0, n_1, ..., n_{M-1}) with sum(n) <= n_max,
    the annihilation operator on mode i acts as:
        a[i] |..., n_i, ...> = sqrt(n_i) |..., n_i - 1, ...>

    Parameters
    ----------
    space : SubSpace
        The subspace to construct the operator on.
        For 780: mode 0=H, 1=V
        For 1517: mode 0=H, 1=V
    mode_id : int
        Which mode to annihilate (default: 0)

    Returns
    -------
    np.ndarray
        Annihilation operator matrix with shape (space.dim, space.dim)

    Examples
    --------
    >>> a_780_H = annihilation_op(SUBSPACE_780, mode_id=0)  # 780 H mode
    >>> a_1517_V = annihilation_op(SUBSPACE_1517, mode_id=1)  # 1517 V mode
    """
    dim = space.dim

    if space == SUBSPACE_780:
        # 780 subspace: vac, H, V
        # mode_id=0 (H): a|H> = |vac>, a|vac> = 0, a|V> = 0
        # mode_id=1 (V): a|V> = |vac>, a|vac> = 0, a|H> = 0
        op = np.zeros((dim, dim), dtype=complex)
        if mode_id == 0:  # H mode
            op[0, 1] = 1.0  # |vac><H|
        elif mode_id == 1:  # V mode
            op[0, 2] = 1.0  # |vac><V|
        else:
            raise ValueError(f"Invalid mode_id {mode_id} for 780 subspace")
        return op

    elif space == SUBSPACE_1517:
        # 1517 subspace: vac, H, V, 2H, 2V, HV
        # Occupancy representation:
        # vac: (0,0), H: (1,0), V: (0,1), 2H: (2,0), 2V: (0,2), HV: (1,1)
        # mode_id=0 (H), mode_id=1 (V)
        op = np.zeros((dim, dim), dtype=complex)
        basis_order = [
            (0, 0),  # 0: vac
            (1, 0),  # 1: H
            (0, 1),  # 2: V
            (2, 0),  # 3: 2H
            (0, 2),  # 4: 2V
            (1, 1),  # 5: HV
        ]

        # Build index map: (n_H, n_V) -> index
        idx_map = {occ: i for i, occ in enumerate(basis_order)}

        for i, (nH, nV) in enumerate(basis_order):
            if mode_id == 0:  # H mode
                nH_target = nH - 1
                nV_target = nV
            else:  # V mode
                nH_target = nH
                nV_target = nV - 1

            if nH_target < 0 or nV_target < 0:
                continue

            target = (nH_target, nV_target)
            if target in idx_map:
                j = idx_map[target]
                # Coefficient is sqrt(n) where n is the original occupancy
                n = nH if mode_id == 0 else nV
                op[j, i] = np.sqrt(n)

        return op

    else:
        raise ValueError(f"Unsupported subspace: {space.name}. "
                        f"Use SUBSPACE_780 or SUBSPACE_1517")


def creation_op(space: SubSpace, mode_id: int = 0) -> np.ndarray:
    """
    Construct creation operator a^†[i] on a Fock subspace.

    This is the Hermitian conjugate of the annihilation operator.

    Parameters
    ----------
    space : SubSpace
        The subspace to construct the operator on
    mode_id : int
        Which mode to create on (default: 0)

    Returns
    -------
    np.ndarray
        Creation operator matrix with shape (space.dim, space.dim)
    """
    a = annihilation_op(space, mode_id)
    return a.conj().T


def atom_transition(which: str) -> np.ndarray:
    """
    Construct atomic transition operators S_+ or S_-.

    Atomic levels (3D):
        |0>: ground state (m_F = +1)
        |1>: ground state (m_F = -1)
        |e>: excited state (m_F = 0)

    Selection rules:
        |e> → |0>: Δm = +1 → σ+ photon (S_+ = |0><e|)
        |e> → |1>: Δm = -1 → σ- photon (S_- = |1><e|)

    Parameters
    ----------
    which : str
        Either '+' for S_+ or '-' for S_-

    Returns
    -------
    np.ndarray
        Transition operator matrix with shape (3, 3)

    Examples
    --------
    >>> S_plus = atom_transition('+')  # |0><e|
    >>> S_minus = atom_transition('-')  # |1><e|
    """
    # Basis order: |0>, |1>, |e>
    op = np.zeros((3, 3), dtype=complex)

    if which == '+':
        # S_+ = |0><e|
        op[0, 2] = 1.0
    elif which == '-':
        # S_- = |1><e|
        op[1, 2] = 1.0
    else:
        raise ValueError(f"which must be '+' or '-', got '{which}'")

    return op


def number_op(space: SubSpace, mode_id: int = 0) -> np.ndarray:
    """
    Construct number operator N = a^† a on a Fock subspace.

    Used for wave packet extraction: <N> gives photon number expectation.

    For the 1517 subspace with basis (vac, H, V, 2H, 2V, HV):
        N_H = diag(0, 1, 0, 2, 0, 1)
        N_V = diag(0, 0, 1, 0, 2, 1)

    Parameters
    ----------
    space : SubSpace
        The subspace to construct the operator on
    mode_id : int
        Which mode to count photons in (default: 0)

    Returns
    -------
    np.ndarray
        Number operator matrix (diagonal) with shape (space.dim, space.dim)

    Examples
    --------
    >>> N_780_H = number_op(SUBSPACE_780, mode_id=0)  # Count 780 H photons
    >>> N_1517_H = number_op(SUBSPACE_1517, mode_id=0)  # Count 1517 H photons
    >>> N_1517_V = number_op(SUBSPACE_1517, mode_id=1)  # Count 1517 V photons
    """
    adag = creation_op(space, mode_id)
    a = annihilation_op(space, mode_id)
    return adag @ a
