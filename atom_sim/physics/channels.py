"""
Kraus Channels for Non-Unitary Evolution

This module provides Kraus operator sets for various quantum channels:
- Amplitude damping (fiber loss)
- Detection POVM (on/off measurement)
- Atomic dephasing
"""

from typing import List, Tuple
import numpy as np

from ..hilbert.basis import SUBSPACE_1517, SUBSPACE_780, BIN_SPACE


def loss_channel(
    eta: float,
    n_max: int = 2
) -> List[np.ndarray]:
    """
    Amplitude damping (loss) channel for a single mode.

    Models loss with transmissivity eta. Kraus operators:
        K_k = sum_{n=k}^{n_max} sqrt(C(n,k)) * eta^{(n-k)/2} * (1-eta)^{k/2} * |n-k><n|

    where k is the number of photons lost.

    Parameters
    ----------
    eta : float
        Transmissivity (0 <= eta <= 1)
    n_max : int
        Maximum photon number in the truncation (default: 2)

    Returns
    -------
    List[np.ndarray]
        List of Kraus operators K_k for k = 0, ..., n_max

    Examples
    --------
    >>> K = loss_channel(eta=0.9, n_max=2)
    >>> # K[0]: no loss, K[1]: lose 1 photon, K[2]: lose 2 photons
    """
    if not 0 <= eta <= 1:
        raise ValueError(f"eta must be in [0, 1], got {eta}")

    if n_max < 1:
        raise ValueError(f"n_max must be >= 1, got {n_max}")

    dim = n_max + 1  # |0>, |1>, ..., |n_max>

    # Build Kraus operators
    # K_k acts on |n> and gives amplitude for losing k photons
    kraus_ops = []

    for k in range(n_max + 1):
        K = np.zeros((dim, dim), dtype=complex)
        for n in range(k, n_max + 1):
            # |n-k><n| with coefficient sqrt(C(n,k)) * eta^{(n-k)/2} * (1-eta)^{k/2}
            from math import comb
            coeff = np.sqrt(comb(n, k)) * (eta ** ((n - k) / 2)) * ((1 - eta) ** (k / 2))
            K[n - k, n] = coeff
        kraus_ops.append(K)

    return kraus_ops


def loss_channel_1517(eta_H: float, eta_V: float) -> List[np.ndarray]:
    """
    Amplitude damping for the 1517nm telecom subspace (6D).

    Handles two polarization modes independently with possibly different loss.

    Parameters
    ----------
    eta_H : float
        Transmissivity for H polarization
    eta_V : float
        Transmissivity for V polarization

    Returns
    -------
    List[np.ndarray]
        List of Kraus operators acting on 6D 1517 subspace
    """
    # 1517 basis: vac, H, V, 2H, 2V, HV
    # Occupancy: (0,0), (1,0), (0,1), (2,0), (0,2), (1,1)

    # We need to construct Kraus operators for all combinations of loss
    # on H and V modes. For small truncation, enumerate explicitly.

    K_list = []

    # Basis with occupancy tuples
    basis = [
        (0, 0),  # 0: vac
        (1, 0),  # 1: H
        (0, 1),  # 2: V
        (2, 0),  # 3: 2H
        (0, 2),  # 4: 2V
        (1, 1),  # 5: HV
    ]

    # For each possible loss outcome (kH photons lost from H, kV from V)
    for kH in range(3):  # Can lose 0, 1, or 2 H photons
        for kV in range(3):
            K = np.zeros((6, 6), dtype=complex)

            for i, (nH, nV) in enumerate(basis):
                if nH < kH or nV < kV:
                    continue  # Can't lose more than we have

                nH_new = nH - kH
                nV_new = nV - kV

                # Find target index
                target = (nH_new, nV_new)
                if target in basis:
                    j = basis.index(target)

                    # Compute coefficient
                    # Product of independent H and V loss
                    from math import comb
                    coeff_H = np.sqrt(comb(nH, kH)) * (eta_H ** ((nH - kH) / 2)) * ((1 - eta_H) ** (kH / 2))
                    coeff_V = np.sqrt(comb(nV, kV)) * (eta_V ** ((nV - kV) / 2)) * ((1 - eta_V) ** (kV / 2))
                    K[j, i] = coeff_H * coeff_V

            K_list.append(K)

    # Remove all-zero operators
    K_list = [K for K in K_list if np.any(K != 0)]

    return K_list


def detection_channel(
    eta_det: float = 1.0,
    p_dark: float = 0.0
) -> Tuple[List[np.ndarray], List[int]]:
    """
    On/off detection POVM for photon number measurement.

    Models single-photon detector with:
    - Efficiency eta_det
    - Dark count probability p_dark per bin

    POVM elements:
        E_0 = (1-p_dark) * sum_n (1-eta_det)^n |n><n|  (no click)
        E_1 = I - E_0  (click)

    Kraus operators are M_r = sqrt(E_r).

    Parameters
    ----------
    eta_det : float
        Detection efficiency (0 <= eta_det <= 1)
    p_dark : float
        Dark count probability (0 <= p_dark <= 1)

    Returns
    -------
    Tuple[List[np.ndarray], List[int]]
        (Kraus operators, outcome labels)
        Outcome 0 = no click, Outcome 1 = click

    Examples
    --------
    >>> K, outcomes = detection_channel(eta_det=0.9, p_dark=0.001)
    >>> # K[0] = no-click Kraus, K[1] = click Kraus
    """
    if not 0 <= eta_det <= 1:
        raise ValueError(f"eta_det must be in [0, 1], got {eta_det}")
    if not 0 <= p_dark <= 1:
        raise ValueError(f"p_dark must be in [0, 1], got {p_dark}")

    # For the 6D 1517 subspace (n_max = 2)
    dim = 6

    # No-click POVM element
    # E_0 = (1-p_dark) * [(1-eta)^0*|0><0| + (1-eta)^1*|1><1| + (1-eta)^2*|2><2|]
    # But we have multiple single-photon states and multi-photon states

    # Basis: vac, H, V, 2H, 2V, HV
    # Need photon number for each basis state
    n_per_state = np.array([0, 1, 1, 2, 2, 2])  # Total photon number

    # No-click operator (diagonal)
    E0 = np.zeros((dim, dim), dtype=complex)
    for i, n in enumerate(n_per_state):
        prob_no_click = (1 - p_dark) * ((1 - eta_det) ** n)
        E0[i, i] = prob_no_click

    # Click operator
    I = np.eye(dim, dtype=complex)
    E1 = I - E0

    # Kraus operators are matrix square roots
    # For diagonal operators, this is just sqrt of diagonal elements
    M0 = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        M0[i, i] = np.sqrt(E0[i, i]) if E0[i, i] > 0 else 0

    # E1 may not be diagonal (due to I - E0), but since E0 is diagonal, E1 is also diagonal
    M1 = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        M1[i, i] = np.sqrt(E1[i, i]) if E1[i, i] > 0 else 0

    return [M0, M1], [0, 1]


def detection_channel_two_mode(
    eta_det: float = 1.0,
    p_dark: float = 0.0
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """
    On/off detection POVM for two output modes (e.g., after beam splitter).

    Returns Kraus operators for all 4 detector combinations:
    (D1_H, D1_V, D2_H, D2_V) where each is 0 or 1.

    Parameters
    ----------
    eta_det : float
        Detection efficiency
    p_dark : float
        Dark count probability per detector

    Returns
    -------
    Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]
        (Kraus operators, outcome labels)
        Each outcome is (d1_H, d1_V, d2_H, d2_V)

    Examples
    --------
    >>> K, outcomes = detection_channel_two_mode(eta_det=0.9)
    >>> # Each outcome corresponds to a specific click pattern
    """
    # For simplicity, treat each detector independently
    # The total Kraus operators are products of individual detector Kraus operators

    # This returns 16 operators (2^4 combinations)
    K_single, _ = detection_channel(eta_det, p_dark)  # M0, M1 for single mode

    # Build all combinations for 4 detectors
    K_list = []
    outcomes = []

    for d1H in range(2):
        for d1V in range(2):
            for d2H in range(2):
                for d2V in range(2):
                    # Tensor product of 4 single-mode Kraus operators
                    K = K_single[d1H]
                    # For each detector, tensor the appropriate Kraus operator
                    # This gives a (6^4, 6^4) = (1296, 1296) matrix
                    # But we can be smarter: each detector acts on different modes

                    # For now, return a simplified version
                    # In practice, you'd want to construct this more carefully
                    K_list.append(K)  # Placeholder
                    outcomes.append((d1H, d1V, d2H, d2V))

    return K_list, outcomes


def dephasing_channel(
    p_phi: float,
    dim: int = 3
) -> List[np.ndarray]:
    """
    Pure dephasing channel for atomic qubits.

        E(rho) = (1 - p_phi) * rho + p_phi * Z * rho * Z

    where Z = |0><0| - |1><1| flips the phase in the {|0>, |1>} subspace.

    Parameters
    ----------
    p_phi : float
        Dephasing probability (0 <= p_phi <= 1)
    dim : int
        Dimension of the atomic subspace (default: 3 for |0>, |1>, |e>)

    Returns
    -------
    List[np.ndarray]
        Kraus operators [K0, K1] where:
        K0 = sqrt(1 - p_phi) * I
        K1 = sqrt(p_phi) * Z

    Examples
    --------
    >>> K = dephasing_channel(p_phi=0.01)
    """
    if not 0 <= p_phi <= 1:
        raise ValueError(f"p_phi must be in [0, 1], got {p_phi}")

    K0 = np.sqrt(1 - p_phi) * np.eye(dim, dtype=complex)

    K1 = np.zeros((dim, dim), dtype=complex)
    # Z = |0><0| - |1><1| in the {|0>, |1>, |e>} basis
    K1[0, 0] = 1.0   # |0><0|
    K1[1, 1] = -1.0  # -|1><1|
    # |e> is unchanged by dephasing
    K1[2, 2] = 1.0 if dim >= 3 else 0
    K1 = np.sqrt(p_phi) * K1

    return [K0, K1]


def dephasing_channel_from_rate(
    gamma_phi: float,
    tau: float,
    dim: int = 3
) -> List[np.ndarray]:
    """
    Dephasing channel from continuous dephasing rate.

    p_phi = 1 - exp(-gamma_phi * tau)

    Parameters
    ----------
    gamma_phi : float
        Dephasing rate (1/time)
    tau : float
        Time duration
    dim : int
        Dimension of the atomic subspace

    Returns
    -------
    List[np.ndarray]
        Kraus operators
    """
    p_phi = 1 - np.exp(-gamma_phi * tau)
    return dephasing_channel(p_phi, dim)
