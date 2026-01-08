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


def loss_channel_780_100() -> np.ndarray:
    """
    100% loss channel for the 780nm subspace (complete filtering).

    All 780nm photons are lost (mapped to vacuum). This is a single Kraus operator:
        K = |vac><vac| + |vac><H| + |vac><V|

    acting on the 780 subspace with basis {|vac>, |H>, |V>}.

    Returns
    -------
    np.ndarray
        Single Kraus operator (3x3) that maps all 780 states to vacuum
    """
    K = np.zeros((3, 3), dtype=complex)
    K[0, 0] = 1.0  # |vac><vac| - vacuum stays vacuum
    K[0, 1] = 1.0  # |vac><H| - H photon is lost
    K[0, 2] = 1.0  # |vac><V| - V photon is lost
    return K


def loss_channel_both_subspaces(
    eta_780: float,
    eta_H_1517: float,
    eta_V_1517: float
) -> List[np.ndarray]:
    """
    Combined loss channel acting on both 780 and 1517 subspaces.

    For QFC applications: 780nm channel typically has eta_780=0 (100% filtered),
    while 1517nm channel has normal transmission loss.

    The Kraus operators are tensor products: K_780^(k) ⊗ K_1517^(j)

    Parameters
    ----------
    eta_780 : float
        Transmissivity for 780nm subspace (0 = 100% loss/filtered)
    eta_H_1517 : float
        Transmissivity for 1517nm H polarization
    eta_V_1517 : float
        Transmissivity for 1517nm V polarization

    Returns
    -------
    List[np.ndarray]
        List of Kraus operators acting on 18D bin space
    """
    # Get Kraus operators for each subspace (NOT embedded)
    K_780_list = loss_channel_780_general(eta_780)  # 3x3 matrices
    K_1517_list = _loss_channel_1517_raw(eta_H_1517, eta_V_1517)  # 6x6 matrices

    # Form all tensor product combinations
    K_combined = []
    for K_780 in K_780_list:
        for K_1517 in K_1517_list:
            # K_780 is (3,3), K_1517 is (6,6), result is (18,18)
            K_combined.append(np.kron(K_780, K_1517))

    return K_combined


def _loss_channel_1517_raw(eta_H: float, eta_V: float) -> List[np.ndarray]:
    """
    Raw 1517nm loss channel (6x6 matrices, NOT embedded into 18D).

    This is an internal function used by loss_channel_both_subspaces.

    Parameters
    ----------
    eta_H : float
        Transmissivity for H polarization
    eta_V : float
        Transmissivity for V polarization

    Returns
    -------
    List[np.ndarray]
        List of 6x6 Kraus operators acting on 1517 subspace only
    """
    K_list_1517 = []

    basis = [
        (0, 0),  # 0: vac
        (1, 0),  # 1: H
        (0, 1),  # 2: V
        (2, 0),  # 3: 2H
        (0, 2),  # 4: 2V
        (1, 1),  # 5: HV
    ]

    for kH in range(3):
        for kV in range(3):
            K = np.zeros((6, 6), dtype=complex)

            for i, (nH, nV) in enumerate(basis):
                if nH < kH or nV < kV:
                    continue

                nH_new = nH - kH
                nV_new = nV - kV

                target = (nH_new, nV_new)
                if target in basis:
                    j = basis.index(target)

                    from math import comb
                    coeff_H = np.sqrt(comb(nH, kH)) * (eta_H ** ((nH - kH) / 2)) * ((1 - eta_H) ** (kH / 2))
                    coeff_V = np.sqrt(comb(nV, kV)) * (eta_V ** ((nV - kV) / 2)) * ((1 - eta_V) ** (kV / 2))
                    K[j, i] = coeff_H * coeff_V

            K_list_1517.append(K)

    # Remove all-zero operators
    K_list_1517 = [K for K in K_list_1517 if np.any(K != 0)]
    return K_list_1517


def loss_channel_780_general(eta: float) -> List[np.ndarray]:
    """
    General loss channel for 780nm subspace (up to 1 photon per mode).

    For eta=0: 100% loss (3 Kraus operators: |vac><vac|, |vac><H|, |vac><V|)
    For eta=1: no loss (K_0 = I)

    Parameters
    ----------
    eta : float
        Transmissivity (0 <= eta <= 1)

    Returns
    -------
    List[np.ndarray]
        Kraus operators for 780 subspace (3x3 matrices)
    """
    # Basis: |vac>, |H>, |V>

    K_list = []

    if eta == 0.0:
        # 100% loss: 3 Kraus operators
        # K_0 = |vac><vac| (vacuum stays vacuum)
        K0 = np.zeros((3, 3), dtype=complex)
        K0[0, 0] = 1.0
        K_list.append(K0)

        # K_1 = |vac><H| (H photon lost -> vacuum)
        K1 = np.zeros((3, 3), dtype=complex)
        K1[0, 1] = 1.0
        K_list.append(K1)

        # K_2 = |vac><V| (V photon lost -> vacuum)
        K2 = np.zeros((3, 3), dtype=complex)
        K2[0, 2] = 1.0
        K_list.append(K2)
    elif eta == 1.0:
        # No loss: identity
        K_list.append(np.eye(3, dtype=complex))
    else:
        # Partial loss: K_0 (no loss) and K_H, K_V (loss per mode)
        K0 = np.zeros((3, 3), dtype=complex)
        K0[0, 0] = 1.0
        K0[1, 1] = np.sqrt(eta)
        K0[2, 2] = np.sqrt(eta)
        K_list.append(K0)

        # Loss operators for each mode
        loss_amp = np.sqrt(1 - eta)

        K_H = np.zeros((3, 3), dtype=complex)
        K_H[0, 1] = loss_amp
        K_list.append(K_H)

        K_V = np.zeros((3, 3), dtype=complex)
        K_V[0, 2] = loss_amp
        K_list.append(K_V)

    return K_list


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
    Amplitude damping for the 1517nm telecom subspace (6D),
    embedded in the 18D bin space (I_780 ⊗ K_1517).

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
        List of Kraus operators acting on 18D bin space (780 × 1517)
    """
    # 1517 basis: vac, H, V, 2H, 2V, HV
    # Occupancy: (0,0), (1,0), (0,1), (2,0), (0,2), (1,1)

    # We need to construct Kraus operators for all combinations of loss
    # on H and V modes. For small truncation, enumerate explicitly.

    K_list_1517 = []

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

            K_list_1517.append(K)

    # Remove all-zero operators
    K_list_1517 = [K for K in K_list_1517 if np.any(K != 0)]

    # Embed each Kraus operator into 18D bin space: I_780 ⊗ K_1517
    I_780 = np.eye(3, dtype=complex)
    K_list_embedded = [np.kron(I_780, K) for K in K_list_1517]

    return K_list_embedded


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
