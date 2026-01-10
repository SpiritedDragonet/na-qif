"""
Wave Packet Visualization

This module provides functions to extract and visualize wave packets
from MPS states, including intensity envelopes and single-photon probabilities.
"""

from typing import Tuple, Optional, List, Union
import numpy as np
import matplotlib.pyplot as plt

from ..core.mps import MPSState
from ..config import TimeGrid
from ..simulation.trajectory import EmissionResult


# ============================================================================
# Operators for Wave Packet Extraction
# ============================================================================

def _telecom_ops_1517():
    """
    Construct projection and number operators for the 1517nm subspace.

    1517 basis: [vac, 1H, 1V, 2H, 2V, HV]

    Returns
    -------
    Tuple of np.ndarray
        (P1_1517, N_1517, P1H_1517, P1V_1517, NH_1517, NV_1517)
        - P1_1517: Single-photon projection (all pol)
        - N_1517: Total photon number operator
        - P1H_1517: Single H-photon projection
        - P1V_1517: Single V-photon projection
        - NH_1517: H-photon number operator
        - NV_1517: V-photon number operator
    """
    # Basis: [vac, 1H, 1V, 2H, 2V, HV]
    # Single-photon projection (all pol)
    P1_1517 = np.diag([0, 1, 1, 0, 0, 0]).astype(complex)

    # Total photon number
    N_1517 = np.diag([0, 1, 1, 2, 2, 2]).astype(complex)

    # H/V single-photon projections
    P1H_1517 = np.diag([0, 1, 0, 0, 0, 0]).astype(complex)
    P1V_1517 = np.diag([0, 0, 1, 0, 0, 0]).astype(complex)

    # H/V photon numbers
    NH_1517 = np.diag([0, 1, 0, 2, 0, 1]).astype(complex)
    NV_1517 = np.diag([0, 0, 1, 0, 2, 1]).astype(complex)

    return P1_1517, N_1517, P1H_1517, P1V_1517, NH_1517, NV_1517


def telecom_ops_bin18():
    """
    Construct telecom operators embedded in the 18D bin space.

    Bin space = 780(3D) x 1517(6D) = 18D
    Assumes flatten order: |i_780> ⊗ |j_1517>, index = i_780 * 6 + j_1517

    Returns
    -------
    Tuple of np.ndarray
        (P1_bin, N_bin, P1H_bin, P1V_bin, NH_bin, NV_bin)
        Each is 18x18 acting on the full bin space
    """
    I_780 = np.eye(3, dtype=complex)

    P1_1517, N_1517, P1H_1517, P1V_1517, NH_1517, NV_1517 = _telecom_ops_1517()

    # Embed: I_780 ⊗ Op_1517
    P1_bin = np.kron(I_780, P1_1517)
    N_bin = np.kron(I_780, N_1517)
    P1H_bin = np.kron(I_780, P1H_1517)
    P1V_bin = np.kron(I_780, P1V_1517)
    NH_bin = np.kron(I_780, NH_1517)
    NV_bin = np.kron(I_780, NV_1517)

    return P1_bin, N_bin, P1H_bin, P1V_bin, NH_bin, NV_bin


# ============================================================================
# Wave Packet Extraction
# ============================================================================

def extract_wavepacket(
    mps: MPSState,
    n_bins: int,
    use_single_photon_prob: bool = True,
    polarized: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract wave packet data from MPS state.

    Chain layout: atomA, atomB, A1, B1, A2, B2, ..., AN, BN
    Site indices: A_n = 2 + 2*(n-1), B_n = 3 + 2*(n-1)

    Parameters
    ----------
    mps : MPSState
        MPS state (access mps._mps for TeNPy MPS)
    n_bins : int
        Number of time bins to extract
    use_single_photon_prob : bool
        If True, return single-photon probability q_n
        If False, return intensity <N_n>
    polarized : bool
        If True, return separate H and V components
        If False, return total (H+V)

    Returns
    -------
    Tuple of np.ndarray
        (data_A, data_B) where each is:
        - (n_bins,) if polarized=False
        - (n_bins, 2) if polarized=True [H, V columns]
    """
    P1_bin, N_bin, P1H_bin, P1V_bin, NH_bin, NV_bin = telecom_ops_bin18()

    # Choose operator based on mode
    if use_single_photon_prob:
        if polarized:
            OpA = P1H_bin  # Will use separate H/V
            OpB = P1V_bin
        else:
            OpA = P1_bin
            OpB = P1_bin
    else:
        if polarized:
            OpA = NH_bin
            OpB = NV_bin
        else:
            OpA = N_bin
            OpB = N_bin

    # Initialize arrays
    if polarized:
        data_A = np.zeros((n_bins, 2))
        data_B = np.zeros((n_bins, 2))
    else:
        data_A = np.zeros(n_bins)
        data_B = np.zeros(n_bins)

    for n in range(1, n_bins + 1):
        idx_A = 2 + 2 * (n - 1)  # A_n site index
        idx_B = 3 + 2 * (n - 1)  # B_n site index

        # Get reduced density matrices
        rhoA = mps.get_reduced_density([idx_A])
        rhoB = mps.get_reduced_density([idx_B])

        if polarized:
            # H component
            data_A[n - 1, 0] = float(np.real(np.trace(rhoA @ P1H_bin)))
            data_B[n - 1, 0] = float(np.real(np.trace(rhoB @ P1H_bin)))
            # V component
            data_A[n - 1, 1] = float(np.real(np.trace(rhoA @ P1V_bin)))
            data_B[n - 1, 1] = float(np.real(np.trace(rhoB @ P1V_bin)))
        else:
            data_A[n - 1] = float(np.real(np.trace(rhoA @ OpA)))
            data_B[n - 1] = float(np.real(np.trace(rhoB @ OpB)))

    return data_A, data_B


def extract_intensity_envelope(
    mps: MPSState,
    n_bins: int,
    polarized: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract intensity envelope <N_n> for each bin.

    Parameters
    ----------
    mps : MPSState
        MPS state
    n_bins : int
        Number of time bins
    polarized : bool
        If True, return H and V separately

    Returns
    -------
    Tuple of np.ndarray
        (pA, pB) intensity arrays
    """
    return extract_wavepacket(
        mps, n_bins,
        use_single_photon_prob=False,
        polarized=polarized
    )


def extract_single_photon_prob(
    mps: MPSState,
    n_bins: int,
    polarized: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract single-photon probability for each bin.

    Parameters
    ----------
    mps : MPSState
        MPS state
    n_bins : int
        Number of time bins
    polarized : bool
        If True, return H and V separately

    Returns
    -------
    Tuple of np.ndarray
        (qA, qB) probability arrays
    """
    return extract_wavepacket(
        mps, n_bins,
        use_single_photon_prob=True,
        polarized=polarized
    )


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_wavepacket(
    data_A: np.ndarray,
    data_B: np.ndarray,
    time_grid: Optional[TimeGrid] = None,
    polarized: bool = False,
    normalize: bool = False,
    title: str = "Wave Packet",
    labels: Optional[Tuple[str, str]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot wave packet for both arms.

    Parameters
    ----------
    data_A : np.ndarray
        Arm A data (n_bins,) or (n_bins, 2) for polarized
    data_B : np.ndarray
        Arm B data
    time_grid : TimeGrid, optional
        Time grid for x-axis
    polarized : bool
        Whether data is polarized (H/V separate)
    normalize : bool
        If True, normalize to sum=1
    title : str
        Plot title
    labels : Tuple[str, str], optional
        Legend labels for arms A and B
    ax : plt.Axes, optional
        Existing axes to plot on

    Returns
    -------
    plt.Axes
        The axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    n_bins = len(data_A)
    if time_grid is None:
        x = np.arange(n_bins)
        xlabel = "Bin index"
    else:
        x = time_grid.t[:n_bins]
        xlabel = "Time (s)"

    # Normalize if requested
    if normalize:
        if polarized:
            data_A = data_A / (data_A.sum() + 1e-15)
            data_B = data_B / (data_B.sum() + 1e-15)
        else:
            data_A = data_A / (data_A.sum() + 1e-15)
            data_B = data_B / (data_B.sum() + 1e-15)

    # Plot
    if polarized:
        # data_A and data_B are (n_bins, 2) with [H, V] columns
        ax.plot(x, data_A[:, 0], '--', label=f"A: H" if labels is None else labels[0] + " H",
                color='tab:blue', alpha=0.7)
        ax.plot(x, data_A[:, 1], '--', label=f"A: V" if labels is None else labels[0] + " V",
                color='tab:blue', alpha=0.9)
        ax.plot(x, data_B[:, 0], '-', label=f"B: H" if labels is None else labels[1] + " H",
                color='tab:orange', alpha=0.7)
        ax.plot(x, data_B[:, 1], '-', label=f"B: V" if labels is None else labels[1] + " V",
                color='tab:orange', alpha=0.9)
    else:
        label_A = "Arm A" if labels is None else labels[0]
        label_B = "Arm B" if labels is None else labels[1]
        ax.plot(x, data_A, '-', label=label_A, color='tab:blue')
        ax.plot(x, data_B, '-', label=label_B, color='tab:orange')

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability" if normalize else "Intensity / <N>")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_intensity_envelope(
    mps: MPSState,
    n_bins: int,
    time_grid: Optional[TimeGrid] = None,
    polarized: bool = False,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot intensity envelope <N_n> for both arms.

    Parameters
    ----------
    mps : MPSState
        MPS state
    n_bins : int
        Number of time bins
    time_grid : TimeGrid, optional
        Time grid
    polarized : bool
        Whether to show H/V separately
    ax : plt.Axes, optional
        Existing axes

    Returns
    -------
    plt.Axes
    """
    data_A, data_B = extract_intensity_envelope(mps, n_bins, polarized=polarized)
    return plot_wavepacket(
        data_A, data_B, time_grid,
        polarized=polarized,
        normalize=False,
        title="Intensity Envelope <N>",
        ax=ax
    )


def plot_single_photon_prob(
    mps: MPSState,
    n_bins: int,
    time_grid: Optional[TimeGrid] = None,
    polarized: bool = False,
    normalize: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot single-photon probability for both arms.

    Parameters
    ----------
    mps : MPSState
        MPS state
    n_bins : int
        Number of time bins
    time_grid : TimeGrid, optional
        Time grid
    polarized : bool
        Whether to show H/V separately
    normalize : bool
        Whether to normalize (recommended)
    ax : plt.Axes, optional
        Existing axes

    Returns
    -------
    plt.Axes
    """
    data_A, data_B = extract_single_photon_prob(mps, n_bins, polarized=polarized)
    return plot_wavepacket(
        data_A, data_B, time_grid,
        polarized=polarized,
        normalize=normalize,
        title="Single-Photon Probability",
        ax=ax
    )


def plot_mode_overlap(
    data_A: np.ndarray,
    data_B: np.ndarray,
    time_grid: Optional[TimeGrid] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot mode overlap between two arms.

    Overlap M = |sum_n (xi_A_n* xi_B_n)|^2

    Parameters
    ----------
    data_A : np.ndarray
        Complex amplitudes for arm A
    data_B : np.ndarray
        Complex amplitudes for arm B
    time_grid : TimeGrid, optional
        Time grid
    ax : plt.Axes, optional
        Existing axes

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    n_bins = len(data_A)
    if time_grid is None:
        x = np.arange(n_bins)
        xlabel = "Bin index"
    else:
        x = time_grid.t[:n_bins]
        xlabel = "Time (s)"

    # Compute overlap
    overlap = np.abs(np.sum(data_A * np.conj(data_B))) ** 2

    ax.plot(x, np.real(data_A), '--', label='Re(A)', color='tab:blue', alpha=0.7)
    ax.plot(x, np.imag(data_A), ':', label='Im(A)', color='tab:blue', alpha=0.5)
    ax.plot(x, np.real(data_B), '--', label='Re(B)', color='tab:orange', alpha=0.7)
    ax.plot(x, np.imag(data_B), ':', label='Im(B)', color='tab:orange', alpha=0.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Mode Overlap M = {overlap:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


# ============================================================================
# Bin State Heatmap Visualization
# ============================================================================

def _get_bin18_state_labels() -> List[str]:
    """
    Get labels for the 18 bin states.

    Bin space = 780(3D) x 1517(6D) with index = i_780 * 6 + i_1517

    780 subspace: only supports 0 or 1 photon (|vac>, |H>, |V>)
    1517 subspace: supports up to 2 photons (|vac>, |H>, |V>, |2H>, |2V>, |HV>)

    Returns
    -------
    List[str]
        18 state labels in the form |780,1517>
    """
    # 780 basis states (single photon only)
    bases_780 = ['|vac>', '|H>', '|V>']
    # 1517 basis states (up to two photons)
    bases_1517 = ['|vac>', '|H>', '|V>', '|2H>', '|2V>', '|HV>']

    labels = []
    for i_780, b780 in enumerate(bases_780):
        for i_1517, b1517 in enumerate(bases_1517):
            # Format: |780_state, 1517_state>
            # Remove angle brackets for cleaner display, keep structure clear
            if b780 == '|vac>' and b1517 == '|vac>':
                label = '|vac,vac>'  # Both vacuum
            elif b780 == '|vac>':
                label = f'|vac,{b1517[1:-1]}>'  # Only 1517 state
            elif b1517 == '|vac>':
                label = f'|{b780[1:-1]},vac>'  # Only 780 state
            else:
                # Both non-vacuum: show both states
                label = f'|{b780[1:-1]},{b1517[1:-1]}>'
            labels.append(label)

    return labels


def extract_bin_state_probabilities(
    mps: MPSState,
    arm: str = 'A',
    n_bins: int = None,
    atom_at_end: bool = False,
) -> np.ndarray:
    """
    Extract probability for each of the 18 bin states across all time bins.

    For each bin, computes the reduced density matrix and extracts
    the diagonal elements (probabilities for each of the 18 states).

    Supports multiple chain layouts:
    1. Dual-atom: atomA, atomB, A1, B1, A2, B2, ..., AN, BN
    2. Single-atom (no SWAP): atom, bin1, bin2, ..., binN
    3. Single-atom (after SWAP): bin1, bin2, ..., binN, atom

    Parameters
    ----------
    mps : MPSState
        MPS state
    arm : str
        Which arm to extract ('A' or 'B'). Ignored for single-atom layout.
    n_bins : int, optional
        Number of time bins. If None, infers from chain length.
    atom_at_end : bool
        If True, assumes single-atom layout with atom at the end
        (after SWAP conveyor belt). Bins are at sites 0 to n_bins-1.

    Returns
    -------
    np.ndarray
        Probability array of shape (n_bins, 18)
        prob[i, j] = probability of state j in bin i
    """
    if n_bins is None:
        # Auto-detect chain layout
        # Count how many 3D sites (atoms) vs 18D sites (bins)
        n_3d = sum(1 for d in mps.d if d == 3)
        if n_3d == 1:
            # Single atom: L = 1 + n_bins
            n_bins = mps.L - 1
        else:
            # Dual atom: L = 2 + 2 * n_bins
            n_bins = (mps.L - 2) // 2

    # Detect chain type by checking dimensions
    # Single-atom: exactly one 3D site (atom), rest are 18D (bins)
    # Dual-atom: exactly two 3D sites (atoms), rest are 18D (bins)
    n_3d = sum(1 for d in mps.d if d == 3)
    is_single_atom = n_3d == 1

    # Array to store probabilities: (n_bins, 18)
    probs = np.zeros((n_bins, 18))

    if is_single_atom:
        if atom_at_end:
            # After SWAP: bins at sites 0, 1, ..., n_bins-1, atom at site n_bins
            # But we need to be careful: mps.L = 1 + n_bins, and atom is at the end
            # The actual number of bins might be less than mps.L - 1 if atom is at the end
            # Find where the 3D atom site is
            atom_site = next(i for i, d in enumerate(mps.d) if d == 3)
            actual_n_bins = min(n_bins, atom_site)  # Don't go past the atom
            for n in range(actual_n_bins):
                idx = n  # bin indices are 0, 1, ..., n_bins-1
                rho = mps.get_reduced_density([idx])
                # Handle both 3D (atom) and 18D (bin) cases
                if rho.shape[0] == 18:
                    probs[n, :] = np.diag(rho).real
                # else: skip 3D sites (atoms)
        else:
            # Before SWAP: atom at site 0, bins at sites 1, 2, ..., n_bins
            # Find atom site (3D) and start from there
            atom_site = next(i for i, d in enumerate(mps.d) if d == 3)
            for n in range(n_bins):
                idx = atom_site + 1 + n  # bins follow atom
                if idx >= mps.L:
                    continue  # skip if out of bounds
                rho = mps.get_reduced_density([idx])
                if rho.shape[0] == 18:
                    probs[n, :] = np.diag(rho).real
    else:
        # Dual-atom layout: atomA, atomB, A1, B1, A2, B2, ..., AN, BN
        for n in range(n_bins):
            if arm.upper() == 'A':
                idx = 2 + 2 * n  # A_n site index
            else:
                idx = 3 + 2 * n  # B_n site index
            rho = mps.get_reduced_density([idx])
            probs[n, :] = np.diag(rho).real

    return probs


def plot_bin_state_heatmap(
    mps: MPSState,
    arm: str = 'A',
    n_bins: int = None,
    time_grid: Optional[TimeGrid] = None,
    subspace: str = 'both',
    group_by: str = '780',
    vmax: float = None,
    figsize: tuple = (12, 8),
    ax: Optional[plt.Axes] = None,
    atom_at_end: bool = False,
    separate_vac_scale: bool = False,
) -> plt.Axes:
    """
    Plot heatmap of bin state probabilities across time bins.

    Creates a heatmap showing the probability of each of the 18 bin states
    for each time bin. Rows are the 18 states (labeled on the left),
    columns are time bins (labeled with time on bottom).

    Parameters
    ----------
    mps : MPSState
        MPS state
    arm : str
        Which arm to plot ('A' or 'B')
    n_bins : int, optional
        Number of time bins. If None, infers from chain length.
    time_grid : TimeGrid, optional
        Time grid for x-axis labels
    subspace : str
        Which subspace to show ('780', '1517', or 'both')
    group_by : str
        How to group separator lines ('780' or '1517')
        - '780': Group by 780 state (vac/H/V) - lines at 5.5, 11.5
        - '1517': Group by 1517 photon number (0/1/2 photons) - lines at 2.5, 5.5 across all 780 states
    vmax : float, optional
        Maximum value for color scale. If None, auto-scales.
    figsize : tuple
        Figure size (width, height)
    ax : plt.Axes, optional
        Existing axes to plot on
    atom_at_end : bool
        If True, assumes single-atom layout with atom at the end
        (after SWAP conveyor belt). Use this for test_emission_wavepacket.py results.
    separate_vac_scale : bool
        If True, uses a separate color scale for the |vac,vac> row (index 0)
        to better visualize small changes in vacuum probability.

    Returns
    -------
    plt.Axes
        The axes object
    """
    import matplotlib

    # Extract probabilities
    probs = extract_bin_state_probabilities(mps, arm=arm, n_bins=n_bins, atom_at_end=atom_at_end)
    n_bins_actual = probs.shape[0]

    # Get state labels
    state_labels = _get_bin18_state_labels()

    # Filter by subspace if requested
    if subspace == '780':
        # Show only states where 780 is not vacuum (indices 6-17)
        # Or reorganize to show 780 subspace structure
        row_indices = list(range(18))
        row_labels = state_labels
        title_suffix = " (780nm subspace highlighted)"
    elif subspace == '1517':
        row_indices = list(range(18))
        row_labels = state_labels
        title_suffix = " (1517nm subspace)"
    else:  # 'both'
        row_indices = list(range(18))
        row_labels = state_labels
        title_suffix = ""

    # Filter data
    probs_filtered = probs[:, row_indices]

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    # Create heatmap with separate scale for vac,vac if requested
    if separate_vac_scale:
        # Split: row 0 (|vac,vac>) and rows 1-17
        from matplotlib.colors import Normalize

        # For vac,vac row: use its own range centered at 1
        vac_row = probs_filtered[:, 0:1].T
        vac_vmin = max(0, vac_row.min() - 0.05)
        vac_vmax = min(1, vac_row.max() + 0.05)

        # For other rows: use auto-scale or provided vmax
        other_rows = probs_filtered[:, 1:].T
        if vmax is None:
            other_vmax = max(0.01, other_rows.max())
        else:
            other_vmax = vmax

        # Create combined data for display with separate normalization
        # We'll use two imshow calls stacked
        display_data = probs_filtered.T

        # Create a masked array for two different normalizations
        im = ax.imshow(
            display_data,
            aspect='auto',
            cmap='viridis',
            vmin=0,
            vmax=max(vac_vmax, other_vmax) if vmax is None else vmax,
            origin='upper'
        )
    else:
        # Standard single-scale heatmap
        im = ax.imshow(
            probs_filtered.T,  # Transpose so states are rows, bins are columns
            aspect='auto',
            cmap='viridis',
            vmin=0,
            vmax=vmax,
            origin='upper'
        )

    # Set y-axis labels (state names)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    # Set x-axis labels (dual: bin index on top, time on bottom)
    n_ticks = min(10, n_bins_actual)
    tick_indices = np.linspace(0, n_bins_actual - 1, n_ticks, dtype=int)

    # Bottom x-axis: time (ns)
    ax.set_xticks(tick_indices)
    if time_grid is not None:
        ax.set_xticklabels([f'{time_grid.t[i]:.1f}' for i in tick_indices])
    else:
        ax.set_xticklabels([str(i) for i in tick_indices])
    ax.set_xlabel('Time (ns)')

    # Top x-axis: bin index
    ax_top = ax.twiny()
    ax_top.set_xticks(tick_indices)
    ax_top.set_xticklabels([str(i) for i in tick_indices])
    ax_top.set_xlabel('Bin index')
    ax_top.set_xlim(ax.get_xlim())  # Sync limits

    ax.set_ylabel('Bin state |780,1517>')
    ax.set_title(f'Arm {arm.upper()} Bin State Probabilities{title_suffix}')

    # Add colorbar with better positioning to avoid overlap
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Probability')

    # Add separator lines based on group_by parameter
    if group_by == '780':
        # Group by 780 state: |vac>, |H>, |V>
        # Rows 0-5: 780=vac, Rows 6-11: 780=H, Rows 12-17: 780=V
        boundaries = [5.5, 11.5]
    else:  # '1517'
        # Group by 1517 photon number: 0/1/2 photons
        # 1517 structure repeats for each 780 state (every 6 rows)
        # vac(0), H(1), V(1) -> boundary after row 2
        # 2H(2), 2V(2), HV(2) -> boundary after row 5
        # This pattern repeats for 780=H (rows 6-11) and 780=V (rows 12-17)
        boundaries = [2.5, 5.5, 8.5, 11.5, 14.5, 17.5]

    for boundary in boundaries:
        ax.axhline(boundary, color='white', linewidth=1, alpha=0.5, linestyle='--')

    return ax


# ============================================================================
# Dual-Arm Heatmap Visualization (General Purpose)
# ============================================================================

def plot_dual_arm_heatmap(
    result: Union[EmissionResult, MPSState],
    save_path: str = "dual_arm_heatmap.png",
    show_atomic: bool = False,
    stage_name: str = "",
    time_grid: Optional[TimeGrid] = None,
    vmax_scale_factor: float = 1.5,
) -> None:
    """
    Visualize dual-arm bin state probabilities with optional atomic state display.

    General-purpose heatmap function that works for any simulation stage:
    - Emission: use show_atomic=True to show atomic state evolution
    - QFC/Jones/Loss/BS: use show_atomic=False (atoms not involved)

    Each arm shows:
    - If show_atomic=True: Top 3 rows (atomic) + Bottom 18 rows (bin states)
    - If show_atomic=False: 18 rows of bin states only

    Three colormaps:
    - Atomic states: YlOrRd (Yellow-Orange-Red)
    - (vac,vac) state: Greys (separate colorbar)
    - Other bin states: plasma

    Parameters
    ----------
    result : Union[EmissionResult, MPSState]
        Simulation result to visualize. If EmissionResult and show_atomic=True,
        atomic state evolution is extracted from result.atom_X_state_evolution.
    save_path : str
        Path to save the figure
    show_atomic : bool
        Whether to display atomic state rows (default: False)
    stage_name : str
        Stage name for title (e.g., "Emission", "QFC", "BS")
    time_grid : TimeGrid, optional
        Time grid for x-axis labels. If None and result is EmissionResult,
        uses result.time_grid.
    vmax_scale_factor : float
        Factor for scaling vmax (relative to max bin probability)
    """
    import matplotlib as mpl
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    mpl.rcParams['image.interpolation'] = 'nearest'

    # Extract MPS and time_grid from result
    if isinstance(result, EmissionResult):
        mps = result.mps
        if time_grid is None:
            time_grid = result.time_grid
        n_bins = result.get_n_bins()
        has_atom_evol = True
    else:  # MPSState
        mps = result
        if time_grid is None:
            time_grid = TimeGrid(dt=1.0, N=1)  # Dummy
        n_bins = (mps.L - 2) // 2  # Infer from chain length
        has_atom_evol = False

    if show_atomic and not has_atom_evol:
        raise ValueError("show_atomic=True requires EmissionResult with atomic state evolution")

    # Create figure with more spacing
    fig, axes = plt.subplots(1, 2, figsize=(24, 13))
    plt.subplots_adjust(left=0.04, right=0.85, top=0.80, bottom=0.06, wspace=0.50)

    # Extract atomic state evolution if needed
    if show_atomic and has_atom_evol:
        atom_A_evol = result.atom_A_state_evolution
        atom_B_evol = result.atom_B_state_evolution
        # For visualization, take every other column (after each full bin processing)
        atom_A_for_bins = atom_A_evol[:, 1::2]
        atom_B_for_bins = atom_B_evol[:, 1::2]
        # If evolution has fewer columns than bins, pad with final state
        if atom_A_for_bins.shape[1] < n_bins:
            padding = np.tile(atom_A_for_bins[:, -1:], (1, n_bins - atom_A_for_bins.shape[1]))
            atom_A_for_bins = np.hstack([atom_A_for_bins, padding])
        if atom_B_for_bins.shape[1] < n_bins:
            padding = np.tile(atom_B_for_bins[:, -1:], (1, n_bins - atom_B_for_bins.shape[1]))
            atom_B_for_bins = np.hstack([atom_B_for_bins, padding])

    # Extract bin probabilities
    probs_A = np.zeros((n_bins, 18))
    probs_B = np.zeros((n_bins, 18))

    for n in range(n_bins):
        # Chain layout: A1(0), B1(1), A2(2), B2(3), ..., AN, BN
        site_A = 2 * n
        site_B = 2 * n + 1
        rho_A = mps.get_reduced_density([site_A])
        rho_B = mps.get_reduced_density([site_B])
        if rho_A.shape[0] == 18:
            probs_A[n, :] = np.diag(rho_A).real
        if rho_B.shape[0] == 18:
            probs_B[n, :] = np.diag(rho_B).real

    # Calculate vmax EXCLUDING (vac,vac) row (index 0)
    vmax_A = max(0.01, probs_A[:, 1:].max() * vmax_scale_factor)
    vmax_B = max(0.01, probs_B[:, 1:].max() * vmax_scale_factor)
    vmax = max(vmax_A, vmax_B)

    # Get state labels
    bin_state_labels = _get_bin18_state_labels()

    # Create combined data matrices
    if show_atomic:
        atomic_labels = ['|e>', '|1>', '|0>']
        combined_labels_A = atomic_labels + bin_state_labels
        combined_labels_B = atomic_labels + bin_state_labels
        total_rows = 3 + 18

        combined_A = np.zeros((total_rows, n_bins))
        combined_A[0, :] = atom_A_for_bins[2, :]  # |e>
        combined_A[1, :] = atom_A_for_bins[1, :]  # |1>
        combined_A[2, :] = atom_A_for_bins[0, :]  # |0>
        combined_A[3:, :] = probs_A.T

        combined_B = np.zeros((total_rows, n_bins))
        combined_B[0, :] = atom_B_for_bins[2, :]  # |e>
        combined_B[1, :] = atom_B_for_bins[1, :]  # |1>
        combined_B[2, :] = atom_B_for_bins[0, :]  # |0>
        combined_B[3:, :] = probs_B.T
    else:
        combined_labels_A = bin_state_labels
        combined_labels_B = bin_state_labels
        total_rows = 18

        combined_A = probs_A.T
        combined_B = probs_B.T

    # Scientific colormaps
    # Atomic states: YlOrRd (Yellow-Orange-Red)
    # (vac,vac) state: Greys - separate colorbar for vacuum probability
    # Other bin states: plasma (purple-yellow)
    atom_cmap = plt.get_cmap('YlOrRd')
    vac_cmap = plt.get_cmap('Greys')
    bin_cmap = plt.get_cmap('plasma')

    # Create masks for different sections
    if show_atomic:
        # Three sections: Atom (rows 0-2), Vac (row 3), Bin (rows 4-20)
        mask_atom = np.zeros((total_rows, n_bins), dtype=bool)
        mask_atom[:3, :] = True
        mask_vac = np.zeros((total_rows, n_bins), dtype=bool)
        mask_vac[3, :] = True
        mask_bin = np.zeros((total_rows, n_bins), dtype=bool)
        mask_bin[4:, :] = True
        atom_row_offset = 3  # Offset for separator lines
    else:
        # Two sections: Vac (row 0), Bin (rows 1-17)
        mask_atom = None
        mask_vac = np.zeros((total_rows, n_bins), dtype=bool)
        mask_vac[0, :] = True
        mask_bin = np.zeros((total_rows, n_bins), dtype=bool)
        mask_bin[1:, :] = True
        atom_row_offset = 0

    # Plot arm A
    # First plot all bin states with plasma colormap
    im_A = axes[0].imshow(
        combined_A,
        aspect='auto',
        cmap=bin_cmap,
        vmin=0,
        vmax=vmax,
        origin='upper'
    )

    # Overlay (vac,vac) row with different colormap
    im_A_vac = axes[0].imshow(
        np.ma.masked_where(~mask_vac, combined_A),
        aspect='auto',
        cmap=vac_cmap,
        vmin=0,
        vmax=1,
        origin='upper',
        interpolation='nearest'
    )

    # Overlay atomic states if needed
    if show_atomic:
        im_A_atom = axes[0].imshow(
            np.ma.masked_where(~mask_atom, combined_A),
            aspect='auto',
            cmap=atom_cmap,
            vmin=0,
            vmax=1,
            origin='upper',
            interpolation='nearest'
        )

    axes[0].set_yticks(range(total_rows))
    axes[0].set_yticklabels(combined_labels_A, fontsize=8)
    axes[0].set_ylabel('State', fontsize=10)
    axes[0].set_title(f'Arm A - Bin State Probabilities (vmax={vmax:.3f})', fontsize=11)

    if show_atomic:
        axes[0].axhline(2.5, color='black', linewidth=2)

    # x-axis (dual: time and bin index)
    n_ticks = min(10, n_bins)
    tick_indices = np.linspace(0, n_bins - 1, n_ticks, dtype=int)
    axes[0].set_xticks(tick_indices)
    axes[0].set_xticklabels([f'{time_grid.t[i] * 1e9:.0f}' for i in tick_indices], fontsize=9)
    axes[0].set_xlabel('Time (ns)', fontsize=10)
    axes[0].top = axes[0].twiny()
    axes[0].top.set_xticks(tick_indices)
    axes[0].top.set_xticklabels([str(i) for i in tick_indices], fontsize=9)
    axes[0].top.set_xlabel('Bin index', fontsize=10)
    axes[0].top.set_xlim(axes[0].get_xlim())

    # Plot arm B
    im_B = axes[1].imshow(
        combined_B,
        aspect='auto',
        cmap=bin_cmap,
        vmin=0,
        vmax=vmax,
        origin='upper'
    )

    im_B_vac = axes[1].imshow(
        np.ma.masked_where(~mask_vac, combined_B),
        aspect='auto',
        cmap=vac_cmap,
        vmin=0,
        vmax=1,
        origin='upper',
        interpolation='nearest'
    )

    if show_atomic:
        im_B_atom = axes[1].imshow(
            np.ma.masked_where(~mask_atom, combined_B),
            aspect='auto',
            cmap=atom_cmap,
            vmin=0,
            vmax=1,
            origin='upper',
            interpolation='nearest'
        )

    axes[1].set_yticks(range(total_rows))
    axes[1].set_yticklabels(combined_labels_B, fontsize=8)
    axes[1].set_ylabel('State', fontsize=10)
    axes[1].set_title(f'Arm B - Bin State Probabilities (vmax={vmax:.3f})', fontsize=11)

    if show_atomic:
        axes[1].axhline(2.5, color='black', linewidth=2)

    axes[1].set_xticks(tick_indices)
    axes[1].set_xticklabels([f'{time_grid.t[i] * 1e9:.0f}' for i in tick_indices], fontsize=9)
    axes[1].set_xlabel('Time (ns)', fontsize=10)
    axes[1].top = axes[1].twiny()
    axes[1].top.set_xticks(tick_indices)
    axes[1].top.set_xticklabels([str(i) for i in tick_indices], fontsize=9)
    axes[1].top.set_xlabel('Bin index', fontsize=10)
    axes[1].top.set_xlim(axes[1].get_xlim())

    # Add separator lines for bin states (group by 780 state)
    for ax in axes:
        for boundary in [5.5, 11.5]:
            ax.axhline(boundary + atom_row_offset, color='white', linewidth=1, alpha=0.5, linestyle='--')

    # Add colorbars aligned with their respective sections
    ax_pos_A = axes[0].get_position()
    ax_pos_B = axes[1].get_position()
    fig_height = ax_pos_A.y1 - ax_pos_A.y0

    if show_atomic:
        # Three colorbars: Atom (3/21), Vac (1/21), Bin (17/21)
        # Atomic colorbar for arm A
        cax_A_atom = fig.add_axes([
            ax_pos_A.x1 + 0.01,
            ax_pos_A.y1 - fig_height * (3/total_rows),
            0.01,
            fig_height * (3/total_rows)
        ])
        cbar_A_atom = fig.colorbar(im_A_atom, cax=cax_A_atom)
        cbar_A_atom.set_ticks([0, 0.5, 1])
        cbar_A_atom.set_label('Atom', fontsize=9)

        # (vac,vac) colorbar for arm A
        cax_A_vac = fig.add_axes([
            ax_pos_A.x1 + 0.01,
            ax_pos_A.y1 - fig_height * (4/total_rows),
            0.01,
            fig_height * (1/total_rows)
        ])
        cbar_A_vac = fig.colorbar(im_A_vac, cax=cax_A_vac)
        cbar_A_vac.set_ticks([0, 0.5, 1])
        cbar_A_vac.set_label('Vac', fontsize=8)

        # Bin colorbar for arm A
        cax_A = fig.add_axes([
            ax_pos_A.x1 + 0.01,
            ax_pos_A.y0,
            0.01,
            fig_height * (17/total_rows)
        ])
        cbar_A = fig.colorbar(im_A, cax=cax_A)
        n_ticks_cb = 4
        tick_vals = np.linspace(0, vmax, n_ticks_cb)
        cbar_A.set_ticks(tick_vals)
        cbar_A.set_label(f'Bin (max={vmax:.3f})', fontsize=9)

        # Atomic colorbar for arm B
        cax_B_atom = fig.add_axes([
            ax_pos_B.x1 + 0.01,
            ax_pos_B.y1 - fig_height * (3/total_rows),
            0.01,
            fig_height * (3/total_rows)
        ])
        cbar_B_atom = fig.colorbar(im_B_atom, cax=cax_B_atom)
        cbar_B_atom.set_ticks([0, 0.5, 1])
        cbar_B_atom.set_label('Atom', fontsize=9)

        # (vac,vac) colorbar for arm B
        cax_B_vac = fig.add_axes([
            ax_pos_B.x1 + 0.01,
            ax_pos_B.y1 - fig_height * (4/total_rows),
            0.01,
            fig_height * (1/total_rows)
        ])
        cbar_B_vac = fig.colorbar(im_B_vac, cax=cax_B_vac)
        cbar_B_vac.set_ticks([0, 0.5, 1])
        cbar_B_vac.set_label('Vac', fontsize=8)

        # Bin colorbar for arm B
        cax_B = fig.add_axes([
            ax_pos_B.x1 + 0.01,
            ax_pos_B.y0,
            0.01,
            fig_height * (17/total_rows)
        ])
        cbar_B = fig.colorbar(im_B, cax=cax_B)
        cbar_B.set_ticks(tick_vals)
        cbar_B.set_label(f'Bin (max={vmax:.3f})', fontsize=9)
    else:
        # Two colorbars: Vac (1/18), Bin (17/18)
        # (vac,vac) colorbar for arm A
        cax_A_vac = fig.add_axes([
            ax_pos_A.x1 + 0.01,
            ax_pos_A.y1 - fig_height * (1/total_rows),
            0.01,
            fig_height * (1/total_rows)
        ])
        cbar_A_vac = fig.colorbar(im_A_vac, cax=cax_A_vac)
        cbar_A_vac.set_ticks([0, 0.5, 1])
        cbar_A_vac.set_label('Vac', fontsize=8)

        # Bin colorbar for arm A
        cax_A = fig.add_axes([
            ax_pos_A.x1 + 0.01,
            ax_pos_A.y0,
            0.01,
            fig_height * (17/total_rows)
        ])
        cbar_A = fig.colorbar(im_A, cax=cax_A)
        n_ticks_cb = 4
        tick_vals = np.linspace(0, vmax, n_ticks_cb)
        cbar_A.set_ticks(tick_vals)
        cbar_A.set_label(f'Bin (max={vmax:.3f})', fontsize=9)

        # (vac,vac) colorbar for arm B
        cax_B_vac = fig.add_axes([
            ax_pos_B.x1 + 0.01,
            ax_pos_B.y1 - fig_height * (1/total_rows),
            0.01,
            fig_height * (1/total_rows)
        ])
        cbar_B_vac = fig.colorbar(im_B_vac, cax=cax_B_vac)
        cbar_B_vac.set_ticks([0, 0.5, 1])
        cbar_B_vac.set_label('Vac', fontsize=8)

        # Bin colorbar for arm B
        cax_B = fig.add_axes([
            ax_pos_B.x1 + 0.01,
            ax_pos_B.y0,
            0.01,
            fig_height * (17/total_rows)
        ])
        cbar_B = fig.colorbar(im_B, cax=cax_B)
        cbar_B.set_ticks(tick_vals)
        cbar_B.set_label(f'Bin (max={vmax:.3f})', fontsize=9)

    # Title
    if stage_name:
        title = f'Dual-Arm Heatmap: {stage_name}'
    else:
        title = 'Dual-Arm Heatmap: Bin State Probabilities'
    plt.suptitle(title, fontsize=16, y=0.97)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved dual-arm heatmaps to: {save_path}")


# ============================================================================
# Cross-Bin First-Order Coherence (for Phase Visualization)
# ============================================================================

def _telecom_annihilation_ops():
    """
    Construct annihilation operators for the 1517nm subspace.

    1517 basis: [vac, 1H, 1V, 2H, 2V, HV]
    a_H: annihilates H photon (maps |1H> -> |vac>)
    a_V: annihilates V photon (maps |1V> -> |vac>)

    Returns
    -------
    Tuple of np.ndarray
        (a_H_1517, a_V_1517, a_H_dag_1517, a_V_dag_1517)
        Each is 6x6 complex matrix
    """
    # a_H: maps |1H> (index 1) to |vac> (index 0)
    a_H_1517 = np.zeros((6, 6), dtype=complex)
    a_H_1517[0, 1] = 1.0

    # a_V: maps |1V> (index 2) to |vac> (index 0)
    a_V_1517 = np.zeros((6, 6), dtype=complex)
    a_V_1517[0, 2] = 1.0

    # Hermitian conjugates (creation operators)
    a_H_dag_1517 = a_H_1517.conj().T
    a_V_dag_1517 = a_V_1517.conj().T

    return a_H_1517, a_V_1517, a_H_dag_1517, a_V_dag_1517


def _bin18_annihilation_ops():
    """
    Construct annihilation operators embedded in the 18D bin space.

    Bin space = 780(3D) x 1517(6D) = 18D
    We care about telecom (1517nm) photons only.

    Returns
    -------
    Tuple of np.ndarray
        (a_H_bin, a_V_bin, a_H_dag_bin, a_V_dag_bin)
        Each is 18x18 complex matrix
    """
    I_780 = np.eye(3, dtype=complex)
    a_H_1517, a_V_1517, a_H_dag_1517, a_V_dag_1517 = _telecom_annihilation_ops()

    # Embed: I_780 ⊗ a_1517
    a_H_bin = np.kron(I_780, a_H_1517)
    a_V_bin = np.kron(I_780, a_V_1517)
    a_H_dag_bin = np.kron(I_780, a_H_dag_1517)
    a_V_dag_bin = np.kron(I_780, a_V_dag_1517)

    return a_H_bin, a_V_bin, a_H_dag_bin, a_V_dag_bin


def extract_first_order_coherence(
    mps: MPSState,
    n_bins: int,
    arm: str = 'A',
    reference_bin: Optional[int] = None,
    coherence_threshold: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract first-order coherence G_{nm} = <a_n^dag a_m> for wave packet phase.

    For single-photon states, G is approximately rank-1: G_{nm} ≈ ξ_n* ξ_m
    where ξ_n is the wave packet amplitude. The phase of ξ_n gives the
    optical phase at each bin.

    Two extraction modes:
    1. reference_bin=None: Extract full G matrix, get phase from eigenvector
    2. reference_bin=int: Extract relative phase g_n = <a_ref^dag a_n>

    Parameters
    ----------
    mps : MPSState
        MPS state
    n_bins : int
        Number of time bins
    arm : str
        Which arm to extract ('A' or 'B')
    reference_bin : int, optional
        Reference bin index. If None, uses eigendecomposition.
        If specified, computes relative phase to this bin.
    coherence_threshold : float
        Minimum coherence magnitude to trust phase. Below this, phase is masked.

    Returns
    -------
    Tuple of np.ndarray
        (phases, amplitudes)
        - phases: (n_bins,) phase array in radians [-π, π]
        - amplitudes: (n_bins,) coherence magnitude array
    """
    # Get annihilation operators
    a_H, a_V, a_H_dag, a_V_dag = _bin18_annihilation_ops()

    # Use H polarization as default (can extend to both)
    a = a_H
    a_dag = a_H_dag

    # Get site indices for this arm
    # Chain layout: A1(0), B1(1), A2(2), B2(3), ..., AN, BN
    arm_indices = []
    if arm.upper() == 'A':
        for n in range(n_bins):
            arm_indices.append(2 * n)
    else:  # arm B
        for n in range(n_bins):
            arm_indices.append(2 * n + 1)

    # Method 1: Reference bin approach (faster, gives relative phase)
    if reference_bin is not None:
        if reference_bin < 0 or reference_bin >= n_bins:
            raise ValueError(f"reference_bin={reference_bin} out of range [0, {n_bins})")

        ref_site = arm_indices[reference_bin]
        phases = np.zeros(n_bins)
        amplitudes = np.zeros(n_bins)

        for i, site in enumerate(arm_indices):
            if i == reference_bin:
                # Self-correlation: <a^dag a> = number operator
                rho_ref = mps.get_reduced_density([ref_site])
                if rho_ref.shape[0] == 18:
                    N_op = a_dag @ a
                    amplitudes[i] = np.abs(np.trace(rho_ref @ N_op))
                else:
                    amplitudes[i] = 0.0
                phases[i] = 0.0  # Reference phase
            else:
                # Get two-site reduced density matrix
                # Must ensure sites are in order for consistent tensor product
                if ref_site < site:
                    sites = [ref_site, site]
                    # Construct two-site operator: a_ref^dag ⊗ a_i
                    # Operator dimension: 18x18 for each site -> 324x324 for two sites
                    op_2site = np.kron(a_dag, a)
                else:
                    sites = [site, ref_site]
                    # Order reversed: a_i ⊗ a_ref^dag
                    op_2site = np.kron(a, a_dag)

                rho_2site = mps.get_reduced_density(sites)

                # rho_2site should be (18*18) x (18*18) = 324x324
                # op_2site should also be 324x324
                if rho_2site.shape[0] == 324 and op_2site.shape[0] == 324:
                    # Compute expectation value: Tr[rho * (a_dag ⊗ a)]
                    g = np.trace(rho_2site @ op_2site)
                    phases[i] = np.angle(g)
                    amplitudes[i] = np.abs(g)
                else:
                    # Dimension mismatch, skip
                    phases[i] = 0.0
                    amplitudes[i] = 0.0

        return phases, amplitudes

    # Method 2: Full correlation matrix with eigendecomposition
    # Build G matrix where G[n,m] = <a_n^dag a_m>
    G = np.zeros((n_bins, n_bins), dtype=complex)

    for n in range(n_bins):
        for m in range(n_bins):
            if m < n:
                # Use Hermitian symmetry: G[n,m] = conj(G[m,n])
                G[n, m] = np.conj(G[m, n])
                continue

            site_n = arm_indices[n]
            site_m = arm_indices[m]

            if n == m:
                # On-site: <a_n^dag a_n> = photon number at bin n
                rho_n = mps.get_reduced_density([site_n])
                if rho_n.shape[0] == 18:
                    # Tr[rho * a^dag a] = number expectation
                    N_op = a_dag @ a
                    G[n, n] = np.trace(rho_n @ N_op)
                else:
                    G[n, n] = 0.0
            else:
                # Cross-bin correlation
                if site_n < site_m:
                    sites = [site_n, site_m]
                    op_2site = np.kron(a_dag, a)
                else:
                    sites = [site_m, site_n]
                    op_2site = np.kron(a, a_dag)

                rho_2site = mps.get_reduced_density(sites)

                if rho_2site.shape[0] == 324:
                    G[n, m] = np.trace(rho_2site @ op_2site)
                else:
                    G[n, m] = 0.0

    # Extract phases from dominant eigenvector
    # For pure single-photon states, G should be rank-1
    eigvals, eigvecs = np.linalg.eigh(G)

    # Dominant eigenvalue and eigenvector
    idx_max = np.argmax(np.abs(eigvals))
    eigenmode = eigvecs[:, idx_max]

    # Phase is the argument of the eigenmode
    # Global phase is arbitrary, so we set mean phase to 0
    phases = np.angle(eigenmode)
    phases = phases - np.mean(phases)  # Remove global phase

    # Amplitude from eigenvalue (sqrt for single-photon)
    amplitudes = np.sqrt(np.abs(eigvals[idx_max])) * np.abs(eigenmode)

    # Apply threshold mask: set phase to 0 where amplitude is too small
    mask = amplitudes < coherence_threshold
    phases[mask] = 0.0

    return phases, amplitudes


# ============================================================================
# Phase-Aware (Domain Coloring) Heatmap Visualization
# ============================================================================

def extract_bin_state_coherences(
    mps: MPSState,
    n_bins: int,
    arm: str = 'A',
    coherence_threshold: float = 1e-10,
    use_crossbin_phase: bool = False,
    reference_bin: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract probabilities and coherence phases for all 18 bin states.

    For each bin, extracts:
    - Probability: rho[k,k] (diagonal elements)
    - Phase: Two methods available:
      1. use_crossbin_phase=False: arg(rho[0,k]) (coherence with vacuum)
         WARNING: This is usually noise for entangled states!
      2. use_crossbin_phase=True: Uses cross-bin first-order coherence
         G_nm = <a_n^dag a_m> for single-photon wavepacket phase.
         This is the physically meaningful phase for entangled atom-photon states.

    Parameters
    ----------
    mps : MPSState
        MPS state
    n_bins : int
        Number of time bins
    arm : str
        Which arm to extract ('A' or 'B')
    coherence_threshold : float
        Threshold for coherence magnitude. When |rho[0,k]| < threshold,
        phase is set to 0 and masked (to avoid displaying numerical noise).
    use_crossbin_phase : bool
        If True, use cross-bin first-order coherence for phase extraction.
        This is the recommended method for atom-photon entangled states.
    reference_bin : int, optional
        Reference bin for cross-bin phase calculation. If None, uses
        bin with maximum intensity as reference.

    Returns
    -------
    Tuple of np.ndarray
        (probs_A, probs_B, phases_A, phases_B)
        - probs: (n_bins, 18) real probability array
        - phases: (n_bins, 18) real phase array (in radians, -π to π)
    """
    probs_A = np.zeros((n_bins, 18), dtype=float)
    probs_B = np.zeros((n_bins, 18), dtype=float)
    phases_A = np.zeros((n_bins, 18), dtype=float)
    phases_B = np.zeros((n_bins, 18), dtype=float)

    # Extract probabilities for all bins
    for n in range(n_bins):
        # Chain layout: A1(0), B1(1), A2(2), B2(3), ..., AN, BN
        site_A = 2 * n
        site_B = 2 * n + 1

        rho_A = mps.get_reduced_density([site_A])
        rho_B = mps.get_reduced_density([site_B])

        if rho_A.shape[0] == 18:
            probs_A[n, :] = np.diag(rho_A).real
        if rho_B.shape[0] == 18:
            probs_B[n, :] = np.diag(rho_B).real

    if use_crossbin_phase:
        # Use cross-bin first-order coherence for physically meaningful phase
        # This extracts the wavepacket phase from G_nm = <a_n^dag a_m>
        if reference_bin is None:
            # Find bin with maximum single-photon probability
            # Single-photon states are at indices 1 (|vac,H>) and 2 (|vac,V>)
            total_1ph_A = probs_A[:, 1] + probs_A[:, 2]
            total_1ph_B = probs_B[:, 1] + probs_B[:, 2]
            ref_A = int(np.argmax(total_1ph_A))
            ref_B = int(np.argmax(total_1ph_B))
        else:
            ref_A = ref_B = reference_bin

        # Extract cross-bin phases
        crossbin_phases_A, crossbin_amps_A = extract_first_order_coherence(
            mps, n_bins, arm='A', reference_bin=ref_A,
            coherence_threshold=coherence_threshold
        )
        crossbin_phases_B, crossbin_amps_B = extract_first_order_coherence(
            mps, n_bins, arm='B', reference_bin=ref_B,
            coherence_threshold=coherence_threshold
        )

        # Broadcast to all 18 states (phase is shared across bin states)
        for n in range(n_bins):
            phases_A[n, :] = crossbin_phases_A[n]
            phases_B[n, :] = crossbin_phases_B[n]

            # Apply amplitude mask: where coherence is small, set phase to 0
            if crossbin_amps_A[n] < coherence_threshold:
                phases_A[n, :] = 0.0
            if crossbin_amps_B[n] < coherence_threshold:
                phases_B[n, :] = 0.0

    else:
        # Original method: use vacuum coherence (often noise for entangled states)
        for n in range(n_bins):
            site_A = 2 * n
            site_B = 2 * n + 1

            rho_A = mps.get_reduced_density([site_A])
            rho_B = mps.get_reduced_density([site_B])

            if rho_A.shape[0] == 18:
                # Get coherence magnitudes
                coh_A = rho_A[0, :]
                coh_mag_A = np.abs(coh_A)

                # Apply threshold mask: if coherence too small, phase is noise
                phases_A[n, :] = np.where(
                    coh_mag_A >= coherence_threshold,
                    np.angle(coh_A),
                    0.0
                )

            if rho_B.shape[0] == 18:
                coh_B = rho_B[0, :]
                coh_mag_B = np.abs(coh_B)

                phases_B[n, :] = np.where(
                    coh_mag_B >= coherence_threshold,
                    np.angle(coh_B),
                    0.0
                )

    return probs_A, probs_B, phases_A, phases_B


def _probs_phases_to_rgb_image(
    probs: np.ndarray,
    phases: np.ndarray,
    saturation: float = 1.0,
    value_power: float = 0.5,
    max_prob: float = None,
) -> np.ndarray:
    """
    Convert probabilities and phases to RGB image using HSV color model.

    For each element:
    - Hue = phase (0 to 2π mapped to 0-1)
    - Saturation = fixed value
    - Value = prob^value_power (normalized)

    Parameters
    ----------
    probs : np.ndarray
        Probability array (n_rows, n_cols), real values >= 0
    phases : np.ndarray
        Phase array (n_rows, n_cols), real values in [-π, π]
    saturation : float
        Color saturation (0-1)
    value_power : float
        Power for intensity mapping
    max_prob : float, optional
        Maximum probability for normalization. If None, uses data max.

    Returns
    -------
    np.ndarray
        RGB image array (n_rows, n_cols, 3) with values in [0, 1]
    """
    from matplotlib.colors import hsv_to_rgb

    # Normalize phases to [0, 1] for hue
    hues = (phases + np.pi) / (2 * np.pi)

    # Normalize probabilities for value channel
    if max_prob is None or max_prob <= 0:
        max_prob = probs.max() if probs.max() > 0 else 1.0
    values = (probs / max_prob) ** value_power
    values = np.clip(values, 0, 1)

    # Create HSV array
    hsv = np.zeros(probs.shape + (3,))
    hsv[..., 0] = hues
    hsv[..., 1] = saturation
    hsv[..., 2] = values

    # Convert to RGB
    rgb = hsv_to_rgb(hsv)

    return rgb


def _create_hsv_phase_colorbar(
    fig: plt.Figure,
    position: list,
    label: str = "Phase",
) -> None:
    """
    Add a horizontal phase colorbar (HSV color wheel) to the figure.

    Parameters
    ----------
    fig : plt.Figure
        Figure to add colorbar to
    position : list
        [left, bottom, width, height] for colorbar axes
    label : str
        Label for the colorbar
    """
    from matplotlib.patches import Rectangle
    from matplotlib.colors import hsv_to_rgb

    # Create axes for colorbar
    cax = fig.add_axes(position)
    cax.set_aspect('auto')
    cax.axis('off')

    # Create phase gradient (0 to 2π)
    n_grad = 256
    phase_vals = np.linspace(0, 1, n_grad)
    grad_hsv = np.zeros((1, n_grad, 3))
    grad_hsv[0, :, 0] = phase_vals  # Hue
    grad_hsv[0, :, 1] = 1.0  # Saturation
    grad_hsv[0, :, 2] = 1.0  # Value

    # Convert to RGB and display as image
    grad_rgb = hsv_to_rgb(grad_hsv).squeeze()
    cax.imshow(grad_rgb[np.newaxis, :], aspect='auto', extent=[0, 1, 0, 1])

    # Add phase labels
    cax.text(0, 1.15, '0', ha='left', va='bottom', transform=cax.transAxes, fontsize=8)
    cax.text(0.25, 1.15, 'π/2', ha='center', va='bottom', transform=cax.transAxes, fontsize=8)
    cax.text(0.5, 1.15, 'π', ha='center', va='bottom', transform=cax.transAxes, fontsize=8)
    cax.text(0.75, 1.15, '3π/2', ha='center', va='bottom', transform=cax.transAxes, fontsize=8)
    cax.text(1, 1.15, '2π', ha='right', va='bottom', transform=cax.transAxes, fontsize=8)

    cax.text(0.5, -0.2, label, ha='center', va='top', transform=cax.transAxes, fontsize=9)


def plot_dual_arm_heatmap_phase(
    result: Union[EmissionResult, MPSState],
    save_path: str = "dual_arm_heatmap_phase.png",
    show_atomic: bool = False,
    stage_name: str = "",
    time_grid: Optional[TimeGrid] = None,
    saturation: float = 1.0,
    value_power: float = 0.5,
    vmax_scale_factor: float = 1.5,
    use_crossbin_phase: bool = False,
    coherence_threshold: float = 1e-10,
    reference_bin: Optional[int] = None,
) -> None:
    """
    Visualize dual-arm bin state amplitudes with phase information.

    This function mimics the layout of plot_dual_arm_heatmap() but uses HSV
    coloring where:
    - Hue = phase of coherence (0 to 2π as color wheel)
    - Saturation = color intensity (default: 1.0)
    - Value = brightness ∝ probability^value_power

    Phase Extraction Methods:
    - use_crossbin_phase=False (default): Uses arg(rho[0,k]) (vacuum coherence)
      with threshold masking. This is fast but may show noise for entangled states.
      Phases where |rho[0,k]| < coherence_threshold are masked to 0.
    - use_crossbin_phase=True: Uses cross-bin first-order coherence
      G_nm = <a_n^dag a_m> to extract the wavepacket phase.
      WARNING: This is O(n_bins^2) and can be very slow for large n_bins.

    Parameters
    ----------
    result : Union[EmissionResult, MPSState]
        Simulation result to visualize. If EmissionResult and show_atomic=True,
        atomic state evolution is extracted from result.atom_X_state_evolution.
    save_path : str
        Path to save the figure
    show_atomic : bool
        Whether to display atomic state rows (default: False)
    stage_name : str
        Stage name for title (e.g., "Emission", "QFC", "BS")
    time_grid : TimeGrid, optional
        Time grid for x-axis labels. If None and result is EmissionResult,
        uses result.time_grid.
    saturation : float
        Color saturation (0-1). Lower values give more pastel colors.
    value_power : float
        Power for intensity mapping. 0.5 = sqrt (default), 1.0 = linear.
        Higher values increase contrast for small amplitudes.
    vmax_scale_factor : float
        Factor for scaling max amplitude (relative to max coherence magnitude).
    use_crossbin_phase : bool
        If True, use cross-bin first-order coherence for phase.
        This extracts the wavepacket phase from G_nm = <a_n^dag a_m>.
        WARNING: Very slow for large n_bins (O(n_bins^2) density matrix calls).
        If False (default), use vacuum coherence arg(rho[0,k]) with masking.
    coherence_threshold : float
        Threshold for coherence magnitude. Below this, phase is masked.
    reference_bin : int, optional
        Reference bin for cross-bin phase calculation. If None, uses
        bin with maximum intensity.
    """
    import matplotlib as mpl
    from matplotlib.colors import hsv_to_rgb

    mpl.rcParams['image.interpolation'] = 'nearest'

    # Extract MPS and time_grid from result
    if isinstance(result, EmissionResult):
        mps = result.mps
        if time_grid is None:
            time_grid = result.time_grid
        n_bins = result.get_n_bins()
        has_atom_evol = True
    else:  # MPSState
        mps = result
        if time_grid is None:
            time_grid = TimeGrid(dt=1.0, N=1)
        n_bins = (mps.L - 2) // 2
        has_atom_evol = False

    if show_atomic and not has_atom_evol:
        raise ValueError("show_atomic=True requires EmissionResult with atomic state evolution")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(24, 13))
    plt.subplots_adjust(left=0.04, right=0.85, top=0.80, bottom=0.06, wspace=0.50)

    # Extract atomic state evolution if needed
    if show_atomic and has_atom_evol:
        atom_A_evol = result.atom_A_state_evolution
        atom_B_evol = result.atom_B_state_evolution
        atom_A_for_bins = atom_A_evol[:, 1::2]
        atom_B_for_bins = atom_B_evol[:, 1::2]
        if atom_A_for_bins.shape[1] < n_bins:
            padding = np.tile(atom_A_for_bins[:, -1:], (1, n_bins - atom_A_for_bins.shape[1]))
            atom_A_for_bins = np.hstack([atom_A_for_bins, padding])
        if atom_B_for_bins.shape[1] < n_bins:
            padding = np.tile(atom_B_for_bins[:, -1:], (1, n_bins - atom_B_for_bins.shape[1]))
            atom_B_for_bins = np.hstack([atom_B_for_bins, padding])

    # Extract probabilities and phases for all 18 states
    probs_A, probs_B, phases_A, phases_B = extract_bin_state_coherences(
        mps, n_bins, arm='A',
        coherence_threshold=coherence_threshold,
        use_crossbin_phase=use_crossbin_phase,
        reference_bin=reference_bin
    )
    _, probs_B, _, phases_B = extract_bin_state_coherences(
        mps, n_bins, arm='B',
        coherence_threshold=coherence_threshold,
        use_crossbin_phase=use_crossbin_phase,
        reference_bin=reference_bin
    )

    # Get state labels
    bin_state_labels = _get_bin18_state_labels()

    # Calculate max probability for normalization (excluding vacuum-vacuum at index 0)
    max_prob_A = np.max(probs_A[:, 1:]) if n_bins > 0 else 1.0
    max_prob_B = np.max(probs_B[:, 1:]) if n_bins > 0 else 1.0
    max_prob = max(max_prob_A, max_prob_B) * vmax_scale_factor

    # Create combined data matrices with HSV coloring
    if show_atomic:
        atomic_labels = ['|e>', '|1>', '|0>']
        combined_labels_A = atomic_labels + bin_state_labels
        combined_labels_B = atomic_labels + bin_state_labels
        total_rows = 3 + 18

        # For atomic states, use the original probability display (no phase)
        atom_cmap = plt.get_cmap('YlOrRd')

        # Create atomic state displays (grayscale, no phase info)
        atom_A_disp = np.zeros((3, n_bins, 3))
        atom_B_disp = np.zeros((3, n_bins, 3))

        atom_A_disp[0, :, 0] = atom_A_for_bins[2, :]  # |e>
        atom_A_disp[0, :, 1] = atom_A_for_bins[2, :]
        atom_A_disp[0, :, 2] = atom_A_for_bins[2, :]
        atom_A_disp[1, :, 0] = atom_A_for_bins[1, :]  # |1>
        atom_A_disp[1, :, 1] = atom_A_for_bins[1, :]
        atom_A_disp[1, :, 2] = atom_A_for_bins[1, :]
        atom_A_disp[2, :, 0] = atom_A_for_bins[0, :]  # |0>
        atom_A_disp[2, :, 1] = atom_A_for_bins[0, :]
        atom_A_disp[2, :, 2] = atom_A_for_bins[0, :]

        atom_B_disp[0, :, 0] = atom_B_for_bins[2, :]
        atom_B_disp[0, :, 1] = atom_B_for_bins[2, :]
        atom_B_disp[0, :, 2] = atom_B_for_bins[2, :]
        atom_B_disp[1, :, 0] = atom_B_for_bins[1, :]
        atom_B_disp[1, :, 1] = atom_B_for_bins[1, :]
        atom_B_disp[1, :, 2] = atom_B_for_bins[1, :]
        atom_B_disp[2, :, 0] = atom_B_for_bins[0, :]
        atom_B_disp[2, :, 1] = atom_B_for_bins[0, :]
        atom_B_disp[2, :, 2] = atom_B_for_bins[0, :]

        # Create bin state displays with HSV coloring (probs for intensity, phases for hue)
        bin_A_rgb = _probs_phases_to_rgb_image(
            probs_A.T,
            phases_A.T,
            saturation=saturation,
            value_power=value_power,
            max_prob=max_prob
        )
        bin_B_rgb = _probs_phases_to_rgb_image(
            probs_B.T,
            phases_B.T,
            saturation=saturation,
            value_power=value_power,
            max_prob=max_prob
        )

        # Combine atomic and bin displays
        combined_A = np.vstack([atom_A_disp, bin_A_rgb])
        combined_B = np.vstack([atom_B_disp, bin_B_rgb])

    else:
        combined_labels_A = bin_state_labels
        combined_labels_B = bin_state_labels
        total_rows = 18

        combined_A = _probs_phases_to_rgb_image(
            probs_A.T,
            phases_A.T,
            saturation=saturation,
            value_power=value_power,
            max_prob=max_prob
        )
        combined_B = _probs_phases_to_rgb_image(
            probs_B.T,
            phases_B.T,
            saturation=saturation,
            value_power=value_power,
            max_prob=max_prob
        )

    # Plot arm A
    axes[0].imshow(combined_A, aspect='auto', origin='upper')
    axes[0].set_yticks(range(total_rows))
    axes[0].set_yticklabels(combined_labels_A, fontsize=8)
    axes[0].set_ylabel('State', fontsize=10)
    axes[0].set_title(f'Arm A - Phase & Amplitude (vmax={max_prob:.3f})', fontsize=11)

    if show_atomic:
        axes[0].axhline(2.5, color='black', linewidth=2)
        atom_row_offset = 3
    else:
        atom_row_offset = 0

    # x-axis (dual: time and bin index)
    n_ticks = min(10, n_bins)
    tick_indices = np.linspace(0, n_bins - 1, n_ticks, dtype=int)
    axes[0].set_xticks(tick_indices)
    axes[0].set_xticklabels([f'{time_grid.t[i] * 1e9:.0f}' for i in tick_indices], fontsize=9)
    axes[0].set_xlabel('Time (ns)', fontsize=10)
    ax_top_A = axes[0].twiny()
    ax_top_A.set_xticks(tick_indices)
    ax_top_A.set_xticklabels([str(i) for i in tick_indices], fontsize=9)
    ax_top_A.set_xlabel('Bin index', fontsize=10)
    ax_top_A.set_xlim(axes[0].get_xlim())

    # Plot arm B
    axes[1].imshow(combined_B, aspect='auto', origin='upper')
    axes[1].set_yticks(range(total_rows))
    axes[1].set_yticklabels(combined_labels_B, fontsize=8)
    axes[1].set_ylabel('State', fontsize=10)
    axes[1].set_title(f'Arm B - Phase & Amplitude (vmax={max_prob:.3f})', fontsize=11)

    if show_atomic:
        axes[1].axhline(2.5, color='black', linewidth=2)

    axes[1].set_xticks(tick_indices)
    axes[1].set_xticklabels([f'{time_grid.t[i] * 1e9:.0f}' for i in tick_indices], fontsize=9)
    axes[1].set_xlabel('Time (ns)', fontsize=10)
    ax_top_B = axes[1].twiny()
    ax_top_B.set_xticks(tick_indices)
    ax_top_B.set_xticklabels([str(i) for i in tick_indices], fontsize=9)
    ax_top_B.set_xlabel('Bin index', fontsize=10)
    ax_top_B.set_xlim(axes[1].get_xlim())

    # Add separator lines for bin states (group by 780 state)
    for ax in axes:
        for boundary in [5.5, 11.5]:
            ax.axhline(boundary + atom_row_offset, color='white', linewidth=1, alpha=0.5, linestyle='--')

    # Add phase colorbar
    ax_pos_A = axes[0].get_position()
    fig_height = ax_pos_A.y1 - ax_pos_A.y0

    if show_atomic:
        # Phase colorbar for bin states only (bottom section)
        cax_phase = fig.add_axes([
            ax_pos_A.x1 + 0.01,
            ax_pos_A.y0,
            0.01,
            fig_height * (17/21)
        ])
    else:
        cax_phase = fig.add_axes([
            ax_pos_A.x1 + 0.01,
            ax_pos_A.y0,
            0.01,
            fig_height * (17/18)
        ])

    # Create phase colorbar manually
    from matplotlib.patches import Rectangle
    n_grad = 256
    phase_hsv = np.zeros((n_grad, 1, 3))
    phase_hsv[:, 0, 0] = np.linspace(0, 1, n_grad)
    phase_hsv[:, 0, 1] = 1.0
    phase_hsv[:, 0, 2] = 1.0
    phase_rgb = hsv_to_rgb(phase_hsv).squeeze()
    for i in range(n_grad):
        cax_phase.add_patch(Rectangle((0, i/n_grad), 1, 1/n_grad,
                                      facecolor=phase_rgb[i], edgecolor='none'))
    cax_phase.set_xlim(0, 1)
    cax_phase.set_ylim(0, 1)
    cax_phase.axis('off')
    cax_phase.text(0.5, 1.02, 'Phase (0 to 2π)', ha='center', va='bottom',
                   transform=cax_phase.transAxes, fontsize=8)

    # Title
    if stage_name:
        title = f'Phase-Aware Heatmap: {stage_name}'
    else:
        title = 'Phase-Aware Heatmap: Complex Amplitudes'
    plt.suptitle(title, fontsize=16, y=0.97)

    # Add explanation text with phase extraction method
    if use_crossbin_phase:
        method_str = "Cross-bin coherence G_nm = <a^dag_n a_m>"
    else:
        method_str = "Vacuum coherence arg(rho[0,k])"
    explanation = f"Color = Phase (0 to 2pi), Brightness = Probability^{value_power}\nPhase extraction: {method_str}"
    fig.text(0.5, 0.01, explanation, ha='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved phase-aware heatmaps to: {save_path}")
