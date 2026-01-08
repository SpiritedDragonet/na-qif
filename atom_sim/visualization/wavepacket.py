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
