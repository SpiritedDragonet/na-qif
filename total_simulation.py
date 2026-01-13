# -*- coding: utf-8 -*-
"""
Total Simulation: Dual-Atom Emission -> Time-Bin Wave Packets

This is the first stage of the total simulation:
- Two atoms (A and B) in excited state
- Emission to time bins (780nm only, no QFC yet)
- Final state ready for next gate (BSM at station)

Chain structure (dual-atom layout):
    atomA, atomB, A1, B1, A2, B2, ..., AN, BN

After SWAP conveyor belt:
    A1, B1, A2, B2, ..., AN, BN, atomA, atomB

This allows A_n and B_n to be adjacent for BSM operations.
"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np
from typing import Optional, Tuple

# Add project root to path (for running as standalone script)
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from atom_sim.config import TimeGrid, EmitParams
from atom_sim.simulation import (
    run_emission_only, EmissionResult, apply_qfc, apply_780_filter, apply_fiber_channel,
    apply_bs, apply_detection, find_bsm_success,
    # New quantum jump detection
    run_two_photon_detection, compute_fidelity_with_bell, compute_photon_statistics,
)
from atom_sim.visualization import plot_dual_arm_heatmap, plot_dual_arm_heatmap_phase
from atom_sim.physics import FiberChannelParams


# Backward compatibility: alias for EmissionResult
DualAtomEmissionResult = EmissionResult


def run_dual_atom_emission(
    n_bins: int = 200,
    dt_ns: float = 0.2,
    chi_max: int = 50,
    Alpha_A: Optional[np.ndarray] = None,
    Alpha_B: Optional[np.ndarray] = None,
    gamma_peak_A: float = 0.2,
    gamma_peak_B: float = 0.2,
    t0_A: Optional[float] = None,
    t0_B: Optional[float] = None,
    sigma: float = 12.0,
    delay_bins_B: int = 0,
    verbose: bool = True,
) -> EmissionResult:
    """
    Run dual-atom emission simulation using SWAP conveyor belt protocol.

    This function creates the necessary parameters and calls the simulation layer.

    Parameters
    ----------
    n_bins : int
        Number of time bins
    dt_ns : float
        Time step in nanoseconds
    chi_max : int
        Maximum bond dimension for MPS
    Alpha_A : np.ndarray, optional
        2x2 polarization matrix for atom A
    Alpha_B : np.ndarray, optional
        2x2 polarization matrix for atom B
    gamma_peak_A : float
        Peak emission rate for atom A
    gamma_peak_B : float
        Peak emission rate for atom B
    t0_A : float, optional
        Peak time for atom A (ns)
    t0_B : float, optional
        Peak time for atom B (ns)
    sigma : float
        Width parameter for Gaussian emission profile (ns)
    delay_bins_B : int
        原子B发射延迟的仓数（负数表示B晚于A）
        例如-10表示A先开始发射，10个仓后B才开始
    verbose : bool
        Whether to print progress information

    Returns
    -------
    EmissionResult
        Container with simulation results
    """
    # Create time grid
    time_grid = TimeGrid(dt=dt_ns * 1e-9, N=n_bins)  # Convert ns to seconds
    t = time_grid.t

    # Set default peak times to center of time window
    if t0_A is None:
        t0_A = n_bins * dt_ns / 2
    if t0_B is None:
        t0_B = n_bins * dt_ns / 2

    # Create Gaussian emission rate functions
    def gamma_A_func(t_sec):
        t_ns = t_sec * 1e9  # Convert to ns for calculation
        return gamma_peak_A * np.exp(-0.5 * ((t_ns - t0_A) / sigma) ** 2)

    def gamma_B_func(t_sec):
        t_ns = t_sec * 1e9
        return gamma_peak_B * np.exp(-0.5 * ((t_ns - t0_B) / sigma) ** 2)

    # Set default Alpha matrices (identity = no polarization mixing)
    if Alpha_A is None:
        Alpha_A = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    if Alpha_B is None:
        Alpha_B = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)

    # Create emission parameters
    emit_params = EmitParams(
        gamma_A=gamma_A_func,
        gamma_B=gamma_B_func,
        Alpha_A=Alpha_A,
        Alpha_B=Alpha_B,
    )

    # Run simulation using the simulation layer
    result = run_emission_only(
        time_grid=time_grid,
        emit_params=emit_params,
        chi_max=chi_max,
        verbose=verbose,
        delay_bins_B=delay_bins_B,
    )

    return result


def run_detection_and_bsm(
    result: EmissionResult,
    eta_det: float = 0.85,
    p_dark: float = 1e-6,
    seed: int = 42,
    verbose: bool = True,
    use_quantum_jump: bool = True,
) -> Tuple:
    """
    Run detection and check for BSM success.

    Parameters
    ----------
    result : EmissionResult
        Simulation result containing MPS state
    eta_det : float
        Detection efficiency (typical SNSPD: 0.85)
    p_dark : float
        Dark count probability per detector per bin
    seed : int
        Random seed for reproducibility
    verbose : bool
        Whether to print progress
    use_quantum_jump : bool
        If True, use new quantum jump method (Path B).
        If False, use old independent bin sampling (deprecated).

    Returns
    -------
    Tuple
        If use_quantum_jump=True: (TwoPhotonDetectionResult,)
        If use_quantum_jump=False: (detection_outcomes, success, success_bin, bell_state)
    """
    rng = np.random.default_rng(seed=seed)

    if use_quantum_jump:
        # New quantum jump method (event-driven, physically correct)
        det_result = run_two_photon_detection(
            mps=result.mps,
            n_bins=result.get_n_bins(),
            eta_det=eta_det,
            rng=rng,
            verbose=verbose,
        )
        return det_result
    else:
        # Old method (independent bin sampling - deprecated)
        result.mps, detection_outcomes = apply_detection(
            mps=result.mps,
            n_bins=result.get_n_bins(),
            eta_det=eta_det,
            p_dark=p_dark,
            rng=rng,
            verbose=verbose,
        )
        success, success_bin, bell_state = find_bsm_success(detection_outcomes)
        return detection_outcomes, success, success_bin, bell_state


def save_detection_summary(
    output_dir: Path,
    detection_outcomes: list,
    success: bool,
    success_bin: int,
    bell_state: str,
    eta_det: float,
    p_dark: float,
):
    """Save detection outcomes to a summary file."""
    psi_plus = [(1,0,0,1), (0,1,1,0)]
    psi_minus = [(0,1,0,1), (1,0,1,0)]

    with open(output_dir / "detection_outcomes.txt", "w") as f:
        f.write("Detection Outcomes Summary\n")
        f.write("===========================\n\n")
        f.write(f"Total bins: {len(detection_outcomes)}\n")
        f.write(f"Eta_det: {eta_det}\n")
        f.write(f"P_dark: {p_dark}\n\n")

        n_psi_plus = sum(1 for o in detection_outcomes if o in psi_plus)
        n_psi_minus = sum(1 for o in detection_outcomes if o in psi_minus)
        n_no_click = sum(1 for o in detection_outcomes if sum(o) == 0)
        n_single_click = sum(1 for o in detection_outcomes if sum(o) == 1)
        n_multi_click = sum(1 for o in detection_outcomes if sum(o) >= 2)

        f.write("Outcome Statistics:\n")
        f.write(f"  No click: {n_no_click} bins\n")
        f.write(f"  Single click: {n_single_click} bins\n")
        f.write(f"  Multi-click: {n_multi_click} bins\n")
        f.write(f"  Psi+ heralding: {n_psi_plus} bins\n")
        f.write(f"  Psi- heralding: {n_psi_minus} bins\n\n")

        f.write("Detailed outcomes (non-zero only):\n")
        for n, outcome in enumerate(detection_outcomes):
            if outcome != (0, 0, 0, 0):
                f.write(f"  Bin {n:3d}: {outcome}\n")

        f.write("\nBSM Result:\n")
        f.write(f"  Success: {success}\n")
        f.write(f"  Success bin: {success_bin}\n")
        f.write(f"  Bell state: {bell_state}\n")


def save_debug_info(
    mps,
    n_bins: int,
    stage: str,
    output_dir: Path,
    step_index: int,
):
    """
    保存调试信息到文件。

    Parameters
    ----------
    mps : MPSState
        当前MPS态
    n_bins : int
        时间仓数量
    stage : str
        当前阶段名称
    output_dir : Path
        输出目录
    step_index : int
        步骤索引
    """
    from atom_sim.simulation.detection import (
        compute_photon_statistics, build_jump_operators_18d,
        extract_spin_state, compute_fidelity_with_bell
    )

    info = {}
    info['stage'] = stage
    info['step'] = step_index

    # MPS维度信息
    chi_list = mps._mps.chi
    d_list = mps.d
    info['n_sites'] = len(d_list)
    info['n_bins'] = n_bins
    info['bond_dimensions'] = f'chi_min={min(chi_list)}, chi_max={max(chi_list)}, chi_mean={np.mean(chi_list):.1f}'
    info['local_dimensions'] = f'first_5={d_list[:5]}, last_5={d_list[-5:]}'

    # 光子统计
    stats = compute_photon_statistics(mps, n_bins, verbose=False)
    info['photon_stats'] = stats

    # 原子态信息
    spin_state = extract_spin_state(mps, n_bins)
    info['spin_state_diag'] = np.diag(spin_state).real.tolist()
    info['spin_purity'] = float(np.real(np.trace(spin_state @ spin_state)))

    # Bell态保真度
    for bell in ['Psi+', 'Psi-', 'Phi+', 'Phi-']:
        info[f'fidelity_{bell.replace("+", "p").replace("-", "m")}'] = compute_fidelity_with_bell(spin_state, bell)

    # 保存到文件
    info_file = output_dir / f'debug_step_{step_index:02d}_{stage.replace(" ", "_").lower()}.txt'
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write(f'调试信息 - {stage}\n')
        f.write('='*60 + '\n\n')
        f.write(f'MPS维度信息:\n')
        f.write(f'  n_sites = {info["n_sites"]}\n')
        f.write(f'  n_bins = {info["n_bins"]}\n')
        f.write(f'  {info["bond_dimensions"]}\n')
        f.write(f'  {info["local_dimensions"]}\n\n')
        f.write(f'光子统计:\n')
        f.write(f'  总期望光子数 = {stats["n_total"]:.4f}\n')
        f.write(f'  H偏振 = {stats["n_H"]:.4f}\n')
        f.write(f'  V偏振 = {stats["n_V"]:.4f}\n')
        f.write(f'  损耗概率 = {stats["loss_prob"]:.4f}\n\n')
        f.write(f'原子态信息:\n')
        f.write(f'  对角元: {info["spin_state_diag"]}\n')
        f.write(f'  纯度: {info["spin_purity"]:.4f}\n\n')
        f.write(f'Bell态保真度:\n')
        f.write(f'  Psi+ = {info["fidelity_Psip"]:.4f}\n')
        f.write(f'  Psi- = {info["fidelity_Psim"]:.4f}\n')
        f.write(f'  Phi+ = {info["fidelity_Phip"]:.4f}\n')
        f.write(f'  Phi- = {info["fidelity_Phim"]:.4f}\n')

    print(f'  调试信息已保存: {info_file.name}')


def main():
    """Main function for testing emission + QFC + fiber channel."""
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = PROJECT_ROOT / "outputs" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print("Running emission + QFC + fiber channel simulation...")

    # Run emission
    # delay_bins_B=-10: B比A晚10个仓开始发射（2ns延迟）
    result = run_dual_atom_emission(
        n_bins=100,  # 仓数
        dt_ns=0.2,   # 时间步长
        chi_max=30,
        gamma_peak_A=0.2,
        gamma_peak_B=0.2,
        sigma=6.0,
        delay_bins_B=-10,  # B延迟10个仓
        verbose=True,
    )

    # Save after-emission visualization
    print("\nGenerating after-emission visualization...")
    plot_dual_arm_heatmap(
        result,
        save_path=str(output_dir / "1_after_emission.png"),
        show_atomic=True,
        stage_name="After Emission"
    )

    # Save debug info
    print("\nSaving debug information...")
    save_debug_info(
        mps=result.mps,
        n_bins=result.get_n_bins(),
        stage='After Emission',
        output_dir=output_dir,
        step_index=1,
    )

    # # Save phase-aware visualization (HSV domain coloring)
    # print("\nGenerating phase-aware visualization...")
    # plot_dual_arm_heatmap_phase(
    #     result,
    #     save_path=str(output_dir / "1_after_emission_phase.png"),
    #     show_atomic=True,
    #     stage_name="After Emission",
    # )

    # Apply QFC
    print("\nApplying QFC...")
    apply_qfc(
        mps=result.mps,
        n_bins=result.get_n_bins(),
        theta_H=np.pi/4,  # 50% conversion
        theta_V=np.pi/4,
        verbose=True,
    )

    # Apply 780nm filter (remove unconverted 780nm photons)
    print("\nApplying 780nm filter...")
    apply_780_filter(
        mps=result.mps,
        n_bins=result.get_n_bins(),
        verbose=True,
    )

    # Save after-QFC visualization (now with 780nm filtered out)
    print("\nGenerating after-QFC+filter visualization...")
    plot_dual_arm_heatmap(
        result.mps,
        save_path=str(output_dir / "2_after_qfc.png"),
        show_atomic=False,
        stage_name="After QFC + 780nm Filter",
        time_grid=result.time_grid,
    )

    # Save debug info
    save_debug_info(
        mps=result.mps,
        n_bins=result.get_n_bins(),
        stage='After QFC + Filter',
        output_dir=output_dir,
        step_index=2,
    )

    # # Save phase-aware visualization after QFC
    # print("\nGenerating phase-aware visualization after QFC...")
    # plot_dual_arm_heatmap_phase(
    #     result.mps,
    #     save_path=str(output_dir / "2_after_qfc_phase.png"),
    #     show_atomic=False,
    #     stage_name="After QFC (50% conversion)",
    #     time_grid=result.time_grid,
    # )

    # =========================================================================
    # Fiber channel (TEMPORARILY DISABLED - will revisit after BS works)
    # =========================================================================
    # Setup fiber channel parameters with random drift
    # This models realistic fiber transmission with:
    # - Polarization drift (Jones matrices)
    # - Phase drift between arms
    # - Loss with small fluctuations
    # fiber_params = FiberChannelParams(
    #     polarization_model="perturb",  # Small random rotations
    #     polarization_sigma=0.1,          # 0.1 rad std (~5.7 degrees)
    #     eta_mean=0.57,                    # Mean transmissivity
    #     eta_std=0.02,                     # 2% fluctuation
    #     phase_drift_std=0.2,              # Phase drift std (rad)
    # )
    #
    # rng = np.random.default_rng(seed=42)  # For reproducibility
    #
    # # Apply fiber channel (Jones + loss) with sampled parameters
    # print(f"\nApplying fiber channel (Jones rotation + loss)...")
    # result.mps, sampled_params = apply_fiber_channel(
    #     mps=result.mps,
    #     n_bins=result.get_n_bins(),
    #     fiber_params=fiber_params,
    #     rng=rng,
    #     verbose=True,
    # )
    #
    # # Save after-fiber visualization
    # print("\nGenerating after-fiber visualization...")
    # U_A, U_B, eta, phase = sampled_params
    # plot_dual_arm_heatmap(
    #     result.mps,
    #     save_path=str(output_dir / "3_after_fiber.png"),
    #     show_atomic=False,
    #     stage_name=f"After Fiber (eta={eta:.2f}, phase={phase:.2f}rad)",
    #     time_grid=result.time_grid,
    # )
    #
    # # Save phase-aware visualization after fiber
    # print("\nGenerating phase-aware visualization after fiber...")
    # plot_dual_arm_heatmap_phase(
    #     result.mps,
    #     save_path=str(output_dir / "3_after_fiber_phase.png"),
    #     show_atomic=False,
    #     stage_name=f"After Fiber (eta={eta:.2f}, phase={phase:.2f}rad)",
    #     time_grid=result.time_grid,
    # )
    # =========================================================================

    # Apply Beam Splitter (BS) to interfere A_n with B_n at each bin
    print("\nApplying Beam Splitter (BS)...")
    apply_bs(
        mps=result.mps,
        n_bins=result.get_n_bins(),
        verbose=True,
    )

    # Save after-BS visualization
    print("\nGenerating after-BS visualization...")
    plot_dual_arm_heatmap(
        result.mps,
        save_path=str(output_dir / "3_after_bs.png"),
        show_atomic=False,
        stage_name="After Beam Splitter",
        time_grid=result.time_grid,
    )

    # Save debug info after BS
    save_debug_info(
        mps=result.mps,
        n_bins=result.get_n_bins(),
        stage='After BS',
        output_dir=output_dir,
        step_index=3,
    )

    # Compute photon statistics before normalization
    print("\nComputing photon statistics after BS...")
    photon_stats = compute_photon_statistics(
        mps=result.mps,
        n_bins=result.get_n_bins(),
        verbose=True,
    )

    # Normalize MPS to condition on two-photon arrival
    # This discards the "photon lost" probability and focuses on successful cases
    print("\nNormalizing MPS to condition on two-photon arrival...")
    result.mps._mps.canonical_form_finite(renormalize=True)
    print(f"  MPS normalized.")

    # Save after-normalization visualization
    print("\nGenerating after-normalization visualization...")
    plot_dual_arm_heatmap(
        result.mps,
        save_path=str(output_dir / "4_after_normalization.png"),
        show_atomic=False,
        stage_name="After Normalization (Two-Photon Branch)",
        time_grid=result.time_grid,
    )

    # 保存归一化后的调试信息
    save_debug_info(
        mps=result.mps,
        n_bins=result.get_n_bins(),
        stage='After Normalization',
        output_dir=output_dir,
        step_index=4,
    )

    # Verify photon statistics after normalization
    photon_stats_norm = compute_photon_statistics(
        mps=result.mps,
        n_bins=result.get_n_bins(),
        verbose=True,
    )

    # =========================================================================
    # Detection: TEMPORARILY DISABLED FOR TESTING
    # =========================================================================
    # # Detection parameters
    # eta_det = 0.85
    # p_dark = 1e-6
    #
    # # Run detection and BSM using quantum jump method
    # print("\nRunning detection and BSM (Quantum Jump Method)...")
    # det_result = run_detection_and_bsm(
    #     result=result,
    #     eta_det=eta_det,
    #     p_dark=p_dark,
    #     seed=42,
    #     verbose=True,
    #     use_quantum_jump=True,  # Use new physically correct method
    # )
    #
    # # Print results
    # if det_result.success:
    #     print(f"\n  BSM SUCCESS!")
    #     print(f"  Bell state heralded: {det_result.bell_state}")
    #     print(f"  Clicks: {[(c.detector, c.bin_index) for c in det_result.clicks]}")
    #
    #     # Compute fidelity with expected Bell state
    #     fidelity = compute_fidelity_with_bell(det_result.spin_state, det_result.bell_state)
    #     print(f"  Fidelity with |{det_result.bell_state}>: {fidelity:.4f}")
    #
    #     # Also compute fidelity with all Bell states for reference
    #     print(f"\n  Fidelity with all Bell states:")
    #     for bell in ["Psi+", "Psi-", "Phi+", "Phi-"]:
    #         f = compute_fidelity_with_bell(det_result.spin_state, bell)
    #         marker = " <-- heralded" if bell == det_result.bell_state else ""
    #         print(f"    F(|{bell}>): {f:.4f}{marker}")
    #
    #     # Print spin state
    #     print(f"\n  Spin density matrix (qubit subspace):")
    #     rho = det_result.spin_state
    #     print(f"    Tr(rho) = {np.trace(rho).real:.4f}")
    #     print(f"    Purity = {np.trace(rho @ rho).real:.4f}")
    # else:
    #     print(f"\n  BSM FAILED - no success pattern found")
    #     print(f"  Number of clicks: {len(det_result.clicks)}")
    #     if det_result.clicks:
    #         print(f"  Clicks: {[(c.detector, c.bin_index) for c in det_result.clicks]}")
    #
    # # 保存探测后的调试信息
    # # 注意：探测后MPS已被修改，需要使用det_result.spin_state
    # print("\nSaving post-detection debug info...")
    # with open(output_dir / 'debug_detection_result.txt', 'w', encoding='utf-8') as file:
    #     file.write(f'探测结果\n')
    #     file.write('='*60 + '\n\n')
    #     file.write(f'成功: {det_result.success}\n')
    #     file.write(f'Bell态: {det_result.bell_state}\n')
    #     file.write(f'点击次数: {len(det_result.clicks)}\n')
    #     if det_result.clicks:
    #         file.write(f'点击详情: {[(c.detector, c.bin_index) for c in det_result.clicks]}\n')
    #
    #     file.write(f'\n自旋密度矩阵:\n')
    #     rho = det_result.spin_state
    #     file.write(f'  基: |00>, |01>, |10>, |11>\n')
    #     for i in range(4):
    #         for j in range(4):
    #             val = rho[i, j]
    #             if abs(val) > 1e-10:
    #                 file.write(f'  rho[{i},{j}] = {val:.4f}\n')
    #
    #     file.write(f'\n纯度: {np.trace(rho @ rho).real:.4f}\n')
    #
    #     file.write(f'\nBell态保真度:\n')
    #     for bell in ["Psi+", "Psi-", "Phi+", "Phi-"]:
    #         fid = compute_fidelity_with_bell(rho, bell)
    #         marker = " <-- 探测到的" if bell == det_result.bell_state else ""
    #         file.write(f'  F({bell}) = {fid:.4f}{marker}\n')
    #
    # print(f"  调试信息已保存: debug_detection_result.txt")

    print(f"\nDone! Files saved to: {output_dir}/")


if __name__ == "__main__":
    main()
