# -*- coding: utf-8 -*-
"""
波包可视化模块

本模块提供从MPS态中提取并可视化波包的函数，
包括双臂热图。
"""

from typing import Tuple, Optional, List, Union
import os
import numpy as np
import matplotlib


def _is_headless() -> bool:
    if os.environ.get("QSIM_NO_SHOW", "").lower() in ("1", "true", "yes"):
        return True
    if os.name == "nt":
        return False
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return False
    return True


if _is_headless():
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from ..core.mps import MPSState  # noqa: E402
from ..simulation.trajectory import EmissionResult  # noqa: E402
from ..physics.gates import qfc_gate  # noqa: E402
from ..physics.channels import loss_channel_both_subspaces, loss_channel_1517_single_photon  # noqa: E402
from ..hilbert.basis import proj_3_from_5, embed_6_from_3, jones_3d  # noqa: E402


# ============================================================================
# 波包提取算符
# ============================================================================

def _maybe_show(wait_s: float = 5.0, show: bool = True) -> None:
    if not show:
        return
    if _is_headless():
        return
    plt.show(block=False)
    plt.pause(wait_s)

# ============================================================================
# 仓状态热图可视化
# ============================================================================

def _get_bin6_state_labels() -> List[str]:
    """
    获取6个仓状态的标签（仅1517nm子空间）。

    1517子空间：支持最多2个光子
    - |vac>: 真空态
    - |H>: 单H光子
    - |V>: 单V光子
    - |2H>: 双H光子
    - |2V>: 双V光子
    - |HV>: H+V各一个

    Returns
    -------
    List[str]
        6个状态标签
    """
    return ['|vac>', '|H>', '|V>', '|2H>', '|2V>', '|HV>']


def _get_bin5_state_labels() -> List[str]:
    """获取5D bin状态标签（vac, H780, V780, H1517, V1517）。"""
    return ['|vac>', '|H_780>', '|V_780>', '|H_1517>', '|V_1517>']


def _get_bin3_state_labels() -> List[str]:
    """获取3D telecom状态标签（vac, H, V）。"""
    return ['|vac>', '|H>', '|V>']


def _build_port_effects(
    bin_dim: int,
    bs_unitary: Optional[np.ndarray] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Build per-port measurement effects.

    - 无 BS：E1=P⊗I, E2=I⊗P
    - 有 BS：E1=U^†(P⊗I)U, E2=U^†(I⊗P)U

    Returns
    -------
    Tuple[List[np.ndarray], List[np.ndarray]]
        (effects_port1, effects_port2) where each list has length bin_dim.
    """
    # ------------------------------------------------------------------
    # 统一“单一路径”：
    #   无论是否有 BS，都用端口测量的 effect 来取概率。
    #   bs_unitary=None 时即退化为对原臂的测量。
    # ------------------------------------------------------------------
    dim_pair = bin_dim * bin_dim
    eye = np.eye(bin_dim, dtype=complex)

    projectors = []
    for idx in range(bin_dim):
        P = np.zeros((bin_dim, bin_dim), dtype=complex)
        P[idx, idx] = 1.0
        projectors.append(P)

    if bs_unitary is not None:
        bs_unitary = np.asarray(bs_unitary, dtype=complex)
        if bs_unitary.shape != (dim_pair, dim_pair):
            raise ValueError(
                f"bs_unitary shape {bs_unitary.shape} != ({dim_pair},{dim_pair})"
            )
        U_dag = bs_unitary.conj().T
        effects_port1 = [U_dag @ np.kron(P, eye) @ bs_unitary for P in projectors]
        effects_port2 = [U_dag @ np.kron(eye, P) @ bs_unitary for P in projectors]
        return effects_port1, effects_port2

    effects_port1 = [np.kron(P, eye) for P in projectors]
    effects_port2 = [np.kron(eye, P) for P in projectors]
    return effects_port1, effects_port2


def _apply_kraus_state(rho: np.ndarray, K_list: List[np.ndarray]) -> np.ndarray:
    """ρ -> Σ K ρ K^†（单端口）。"""
    acc = np.zeros_like(rho)
    for K in K_list:
        acc += K @ rho @ K.conj().T
    return acc


def _apply_kraus_state_pair(rho: np.ndarray, K_list_A: List[np.ndarray], K_list_B: List[np.ndarray]) -> np.ndarray:
    """ρ -> Σ (K_A ⊗ K_B) ρ (K_A ⊗ K_B)^†（双端口）。"""
    acc = np.zeros_like(rho)
    for K_A in K_list_A:
        for K_B in K_list_B:
            K = np.kron(K_A, K_B)
            acc += K @ rho @ K.conj().T
    return acc


def _infer_first_bin_site(mps: MPSState) -> int:
    """Infer the first bin site index based on emitter-like leading sites."""
    if len(mps.d) >= 4 and mps.d[2] == 5 and mps.d[3] == 5:
        return 2
    return 0


def _validate_bin_rho_traces(
    mps: MPSState,
    n_bins: int,
    tol: float = 1e-6,
) -> None:
    """Check that per-bin reduced density matrices are properly normalized."""
    first_bin_site = _infer_first_bin_site(mps)
    issues = []
    for n in range(n_bins):
        for offset, arm in ((0, "A"), (1, "B")):
            site = first_bin_site + 2 * n + offset
            if site >= mps.L:
                continue
            rho = mps.get_reduced_density([site])
            tr = np.trace(rho).real
            if abs(tr - 1.0) > tol:
                issues.append((arm, n, tr))

    if issues:
        print("Warning: bin reduced density traces deviate from 1.0")
        for arm, n, tr in issues[:8]:
            print(f"  Arm {arm} bin {n}: Tr={tr:.6f}")
        if len(issues) > 8:
            print(f"  ... {len(issues) - 8} more bins")


# ============================================================================
# 双臂热图可视化（通用）
# ============================================================================

def plot_dual_arm_heatmap(
    result: Union[EmissionResult, MPSState],
    save_path: str = "dual_arm_heatmap.png",
    show_atomic: bool = False,
    stage_name: str = "",
    time_grid: Optional[dict] = None,
    vmax_scale_factor: float = 1.5,
    show: bool = True,
    validate: bool = True,
    trace_tol: float = 1e-6,
    bs_unitary: Optional[np.ndarray] = None,
    qfc_params: Optional[Tuple[float, float]] = None,
    fiber_sample: Optional[tuple] = None,
    apply_filter_780: bool = True,
    arm_labels: Optional[Tuple[str, str]] = None,
) -> None:
    """
    可视化双臂仓状态概率，可选显示原子状态。

    通用热图函数，适用于任何仿真阶段：
    - 发射：使用 show_atomic=True 显示原子状态演化
    - QFC/Jones/Loss/BS：使用 show_atomic=False（原子不参与）

    每个臂显示：
    - 若 show_atomic=True：顶部4行（原子）+ 底部 bin_dim 行（仓状态）
    - 若 show_atomic=False：仅 bin_dim 行仓状态

    三种色图：
    - 原子态：YlOrRd（黄-橙-红）
    - (vac,vac) 态：Greys（单独色条）
    - 其他仓态：plasma

    Parameters
    ----------
    result : Union[EmissionResult, MPSState]
        要可视化的仿真结果。如果是EmissionResult且show_atomic=True，
        从result.atom_X_state_evolution提取原子状态演化。
    save_path : str
        保存图形的路径
    show_atomic : bool
        是否显示原子状态行（默认：False）
    stage_name : str
        标题的阶段名称（如 "Emission", "QFC", "BS"）
    time_grid : dict, optional
        x轴标签的时间参数字典，可包含 dt_s / dt / dt_ns。
        若为None且result是EmissionResult，使用result.dt_s。
    vmax_scale_factor : float
        缩放vmax的因子（相对于最大仓概率）
    show : bool
        若为 True 且非无屏幕环境，显示图像并暂停 5 秒
    validate : bool
        若为 True，先检查单 bin 约化密度矩阵的归一化
    trace_tol : float
        归一化检查的误差阈值
    bs_unitary : np.ndarray, optional
        若提供，则在测量端使用 U^† (P ⊗ I) U / U^† (I ⊗ P) U 计算端口概率，
        可在不显式作用 BS 的情况下绘制 after-BS 热图。
        仅支持 6D bin（1517nm）。
    qfc_params : Tuple[float, float], optional
        QFC 的 (theta_H, theta_V)，用于可视化重建（默认 π/4, π/4）。
    fiber_sample : tuple, optional
        光纤采样参数：
        (U_A, U_B, eta_H_A, eta_V_A, eta_H_B, eta_V_B, phase, phase_slope, phase_jitter_std)。
        用于 after_fiber / after_bs 的局域重建。
    apply_filter_780 : bool
        是否在可视化重建中应用 780 过滤（默认 True）。
    arm_labels : Tuple[str, str], optional
        自定义左右臂标题标签，默认使用 ("Arm A","Arm B")；
        若 bs_unitary 给出且未显式设置，则默认 ("Port 1","Port 2")。
    """
    # ------------------------------------------------------------------
    # 该热图可以工作在两种模式：
    #   1) 直接从 MPS 的单臂约化密度矩阵取对角元 (默认)
    #   2) 若传入 bs_unitary，则用 U^† P U 在测量端求概率，
    #      从而得到“After BS”的端口分布而无需显式作用 BS。
    # ------------------------------------------------------------------
    import matplotlib as mpl

    mpl.rcParams['image.interpolation'] = 'nearest'

    def _resolve_dt_s(grid: Optional[dict], fallback_dt: float) -> float:
        if grid is None:
            return fallback_dt
        if isinstance(grid, dict):
            if "dt_s" in grid:
                return float(grid["dt_s"])
            if "dt" in grid:
                return float(grid["dt"])
            if "dt_ns" in grid:
                return float(grid["dt_ns"]) * 1e-9
            return fallback_dt
        dt_s = getattr(grid, "dt_s", None)
        if dt_s is not None:
            return float(dt_s)
        dt = getattr(grid, "dt", None)
        if dt is not None:
            return float(dt)
        dt_ns = getattr(grid, "dt_ns", None)
        if dt_ns is not None:
            return float(dt_ns) * 1e-9
        return fallback_dt

    # 从结果中提取MPS和时间参数
    if isinstance(result, EmissionResult):
        mps = result.mps
        n_bins = result.get_n_bins()
        dt_s = _resolve_dt_s(time_grid, result.dt_s)
        has_atom_evol = True
    else:  # MPSState
        mps = result
        n_bins = (mps.L - 2) // 2  # 从链长度推断
        dt_s = _resolve_dt_s(time_grid, 1.0)
        has_atom_evol = False
    time_axis_s = np.arange(n_bins) * dt_s

    # 创建具有更大间距的图形 (1080x720 aspect ratio)
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 7.2))
    # 注意：subplots_adjust 将在确定 display_atomic 后设置

    # 检测是否有前导 emitter/原子站点（只要前两站不是 bin=5 即认为有）
    has_atomic_sites = bool(len(mps.d) >= 4 and mps.d[2] == 5 and mps.d[3] == 5)

    # 确定first_bin_site
    if has_atomic_sites:
        first_bin_site = 2  #前两个站点是原子
    else:
        first_bin_site = 0  #没有原子站点

    if validate:
        _validate_bin_rho_traces(mps, n_bins, tol=trace_tol)

    # 决定是否显示原子态行
    # show_atomic=True: 显示原子态时间演化（仅emission阶段，需要has_atom_evol）
    # show_atomic=False但has_atomic_sites: 显示原子态静态概率（QFC/BS阶段）
    display_atomic_evolution = show_atomic and has_atom_evol
    display_atomic_static = (not show_atomic) and has_atomic_sites
    display_atomic = display_atomic_evolution or display_atomic_static

    # 提取原子状态数据
    if display_atomic_evolution:
        # 模式1：显示原子态时间演化（emission阶段）
        atom_A_evol = result.atom_A_state_evolution
        atom_B_evol = result.atom_B_state_evolution
        # 用于可视化，取每隔一列（每次完整仓处理后）
        atom_A_for_bins = atom_A_evol[:, 1::2]
        atom_B_for_bins = atom_B_evol[:, 1::2]
        # 如果演化列数少于仓数，用末态填充
        if atom_A_for_bins.shape[1] < n_bins:
            padding = np.tile(atom_A_for_bins[:, -1:], (1, n_bins - atom_A_for_bins.shape[1]))
            atom_A_for_bins = np.hstack([atom_A_for_bins, padding])
        if atom_B_for_bins.shape[1] < n_bins:
            padding = np.tile(atom_B_for_bins[:, -1:], (1, n_bins - atom_B_for_bins.shape[1]))
            atom_B_for_bins = np.hstack([atom_B_for_bins, padding])

        # 反转原子状态数据：使得最右列对应t=0（与AN相互作用），最左列对应t=(N-1)dt（与A1相互作用）
        atom_A_for_bins = np.fliplr(atom_A_for_bins)
        atom_B_for_bins = np.fliplr(atom_B_for_bins)
    elif display_atomic_static:
        # 模式2：显示原子态静态概率（QFC/BS阶段）
        # 从MPS中提取原子站点的约化密度矩阵
        rho_A = mps.get_reduced_density([0])
        rho_B = mps.get_reduced_density([1])
        # 提取对角元素（概率）
        atom_A_probs = np.diag(rho_A).real  # shape: (4,)
        atom_B_probs = np.diag(rho_B).real  # shape: (4,)
        # 扩展为所有bin列显示相同的概率（整行同色）
        atom_A_for_bins = np.tile(atom_A_probs.reshape(4, 1), (1, n_bins))
        atom_B_for_bins = np.tile(atom_B_probs.reshape(4, 1), (1, n_bins))

    # 选择可视化模式（默认按参数推断）
    if stage_name:
        stage_lower = stage_name.lower()
    else:
        stage_lower = ""
    if stage_lower and "bs" in stage_lower:
        stage_mode = "after_bs"
    elif stage_lower and "fiber" in stage_lower:
        stage_mode = "after_fiber"
    elif stage_lower and "qfc" in stage_lower:
        stage_mode = "after_qfc"
    else:
        stage_mode = "emission"

    if bs_unitary is not None:
        stage_mode = "after_bs"

    if qfc_params is None:
        qfc_params = (np.pi / 4, np.pi / 4)
    theta_H, theta_V = qfc_params

    if fiber_sample is None:
        U_A = np.eye(2, dtype=complex)
        U_B = np.eye(2, dtype=complex)
        eta_H_A = eta_V_A = eta_H_B = eta_V_B = 1.0
        phase_slope = 0.0
    else:
        U_A, U_B, eta_H_A, eta_V_A, eta_H_B, eta_V_B, _phase, phase_slope, _phase_jitter_std = fiber_sample

    # 设置状态标签与显示维度
    if stage_mode == "after_bs":
        bin_dim = 6
        bin_state_labels = _get_bin6_state_labels()
        if bs_unitary is None:
            raise ValueError("after_bs 模式需要 bs_unitary (36x36)")
        if arm_labels is None:
            arm_labels = ("Port 1", "Port 2")
    elif stage_mode in ("after_qfc", "after_fiber"):
        bin_dim = 3
        bin_state_labels = _get_bin3_state_labels()
    else:
        bin_dim = 5
        bin_state_labels = _get_bin5_state_labels()

    # 提取仓概率
    probs_A = np.zeros((n_bins, bin_dim))
    probs_B = np.zeros((n_bins, bin_dim))

    # 预计算通道算符
    U_qfc = qfc_gate(theta_H=theta_H, theta_V=theta_V)
    eta_780 = 0.0 if apply_filter_780 else 1.0
    K_filter = loss_channel_both_subspaces(
        eta_780=eta_780,
        eta_H_1517=1.0,
        eta_V_1517=1.0,
    )
    P_3_from_5 = proj_3_from_5()
    P_6_from_3 = embed_6_from_3()
    U_A_3 = jones_3d(U_A)

    phase_center = 0.5 * (n_bins - 1)

    if stage_mode == "after_bs":
        # 先构造端口测量 effect
        effects_port1, effects_port2 = _build_port_effects(6, bs_unitary)
        dim_pair_5 = 25
        for n in range(n_bins):
            site_A = first_bin_site + 2 * n
            site_B = first_bin_site + 2 * n + 1
            if site_A >= len(mps.d) or site_B >= len(mps.d):
                continue
            rho_pair = mps.get_reduced_density([site_A, site_B])
            if rho_pair.ndim == 4:
                rho_pair = rho_pair.reshape(dim_pair_5, dim_pair_5)

            # QFC + 过滤
            U_qfc_pair = np.kron(U_qfc, U_qfc)
            rho_pair = U_qfc_pair @ rho_pair @ U_qfc_pair.conj().T
            rho_pair = _apply_kraus_state_pair(rho_pair, K_filter, K_filter)

            # 投影到 3D×3D
            P_pair_3 = np.kron(P_3_from_5, P_3_from_5)
            rho_3 = P_pair_3 @ rho_pair @ P_pair_3.conj().T

            # 光纤（Jones + 相位）+ 损耗
            phase_n = phase_slope * (n - phase_center)
            U_B_n = np.exp(1j * phase_n) * U_B
            U_B_3 = jones_3d(U_B_n)
            U_pair_3 = np.kron(U_A_3, U_B_3)
            rho_3 = U_pair_3 @ rho_3 @ U_pair_3.conj().T
            K_A_3 = loss_channel_1517_single_photon(float(eta_H_A), float(eta_V_A))
            K_B_3 = loss_channel_1517_single_photon(float(eta_H_B), float(eta_V_B))
            rho_3 = _apply_kraus_state_pair(rho_3, K_A_3, K_B_3)

            # 嵌入到 6D×6D，并用端口 effect 取概率
            P_pair_6 = np.kron(P_6_from_3, P_6_from_3)
            rho_6 = P_pair_6 @ rho_3 @ P_pair_6.conj().T
            for idx in range(bin_dim):
                p1 = np.real(np.trace(effects_port1[idx] @ rho_6))
                p2 = np.real(np.trace(effects_port2[idx] @ rho_6))
                probs_A[n, idx] = max(0.0, float(p1))
                probs_B[n, idx] = max(0.0, float(p2))
    else:
        for n in range(n_bins):
            site_A = first_bin_site + 2 * n
            site_B = first_bin_site + 2 * n + 1
            if site_A >= len(mps.d) or site_B >= len(mps.d):
                continue
            rho_A = mps.get_reduced_density([site_A])
            rho_B = mps.get_reduced_density([site_B])
            if rho_A.ndim == 4:
                rho_A = rho_A.reshape(5, 5)
            if rho_B.ndim == 4:
                rho_B = rho_B.reshape(5, 5)

            if stage_mode in ("after_qfc", "after_fiber"):
                rho_A = U_qfc @ rho_A @ U_qfc.conj().T
                rho_B = U_qfc @ rho_B @ U_qfc.conj().T
                rho_A = _apply_kraus_state(rho_A, K_filter)
                rho_B = _apply_kraus_state(rho_B, K_filter)

                rho_A = P_3_from_5 @ rho_A @ P_3_from_5.conj().T
                rho_B = P_3_from_5 @ rho_B @ P_3_from_5.conj().T

                if stage_mode == "after_fiber":
                    phase_n = phase_slope * (n - phase_center)
                    U_B_n = np.exp(1j * phase_n) * U_B
                    U_B_3 = jones_3d(U_B_n)
                    rho_A = U_A_3 @ rho_A @ U_A_3.conj().T
                    rho_B = U_B_3 @ rho_B @ U_B_3.conj().T
                    K_A_3 = loss_channel_1517_single_photon(float(eta_H_A), float(eta_V_A))
                    K_B_3 = loss_channel_1517_single_photon(float(eta_H_B), float(eta_V_B))
                    rho_A = _apply_kraus_state(rho_A, K_A_3)
                    rho_B = _apply_kraus_state(rho_B, K_B_3)

                probs_A[n, :] = np.maximum(0.0, np.real(np.diag(rho_A)))
                probs_B[n, :] = np.maximum(0.0, np.real(np.diag(rho_B)))
            else:
                probs_A[n, :] = np.maximum(0.0, np.real(np.diag(rho_A)))
                probs_B[n, :] = np.maximum(0.0, np.real(np.diag(rho_B)))

    # Calculate vmax EXCLUDING (vac,vac) row (index 0)
    vmax_A = max(0.01, probs_A[:, 1:].max() * vmax_scale_factor)
    vmax_B = max(0.01, probs_B[:, 1:].max() * vmax_scale_factor)
    vmax = max(vmax_A, vmax_B)

    # Create combined data matrices
    if display_atomic:
        atomic_labels = ['|e>', '|u>', '|1>', '|0>']
        combined_labels_A = atomic_labels + bin_state_labels
        combined_labels_B = atomic_labels + bin_state_labels
        total_rows = 4 + bin_dim

        combined_A = np.zeros((total_rows, n_bins))
        combined_A[0, :] = atom_A_for_bins[2, :]  # |e>
        combined_A[1, :] = atom_A_for_bins[3, :]  # |u>
        combined_A[2, :] = atom_A_for_bins[1, :]  # |1>
        combined_A[3, :] = atom_A_for_bins[0, :]  # |0>
        combined_A[4:, :] = probs_A.T

        combined_B = np.zeros((total_rows, n_bins))
        combined_B[0, :] = atom_B_for_bins[2, :]  # |e>
        combined_B[1, :] = atom_B_for_bins[3, :]  # |u>
        combined_B[2, :] = atom_B_for_bins[1, :]  # |1>
        combined_B[3, :] = atom_B_for_bins[0, :]  # |0>
        combined_B[4:, :] = probs_B.T
    else:
        combined_labels_A = bin_state_labels
        combined_labels_B = bin_state_labels
        total_rows = bin_dim

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
    if display_atomic:
        # Three sections: Atom (rows 0-3), Vac (row 4), Bin (rows 5+)
        mask_atom = np.zeros((total_rows, n_bins), dtype=bool)
        mask_atom[:4, :] = True
        mask_vac = np.zeros((total_rows, n_bins), dtype=bool)
        mask_vac[4, :] = True
        mask_bin = np.zeros((total_rows, n_bins), dtype=bool)
        mask_bin[5:, :] = True
        atom_row_offset = 4  # Offset for separator lines
    else:
        # Two sections: Vac (row 0), Bin (rows 1-17)
        mask_atom = None
        mask_vac = np.zeros((total_rows, n_bins), dtype=bool)
        mask_vac[0, :] = True
        mask_bin = np.zeros((total_rows, n_bins), dtype=bool)
        mask_bin[1:, :] = True
        atom_row_offset = 0

    # 根据是否显示原子态调整子图布局
    if display_atomic:
        # 有原子态：使用原来的布局
        plt.subplots_adjust(left=0.04, right=0.85, top=0.80, bottom=0.06, wspace=0.50)
    else:
        # 无原子态：增加底部边距，减少顶部边距，让行填满整个区域
        plt.subplots_adjust(left=0.04, right=0.85, top=0.92, bottom=0.08, wspace=0.50)

    # Resolve labels for plotting
    if arm_labels is not None:
        label_A, label_B = arm_labels
    elif bs_unitary is not None:
        label_A, label_B = "Port 1", "Port 2"
    else:
        label_A, label_B = "Arm A", "Arm B"

    # Plot arm A
    # First plot all bin states with plasma colormap
    im_A = axes[0].imshow(
        combined_A,
        aspect='auto',
        cmap=bin_cmap,
        vmin=0,
        vmax=vmax,
        origin='upper',
        extent=[0, n_bins, total_rows, 0]  # 明确设置范围
    )

    # Overlay (vac,vac) row with different colormap
    im_A_vac = axes[0].imshow(
        np.ma.masked_where(~mask_vac, combined_A),
        aspect='auto',
        cmap=vac_cmap,
        vmin=0,
        vmax=1,
        origin='upper',
        interpolation='nearest',
        extent=[0, n_bins, total_rows, 0]
    )

    # Overlay atomic states if needed
    if display_atomic:
        im_A_atom = axes[0].imshow(
            np.ma.masked_where(~mask_atom, combined_A),
            aspect='auto',
            cmap=atom_cmap,
            vmin=0,
            vmax=1,
            origin='upper',
            interpolation='nearest',
            extent=[0, n_bins, total_rows, 0]
        )

    axes[0].set_yticks(range(total_rows))
    axes[0].set_yticklabels(combined_labels_A, fontsize=8)
    axes[0].set_ylabel('State', fontsize=10)
    axes[0].set_title(f'{label_A} - Bin State Probabilities (vmax={vmax:.3f})', fontsize=11)

    if display_atomic:
        # 黑色实线分隔原子态（行0-3）和仓态（行4+），应该在第3行下方（y=4）
        axes[0].axhline(4, color='black', linewidth=2)

    # x-axis (dual: time and bin index)
    # 时间轴：从左到右显示 (N-1)dt, ..., dt, 0（递减）
    # 仓索引：从左到右显示 0, 1, ..., N-1（递增，因为A1索引是0，AN索引是N-1）
    n_ticks = min(10, n_bins)
    tick_indices = np.linspace(0, n_bins - 1, n_ticks, dtype=int)
    axes[0].set_xticks(tick_indices)
    # 反转时间标签：tick_indices[i] 对应的时间是 t[(n_bins-1) - tick_indices[i]]
    axes[0].set_xticklabels([f'{time_axis_s[(n_bins - 1) - i] * 1e9:.0f}' for i in tick_indices], fontsize=9)
    axes[0].set_xlabel('Time (ns)', fontsize=10)
    axes[0].top = axes[0].twiny()
    axes[0].top.set_xticks(tick_indices)
    # bin index保持原样：最左边是0（A1），最右边是N-1（AN）
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
        origin='upper',
        extent=[0, n_bins, total_rows, 0]
    )

    im_B_vac = axes[1].imshow(
        np.ma.masked_where(~mask_vac, combined_B),
        aspect='auto',
        cmap=vac_cmap,
        vmin=0,
        vmax=1,
        origin='upper',
        interpolation='nearest',
        extent=[0, n_bins, total_rows, 0]
    )

    if display_atomic:
        im_B_atom = axes[1].imshow(
            np.ma.masked_where(~mask_atom, combined_B),
            aspect='auto',
            cmap=atom_cmap,
            vmin=0,
            vmax=1,
            origin='upper',
            interpolation='nearest',
            extent=[0, n_bins, total_rows, 0]
        )

    axes[1].set_yticks(range(total_rows))
    axes[1].set_yticklabels(combined_labels_B, fontsize=8)
    axes[1].set_ylabel('State', fontsize=10)
    axes[1].set_title(f'{label_B} - Bin State Probabilities (vmax={vmax:.3f})', fontsize=11)

    if display_atomic:
        # 黑色实线分隔原子态（行0-3）和仓态（行4+），应该在第3行下方（y=4）
        axes[1].axhline(4, color='black', linewidth=2)

    axes[1].set_xticks(tick_indices)
    # 反转时间标签：tick_indices[i] 对应的时间是 t[(n_bins-1) - tick_indices[i]]
    axes[1].set_xticklabels([f'{time_axis_s[(n_bins - 1) - i] * 1e9:.0f}' for i in tick_indices], fontsize=9)
    axes[1].set_xlabel('Time (ns)', fontsize=10)
    axes[1].top = axes[1].twiny()
    axes[1].top.set_xticks(tick_indices)
    # bin index保持原样：最左边是0（B1），最右边是N-1（BN）
    axes[1].top.set_xticklabels([str(i) for i in tick_indices], fontsize=9)
    axes[1].top.set_xlabel('Bin index', fontsize=10)
    axes[1].top.set_xlim(axes[1].get_xlim())

    # Add separator lines for bin states
    # 根据bin维度确定分界线位置
    if bin_dim == 6:
        # 6维：按1517光子数分组
        # 行0：0光子(vac)，行1-2：1光子(H/V)，行3-5：2光子(2H/2V/HV)
        boundaries = [1, 3]
    elif bin_dim == 5:
        # 5维：按 780 / 1517 子空间分组
        # 行0：vac，行1-2：780，行3-4：1517
        boundaries = [3]
    else:
        boundaries = []

    for ax in axes:
        for boundary in boundaries:
            ax.axhline(boundary + atom_row_offset, color='white', linewidth=1, alpha=0.5, linestyle='--')

    # Add colorbars aligned with their respective sections
    ax_pos_A = axes[0].get_position()
    ax_pos_B = axes[1].get_position()
    fig_height = ax_pos_A.y1 - ax_pos_A.y0

    if display_atomic:
        # Three colorbars: Atom (4/total), Vac (1/total), Bin (rest)
        # Atomic colorbar for arm A
        cax_A_atom = fig.add_axes([
            ax_pos_A.x1 + 0.01,
            ax_pos_A.y1 - fig_height * (4/total_rows),
            0.01,
            fig_height * (4/total_rows)
        ])
        cbar_A_atom = fig.colorbar(im_A_atom, cax=cax_A_atom)
        cbar_A_atom.set_ticks([0, 0.5, 1])
        cbar_A_atom.set_label('Atom', fontsize=9)

        # (vac,vac) colorbar for arm A
        cax_A_vac = fig.add_axes([
            ax_pos_A.x1 + 0.01,
            ax_pos_A.y1 - fig_height * (5/total_rows),
            0.01,
            fig_height * (1/total_rows)
        ])
        cbar_A_vac = fig.colorbar(im_A_vac, cax=cax_A_vac)
        cbar_A_vac.set_ticks([0.5])
        cbar_A_vac.set_label('Vac', fontsize=8)

        # Bin colorbar for arm A
        bin_rows = total_rows - 5  # 除去4个原子行和1个vac行
        cax_A = fig.add_axes([
            ax_pos_A.x1 + 0.01,
            ax_pos_A.y0,
            0.01,
            fig_height * (bin_rows/total_rows)
        ])
        cbar_A = fig.colorbar(im_A, cax=cax_A)
        n_ticks_cb = 4
        tick_vals = np.linspace(0, vmax, n_ticks_cb)
        cbar_A.set_ticks(tick_vals)
        cbar_A.set_label(f'Bin (max={vmax:.3f})', fontsize=9)

        # Atomic colorbar for arm B
        cax_B_atom = fig.add_axes([
            ax_pos_B.x1 + 0.01,
            ax_pos_B.y1 - fig_height * (4/total_rows),
            0.01,
            fig_height * (4/total_rows)
        ])
        cbar_B_atom = fig.colorbar(im_B_atom, cax=cax_B_atom)
        cbar_B_atom.set_ticks([0, 0.5, 1])
        cbar_B_atom.set_label('Atom', fontsize=9)

        # (vac,vac) colorbar for arm B
        cax_B_vac = fig.add_axes([
            ax_pos_B.x1 + 0.01,
            ax_pos_B.y1 - fig_height * (5/total_rows),
            0.01,
            fig_height * (1/total_rows)
        ])
        cbar_B_vac = fig.colorbar(im_B_vac, cax=cax_B_vac)
        cbar_B_vac.set_ticks([0.5])
        cbar_B_vac.set_label('Vac', fontsize=8)

        # Bin colorbar for arm B
        bin_rows = total_rows - 5  # 除去4个原子行和1个vac行
        cax_B = fig.add_axes([
            ax_pos_B.x1 + 0.01,
            ax_pos_B.y0,
            0.01,
            fig_height * (bin_rows/total_rows)
        ])
        cbar_B = fig.colorbar(im_B, cax=cax_B)
        cbar_B.set_ticks(tick_vals)
        cbar_B.set_label(f'Bin (max={vmax:.3f})', fontsize=9)
    else:
        # Two colorbars: Vac (1/total_rows), Bin (rest)
        # (vac,vac) colorbar for arm A
        cax_A_vac = fig.add_axes([
            ax_pos_A.x1 + 0.01,
            ax_pos_A.y1 - fig_height * (1/total_rows),
            0.01,
            fig_height * (1/total_rows)
        ])
        cbar_A_vac = fig.colorbar(im_A_vac, cax=cax_A_vac)
        cbar_A_vac.set_ticks([0.5])
        cbar_A_vac.set_label('Vac', fontsize=8)

        # Bin colorbar for arm A
        bin_rows = total_rows - 1  # 除去1个vac行
        cax_A = fig.add_axes([
            ax_pos_A.x1 + 0.01,
            ax_pos_A.y0,
            0.01,
            fig_height * (bin_rows/total_rows)
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
        cbar_B_vac.set_ticks([0.5])
        cbar_B_vac.set_label('Vac', fontsize=8)

        # Bin colorbar for arm B
        bin_rows = total_rows - 1  # 除去1个vac行
        cax_B = fig.add_axes([
            ax_pos_B.x1 + 0.01,
            ax_pos_B.y0,
            0.01,
            fig_height * (bin_rows/total_rows)
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
    _maybe_show(show=show)
    plt.close()  # 自动关闭
    print(f"  Saved dual-arm heatmaps to: {save_path}")



