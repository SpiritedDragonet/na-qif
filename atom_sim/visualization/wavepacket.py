# -*- coding: utf-8 -*-
"""
波包可视化模块

本模块提供从MPS态中提取和可视化波包的函数，
包括强度包络和单光子概率。
"""

from typing import Tuple, Optional, List, Union
import numpy as np
import matplotlib.pyplot as plt

from ..core.mps import MPSState
from ..config import TimeGrid
from ..simulation.trajectory import EmissionResult


# ============================================================================
# 波包提取算符
# ============================================================================

def _telecom_ops_1517():
    """
    构造1517nm子空间的投影和数算符。

    1517基：[vac, 1H, 1V, 2H, 2V, HV]

    Returns
    -------
    Tuple of np.ndarray
        (P1_1517, N_1517, P1H_1517, P1V_1517, NH_1517, NV_1517)
        - P1_1517: 单光子投影（所有偏振）
        - N_1517: 总光子数算符
        - P1H_1517: 单H光子投影
        - P1V_1517: 单V光子投影
        - NH_1517: H光子数算符
        - NV_1517: V光子数算符
    """
    # 基：[vac, 1H, 1V, 2H, 2V, HV]
    # 单光子投影（所有偏振）
    P1_1517 = np.diag([0, 1, 1, 0, 0, 0]).astype(complex)

    # 总光子数
    N_1517 = np.diag([0, 1, 1, 2, 2, 2]).astype(complex)

    # H/V单光子投影
    P1H_1517 = np.diag([0, 1, 0, 0, 0, 0]).astype(complex)
    P1V_1517 = np.diag([0, 0, 1, 0, 0, 0]).astype(complex)

    # H/V光子数
    NH_1517 = np.diag([0, 1, 0, 2, 0, 1]).astype(complex)
    NV_1517 = np.diag([0, 0, 1, 0, 2, 1]).astype(complex)

    return P1_1517, N_1517, P1H_1517, P1V_1517, NH_1517, NV_1517


def telecom_ops_bin18():
    """
    构造嵌入18维仓空间的通信算符。

    仓空间 = 780(3D) x 1517(6D) = 18D
    假设展平顺序：|i_780> ⊗ |j_1517>，索引 = i_780 * 6 + j_1517

    Returns
    -------
    Tuple of np.ndarray
        (P1_bin, N_bin, P1H_bin, P1V_bin, NH_bin, NV_bin)
        每个都是18x18，作用于完整仓空间
    """
    I_780 = np.eye(3, dtype=complex)

    P1_1517, N_1517, P1H_1517, P1V_1517, NH_1517, NV_1517 = _telecom_ops_1517()

    # 嵌入：I_780 ⊗ Op_1517
    P1_bin = np.kron(I_780, P1_1517)
    N_bin = np.kron(I_780, N_1517)
    P1H_bin = np.kron(I_780, P1H_1517)
    P1V_bin = np.kron(I_780, P1V_1517)
    NH_bin = np.kron(I_780, NH_1517)
    NV_bin = np.kron(I_780, NV_1517)

    return P1_bin, N_bin, P1H_bin, P1V_bin, NH_bin, NV_bin


# ============================================================================
# 波包提取
# ============================================================================

def extract_wavepacket(
    mps: MPSState,
    n_bins: int,
    use_single_photon_prob: bool = True,
    polarized: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从MPS态中提取波包数据。

    链布局：atomA, atomB, A1, B1, A2, B2, ..., AN, BN
    格点索引：A_n = 2 + 2*(n-1), B_n = 3 + 2*(n-1)

    Parameters
    ----------
    mps : MPSState
        MPS态（访问 mps._mps 获取TeNPy MPS）
    n_bins : int
        要提取的时间仓数量
    use_single_photon_prob : bool
        若为True，返回单光子概率 q_n
        若为False，返回强度 <N_n>
    polarized : bool
        若为True，分别返回H和V分量
        若为False，返回总量（H+V）

    Returns
    -------
    Tuple of np.ndarray
        (data_A, data_B) 其中每个为：
        - (n_bins,) 当 polarized=False
        - (n_bins, 2) 当 polarized=True [H, V 列]
    """
    P1_bin, N_bin, P1H_bin, P1V_bin, NH_bin, NV_bin = telecom_ops_bin18()

    # 根据模式选择算符
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

    # 初始化数组
    if polarized:
        data_A = np.zeros((n_bins, 2))
        data_B = np.zeros((n_bins, 2))
    else:
        data_A = np.zeros(n_bins)
        data_B = np.zeros(n_bins)

    for n in range(1, n_bins + 1):
        idx_A = 2 + 2 * (n - 1)  # A_n site index
        idx_B = 3 + 2 * (n - 1)  # B_n site index

        # 获取约化密度矩阵
        rhoA = mps.get_reduced_density([idx_A])
        rhoB = mps.get_reduced_density([idx_B])

        if polarized:
            # H分量
            data_A[n - 1, 0] = float(np.real(np.trace(rhoA @ P1H_bin)))
            data_B[n - 1, 0] = float(np.real(np.trace(rhoB @ P1H_bin)))
            # V分量
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
    提取每个仓的强度包络 <N_n>。

    Parameters
    ----------
    mps : MPSState
        MPS态
    n_bins : int
        时间仓数量
    polarized : bool
        若为True，分别返回H和V

    Returns
    -------
    Tuple of np.ndarray
        (pA, pB) 强度数组
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
    提取每个仓的单光子概率。

    Parameters
    ----------
    mps : MPSState
        MPS态
    n_bins : int
        时间仓数量
    polarized : bool
        若为True，分别返回H和V

    Returns
    -------
    Tuple of np.ndarray
        (qA, qB) 概率数组
    """
    return extract_wavepacket(
        mps, n_bins,
        use_single_photon_prob=True,
        polarized=polarized
    )


# ============================================================================
# 绘图函数
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
    绘制双臂的波包。

    Parameters
    ----------
    data_A : np.ndarray
        A臂数据 (n_bins,) 或 (n_bins, 2) 用于偏振情况
    data_B : np.ndarray
        B臂数据
    time_grid : TimeGrid, optional
        x轴的时间网格
    polarized : bool
        数据是否为偏振（H/V分离）
    normalize : bool
        若为True，归一化使和为1
    title : str
        图标题
    labels : Tuple[str, str], optional
        A臂和B臂的图例标签
    ax : plt.Axes, optional
        现有坐标轴用于绘图

    Returns
    -------
    plt.Axes
        坐标轴对象
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    n_bins = len(data_A)
    if time_grid is None:
        x = np.arange(n_bins)
        xlabel = "Bin index"
    else:
        # 反转时间轴：data_A[0]对应A1(最晚发射)，data_A[N-1]对应AN(最早发射)
        # 所以 x[0] 应该是 t[N-1]，x[N-1] 应该是 t[0]
        x = time_grid.t[:n_bins][::-1]  # 反转时间数组
        xlabel = "Time (s)"

    # 如果需要则归一化
    if normalize:
        if polarized:
            data_A = data_A / (data_A.sum() + 1e-15)
            data_B = data_B / (data_B.sum() + 1e-15)
        else:
            data_A = data_A / (data_A.sum() + 1e-15)
            data_B = data_B / (data_B.sum() + 1e-15)

    # 绘图
    if polarized:
        # data_A 和 data_B 是 (n_bins, 2) 含 [H, V] 列
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
    绘制双臂的强度包络 <N_n>。

    Parameters
    ----------
    mps : MPSState
        MPS态
    n_bins : int
        时间仓数量
    time_grid : TimeGrid, optional
        时间网格
    polarized : bool
        是否分别显示H/V
    ax : plt.Axes, optional
        现有坐标轴

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
    绘制双臂的单光子概率。

    Parameters
    ----------
    mps : MPSState
        MPS态
    n_bins : int
        时间仓数量
    time_grid : TimeGrid, optional
        时间网格
    polarized : bool
        是否分别显示H/V
    normalize : bool
        是否归一化（推荐）
    ax : plt.Axes, optional
        现有坐标轴

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


# ============================================================================
# 仓状态热图可视化
# ============================================================================

def _get_bin18_state_labels() -> List[str]:
    """
    获取18个仓状态的标签。

    仓空间 = 780(3D) x 1517(6D)，索引 = i_780 * 6 + i_1517

    780子空间：仅支持0或1个光子（|vac>, |H>, |V>）
    1517子空间：支持最多2个光子（|vac>, |H>, |V>, |2H>, |2V>, |HV>）

    Returns
    -------
    List[str]
        18个状态标签，格式为 |780,1517>
    """
    # 780基态（仅单光子）
    bases_780 = ['|vac>', '|H>', '|V>']
    # 1517基态（最多两个光子）
    bases_1517 = ['|vac>', '|H>', '|V>', '|2H>', '|2V>', '|HV>']

    labels = []
    for i_780, b780 in enumerate(bases_780):
        for i_1517, b1517 in enumerate(bases_1517):
            # 格式：|780_state, 1517_state>
            # 移除尖括号以更清晰地显示，保持结构清晰
            if b780 == '|vac>' and b1517 == '|vac>':
                label = '|vac,vac>'  # 两者都是真空
            elif b780 == '|vac>':
                label = f'|vac,{b1517[1:-1]}>'  # 仅1517态
            elif b1517 == '|vac>':
                label = f'|{b780[1:-1]},vac>'  # 仅780态
            else:
                # 两者都不是真空：显示两个态
                label = f'|{b780[1:-1]},{b1517[1:-1]}>'
            labels.append(label)

    return labels


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


def _get_bin3_state_labels() -> List[str]:
    """
    获取3个仓状态的标签（发射后的原始780nm态）。

    780子空间：支持最多1个光子
    - |vac>: 真空态
    - |H>: 单H光子（780nm）
    - |V>: 单V光子（780nm）

    Returns
    -------
    List[str]
        3个状态标签
    """
    return ['|vac>', '|H>', '|V>']


def _infer_first_bin_site(mps: MPSState) -> int:
    """Infer the first bin site index based on 3D atom sites."""
    n_3d_sites = sum(1 for d in mps.d if d == 3)
    return 2 if n_3d_sites >= 2 else 0


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


def extract_bin_state_probabilities(
    mps: MPSState,
    arm: str = 'A',
    n_bins: int = None,
    atom_at_end: bool = False,
) -> np.ndarray:
    """
    提取所有时间仓中18个仓状态各自的概率。

    对每个仓，计算约化密度矩阵并提取对角元素
    （18个状态中每个状态的概率）。

    支持多种链布局：
    1. 双原子：atomA, atomB, A1, B1, A2, B2, ..., AN, BN
    2. 单原子（无SWAP）：atom, bin1, bin2, ..., binN
    3. 单原子（SWAP后）：bin1, bin2, ..., binN, atom

    Parameters
    ----------
    mps : MPSState
        MPS态
    arm : str
        要提取的臂（'A' 或 'B'）。单原子布局时忽略。
    n_bins : int, optional
        时间仓数量。若为None，从链长度推断。
    atom_at_end : bool
        若为True，假设单原子布局且原子在末端
        （SWAP传送带后）。仓位于格点0到n_bins-1。

    Returns
    -------
    np.ndarray
        概率数组，形状为 (n_bins, 18)
        prob[i, j] = 仓i中状态j的概率
    """
    if n_bins is None:
        # 自动检测链布局
        # 统计有多少个3维格点（原子）vs 18维格点（仓）
        n_3d = sum(1 for d in mps.d if d == 3)
        if n_3d == 1:
            # 单原子：L = 1 + n_bins
            n_bins = mps.L - 1
        else:
            # 双原子：L = 2 + 2 * n_bins
            n_bins = (mps.L - 2) // 2

    # 通过检查维度检测链类型
    # 单原子：恰好一个3维格点（原子），其余为18维（仓）
    # 双原子：恰好两个3维格点（原子），其余为18维（仓）
    n_3d = sum(1 for d in mps.d if d == 3)
    is_single_atom = n_3d == 1

    # 存储概率的数组：(n_bins, 18)
    probs = np.zeros((n_bins, 18))

    if is_single_atom:
        if atom_at_end:
            # SWAP后：仓位于格点0, 1, ..., n_bins-1，原子位于格点n_bins
            # 需要注意：mps.L = 1 + n_bins，原子在末端
            # 实际仓数可能少于 mps.L - 1，如果原子在末端
            # 找到3维原子格点的位置
            atom_site = next(i for i, d in enumerate(mps.d) if d == 3)
            actual_n_bins = min(n_bins, atom_site)  # 不要越过原子
            for n in range(actual_n_bins):
                idx = n  # 仓索引为 0, 1, ..., n_bins-1
                rho = mps.get_reduced_density([idx])
                # 处理3D（原子）和18D（仓）两种情况
                if rho.shape[0] == 18:
                    probs[n, :] = np.diag(rho).real
                # 否则：跳过3D格点（原子）
        else:
            # SWAP前：原子在格点0，仓在格点1, 2, ..., n_bins
            # 找到原子格点（3D）并从那里开始
            atom_site = next(i for i, d in enumerate(mps.d) if d == 3)
            for n in range(n_bins):
                idx = atom_site + 1 + n  # 仓跟随原子
                if idx >= mps.L:
                    continue  # 若越界则跳过
                rho = mps.get_reduced_density([idx])
                if rho.shape[0] == 18:
                    probs[n, :] = np.diag(rho).real
    else:
        # 双原子布局：atomA, atomB, A1, B1, A2, B2, ..., AN, BN
        for n in range(n_bins):
            if arm.upper() == 'A':
                idx = 2 + 2 * n  # A_n格点索引
            else:
                idx = 3 + 2 * n  # B_n格点索引
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
    绘制时间仓上仓状态概率的热图。

    创建一个热图，显示每个时间仓中18个仓状态各自的概率。
    行是18个状态（左侧标注），列是时间仓（底部标注时间）。

    Parameters
    ----------
    mps : MPSState
        MPS态
    arm : str
        要绘制的臂（'A' 或 'B'）
    n_bins : int, optional
        时间仓数量。若为None，从链长度推断。
    time_grid : TimeGrid, optional
        x轴标签的时间网格
    subspace : str
        显示哪个子空间（'780'、'1517' 或 'both'）
    group_by : str
        如何分组分隔线（'780' 或 '1517'）
        - '780'：按780态分组（vac/H/V）- 线在6, 12
        - '1517'：按1517光子数分组（0/1/2光子）- 线在3, 6，跨越所有780态
    vmax : float, optional
        色标的最大值。若为None，自动缩放。
    figsize : tuple
        图形大小（宽度，高度）
    ax : plt.Axes, optional
        现有坐标轴用于绘图
    atom_at_end : bool
        若为True，假设单原子布局且原子在末端
        （SWAP传送带后）。用于test_emission_wavepacket.py的结果。
    separate_vac_scale : bool
        若为True，为 |vac,vac> 行（索引0）使用单独的色标
        以更好地可视化真空概率的小变化。

    Returns
    -------
    plt.Axes
        坐标轴对象
    """
    import matplotlib

    # 提取概率
    probs = extract_bin_state_probabilities(mps, arm=arm, n_bins=n_bins, atom_at_end=atom_at_end)
    n_bins_actual = probs.shape[0]

    # 获取状态标签
    state_labels = _get_bin18_state_labels()

    # 根据请求过滤子空间
    if subspace == '780':
        # 仅显示780非真空的状态（索引6-17）
        # 或重新组织以显示780子空间结构
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

    # 过滤数据
    probs_filtered = probs[:, row_indices]

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    # 如果请求，为vac,vac创建单独色标的热图
    if separate_vac_scale:
        # 分割：第0行（|vac,vac>）和第1-17行
        from matplotlib.colors import Normalize

        # 对于vac,vac行：使用自身范围，中心在1
        vac_row = probs_filtered[:, 0:1].T
        vac_vmin = max(0, vac_row.min() - 0.05)
        vac_vmax = min(1, vac_row.max() + 0.05)

        # 对于其他行：使用自动缩放或提供的vmax
        other_rows = probs_filtered[:, 1:].T
        if vmax is None:
            other_vmax = max(0.01, other_rows.max())
        else:
            other_vmax = vmax

        # 创建用于显示的组合数据，使用单独归一化
        # 我们将使用堆叠的两个imshow调用
        display_data = probs_filtered.T

        # 为两种不同归一化创建掩码数组
        im = ax.imshow(
            display_data,
            aspect='auto',
            cmap='viridis',
            vmin=0,
            vmax=max(vac_vmax, other_vmax) if vmax is None else vmax,
            origin='upper'
        )
    else:
        # 标准单色标热图
        im = ax.imshow(
            probs_filtered.T,  # 转置使状态为行，仓为列
            aspect='auto',
            cmap='viridis',
            vmin=0,
            vmax=vmax,
            origin='upper'
        )

    # 设置y轴标签（状态名称）
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    # 设置x轴标签（双重：顶部仓索引，底部时间）
    n_ticks = min(10, n_bins_actual)
    tick_indices = np.linspace(0, n_bins_actual - 1, n_ticks, dtype=int)

    # 底部x轴：时间（纳秒）
    # 反转时间轴：probs[0]对应A1/B1(最晚发射)，probs[N-1]对应AN/BN(最早发射)
    ax.set_xticks(tick_indices)
    if time_grid is not None:
        # 反转时间标签：索引0显示t[N-1]，索引N-1显示t[0]
        reversed_times = [time_grid.t[n_bins_actual - 1 - i] for i in tick_indices]
        ax.set_xticklabels([f'{t:.1f}' for t in reversed_times])
    else:
        ax.set_xticklabels([str(i) for i in tick_indices])
    ax.set_xlabel('Time (ns)')

    # 顶部x轴：仓索引
    ax_top = ax.twiny()
    ax_top.set_xticks(tick_indices)
    ax_top.set_xticklabels([str(i) for i in tick_indices])
    ax_top.set_xlabel('Bin index')
    ax_top.set_xlim(ax.get_xlim())  # 同步限制

    ax.set_ylabel('Bin state |780,1517>')
    ax.set_title(f'Arm {arm.upper()} Bin State Probabilities{title_suffix}')

    # 添加色条，更好地定位避免重叠
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Probability')

    # 根据group_by参数添加分隔线
    if group_by == '780':
        # 按780态分组：|vac>, |H>, |V>
        # 行0-5：780=vac，行6-11：780=H，行12-17：780=V
        boundaries = [6, 12]
    else:  # '1517'
        # 按1517光子数分组：0/1/2光子
        # 1517结构对每个780态重复（每6行）
        # vac(0), H(1), V(1) -> 第2行后边界
        # 2H(2), 2V(2), HV(2) -> 第5行后边界
        # 此模式对780=H（行6-11）和780=V（行12-17）重复
        boundaries = [3, 6, 9, 12, 15, 18]

    for boundary in boundaries:
        ax.axhline(boundary, color='white', linewidth=1, alpha=0.5, linestyle='--')

    return ax


# ============================================================================
# 双臂热图可视化（通用）
# ============================================================================

def plot_dual_arm_heatmap(
    result: Union[EmissionResult, MPSState],
    save_path: str = "dual_arm_heatmap.png",
    show_atomic: bool = False,
    stage_name: str = "",
    time_grid: Optional[TimeGrid] = None,
    vmax_scale_factor: float = 1.5,
    validate: bool = True,
    trace_tol: float = 1e-6,
) -> None:
    """
    可视化双臂仓状态概率，可选显示原子状态。

    通用热图函数，适用于任何仿真阶段：
    - 发射：使用 show_atomic=True 显示原子状态演化
    - QFC/Jones/Loss/BS：使用 show_atomic=False（原子不参与）

    每个臂显示：
    - 若 show_atomic=True：顶部3行（原子）+ 底部18行（仓状态）
    - 若 show_atomic=False：仅18行仓状态

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
    time_grid : TimeGrid, optional
        x轴标签的时间网格。若为None且result是EmissionResult，
        使用result.time_grid。
    vmax_scale_factor : float
        缩放vmax的因子（相对于最大仓概率）
    validate : bool
        若为 True，先检查单 bin 约化密度矩阵的归一化
    trace_tol : float
        归一化检查的误差阈值
    """
    import matplotlib as mpl
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    mpl.rcParams['image.interpolation'] = 'nearest'

    # 从结果中提取MPS和time_grid
    if isinstance(result, EmissionResult):
        mps = result.mps
        if time_grid is None:
            time_grid = result.time_grid
        n_bins = result.get_n_bins()
        has_atom_evol = True
    else:  # MPSState
        mps = result
        if time_grid is None:
            time_grid = TimeGrid(dt=1.0, N=1)  # 虚拟
        n_bins = (mps.L - 2) // 2  # 从链长度推断
        has_atom_evol = False

    # 创建具有更大间距的图形 (1080x720 aspect ratio)
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 7.2))
    # 注意：subplots_adjust 将在确定 display_atomic 后设置

    # 检测是否有原子站点（3D站点）
    n_3d_sites = sum(1 for d in mps.d if d == 3)
    has_atomic_sites = n_3d_sites >= 2

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
        atom_A_probs = np.diag(rho_A).real  # shape: (3,)
        atom_B_probs = np.diag(rho_B).real  # shape: (3,)
        # 扩展为所有bin列显示相同的概率（整行同色）
        atom_A_for_bins = np.tile(atom_A_probs.reshape(3, 1), (1, n_bins))
        atom_B_for_bins = np.tile(atom_B_probs.reshape(3, 1), (1, n_bins))

    # 获取第一个bin的维度
    if first_bin_site < len(mps.d):
        first_bin_dim = mps.d[first_bin_site]
    else:
        raise ValueError(f"Cannot find bin site: chain has only {len(mps.d)} sites")

    if first_bin_dim == 18:
        bin_dim = 18
        bin_state_labels = _get_bin18_state_labels()
    elif first_bin_dim == 6:
        bin_dim = 6
        bin_state_labels = _get_bin6_state_labels()
    else:
        raise ValueError(f"Unsupported bin dimension: {first_bin_dim}. Expected 18 or 6.")

    # 提取仓概率
    probs_A = np.zeros((n_bins, bin_dim))
    probs_B = np.zeros((n_bins, bin_dim))

    for n in range(n_bins):
        # 链布局：[atomA, atomB, A1, B1, A2, B2, ..., AN, BN]
        # 原子占据前2个站点（如果存在），bin从站点2开始
        site_A = first_bin_site + 2 * n
        site_B = first_bin_site + 2 * n + 1

        if site_A < len(mps.d) and site_B < len(mps.d):
            rho_A = mps.get_reduced_density([site_A])
            rho_B = mps.get_reduced_density([site_B])
            if rho_A.shape[0] == bin_dim:
                probs_A[n, :] = np.diag(rho_A).real
            if rho_B.shape[0] == bin_dim:
                probs_B[n, :] = np.diag(rho_B).real

    # Calculate vmax EXCLUDING (vac,vac) row (index 0)
    vmax_A = max(0.01, probs_A[:, 1:].max() * vmax_scale_factor)
    vmax_B = max(0.01, probs_B[:, 1:].max() * vmax_scale_factor)
    vmax = max(vmax_A, vmax_B)

    # Create combined data matrices
    if display_atomic:
        atomic_labels = ['|e>', '|1>', '|0>']
        combined_labels_A = atomic_labels + bin_state_labels
        combined_labels_B = atomic_labels + bin_state_labels
        total_rows = 3 + bin_dim

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

    # 根据是否显示原子态调整子图布局
    if display_atomic:
        # 有原子态：使用原来的布局
        plt.subplots_adjust(left=0.04, right=0.85, top=0.80, bottom=0.06, wspace=0.50)
    else:
        # 无原子态：增加底部边距，减少顶部边距，让行填满整个区域
        plt.subplots_adjust(left=0.04, right=0.85, top=0.92, bottom=0.08, wspace=0.50)

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
    axes[0].set_title(f'Arm A - Bin State Probabilities (vmax={vmax:.3f})', fontsize=11)

    if display_atomic:
        # 黑色实线分隔原子态（行0-2）和仓态（行3+），应该在第2行下方（y=3）
        axes[0].axhline(3, color='black', linewidth=2)

    # x-axis (dual: time and bin index)
    # 时间轴：从左到右显示 (N-1)dt, ..., dt, 0（递减）
    # 仓索引：从左到右显示 0, 1, ..., N-1（递增，因为A1索引是0，AN索引是N-1）
    n_ticks = min(10, n_bins)
    tick_indices = np.linspace(0, n_bins - 1, n_ticks, dtype=int)
    axes[0].set_xticks(tick_indices)
    # 反转时间标签：tick_indices[i] 对应的时间是 t[(n_bins-1) - tick_indices[i]]
    axes[0].set_xticklabels([f'{time_grid.t[(n_bins - 1) - i] * 1e9:.0f}' for i in tick_indices], fontsize=9)
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
    axes[1].set_title(f'Arm B - Bin State Probabilities (vmax={vmax:.3f})', fontsize=11)

    if display_atomic:
        # 黑色实线分隔原子态（行0-2）和仓态（行3+），应该在第2行下方（y=3）
        axes[1].axhline(3, color='black', linewidth=2)

    axes[1].set_xticks(tick_indices)
    # 反转时间标签：tick_indices[i] 对应的时间是 t[(n_bins-1) - tick_indices[i]]
    axes[1].set_xticklabels([f'{time_grid.t[(n_bins - 1) - i] * 1e9:.0f}' for i in tick_indices], fontsize=9)
    axes[1].set_xlabel('Time (ns)', fontsize=10)
    axes[1].top = axes[1].twiny()
    axes[1].top.set_xticks(tick_indices)
    # bin index保持原样：最左边是0（B1），最右边是N-1（BN）
    axes[1].top.set_xticklabels([str(i) for i in tick_indices], fontsize=9)
    axes[1].top.set_xlabel('Bin index', fontsize=10)
    axes[1].top.set_xlim(axes[1].get_xlim())

    # Add separator lines for bin states
    # 根据bin维度确定分界线位置
    if bin_dim == 18:
        # 18维：按780态分组 (vac/H/V)，每组6个1517态
        # 行0-5：780=vac，行6-11：780=H，行12-17：780=V
        boundaries = [6, 12]
    elif bin_dim == 6:
        # 6维：按1517光子数分组
        # 行0：0光子(vac)，行1-2：1光子(H/V)，行3-5：2光子(2H/2V/HV)
        boundaries = [1, 3]
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
        cbar_A_vac.set_ticks([0.5])
        cbar_A_vac.set_label('Vac', fontsize=8)

        # Bin colorbar for arm A
        bin_rows = total_rows - 4  # 除去3个原子行和1个vac行
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
        cbar_B_vac.set_ticks([0.5])
        cbar_B_vac.set_label('Vac', fontsize=8)

        # Bin colorbar for arm B
        bin_rows = total_rows - 4  # 除去3个原子行和1个vac行
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
        # Two colorbars: Vac (1/18), Bin (17/18)
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
    plt.show(block=False)  # 非阻塞显示
    plt.pause(5)  # 显示5秒
    plt.close()  # 自动关闭
    print(f"  Saved dual-arm heatmaps to: {save_path}")


# ============================================================================
# 跨 bin 联合分布热图
# ============================================================================

def plot_cross_bin_joint_heatmap(
    mps: MPSState,
    n_bins: int,
    save_path: str = "cross_bin_joint_heatmap.png",
    arm_pair: Tuple[str, str] = ("A", "B"),
    normalize: bool = False,
    vmax: Optional[float] = None,
    validate: bool = True,
    trace_tol: float = 1e-6,
) -> np.ndarray:
    """
    绘制跨 bin 的联合分布热图（两端口/两臂）。

    joint[i, j] 表示：臂 arm_pair[0] 的第 i 个 bin 与
    臂 arm_pair[1] 的第 j 个 bin 同时处于“单光子子空间”的概率
    （H/V 均算单光子）。这是两站点约化密度矩阵的对角边缘分布，
    能显式显示“不同 bin 的两光子”相关性。

    Parameters
    ----------
    mps : MPSState
        MPS态（布局：atomA, atomB, A1, B1, ...）
    n_bins : int
        时间仓数量
    save_path : str
        保存图像路径
    arm_pair : Tuple[str, str]
        选择的两臂/端口，默认 ("A", "B")
    normalize : bool
        若为 True，归一化 joint 使其总和为 1
    vmax : float, optional
        色标上限，None 则自动
    validate : bool
        若为 True，先检查单 bin 约化密度矩阵的归一化
    trace_tol : float
        归一化检查的误差阈值

    Returns
    -------
    np.ndarray
        joint 分布矩阵，形状 (n_bins, n_bins)
    """
    arm_left = arm_pair[0].upper()
    arm_right = arm_pair[1].upper()
    if arm_left not in ("A", "B") or arm_right not in ("A", "B"):
        raise ValueError(f"arm_pair must be ('A','B') or ('B','A'), got {arm_pair}")

    # Validate per-bin reduced density traces before heavy pair extraction
    if validate:
        _validate_bin_rho_traces(mps, n_bins, tol=trace_tol)

    first_bin_site = _infer_first_bin_site(mps)

    def _site_index(arm: str, n: int) -> int:
        return first_bin_site + 2 * n + (0 if arm == "A" else 1)

    # 识别 bin 维度并构造单光子索引
    if first_bin_site >= len(mps.d):
        raise ValueError("Cannot find bin sites in MPS.")
    bin_dim = mps.d[first_bin_site]
    if bin_dim == 6:
        n_diag = np.array([0, 1, 1, 2, 2, 2], dtype=float)
    elif bin_dim == 18:
        _, n_bin, _, _, _, _ = telecom_ops_bin18()
        n_diag = np.real(np.diag(n_bin))
    else:
        raise ValueError(f"Unsupported bin dimension: {bin_dim}. Expected 6 or 18.")

    single_photon_idx = [i for i, n in enumerate(n_diag) if np.isclose(n, 1.0)]
    if not single_photon_idx:
        raise ValueError("No single-photon indices found for the bin basis.")

    joint = np.zeros((n_bins, n_bins), dtype=float)

    for i in range(n_bins):
        site_i = _site_index(arm_left, i)
        if site_i >= mps.L:
            continue
        for j in range(n_bins):
            site_j = _site_index(arm_right, j)
            if site_j >= mps.L:
                continue
            if site_i == site_j:
                raise ValueError("arm_pair refers to the same site; choose different arms.")

            sites = [site_i, site_j]
            if site_i > site_j:
                sites = [site_j, site_i]
            rho_ij = mps.get_reduced_density(sites)
            prob = 0.0
            for a in single_photon_idx:
                for b in single_photon_idx:
                    prob += rho_ij[a, b, a, b].real
            joint[i, j] = prob

    if normalize:
        total = joint.sum()
        if total > 0:
            joint = joint / total

    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    im = ax.imshow(
        joint,
        origin='lower',
        aspect='auto',
        cmap='magma',
        vmin=0,
        vmax=vmax,
    )

    n_ticks = min(10, n_bins)
    tick_indices = np.linspace(0, n_bins - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_indices)
    ax.set_yticks(tick_indices)
    ax.set_xticklabels([str(i) for i in tick_indices])
    ax.set_yticklabels([str(i) for i in tick_indices])

    ax.set_xlabel(f"Bin index (Arm {arm_left})")
    ax.set_ylabel(f"Bin index (Arm {arm_right})")
    title = "Cross-bin joint single-photon distribution"
    if normalize:
        title += " (normalized)"
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Probability" if not normalize else "Normalized probability")

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(5)
    plt.close(fig)
    print(f"  Saved cross-bin joint heatmap to: {save_path}")

    return joint


