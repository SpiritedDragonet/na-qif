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
        x = time_grid.t[:n_bins]
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
        - '780'：按780态分组（vac/H/V）- 线在5.5, 11.5
        - '1517'：按1517光子数分组（0/1/2光子）- 线在2.5, 5.5，跨越所有780态
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
    ax.set_xticks(tick_indices)
    if time_grid is not None:
        ax.set_xticklabels([f'{time_grid.t[i]:.1f}' for i in tick_indices])
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
        boundaries = [5.5, 11.5]
    else:  # '1517'
        # 按1517光子数分组：0/1/2光子
        # 1517结构对每个780态重复（每6行）
        # vac(0), H(1), V(1) -> 第2行后边界
        # 2H(2), 2V(2), HV(2) -> 第5行后边界
        # 此模式对780=H（行6-11）和780=V（行12-17）重复
        boundaries = [2.5, 5.5, 8.5, 11.5, 14.5, 17.5]

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

    if show_atomic and not has_atom_evol:
        raise ValueError("show_atomic=True requires EmissionResult with atomic state evolution")

    # 创建具有更大间距的图形
    fig, axes = plt.subplots(1, 2, figsize=(24, 13))
    plt.subplots_adjust(left=0.04, right=0.85, top=0.80, bottom=0.06, wspace=0.50)

    # 如果需要，提取原子状态演化
    if show_atomic and has_atom_evol:
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

    # 提取仓概率
    probs_A = np.zeros((n_bins, 18))
    probs_B = np.zeros((n_bins, 18))

    for n in range(n_bins):
        # 链布局：A1(0), B1(1), A2(2), B2(3), ..., AN, BN
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
# 跨仓一阶相干性（用于相位可视化）
# ============================================================================

def _telecom_annihilation_ops():
    """
    构造1517nm子空间的湮灭算符。

    1517基：[vac, 1H, 1V, 2H, 2V, HV]
    a_H：湮灭H光子（映射 |1H> -> |vac>）
    a_V：湮灭V光子（映射 |1V> -> |vac>）

    Returns
    -------
    Tuple of np.ndarray
        (a_H_1517, a_V_1517, a_H_dag_1517, a_V_dag_1517)
        每个都是6x6复数矩阵
    """
    # a_H：映射 |1H>（索引1）到 |vac>（索引0）
    a_H_1517 = np.zeros((6, 6), dtype=complex)
    a_H_1517[0, 1] = 1.0

    # a_V：映射 |1V>（索引2）到 |vac>（索引0）
    a_V_1517 = np.zeros((6, 6), dtype=complex)
    a_V_1517[0, 2] = 1.0

    # 厄米共轭（产生算符）
    a_H_dag_1517 = a_H_1517.conj().T
    a_V_dag_1517 = a_V_1517.conj().T

    return a_H_1517, a_V_1517, a_H_dag_1517, a_V_dag_1517


def _bin18_annihilation_ops():
    """
    构造嵌入18维仓空间的湮灭算符。

    仓空间 = 780(3D) x 1517(6D) = 18D
    我们只关心通信（1517nm）光子。

    Returns
    -------
    Tuple of np.ndarray
        (a_H_bin, a_V_bin, a_H_dag_bin, a_V_dag_bin)
        每个都是18x18复数矩阵
    """
    I_780 = np.eye(3, dtype=complex)
    a_H_1517, a_V_1517, a_H_dag_1517, a_V_dag_1517 = _telecom_annihilation_ops()

    # 嵌入：I_780 ⊗ a_1517
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
    提取一阶相干性 G_{nm} = <a_n^dag a_m> 用于波包相位。

    对于单光子态，G近似秩-1：G_{nm} ≈ ξ_n* ξ_m
    其中ξ_n是波包振幅。ξ_n的相位给出每个仓的光学相位。

    两种提取模式：
    1. reference_bin=None：提取完整G矩阵，从特征向量获取相位
    2. reference_bin=int：提取相对相位 g_n = <a_ref^dag a_n>

    Parameters
    ----------
    mps : MPSState
        MPS态
    n_bins : int
        时间仓数量
    arm : str
        要提取的臂（'A' 或 'B'）
    reference_bin : int, optional
        参考仓索引。若为None，使用特征分解。
        若指定，计算相对此仓的相位。
    coherence_threshold : float
        信任相位的相干幅度最小值。低于此值，相位被屏蔽。

    Returns
    -------
    Tuple of np.ndarray
        (phases, amplitudes)
        - phases: (n_bins,) 相位数组，单位弧度 [-π, π]
        - amplitudes: (n_bins,) 相干幅度数组
    """
    # 获取湮灭算符
    a_H, a_V, a_H_dag, a_V_dag = _bin18_annihilation_ops()

    # 默认使用H偏振（可扩展到两者）
    a = a_H
    a_dag = a_H_dag

    # 获取此臂的格点索引
    # 链布局：A1(0), B1(1), A2(2), B2(3), ..., AN, BN
    arm_indices = []
    if arm.upper() == 'A':
        for n in range(n_bins):
            arm_indices.append(2 * n)
    else:  # B臂
        for n in range(n_bins):
            arm_indices.append(2 * n + 1)

    # 方法1：参考仓方法（更快，给出相对相位）
    if reference_bin is not None:
        if reference_bin < 0 or reference_bin >= n_bins:
            raise ValueError(f"reference_bin={reference_bin} out of range [0, {n_bins})")

        ref_site = arm_indices[reference_bin]
        phases = np.zeros(n_bins)
        amplitudes = np.zeros(n_bins)

        for i, site in enumerate(arm_indices):
            if i == reference_bin:
                # 自相关：<a^dag a> = 数算符
                rho_ref = mps.get_reduced_density([ref_site])
                if rho_ref.shape[0] == 18:
                    N_op = a_dag @ a
                    amplitudes[i] = np.abs(np.trace(rho_ref @ N_op))
                else:
                    amplitudes[i] = 0.0
                phases[i] = 0.0  # 参考相位
            else:
                # 获取两格点约化密度矩阵
                # 必须确保格点按张量积顺序排列
                if ref_site < site:
                    sites = [ref_site, site]
                    # 构造两格点算符：a_ref^dag ⊗ a_i
                    # 算符维度：每个格点18x18 -> 两格点324x324
                    op_2site = np.kron(a_dag, a)
                else:
                    sites = [site, ref_site]
                    # 顺序反转：a_i ⊗ a_ref^dag
                    op_2site = np.kron(a, a_dag)

                rho_2site = mps.get_reduced_density(sites)

                # rho_2site 应该是 (18*18) x (18*18) = 324x324
                # op_2site 也应该是 324x324
                if rho_2site.shape[0] == 324 and op_2site.shape[0] == 324:
                    # 计算期望值：Tr[rho * (a_dag ⊗ a)]
                    g = np.trace(rho_2site @ op_2site)
                    phases[i] = np.angle(g)
                    amplitudes[i] = np.abs(g)
                else:
                    # 维度不匹配，跳过
                    phases[i] = 0.0
                    amplitudes[i] = 0.0

        return phases, amplitudes

    # 方法2：完整相关矩阵和特征分解
    # 构造G矩阵，其中 G[n,m] = <a_n^dag a_m>
    G = np.zeros((n_bins, n_bins), dtype=complex)

    for n in range(n_bins):
        for m in range(n_bins):
            if m < n:
                # 利用厄米对称性：G[n,m] = conj(G[m,n])
                G[n, m] = np.conj(G[m, n])
                continue

            site_n = arm_indices[n]
            site_m = arm_indices[m]

            if n == m:
                # 在位：<a_n^dag a_n> = 仓n的光子数
                rho_n = mps.get_reduced_density([site_n])
                if rho_n.shape[0] == 18:
                    # Tr[rho * a^dag a] = 数期望
                    N_op = a_dag @ a
                    G[n, n] = np.trace(rho_n @ N_op)
                else:
                    G[n, n] = 0.0
            else:
                # 跨仓相关
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

    # 从主特征向量提取相位
    # 对于纯单光子态，G应该是秩-1
    eigvals, eigvecs = np.linalg.eigh(G)

    # 主特征值和特征向量
    idx_max = np.argmax(np.abs(eigvals))
    eigenmode = eigvecs[:, idx_max]

    # 相位是特征变量的辐角
    # 全局相位是任意的，所以我们设平均相位为0
    phases = np.angle(eigenmode)
    phases = phases - np.mean(phases)  # 移除全局相位

    # 从特征值获取振幅（单光子取平方根）
    amplitudes = np.sqrt(np.abs(eigvals[idx_max])) * np.abs(eigenmode)

    # 应用阈值屏蔽：在振幅太小的位置将相位设为0
    mask = amplitudes < coherence_threshold
    phases[mask] = 0.0

    return phases, amplitudes


# ============================================================================
# 相位感知（域着色）热图可视化
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
    提取所有18个仓状态的概率和相干相位。

    对每个仓，提��：
    - 概率：rho[k,k]（对角元素）
    - 相位：两种可用方法：
      1. use_crossbin_phase=False：arg(rho[0,k])（与真空的相干）
         警告：对于纠缠态这通常是噪声！
      2. use_crossbin_phase=True：使用跨仓一阶相干性
         G_nm = <a_n^dag a_m> 用于单光子波包相位。
         这是原子-光子纠缠态的物理上有意义的相位。

    Parameters
    ----------
    mps : MPSState
        MPS态
    n_bins : int
        时间仓数量
    arm : str
        要提取的臂（'A' 或 'B'）
    coherence_threshold : float
        相干幅度阈值。当 |rho[0,k]| < threshold 时，
        相位被设为0并被屏蔽（以避免显示数值噪声）。
    use_crossbin_phase : bool
        若为True，使用跨仓一阶相干性进行相位提取。
        这是原子-光子纠缠态的推荐方法。
    reference_bin : int, optional
        跨仓相位计算的参考仓。若为None，使用
        最大强度的仓作为参考。

    Returns
    -------
    Tuple of np.ndarray
        (probs_A, probs_B, phases_A, phases_B)
        - probs: (n_bins, 18) 实数概率数组
        - phases: (n_bins, 18) 实数相位数组（单位弧度，-π到π）
    """
    probs_A = np.zeros((n_bins, 18), dtype=float)
    probs_B = np.zeros((n_bins, 18), dtype=float)
    phases_A = np.zeros((n_bins, 18), dtype=float)
    phases_B = np.zeros((n_bins, 18), dtype=float)

    # 提取所有仓的概率
    for n in range(n_bins):
        # 链布局：A1(0), B1(1), A2(2), B2(3), ..., AN, BN
        site_A = 2 * n
        site_B = 2 * n + 1

        rho_A = mps.get_reduced_density([site_A])
        rho_B = mps.get_reduced_density([site_B])

        if rho_A.shape[0] == 18:
            probs_A[n, :] = np.diag(rho_A).real
        if rho_B.shape[0] == 18:
            probs_B[n, :] = np.diag(rho_B).real

    if use_crossbin_phase:
        # 使用跨仓一阶相干性获取物理上有意义的相位
        # 这从 G_nm = <a_n^dag a_m> 提取波包相位
        if reference_bin is None:
            # 找到具有最大单光子概率的仓
            # 单光子态位于索引1（|vac,H>）和2（|vac,V>）
            total_1ph_A = probs_A[:, 1] + probs_A[:, 2]
            total_1ph_B = probs_B[:, 1] + probs_B[:, 2]
            ref_A = int(np.argmax(total_1ph_A))
            ref_B = int(np.argmax(total_1ph_B))
        else:
            ref_A = ref_B = reference_bin

        # 提取跨仓相位
        crossbin_phases_A, crossbin_amps_A = extract_first_order_coherence(
            mps, n_bins, arm='A', reference_bin=ref_A,
            coherence_threshold=coherence_threshold
        )
        crossbin_phases_B, crossbin_amps_B = extract_first_order_coherence(
            mps, n_bins, arm='B', reference_bin=ref_B,
            coherence_threshold=coherence_threshold
        )

        # 广播到所有18个状态（相位在仓状态间共享）
        for n in range(n_bins):
            phases_A[n, :] = crossbin_phases_A[n]
            phases_B[n, :] = crossbin_phases_B[n]

            # 应用幅度屏蔽：在相干性小的位置将相位设为0
            if crossbin_amps_A[n] < coherence_threshold:
                phases_A[n, :] = 0.0
            if crossbin_amps_B[n] < coherence_threshold:
                phases_B[n, :] = 0.0

    else:
        # 原始方法：使用真空相干性（对纠缠态通常是噪声）
        for n in range(n_bins):
            site_A = 2 * n
            site_B = 2 * n + 1

            rho_A = mps.get_reduced_density([site_A])
            rho_B = mps.get_reduced_density([site_B])

            if rho_A.shape[0] == 18:
                # 获取相干幅度
                coh_A = rho_A[0, :]
                coh_mag_A = np.abs(coh_A)

                # 应用阈值屏蔽：若相干性太小，相位是噪声
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
    使用HSV颜色模型将概率和相位转换为RGB图像。

    对每个元素：
    - Hue = 相位（0到2π映射到0-1）
    - Saturation = 固定值
    - Value = prob^value_power（归一化）

    Parameters
    ----------
    probs : np.ndarray
        概率数组 (n_rows, n_cols)，实数值 >= 0
    phases : np.ndarray
        相位数组 (n_rows, n_cols)，[-π, π]内的实数值
    saturation : float
        颜色饱和度（0-1）
    value_power : float
        强度映射的幂次
    max_prob : float, optional
        归一化的最大概率。若为None，使用数据最大值。

    Returns
    -------
    np.ndarray
        RGB图像数组 (n_rows, n_cols, 3)，值在[0, 1]内
    """
    from matplotlib.colors import hsv_to_rgb

    # 将相位归一化到[0, 1]作为色相
    hues = (phases + np.pi) / (2 * np.pi)

    # 为亮度通道归一化概率
    if max_prob is None or max_prob <= 0:
        max_prob = probs.max() if probs.max() > 0 else 1.0
    values = (probs / max_prob) ** value_power
    values = np.clip(values, 0, 1)

    # 创建HSV数组
    hsv = np.zeros(probs.shape + (3,))
    hsv[..., 0] = hues
    hsv[..., 1] = saturation
    hsv[..., 2] = values

    # 转换为RGB
    rgb = hsv_to_rgb(hsv)

    return rgb


def _create_hsv_phase_colorbar(
    fig: plt.Figure,
    position: list,
    label: str = "Phase",
) -> None:
    """
    向图形添加水平相位色条（HSV色轮）。

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
    可视化带有相位信息的双臂仓状态振幅。

    此函数模仿 plot_dual_arm_heatmap() 的布局，但使用HSV着色：
    - 色相 = 相干性相位（0到2π作为色轮）
    - 饱和度 = 颜色强度（默认：1.0）
    - 明度 = 亮度 ∝ 概率^value_power

    相位提取方法：
    - use_crossbin_phase=False（默认）：使用 arg(rho[0,k])（真空相干）
      配阈值屏蔽。这很快但对纠缠态可能显示噪声。
      |rho[0,k]| < coherence_threshold 的相位被屏蔽为0。
    - use_crossbin_phase=True：使用跨仓一阶相干性
      G_nm = <a_n^dag a_m> 提取波包相位。
      警告：这是 O(n_bins^2)，对大n_bins可能很慢。

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
    saturation : float
        颜色饱和度（0-1）。较低值产生更柔和的颜色。
    value_power : float
        强度映射的幂次。0.5 = 平方根（默认），1.0 = 线性。
        较高值增加小幅振幅的对比度。
    vmax_scale_factor : float
        缩放最大振幅的因子（相对于最大相干幅度）。
    use_crossbin_phase : bool
        若为True，使用跨仓一阶相干性获取相位。
        这从 G_nm = <a_n^dag a_m> 提取波包相位。
        警告：对大n_bins非常慢（O(n_bins^2)密度矩阵调用）。
        若为False（默认），使用带屏蔽的真空相干 arg(rho[0,k])。
    coherence_threshold : float
        相干幅度的阈值。低于此值，相位被屏蔽。
    reference_bin : int, optional
        跨仓相位计算的参考仓。若为None，使用
        最大强度的仓。
    """
    import matplotlib as mpl
    from matplotlib.colors import hsv_to_rgb

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
