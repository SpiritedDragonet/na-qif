# -*- coding: utf-8 -*-
"""
单轨迹执行模块

本模块实现时间仓仿真的"传送带"主循环。
每个时间仓按顺序处理：发射、QFC、损耗、琼斯旋转、分束器、探测。
"""

from typing import Optional, Tuple
from dataclasses import dataclass, field
import math
import numpy as np

from ..core.mps import MPSState
from ..hilbert.basis import BIN_SPACE, SUBSPACE_780, SUBSPACE_1517
from ..physics.gates import (
    emission_gate, jones_gate
)


# 维度常量，便于代码阅读
DIM_ATOM = 4
DIM_BIN = BIN_SPACE.dim  # 18
DIM_780 = SUBSPACE_780.dim  # 3
DIM_1517 = SUBSPACE_1517.dim  # 6


@dataclass
class EmissionResult:
    """
    双原子发射仿真的结果（仅发射阶段）。
    发射前的链布局：A1, B1, A2, B2, ..., AN, BN, atomA, atomB
    先交换为：A1, B1, A2, B2, ..., AN, atomA, BN, atomB
    随后逐步与相邻 bin 作用并交换，作用到最后一个 bin 时布局近似为：
    A1, atomA, B1, atomB, A2, B2, ..., AN, BN
    最后再按既定顺序移动 atomA 与 atomB，使原子回到链首：
    atomA, atomB, A1, B1, A2, B2, ..., AN, BN
    （原子在前，仓在后）

    Attributes
    ----------
    mps : MPSState
        发射后的最终MPS态
    dt_s : float
        时间步长（秒）
    per_bin_prob_A : np.ndarray
        A臂每个仓的发射概率（形状：n_bins）
    per_bin_prob_B : np.ndarray
        B臂每个仓的发射概率（形状：n_bins）
    atom_states : dict
        最终原子状态 {'A': rho_A, 'B': rho_B}
    atom_A_state_evolution : np.ndarray
        原子A的状态演化（形状：4 x 2*n_bins）
        行：P(|0>), P(|1>), P(|e>), P(|u>)
        列：每次SWAP后的记录
    atom_B_state_evolution : np.ndarray
        原子B的状态演化（形状：4 x 2*n_bins）
        行：P(|0>), P(|1>), P(|e>), P(|u>)
        列：每次SWAP后的记录
    delay_ns_base : float
        设定的A-B时间延迟（纳秒）
    delay_jitter_ns : float
        延迟随机抖动范围（纳秒，均匀分布的半宽）
    delay_jitter_actual_ns : float
        本次采样的延迟抖动（纳秒）
    delay_ns_used : float
        实际使用的A-B时间延迟（纳秒）
    p_source_A : float
        A臂外耦合通道的出射成功概率（由波包生成器估计）
    p_source_B : float
        B臂外耦合通道的出射成功概率（由波包生成器估计）
    """
    mps: MPSState
    dt_s: float
    per_bin_prob_A: np.ndarray
    per_bin_prob_B: np.ndarray
    atom_states: dict
    atom_A_state_evolution: np.ndarray = field(default_factory=lambda: np.zeros((4, 1)))
    atom_B_state_evolution: np.ndarray = field(default_factory=lambda: np.zeros((4, 1)))
    delay_ns_base: float = 0.0
    delay_jitter_ns: float = 0.0
    delay_jitter_actual_ns: float = 0.0
    delay_ns_used: float = 0.0
    p_source_A: float = 0.0
    p_source_B: float = 0.0

    def get_bin_indices(self, n: int) -> Tuple[int, int]:
        """
        获取仓n在A臂和B臂的MPS格点索引。

        发射后的链布局：
        - atomA(0) - atomB(1) - A1(2) - B1(3) - A2(4) - B2(5) - ... - AN(2*n_bins) - BN(2*n_bins+1)
        - A_n 位于格点 2 + 2*n，B_n 位于格点 2 + 2*n + 1

        Parameters
        ----------
        n : int
            仓索引（从0开始）

        Returns
        -------
        Tuple[int, int]
            (site_A, site_B) - A_n和 B_n的MPS格点索引
        """
        # atomA在格点0，atomB在格点1
        # A臂和B臂的仓交错排列：A1(2), B1(3), A2(4), B2(5), ...
        return 2 + 2 * n, 2 + 2 * n + 1

    def get_atom_site_indices(self) -> Tuple[int, int]:
        """
        获取原子A和B的MPS格点索引。

        发射后，原子位于链的最左端。

        Returns
        -------
        Tuple[int, int]
            (site_A, site_B) - atomA和atomB的MPS格点索引
        """
        return 0, 1

    def get_n_bins(self) -> int:
        """获取时间仓的数量。"""
        return len(self.per_bin_prob_A)

    def get_mps_for_next_stage(self) -> MPSState:
        """
        获取准备进入下一阶段的MPS态（如QFC、BSM）。

        当前布局为：atomA, atomB, A1, B1, A2, B2, ..., AN, BN
        其中每对 A_n, B_n 相邻以便进行操作。

        Returns
        -------
        MPSState
            准备好进行下一处理的MPS态
        """
        return self.mps


# ============================================================================
# 统一处理函数（apply_* 模式）
# 所有函数遵循相同接口：
#   - 输入：mps (MPSState), params, verbose (bool)
#   - 输出：mps (MPSState)
#   - 打印格式：所有函数保持一致
# ============================================================================

def apply_qfc(
    mps: MPSState,
    n_bins: int,
    theta_H: float = np.pi/4,
    theta_V: float = np.pi/4,
    verbose: bool = True,
) -> MPSState:
    """
    对所有仓应用QFC门。

    Parameters
    ----------
    mps : MPSState
        MPS态（布局：atomA, atomB, A1, B1, A2, B2, ..., AN, BN）
    n_bins : int
        时间仓数量
    theta_H : float
        H偏振的QFC角度（sin² = 转换概率）
    theta_V : float
        V偏振的QFC角度
    verbose : bool
        是否打印进度

    Returns
    -------
    MPSState
        应用了QFC的MPS态（原地修改）
    """
    from ..physics.gates import qfc_gate

    _print_header("QFC", verbose)
    if verbose:
        print(f"  theta_H = {theta_H:.4f} (sin² = {np.sin(theta_H)**2:.3f})")
        print(f"  theta_V = {theta_V:.4f} (sin² = {np.sin(theta_V)**2:.3f})")

    # 获取QFC门（18x18，作用于单个仓）
    U_qfc = qfc_gate(theta_H=theta_H, theta_V=theta_V)

    if verbose:
        print(f"  U_qfc shape: {U_qfc.shape}")
        print(f"  n_bins={n_bins}, MPS L={mps.L}")
        print(f"  MPS d[:5]={mps.d[:5]}, d[-5:]={mps.d[-5:]}")

    # 对每个仓应用QFC
    # 链布局：atomA(0), atomB(1), A1(2), B1(3), A2(4), B2(5), ...
    for n in range(n_bins):
        site_A = 2 + 2 * n
        site_B = 2 + 2 * n + 1

        mps.apply_one_site_gate(site_A, U_qfc)
        mps.apply_one_site_gate(site_B, U_qfc)

        _print_progress(n + 1, n_bins, verbose)

    _print_footer(mps, verbose, stage="QFC")
    return mps


# 一致打印格式的辅助函数
def _print_header(stage: str, verbose: bool):
    """以一致格式打印阶段标题。"""
    if verbose:
        print(f"\n{'='*60}")
        print(f"{stage:>56} <<<")
        print(f"{'='*60}")

def _print_progress(current: int, total: int, verbose: bool):
    """以一致格式打印进度。"""
    if verbose and (current % 50 == 0 or current == total):
        print(f"  Processed {current}/{total} bins...")

def _print_footer(mps: MPSState, verbose: bool, stage: str = ""):
    """以一致格式打印阶段尾部。"""
    if verbose:
        print(f"  Final chi: {mps.get_bond_dimensions()}")
        print(f"{stage} complete.")


def apply_fiber_channel(
    mps: MPSState,
    n_bins: int,
    fiber_params,
    rng: np.random.Generator,
    verbose: bool = True,
    bin_start: Optional[int] = None,
) -> tuple:
    """
    应用光纤信道效应：琼斯旋转 + 相位漂移（含随机采样）。

    这一步同时处理琼斯旋转与损耗，并从 FiberChannelParams
    为每次轨迹采样参数（模拟光纤漂移）。
    仅支持6D（1517-only）bin空间。

    Parameters
    ----------
    mps : MPSState
        MPS态（布局：atomA, atomB, A1, B1, A2, B2, ..., AN, BN）
    n_bins : int
        时间仓数量
    fiber_params : FiberChannelParams
        光纤信道参数（将采样新的琼斯矩阵和eta）
    rng : np.random.Generator
        随机数生成器
    verbose : bool
        是否打印进度
    bin_start : Optional[int]
        bin起始索引（默认自动推断：若前两个站点为原子则为2，否则为0）

    Returns
    -------
    tuple
        (mps, sampled_params) 其中 sampled_params =
        (U_A, U_B, eta_H_A, eta_V_A, eta_H_B, eta_V_B, phase, phase_slope, phase_jitter_std)
    """
    _print_header("Fiber Channel", verbose)

    # 为本次轨迹采样参数（残余Jones旋转 + 小PDL）
    U_A, U_B, eta_H_A, eta_V_A, eta_H_B, eta_V_B, phase = fiber_params.sample_all(rng)
    phase_slope = fiber_params.sample_phase_slope(rng)
    phase_jitter_std = float(fiber_params.phase_jitter_std)

    if verbose:
        print(f"  Sampled Jones_A:\n{U_A}")
        print(f"  Sampled Jones_B:\n{U_B}")
        print(f"  Phase drift: {phase:.4f} rad")
        print(f"  Sampled eta_A(H/V): {eta_H_A:.4f} / {eta_V_A:.4f}")
        print(f"  Sampled eta_B(H/V): {eta_H_B:.4f} / {eta_V_B:.4f}")
        print(f"  PDL_A (H-V): {eta_H_A - eta_V_A:+.4f}")
        print(f"  PDL_B (H-V): {eta_H_B - eta_V_B:+.4f}")
        if abs(phase_slope) > 0.0 or phase_jitter_std > 0.0:
            print(f"  Phase slope: {phase_slope:+.4e} rad/bin")
            print(f"  Phase jitter std: {phase_jitter_std:.4e} rad")

    if bin_start is None:
        if len(mps.d) >= 2 and mps.d[0] == DIM_ATOM and mps.d[1] == DIM_ATOM:
            bin_start = 2
        else:
            bin_start = 0

    if bin_start >= mps.L:
        raise ValueError(f"bin_start={bin_start} 超出MPS长度 {mps.L}")

    bin_dim = mps.d[bin_start]
    if bin_dim not in (DIM_BIN, DIM_1517):
        raise ValueError(f"Unsupported bin dimension: {bin_dim}. Expected 18 or 6.")

    def _to_tuple(U: np.ndarray) -> Tuple[Tuple[complex, complex], Tuple[complex, complex]]:
        return (
            (complex(U[0, 0]), complex(U[0, 1])),
            (complex(U[1, 0]), complex(U[1, 1])),
        )

    # 应用琼斯旋转（仅作用于1517子空间）
    U_J_1517_A = jones_gate(_to_tuple(U_A))
    U_J_1517_B = jones_gate(_to_tuple(U_B))
    if bin_dim == DIM_BIN:
        I_780 = np.eye(DIM_780, dtype=complex)
        U_J_A = np.kron(I_780, U_J_1517_A)
        U_J_B = np.kron(I_780, U_J_1517_B)
    else:
        U_J_A = U_J_1517_A
        U_J_B = U_J_1517_B

    phase_center = 0.5 * (n_bins - 1)
    apply_phase_profile = abs(phase_slope) > 0.0 or phase_jitter_std > 0.0

    for n in range(n_bins):
        site_A = bin_start + 2 * n
        site_B = bin_start + 2 * n + 1

        # 先应用琼斯旋转
        mps.apply_one_site_gate(site_A, U_J_A)
        mps.apply_one_site_gate(site_B, U_J_B)

        if apply_phase_profile:
            phase_n = phase_slope * (n - phase_center)
            if phase_jitter_std > 0.0:
                phase_n += rng.normal(0.0, phase_jitter_std)
            if abs(phase_n) > 0.0:
                phase_factor = np.exp(1j * phase_n)
                U_phase_1517 = jones_gate(((phase_factor, 0.0), (0.0, phase_factor)))
                if bin_dim == DIM_BIN:
                    U_phase = np.kron(I_780, U_phase_1517)
                else:
                    U_phase = U_phase_1517
                # 只对B臂施加时间相关相位，形成相对失配
                mps.apply_one_site_gate(site_B, U_phase)

        _print_progress(n + 1, n_bins, verbose)

    _print_footer(mps, verbose, stage="Fiber Channel")
    return mps, (U_A, U_B, eta_H_A, eta_V_A, eta_H_B, eta_V_B, phase, phase_slope, phase_jitter_std)


# ============================================================================
# 发射门辅助函数（文档26：H_sys + Lμ）
# ============================================================================

def _omega_gaussian(t_ns: float, t0_ns: float, sigma_ns: float, omega_peak: float) -> complex:
    if sigma_ns <= 0.0:
        return np.zeros_like(t_ns, dtype=complex)
    return complex(omega_peak) * np.exp(-0.5 * ((t_ns - t0_ns) / sigma_ns) ** 2)

def _effective_gamma_per_channel(
    g: float,
    kappa_ex: float,
    kappa_in: float,
    eps: float = 1e-30,
) -> float:
    """
    在坏腔近似下估计原子到外耦合通道的有效耦合率（单通道）。
    L_eff ≈ sqrt(2*kappa_ex) * g / kappa，因此 gamma ≈ |L_eff|^2。
    """
    kappa = kappa_ex + kappa_in
    if kappa <= eps or kappa_ex <= 0.0:
        return 0.0
    return 2.0 * kappa_ex * (g / kappa) ** 2


def _build_h_sys(omega: complex, delta_u: float, delta_e: float) -> np.ndarray:
    """
    构造单原子 H_sys（基顺序：|0>, |1>, |e>, |u>）。
    """
    h_sys = np.zeros((DIM_ATOM, DIM_ATOM), dtype=complex)
    h_sys[2, 2] = delta_e
    h_sys[3, 3] = delta_u
    h_sys[2, 3] = omega
    h_sys[3, 2] = np.conj(omega)
    return h_sys





# ============================================================================
# 高层便捷函数
# ============================================================================

def run_dual_atom_emission(
    n_bins: int = 200,
    dt_ns: float = 0.2,
    chi_max: int = 50,
    Alpha_A: Optional[np.ndarray] = None,
    Alpha_B: Optional[np.ndarray] = None,
    gamma_peak_A: float = 2 * np.pi * 20e6,
    gamma_peak_B: float = 2 * np.pi * 20e6,
    t0_A: Optional[float] = None,
    t0_B: Optional[float] = None,
    sigma: float = 12.0,
    delay_ns: float = 0.0,
    delay_jitter_ns: float = 0.0,
    g: float = 2 * np.pi * 20e6,
    kappa_ex: float = 2 * np.pi * 20e6,
    kappa_in: float = 2 * np.pi * 1e6,
    gamma_atom: float = 2 * np.pi * 3e6,
    delta_u: float = 0.0,
    delta_e: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
) -> EmissionResult:
    """
    运行双原子发射仿真（原子向左移动方案）。
    发射前的链布局：A1, B1, A2, B2, ..., AN, BN, atomA, atomB
    需要先交换为：A1, B1, A2, B2, ..., AN, atomA, BN, atomB
    然后通过和atomA和atomB逐渐地和门进行的作用，作用到头，当作用到最后一个原子时候布局差不多是：A1, atomA, B1, atomB, A2, B2, ..., AN, BN
    拿发射矩阵作用完最后的原子了以后，再像往常顺序那样同时移动一下atomA和atomB：atomA，A1, atomB, B1, A2, B2, ..., AN, BN
    最后移动一下atomB，就得到了发射后的链布局：atomA, atomB, A1, B1, A2, B2, ..., AN, BN
    原子从链的右端开始，每个时间步与最左边的仓相互作用后向左移动一步。
    最终布局：atomA, atomB, A1, B1, A2, B2, ..., AN, BN

    Parameters
    ----------
    n_bins : int
        时间仓数量
    dt_ns : float
        时间步长（纳秒）
    chi_max : int
        MPS最大键维度
    Alpha_A : np.ndarray, optional
        原子A的2x2偏振矩阵
    Alpha_B : np.ndarray, optional
        原子B的2x2偏振矩阵
    gamma_peak_A : float
        原子A的驱动脉冲峰值幅度（Ω峰值，rad/s）
    gamma_peak_B : float
        原子B的驱动脉冲峰值幅度（Ω峰值，rad/s）
    t0_A : float, optional
        原子A的峰值时间（纳秒）
    t0_B : float, optional
        原子B的峰值时间（纳秒）
    sigma : float
        驱动脉冲的高斯宽度参数（纳秒）
    delay_ns : float
        原子B相对于A的时间延迟（纳秒）
        正值表示B的高斯峰晚于A，负值表示B早于A
        注意：这是时间延迟，不改变bin索引
    delay_jitter_ns : float
        延迟随机抖动范围（纳秒，均匀分布的半宽）
    g : float
        原子-腔耦合强度（rad/s）
    kappa_ex : float
        腔外耦合衰减率（rad/s）
    kappa_in : float
        腔内损耗衰减率（rad/s）
    gamma_atom : float
        原子极化衰减率（rad/s）
    delta_u : float
        |u> 态失谐（rad/s）
    delta_e : float
        |e> 态失谐（rad/s）
    rng : np.random.Generator, optional
        随机数生成器（用于延迟抖动）
    verbose : bool
        是否打印进度信息

    Returns
    -------
    EmissionResult
        仿真结果容器
    """
    if verbose:
        print("=" * 70)
        print("双原子发射仿真（原子向左移动方案）")
        print("=" * 70)

    # 时间参数
    dt_s = dt_ns * 1e-9
    t_sec = np.arange(n_bins) * dt_s
    t_ns = t_sec * 1e9

    # 设置默认峰值为bin中心
    # 时间数组是 [0, dt, 2*dt, ..., (N-1)*dt]
    # 中心bin索引是 (N-1)/2，对应时间是 (N-1)*dt/2
    if t0_A is None:
        t0_A = (n_bins - 1) * dt_ns / 2
    if t0_B is None:
        t0_B = (n_bins - 1) * dt_ns / 2

    # 延迟抖动（一次采样，作用于整个波包）
    delay_jitter_actual_ns = 0.0
    delay_ns_used = delay_ns
    if delay_jitter_ns > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        delay_jitter_actual_ns = rng.uniform(-delay_jitter_ns, delay_jitter_ns)
        delay_ns_used = delay_ns + delay_jitter_actual_ns

    # 应用时间延迟到B的峰值时间
    t0_B = t0_B + delay_ns_used

    omega_A_values = _omega_gaussian(t_ns, t0_A, sigma, gamma_peak_A)
    omega_B_values = _omega_gaussian(t_ns, t0_B, sigma, gamma_peak_B)
    gamma_per_channel = _effective_gamma_per_channel(g, kappa_ex, kappa_in)
    p_source_A = 0.0
    p_source_B = 0.0

    # 设置默认Alpha矩阵
    if Alpha_A is None:
        Alpha_A = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    if Alpha_B is None:
        Alpha_B = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)

    if verbose:
        print("\n时间网格:")
        print(f"  N_bins = {n_bins}")
        print(f"  dt = {dt_ns} ns")
        print(f"  总时间 = {n_bins * dt_ns} ns")
        print("\n发射参数:")
        print(f"  原子A: Omega_peak={gamma_peak_A:.3e} rad/s, t0={t0_A:.1f} ns, sigma={sigma:.1f} ns")
        print(f"  原子B: Omega_peak={gamma_peak_B:.3e} rad/s, t0={t0_B:.1f} ns, sigma={sigma:.1f} ns")
        print(f"  g={g:.3e} rad/s, kappa_ex={kappa_ex:.3e} rad/s, kappa_in={kappa_in:.3e} rad/s")
        print(f"  gamma_atom={gamma_atom:.3e} rad/s（暂未显式引入）, delta_u={delta_u:.3e}, delta_e={delta_e:.3e}")
        print(f"  有效耦合速率: gamma_eff={gamma_per_channel:.3e} 1/s (单通道)")
        if delay_jitter_ns > 0.0:
            print(
                f"  时间延迟: base={delay_ns:.1f} ns, "
                f"jitter_range=+/-{delay_jitter_ns:.1f} ns, "
                f"jitter={delay_jitter_actual_ns:.2f} ns, "
                f"used={delay_ns_used:.2f} ns"
            )
        else:
            print(f"  时间延迟: delay_ns={delay_ns:.1f} ns")

    # ========================================================================
    # 初始化 MPS: 交错布局 A1, B1, A2, B2, ..., AN, BN, atomA, atomB
    # ========================================================================
    # 布局：A1(0) - B1(1) - A2(2) - B2(3) - ... - AN(2n-2) - BN(2n-1) - atomA(2n) - atomB(2n+1)
    # 所有仓初始为真空态，原子在激发态
    local_dims = []
    init_state = []

    # 交错添加 A 和 B 仓
    for i in range(n_bins):
        local_dims.append(DIM_BIN)  # A_i
        local_dims.append(DIM_BIN)  # B_i
        init_state.append(0)  # A_i 真空
        init_state.append(0)  # B_i 真空

    # 添加两个原子
    local_dims.append(DIM_ATOM)  # atomA
    local_dims.append(DIM_ATOM)  # atomB
    init_state.append(3)  # atomA 在 |u>
    init_state.append(3)  # atomB 在 |u>

    mps = MPSState(local_dims=local_dims, init_state=init_state, max_bond=chi_max)

    if verbose:
        print("\nMPS 初始化:")
        print(f"  链长度 L = {mps.L}")
        print(f"  布局: A1(0) - B1(1) - A2(2) - B2(3) - ... - AN({2*n_bins-2}) - BN({2*n_bins-1}) - atomA({2*n_bins}) - atomB({2*n_bins+1})")
        print("  初始态: 两原子在 |u>, 所有仓在 |vac>")
        print(f"  max_bond = {chi_max}")

    # ========================================================================
    # 预处理：交换 BN 和 atomA
    # ========================================================================
    # 初始布局：A1(0) - B1(1) - ... - AN(2n-2) - BN(2n-1) - atomA(2n) - atomB(2n+1)
    # 目标布局：A1(0) - B1(1) - ... - AN(2n-2) - atomA(2n-1) - BN(2n) - atomB(2n+1)
    if verbose:
        print(f"\n预处理：交换 BN({2*n_bins-1}) 和 atomA({2*n_bins})...")

    # 使用 swap_sites 方法（自动更新维度）
    mps.swap_sites(2*n_bins-1)

    # 当前原子位置
    site_atomA = 2*n_bins - 1
    site_atomB = 2*n_bins + 1

    if verbose:
        print(f"  预处理后布局: A1(0) - B1(1) - ... - AN({2*n_bins-2}) - atomA({site_atomA}) - BN({2*n_bins}) - atomB({site_atomB})")
        print(f"  维度验证: d[{site_atomA}]={mps.d[site_atomA]} (应为{DIM_ATOM}), d[{2*n_bins}]={mps.d[2*n_bins]} (应为{DIM_BIN})")

    # ========================================================================
    # 发射循环：原子从右向左移动，依次与仓发射
    # ========================================================================
    if verbose:
        print("\n开始发射循环（原子向左移动方案）...")

    # 用于记录每个仓的发射概率
    per_bin_prob_A = np.zeros(n_bins)
    per_bin_prob_B = np.zeros(n_bins)

    # 用于记录原子状态演化
    atom_A_evolution = []
    atom_B_evolution = []

    def _bin_to_time_index(bin_index: int) -> int:
        """
        时间索引映射：
        - n_bins-1 是最早发射的 bin
        - 0 是最晚发射的 bin
        """
        return (n_bins - 1) - bin_index

    for n in range(n_bins-1, -1, -1):  # 从 n_bins-1 到 0（空间索引）
        # 物理图景：先发射的光子先到达 QFC/BS，存储在空间上靠后的 bin
        time_idx = _bin_to_time_index(n)
        omega_A_n = omega_A_values[time_idx]
        omega_B_n = omega_B_values[time_idx]
        h_sys_A = _build_h_sys(omega_A_n, delta_u, delta_e)
        h_sys_B = _build_h_sys(omega_B_n, delta_u, delta_e)
        gamma_A_n = gamma_per_channel
        gamma_B_n = gamma_per_channel

        # 当前目标仓的位置
        site_A_n = 2 * n      # A_n 在位置 2n
        site_B_n = 2 * n + 1  # B_n 在位置 2n+1

        # 记录发射前的原子状态
        rho_A = mps.get_reduced_density([site_atomA])
        rho_B = mps.get_reduced_density([site_atomB])
        atom_A_evolution.append(np.diag(rho_A).real)
        atom_B_evolution.append(np.diag(rho_B).real)

        # ====================================================================
        # 步骤1：原子A与左边的A仓发射
        # ====================================================================
        if gamma_A_n > 1e-12 or abs(omega_A_n) > 1e-12 or delta_u != 0.0 or delta_e != 0.0:
            # atomA 应该在 site_A_n 的右边（site_A_n + 1）
            # 但实际位置是 site_atomA，所以发射门作用在 bond(site_atomA-1, site_atomA)
            # 其中 site_atomA-1 应该是 A_n 仓
            U_emit_A = emission_gate(
                gamma=gamma_A_n,
                dt=dt_ns * 1e-9,
                Alpha=Alpha_A,
                phase=0.0,
                H_sys=h_sys_A,
                bin_first=True  # bin × atom（仓在左，原子在右）
            )
            mps.apply_bond_op(site_atomA - 1, U_emit_A)

        # ====================================================================
        # 步骤2：原子B与左边的B仓发射
        # ====================================================================
        if gamma_B_n > 1e-12 or abs(omega_B_n) > 1e-12 or delta_u != 0.0 or delta_e != 0.0:
            # atomB 应该在 site_B_n 的右边
            # 但实际位置是 site_atomB，所以发射门作用在 bond(site_atomB-1, site_atomB)
            U_emit_B = emission_gate(
                gamma=gamma_B_n,
                dt=dt_ns * 1e-9,
                Alpha=Alpha_B,
                phase=0.0,
                H_sys=h_sys_B,
                bin_first=True  # bin × atom
            )
            mps.apply_bond_op(site_atomB - 1, U_emit_B)

        # ====================================================================
        # 步骤3：移动原子（使用 swap_sites 方法）
        # ====================================================================
        if n > 0:  # 不是最后一次循环
            # atomA 向左移动2步
            mps.swap_sites(site_atomA - 1)
            site_atomA -= 1
            mps.swap_sites(site_atomA - 1)
            site_atomA -= 1

            # atomB 向左移动2步
            mps.swap_sites(site_atomB - 1)
            site_atomB -= 1
            mps.swap_sites(site_atomB - 1)
            site_atomB -= 1

        elif n == 0:  # 最后一次循环，特殊处理
            # atomA 向左移动1步
            mps.swap_sites(site_atomA - 1)
            site_atomA -= 1

            # atomB 向左移动2步
            mps.swap_sites(site_atomB - 1)
            site_atomB -= 1
            mps.swap_sites(site_atomB - 1)
            site_atomB -= 1

        # 记录原子状态（每个仓都记录）
        rho_A = mps.get_reduced_density([site_atomA])
        rho_B = mps.get_reduced_density([site_atomB])
        atom_A_evolution.append(np.diag(rho_A).real)
        atom_B_evolution.append(np.diag(rho_B).real)

        # 打印进度
        if verbose and (n % 10 == 0 or n == 0):
            chi = mps.get_bond_dimensions()
            print(
                f"  仓 {n+1:3d}/{n_bins}: "
                f"|Omega_A|={abs(omega_A_n) * 1e-9:.4f}/ns, "
                f"|Omega_B|={abs(omega_B_n) * 1e-9:.4f}/ns, "
                f"gamma_eff={gamma_per_channel * 1e-9:.4f}/ns, "
                f"atomA@{site_atomA}, atomB@{site_atomB}, chi_max={max(chi)}"
            )

    if verbose:
        print("\n发射完成!")
        print(f"  原子位置: atomA@{site_atomA}, atomB@{site_atomB}")
        print(f"  最终键维度: max={max(mps.get_bond_dimensions())}")
        print(f"  最终态归一化: {mps.norm():.6f}")

    # ========================================================================
    # 计算每个仓的发射概率
    # ========================================================================
    if verbose:
        print("\n计算每个仓的发射概率...")

    # 最终布局：atomA(0) - atomB(1) - A1(2) - B1(3) - A2(4) - B2(5) - ... - AN(2n) - BN(2n+1)
    for n in range(n_bins):
        site_A_n = 2 + 2 * n      # A_n 在位置 2 + 2n
        site_B_n = 2 + 2 * n + 1  # B_n 在位置 2 + 2n + 1

        # 计算非真空概率
        rho_A_n = mps.get_reduced_density([site_A_n])
        per_bin_prob_A[n] = 1.0 - rho_A_n[0, 0].real

        rho_B_n = mps.get_reduced_density([site_B_n])
        per_bin_prob_B[n] = 1.0 - rho_B_n[0, 0].real

    # ========================================================================
    # 获取最终原子状态
    # ========================================================================
    rho_A_final = mps.get_reduced_density([site_atomA])  # atomA在位置0
    rho_B_final = mps.get_reduced_density([site_atomB])  # atomB在位置1

    atom_states = {
        'A': rho_A_final,
        'B': rho_B_final
    }

    if verbose:
        print("\n最终原子状态:")
        print(
            "  原子A: "
            f"P(|0>)={rho_A_final[0,0].real:.4f}, "
            f"P(|1>)={rho_A_final[1,1].real:.4f}, "
            f"P(|e>)={rho_A_final[2,2].real:.4f}, "
            f"P(|u>)={rho_A_final[3,3].real:.4f}"
        )
        print(
            "  原子B: "
            f"P(|0>)={rho_B_final[0,0].real:.4f}, "
            f"P(|1>)={rho_B_final[1,1].real:.4f}, "
            f"P(|e>)={rho_B_final[2,2].real:.4f}, "
            f"P(|u>)={rho_B_final[3,3].real:.4f}"
        )
    p_source_A = float(per_bin_prob_A.sum())
    p_source_B = float(per_bin_prob_B.sum())

    if verbose:
        print("\n总发射概率:")
        print(f"  A臂: {p_source_A:.4f}")
        print(f"  B臂: {p_source_B:.4f}")

    # ========================================================================
    # 构造返回结果
    # ========================================================================
    atom_A_evolution_array = np.array(atom_A_evolution).T if atom_A_evolution else np.zeros((4, 1))
    atom_B_evolution_array = np.array(atom_B_evolution).T if atom_B_evolution else np.zeros((4, 1))

    result = EmissionResult(
        mps=mps,
        dt_s=dt_s,
        per_bin_prob_A=per_bin_prob_A,
        per_bin_prob_B=per_bin_prob_B,
        atom_states=atom_states,
        atom_A_state_evolution=atom_A_evolution_array,
        atom_B_state_evolution=atom_B_evolution_array,
        delay_ns_base=delay_ns,
        delay_jitter_ns=delay_jitter_ns,
        delay_jitter_actual_ns=delay_jitter_actual_ns,
        delay_ns_used=delay_ns_used,
        p_source_A=p_source_A,
        p_source_B=p_source_B,
    )

    if verbose:
        print("=" * 70)

    return result
