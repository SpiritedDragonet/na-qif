# -*- coding: utf-8 -*-
"""
单轨迹执行模块

本模块实现时间仓仿真的"传送带"主循环。
每个时间仓按顺序处理：发射（态端），其余光学链路参数采样后推入测量端。
"""

from typing import Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

from ..core.mps import MPSState
from ..hilbert.basis import BIN_SPACE
from ..physics import kraus_from_collapse_ops
from ..physics.gates import emission_gate, build_emitter_operators_12d


# 维度常量，便于代码阅读
DIM_ATOM_LOGICAL = 4
DIM_CAVITY = 3
DIM_EMITTER = DIM_ATOM_LOGICAL * DIM_CAVITY  # 12
DIM_BIN = BIN_SPACE.dim  # 5


def _extract_atom_density_from_emitter_rho(rho_emitter: np.ndarray) -> np.ndarray:
    """
    从 emitter(12D) 约化密度矩阵中提取原子 4D 边缘态。

    emitter 基序为 atom-major：|atom> ⊗ |cavity>，其中 cavity 维度为 3。
    """
    dim = int(rho_emitter.shape[0])
    if rho_emitter.shape != (dim, dim):
        raise ValueError(f"rho_emitter 需为方阵，得到 {rho_emitter.shape}")
    if dim % DIM_ATOM_LOGICAL != 0:
        raise ValueError(f"emitter 维度 {dim} 无法按原子维度 {DIM_ATOM_LOGICAL} 切分")
    cavity_dim = dim // DIM_ATOM_LOGICAL
    rho_atom = np.zeros((DIM_ATOM_LOGICAL, DIM_ATOM_LOGICAL), dtype=complex)
    for atom_row in range(DIM_ATOM_LOGICAL):
        for atom_col in range(DIM_ATOM_LOGICAL):
            value = 0.0 + 0.0j
            for cavity_index in range(cavity_dim):
                row = atom_row * cavity_dim + cavity_index
                col = atom_col * cavity_dim + cavity_index
                value += rho_emitter[row, col]
            rho_atom[atom_row, atom_col] = value
    return rho_atom


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
    diagnostics_enabled : bool
        是否启用发射诊断提取（逐步原子态+逐bin发射概率）。
        关闭时，per_bin_prob_* 仅保留占位数组（NaN），
        atom_*_state_evolution 为空矩阵 (4x0)。
    """
    mps: MPSState
    dt_s: float
    per_bin_prob_A: np.ndarray
    per_bin_prob_B: np.ndarray
    atom_states: dict
    atom_A_state_evolution: np.ndarray = field(default_factory=lambda: np.zeros((4, 1)))
    atom_B_state_evolution: np.ndarray = field(default_factory=lambda: np.zeros((4, 1)))
    diagnostics_enabled: bool = False

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

# 一致打印格式的辅助函数
def _print_header(stage: str, verbose: bool):
    """以一致格式打印阶段标题。"""
    if verbose:
        print(f"\n{'='*60}")
        print(f"{stage:>56} <<<")
        print(f"{'='*60}")

def _print_footer(mps: MPSState, verbose: bool, stage: str = ""):
    """以一致格式打印阶段尾部。"""
    if verbose:
        print(f"  Final chi: {mps.get_bond_dimensions()}")
        print(f"{stage} complete.")


def sample_fiber_realization(
    mps: MPSState,
    n_bins: int,
    fiber_params,
    rng: np.random.Generator,
    verbose: bool = True,
) -> tuple:
    """
    采样光纤信道参数（琼斯旋转 + 损耗 + 相位漂移）。

    方案B下光纤效应推入 POVM，对态端不显式作用。

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
    Returns
    -------
    tuple
        (mps, sampled_params) 其中 sampled_params =
        (U_A, U_B, eta_H_A, eta_V_A, eta_H_B, eta_V_B, phase, phase_slope, phase_jitter_std)
    """
    # ------------------------------------------------------------------
    # 方案B：仅采样参数，不对态端施加任何门/损耗。
    # ------------------------------------------------------------------
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

    _print_footer(mps, verbose, stage="Fiber Channel")

    return mps, (U_A, U_B, eta_H_A, eta_V_A, eta_H_B, eta_V_B, phase, phase_slope, phase_jitter_std)


# ============================================================================
# 发射门辅助函数（文档26：H_sys + Lμ）
# ============================================================================

def _omega_envelope(
    t_ns,
    t0_ns: float,
    sigma_ns: float,
    omega_peak: float,
    waveform: str = "gaussian",
):
    """驱动场复 Rabi 频率包络 Ω(t)，支持多种波形。"""
    if sigma_ns <= 0.0:
        return np.zeros_like(t_ns, dtype=complex)

    x = (t_ns - t0_ns) / sigma_ns
    waveform_norm = str(waveform).strip().lower()
    if waveform_norm == "gaussian":
        env = np.exp(-0.5 * x ** 2)
    elif waveform_norm == "sech":
        env = 1.0 / np.cosh(x)
    elif waveform_norm == "square":
        env = (np.abs(x) <= 1.0).astype(float)
    else:
        raise ValueError(f"未知驱动波形: {waveform}")
    return complex(omega_peak) * env

def _build_h_sys(omega: complex, delta_u: float, delta_e: float) -> np.ndarray:
    """
    构造单原子 H_sys（基顺序：|0>, |1>, |e>, |u>）。
    """
    # H_sys 只作用在激发态 |e> 与光学辅助态 |u> 子空间，
    # 逻辑基态 |0>, |1> 在这里保持能量 0（旋转参考系下）。
    h_sys = np.zeros((DIM_ATOM_LOGICAL, DIM_ATOM_LOGICAL), dtype=complex)
    # 对角线：失谐项（在旋转系中表现为能级偏移）
    h_sys[2, 2] = delta_e
    h_sys[3, 3] = delta_u
    # 非对角：驱动耦合 Ω |e><u| + h.c.
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
    omega_peak_A: float = 2 * np.pi * 20e6,
    omega_peak_B: float = 2 * np.pi * 20e6,
    drive_waveform_A: str = "gaussian",
    drive_waveform_B: str = "gaussian",
    t0_A: Optional[float] = None,
    t0_B: Optional[float] = None,
    sigma: float = 12.0,
    delay_ns: float = 0.0,
    delay_jitter_ns: float = 0.0,
    g_A: float = 2 * np.pi * 20e6,
    g_B: float = 2 * np.pi * 20e6,
    kappa_ex_A: float = 2 * np.pi * 20e6,
    kappa_ex_B: float = 2 * np.pi * 20e6,
    kappa_in_A: float = 2 * np.pi * 1e6,
    kappa_in_B: float = 2 * np.pi * 1e6,
    kappa_ex_H_A: Optional[float] = None,
    kappa_ex_V_A: Optional[float] = None,
    kappa_in_H_A: Optional[float] = None,
    kappa_in_V_A: Optional[float] = None,
    kappa_ex_H_B: Optional[float] = None,
    kappa_ex_V_B: Optional[float] = None,
    kappa_in_H_B: Optional[float] = None,
    kappa_in_V_B: Optional[float] = None,
    gamma_sigma_plus_A: float = 0.0,
    gamma_sigma_minus_A: float = 0.0,
    gamma_sigma_plus_B: float = 0.0,
    gamma_sigma_minus_B: float = 0.0,
    delta_u_A: float = 0.0,
    delta_u_B: float = 0.0,
    delta_e_A: float = 0.0,
    delta_e_B: float = 0.0,
    delta_c_H_A: float = 0.0,
    delta_c_V_A: float = 0.0,
    delta_c_H_B: float = 0.0,
    delta_c_V_B: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
    diagnostics: bool = False,
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
    omega_peak_A : float
        原子A的驱动脉冲峰值幅度（Ω峰值，rad/s）
    omega_peak_B : float
        原子B的驱动脉冲峰值幅度（Ω峰值，rad/s）
    t0_A : float, optional
        原子A的峰值时间（纳秒）
    t0_B : float, optional
        原子B的峰值时间（纳秒）
    drive_waveform_A : str
        A 臂驱动包络（gaussian/sech/square）
    drive_waveform_B : str
        B 臂驱动包络（gaussian/sech/square）
    sigma : float
        驱动脉冲的高斯宽度参数（纳秒）
    delay_ns : float
        原子B相对于A的时间延迟（纳秒）
        正值表示B的高斯峰晚于A，负值表示B早于A
        注意：这是时间延迟，不改变bin索引
    delay_jitter_ns : float
        延迟随机抖动范围（纳秒，均匀分布的半宽）
    g_A, g_B : float
        A/B 臂原子-腔耦合强度（rad/s）
    kappa_ex_A, kappa_ex_B : float
        A/B 臂腔外耦合衰减率（rad/s，H/V 公共默认值）
    kappa_in_A, kappa_in_B : float
        A/B 臂腔内损耗衰减率（rad/s，H/V 公共默认值）
    kappa_ex_H_*, kappa_ex_V_*, kappa_in_H_*, kappa_in_V_* : float, optional
        H/V 分偏振通道参数；若未给出则回退到对应公共 kappa_*。
    gamma_sigma_plus_*, gamma_sigma_minus_* : float
        原子自由空间自发辐射通道速率（1/s）
    delta_u_A, delta_u_B : float
        A/B 臂 |u> 态失谐（rad/s）
    delta_e_A, delta_e_B : float
        A/B 臂 |e> 态失谐（rad/s）
    delta_c_H_*, delta_c_V_* : float
        H/V 腔模失谐（rad/s）
    rng : np.random.Generator, optional
        随机数生成器（用于延迟抖动）
    verbose : bool
        是否打印进度信息
    diagnostics : bool
        是否提取发射诊断信息（逐步原子态演化、逐bin发射概率）。
        关闭可显著减少大规模扫描时的收缩开销。

    Returns
    -------
    EmissionResult
        仿真结果容器
    """
    # ------------------------------------------------------------------
    # 物理/数值要点（简版）：
    #   - “碰撞模型”：每个 time-bin 视为一段真空浴，与原子短时相互作用
    #   - 原子沿链移动，使每个 bin 依次与原子相互作用
    #   - 在每一步：对 (atom, bin) 施加 emission_gate（TEBD 局部更新）
    #   - 记录每个 bin 的发射概率与原子态演化
    #
    # 近似：把连续输出场离散化为 N 个 time-bin
    #   dt = Δt，离散化误差由 dt、chi_max 控制
    # ------------------------------------------------------------------
    if verbose:
        print("=" * 70)
        print("双原子发射仿真（原子向左移动方案）")
        print("=" * 70)

    # 时间参数
    dt_s = dt_ns * 1e-9
    # t_sec / t_ns 用于构造高斯脉冲包络
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
        # 延迟抖动：一次采样，作用于整条波包（而非逐 bin）
        delay_jitter_actual_ns = rng.uniform(-delay_jitter_ns, delay_jitter_ns)
        delay_ns_used = delay_ns + delay_jitter_actual_ns

    # tau/delay 的作用口径：只作用在 Ω_B(t) 包络的时间平移上。
    # 即通过平移 B 臂驱动中心 t0_B -> t0_B + delay 实现，而不改 bin 索引。
    # 这与 HOM 的物理定义一致（延迟改变可干涉重叠）。
    # 应用时间延迟到 B 的驱动中心时间
    t0_B = t0_B + delay_ns_used

    omega_A_values = _omega_envelope(t_ns, t0_A, sigma, omega_peak_A, waveform=drive_waveform_A)
    omega_B_values = _omega_envelope(t_ns, t0_B, sigma, omega_peak_B, waveform=drive_waveform_B)

    def _resolve_kappa_polarized(
        kappa_ex: float,
        kappa_in: float,
        kappa_ex_H: Optional[float],
        kappa_ex_V: Optional[float],
        kappa_in_H: Optional[float],
        kappa_in_V: Optional[float],
    ) -> tuple:
        return (
            float(kappa_ex if kappa_ex_H is None else kappa_ex_H),
            float(kappa_ex if kappa_ex_V is None else kappa_ex_V),
            float(kappa_in if kappa_in_H is None else kappa_in_H),
            float(kappa_in if kappa_in_V is None else kappa_in_V),
        )

    kappa_ex_H_A_used, kappa_ex_V_A_used, kappa_in_H_A_used, kappa_in_V_A_used = _resolve_kappa_polarized(
        kappa_ex_A, kappa_in_A, kappa_ex_H_A, kappa_ex_V_A, kappa_in_H_A, kappa_in_V_A
    )
    kappa_ex_H_B_used, kappa_ex_V_B_used, kappa_in_H_B_used, kappa_in_V_B_used = _resolve_kappa_polarized(
        kappa_ex_B, kappa_in_B, kappa_ex_H_B, kappa_ex_V_B, kappa_in_H_B, kappa_in_V_B
    )
    # 设置默认Alpha矩阵
    if Alpha_A is None:
        # Alpha 是 2×2 偏振耦合矩阵（默认单位阵）
        Alpha_A = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    if Alpha_B is None:
        Alpha_B = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)

    if verbose:
        print("\n时间网格:")
        print(f"  N_bins = {n_bins}")
        print(f"  dt = {dt_ns} ns")
        print(f"  总时间 = {n_bins * dt_ns} ns")
        print("\n发射参数:")
        print(
            f"  原子A: Omega_peak={omega_peak_A:.3e} rad/s, "
            f"waveform={drive_waveform_A}, t0={t0_A:.1f} ns, sigma={sigma:.1f} ns"
        )
        print(
            f"  原子B: Omega_peak={omega_peak_B:.3e} rad/s, "
            f"waveform={drive_waveform_B}, t0={t0_B:.1f} ns, sigma={sigma:.1f} ns"
        )
        print(
            f"  A臂: g={g_A:.3e}, kappa_ex={kappa_ex_A:.3e}, kappa_in={kappa_in_A:.3e}, "
            f"delta_u={delta_u_A:.3e}, delta_e={delta_e_A:.3e}"
        )
        print(
            f"      polarized: kappa_ex(H/V)=({kappa_ex_H_A_used:.3e}, {kappa_ex_V_A_used:.3e}), "
            f"kappa_in(H/V)=({kappa_in_H_A_used:.3e}, {kappa_in_V_A_used:.3e}), "
            f"gamma_sigma(+/-)=({gamma_sigma_plus_A:.3e}, {gamma_sigma_minus_A:.3e}), "
            f"delta_c(H/V)=({delta_c_H_A:.3e}, {delta_c_V_A:.3e})"
        )
        print(
            f"  B臂: g={g_B:.3e}, kappa_ex={kappa_ex_B:.3e}, kappa_in={kappa_in_B:.3e}, "
            f"delta_u={delta_u_B:.3e}, delta_e={delta_e_B:.3e}"
        )
        print(
            f"      polarized: kappa_ex(H/V)=({kappa_ex_H_B_used:.3e}, {kappa_ex_V_B_used:.3e}), "
            f"kappa_in(H/V)=({kappa_in_H_B_used:.3e}, {kappa_in_V_B_used:.3e}), "
            f"gamma_sigma(+/-)=({gamma_sigma_plus_B:.3e}, {gamma_sigma_minus_B:.3e}), "
            f"delta_c(H/V)=({delta_c_H_B:.3e}, {delta_c_V_B:.3e})"
        )
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
    # 初始化 MPS: 交错布局 A1, B1, A2, B2, ..., AN, BN, emitterA, emitterB
    # ========================================================================
    # 布局：A1(0) - B1(1) - A2(2) - B2(3) - ... - AN(2n-2) - BN(2n-1) - atomA(2n) - atomB(2n+1)
    # 所有仓初始为真空态，emitter 初态为 |u,cav_vac>
    local_dims = []
    init_state = []

    # 交错添加 A 和 B 仓
    for i in range(n_bins):
        local_dims.append(DIM_BIN)  # A_i
        local_dims.append(DIM_BIN)  # B_i
        # 初态：时间仓真空（索引 0）
        init_state.append(0)  # A_i 真空
        init_state.append(0)  # B_i 真空

    # 添加两个 emitter（12D）
    local_dims.append(DIM_EMITTER)  # emitterA
    local_dims.append(DIM_EMITTER)  # emitterB
    # emitter 初态：|u,cav_vac>，atom-major 编码下索引 = 3 * DIM_CAVITY + 0 = 9
    init_state.append(3 * DIM_CAVITY)  # emitterA
    init_state.append(3 * DIM_CAVITY)  # emitterB

    mps = MPSState(local_dims=local_dims, init_state=init_state, max_bond=chi_max)

    if verbose:
        print("\nMPS 初始化:")
        print(f"  链长度 L = {mps.L}")
        print(f"  布局: A1(0) - B1(1) - A2(2) - B2(3) - ... - AN({2*n_bins-2}) - BN({2*n_bins-1}) - atomA({2*n_bins}) - atomB({2*n_bins+1})")
        print("  初始态: 两个 emitter 在 |u,cav_vac>, 所有仓在 |vac>")
        print(f"  max_bond = {chi_max}")

    # ========================================================================
    # 预处理：交换 BN 和 atomA
    # ========================================================================
    # 初始布局：A1(0) - B1(1) - ... - AN(2n-2) - BN(2n-1) - atomA(2n) - atomB(2n+1)
    # 目标布局：A1(0) - B1(1) - ... - AN(2n-2) - atomA(2n-1) - BN(2n) - atomB(2n+1)
    if verbose:
        print(f"\n预处理：交换 BN({2*n_bins-1}) 和 atomA({2*n_bins})...")

    # 使用 swap_sites 方法（自动更新维度）
    # 目的：让 atomA 位于 BN 的左侧，方便“向左移动”发射流程
    mps.swap_sites(2*n_bins-1)

    # 当前原子位置
    site_atomA = 2*n_bins - 1
    site_atomB = 2*n_bins + 1

    if verbose:
        print(f"  预处理后布局: A1(0) - B1(1) - ... - AN({2*n_bins-2}) - atomA({site_atomA}) - BN({2*n_bins}) - atomB({site_atomB})")
        print(f"  维度验证: d[{site_atomA}]={mps.d[site_atomA]} (应为{DIM_EMITTER}), d[{2*n_bins}]={mps.d[2*n_bins]} (应为{DIM_BIN})")

    # ========================================================================
    # 发射循环：原子从右向左移动，依次与仓发射
    # ========================================================================
    if verbose:
        print("\n开始发射循环（原子向左移动方案）...")

    # 用于记录每个仓的发射概率
    per_bin_prob_A = np.full(n_bins, np.nan, dtype=float)
    per_bin_prob_B = np.full(n_bins, np.nan, dtype=float)

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
        # --------------------------------------------------------------
        # 这里用“反向遍历”来对齐物理时间顺序：
        #   - 空间索引 n 大的 bin 对应更早的发射时间
        #   - time_idx = (N-1)-n
        # --------------------------------------------------------------
        # 物理图景：先发射的光子先到达 QFC/BS，存储在空间上靠后的 bin
        time_idx = _bin_to_time_index(n)
        # 从预计算的高斯包络里取出当前时间步的驱动幅度
        omega_A_n = omega_A_values[time_idx]
        omega_B_n = omega_B_values[time_idx]
        h_atom_A = _build_h_sys(omega_A_n, delta_u_A, delta_e_A)
        h_atom_B = _build_h_sys(omega_B_n, delta_u_B, delta_e_B)
        emitter_ops_A = build_emitter_operators_12d(
            alpha=Alpha_A,
            g=g_A,
            h_atom=h_atom_A,
            kappa_ex_H=kappa_ex_H_A_used,
            kappa_ex_V=kappa_ex_V_A_used,
            kappa_in_H=kappa_in_H_A_used,
            kappa_in_V=kappa_in_V_A_used,
            gamma_sigma_plus=gamma_sigma_plus_A,
            gamma_sigma_minus=gamma_sigma_minus_A,
            delta_c_H=delta_c_H_A,
            delta_c_V=delta_c_V_A,
        )
        emitter_ops_B = build_emitter_operators_12d(
            alpha=Alpha_B,
            g=g_B,
            h_atom=h_atom_B,
            kappa_ex_H=kappa_ex_H_B_used,
            kappa_ex_V=kappa_ex_V_B_used,
            kappa_in_H=kappa_in_H_B_used,
            kappa_in_V=kappa_in_V_B_used,
            gamma_sigma_plus=gamma_sigma_plus_B,
            gamma_sigma_minus=gamma_sigma_minus_B,
            delta_c_H=delta_c_H_B,
            delta_c_V=delta_c_V_B,
        )
        h_emitter_A = emitter_ops_A["h_emitter"]
        h_emitter_B = emitter_ops_B["h_emitter"]
        l_out_A_H, l_out_A_V = emitter_ops_A["l_out"]
        l_out_B_H, l_out_B_V = emitter_ops_B["l_out"]
        c_ops_A = emitter_ops_A["collapse_ops"]
        c_ops_B = emitter_ops_B["collapse_ops"]

        if diagnostics:
            # 记录发射前原子状态（用于演化可视化）
            rho_A = mps.get_reduced_density([site_atomA])
            rho_B = mps.get_reduced_density([site_atomB])
            atom_A_evolution.append(np.diag(_extract_atom_density_from_emitter_rho(rho_A)).real)
            atom_B_evolution.append(np.diag(_extract_atom_density_from_emitter_rho(rho_B)).real)

        # ====================================================================
        # 步骤1：原子A与左边的A仓发射
        # ====================================================================
        if np.linalg.norm(l_out_A_H) > 0.0 or np.linalg.norm(l_out_A_V) > 0.0 or np.linalg.norm(h_emitter_A) > 0.0:
            # atomA 应该在 site_A_n 的右边（site_A_n + 1）
            # 但实际位置是 site_atomA，所以发射门作用在 bond(site_atomA-1, site_atomA)
            # 其中 site_atomA-1 应该是 A_n 仓
            U_emit_A = emission_gate(
                dt=dt_ns * 1e-9,
                phase=0.0,
                h_emitter=h_emitter_A,
                l_out_h=l_out_A_H,
                l_out_v=l_out_A_V,
                bin_first=True  # bin × atom（仓在左，原子在右）
            )
            # 作用在 (bin, atom) 相邻键上：bin 在左、atom 在右
            mps.apply_bond_op(site_atomA - 1, U_emit_A)

        # ====================================================================
        # 步骤2：原子B与左边的B仓发射
        # ====================================================================
        if np.linalg.norm(l_out_B_H) > 0.0 or np.linalg.norm(l_out_B_V) > 0.0 or np.linalg.norm(h_emitter_B) > 0.0:
            # atomB 应该在 site_B_n 的右边
            # 但实际位置是 site_atomB，所以发射门作用在 bond(site_atomB-1, site_atomB)
            U_emit_B = emission_gate(
                dt=dt_ns * 1e-9,
                phase=0.0,
                h_emitter=h_emitter_B,
                l_out_h=l_out_B_H,
                l_out_v=l_out_B_V,
                bin_first=True  # bin × atom
            )
            # B 臂：作用在 (B_n, atomB) 的相邻键上
            mps.apply_bond_op(site_atomB - 1, U_emit_B)

        loss_kraus_A = kraus_from_collapse_ops(c_ops_A, dt_s) if c_ops_A else []
        loss_kraus_B = kraus_from_collapse_ops(c_ops_B, dt_s) if c_ops_B else []
        if loss_kraus_A:
            mps.apply_kraus_one_site(site_atomA, loss_kraus_A, rng=rng)
        if loss_kraus_B:
            mps.apply_kraus_one_site(site_atomB, loss_kraus_B, rng=rng)
        if loss_kraus_A or loss_kraus_B:
            mps.canonicalize(renormalize=True)

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

        if diagnostics:
            # 记录发射后原子状态（每个仓都记录）
            rho_A = mps.get_reduced_density([site_atomA])
            rho_B = mps.get_reduced_density([site_atomB])
            atom_A_evolution.append(np.diag(_extract_atom_density_from_emitter_rho(rho_A)).real)
            atom_B_evolution.append(np.diag(_extract_atom_density_from_emitter_rho(rho_B)).real)

        # 打印进度
        if verbose and (n % 10 == 0 or n == 0):
            chi = mps.get_bond_dimensions()
            print(
                f"  仓 {n+1:3d}/{n_bins}: "
                f"|Omega_A|={abs(omega_A_n) * 1e-9:.4f}/ns, "
                f"|Omega_B|={abs(omega_B_n) * 1e-9:.4f}/ns, "
                f"|L_A|={np.linalg.norm(l_out_A_H) + np.linalg.norm(l_out_A_V):.3e}, "
                f"|L_B|={np.linalg.norm(l_out_B_H) + np.linalg.norm(l_out_B_V):.3e}, "
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
    if diagnostics:
        if verbose:
            print("\n计算每个仓的发射概率...")

        # 最终布局：atomA(0) - atomB(1) - A1(2) - B1(3) - A2(4) - B2(5) - ... - AN(2n) - BN(2n+1)
        for n in range(n_bins):
            site_A_n = 2 + 2 * n      # A_n 在位置 2 + 2n
            site_B_n = 2 + 2 * n + 1  # B_n 在位置 2 + 2n + 1

            # 约化密度矩阵的 vacuum 元素 = P(vac)，非真空概率 = 1 - P(vac)
            rho_A_n = mps.get_reduced_density([site_A_n])
            per_bin_prob_A[n] = 1.0 - rho_A_n[0, 0].real

            rho_B_n = mps.get_reduced_density([site_B_n])
            per_bin_prob_B[n] = 1.0 - rho_B_n[0, 0].real
    elif verbose:
        print("\n发射诊断关闭：跳过逐bin发射概率提取。")

    # ========================================================================
    # 获取最终原子状态
    # ========================================================================
    rho_A_final_emitter = mps.get_reduced_density([site_atomA])  # emitterA在位置0
    rho_B_final_emitter = mps.get_reduced_density([site_atomB])  # emitterB在位置1
    rho_A_final = _extract_atom_density_from_emitter_rho(rho_A_final_emitter)
    rho_B_final = _extract_atom_density_from_emitter_rho(rho_B_final_emitter)

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
    # ========================================================================
    # 构造返回结果
    # ========================================================================
    atom_A_evolution_array = np.array(atom_A_evolution).T if atom_A_evolution else np.zeros((4, 0))
    atom_B_evolution_array = np.array(atom_B_evolution).T if atom_B_evolution else np.zeros((4, 0))

    result = EmissionResult(
        mps=mps,
        dt_s=dt_s,
        per_bin_prob_A=per_bin_prob_A,
        per_bin_prob_B=per_bin_prob_B,
        atom_states=atom_states,
        atom_A_state_evolution=atom_A_evolution_array,
        atom_B_state_evolution=atom_B_evolution_array,
        diagnostics_enabled=bool(diagnostics),
    )

    if verbose:
        print("=" * 70)

    return result
