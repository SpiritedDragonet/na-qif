# -*- coding: utf-8 -*-
"""
实验流程中的通用配置与工具函数。
"""

from typing import Optional, Tuple
import numpy as np

from ..physics import FiberChannelParams

# 探测噪声默认参数（可用CLI覆盖）
DEFAULT_DARK_RATE_INTRINSIC_HZ = 65.0
DEFAULT_BG_RATE_MEAN_HZ = 165.0
DEFAULT_BG_RATE_STD_HZ = float(np.sqrt(5.0))

# 原子态极端判据
ATOM_EXTREME_EPS = 1e-2


def _get_emission_params(delay_ns: float) -> dict:
    # 发射参数集中管理，避免多处硬编码
    return {
        "n_bins": 100,
        "dt_ns": 0.5,
        "chi_max": 50,
        "gamma_peak_A": 0.5,
        "gamma_peak_B": 0.5,
        "sigma": 10.0,
        "delay_ns": delay_ns,
        "delay_jitter_ns": 0.5,
    }


def _make_fiber_params() -> FiberChannelParams:
    # 残余Jones旋转 + 小PDL + 相位斜率噪声
    compensation_sigma = 0.1  # 补偿后的残差旋转（弧度，可调）
    pdl_sigma = 0.02  # 小PDL：H/V透过率相对差异的标准差（线性）
    phase_slope_std = 0.05  # 相位斜率标准差（rad/bin）
    return FiberChannelParams(
        polarization_model="perturb",
        polarization_sigma=compensation_sigma,
        pdl_sigma=pdl_sigma,
        phase_slope_std=phase_slope_std,
    )


def _compute_window_bins(window_ns: float, bin_dt_ns: float) -> int:
    if bin_dt_ns <= 0:
        return 0
    return int(round(window_ns / bin_dt_ns))


def _compute_noise_params(
    noise_cfg: Optional[dict],
    bin_dt_s: float,
    rng: np.random.Generator,
) -> dict:
    if noise_cfg is None:
        noise_cfg = {}
    dark_rate_intrinsic_hz = max(
        0.0, float(noise_cfg.get("dark_rate_intrinsic_hz", DEFAULT_DARK_RATE_INTRINSIC_HZ))
    )
    bg_rate_mean_hz = max(
        0.0, float(noise_cfg.get("bg_rate_mean_hz", DEFAULT_BG_RATE_MEAN_HZ))
    )
    bg_rate_std_hz = max(
        0.0, float(noise_cfg.get("bg_rate_std_hz", DEFAULT_BG_RATE_STD_HZ))
    )
    dark_rate_bg_hz = max(0.0, rng.normal(bg_rate_mean_hz, bg_rate_std_hz))
    p_dark_intrinsic = 1.0 - np.exp(-dark_rate_intrinsic_hz * bin_dt_s)
    p_bg = 1.0 - np.exp(-dark_rate_bg_hz * bin_dt_s)
    p_noise = 1.0 - (1.0 - p_dark_intrinsic) * (1.0 - p_bg)
    p_noise = min(max(p_noise, 0.0), 1.0)
    return {
        "dark_rate_intrinsic_hz": dark_rate_intrinsic_hz,
        "bg_rate_mean_hz": bg_rate_mean_hz,
        "bg_rate_std_hz": bg_rate_std_hz,
        "dark_rate_bg_hz": dark_rate_bg_hz,
        "p_dark_intrinsic": p_dark_intrinsic,
        "p_bg": p_bg,
        "p_noise": p_noise,
    }


def _atom_extreme_state(mps: "MPSState", eps: float = ATOM_EXTREME_EPS) -> tuple:
    rho_A = mps.get_reduced_density([0])
    rho_B = mps.get_reduced_density([1])

    pA0 = float(np.real(rho_A[0, 0]))
    pA1 = float(np.real(rho_A[1, 1]))
    pAe = float(np.real(rho_A[2, 2]))
    pB0 = float(np.real(rho_B[0, 0]))
    pB1 = float(np.real(rho_B[1, 1]))
    pBe = float(np.real(rho_B[2, 2]))

    extreme_A = (pA0 < eps or pA1 < eps or pA0 > 1.0 - eps or pA1 > 1.0 - eps)
    extreme_B = (pB0 < eps or pB1 < eps or pB0 > 1.0 - eps or pB1 > 1.0 - eps)
    return (extreme_A or extreme_B), (pA0, pA1, pAe, pB0, pB1, pBe)


def _apply_atomic_dephasing(
    mps,
    p_dephase: float,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
) -> None:
    """
    对双原子施加纯退相干通道（Z退相干）。
    """
    if p_dephase <= 0.0:
        if verbose:
            print("原子退相干：p_dephase=0，跳过。")
        return

    p_dephase = min(max(p_dephase, 0.0), 1.0)
    if rng is None:
        rng = np.random.default_rng()

    K0 = np.sqrt(1.0 - p_dephase) * np.eye(3, dtype=complex)
    Z = np.diag([1.0, -1.0, 1.0]).astype(complex)
    K1 = np.sqrt(p_dephase) * Z
    kraus_list = [K0, K1]

    # 原子位于链最左端：atomA(0), atomB(1)
    for site in (0, 1):
        mps.apply_kraus_one_site(site, kraus_list, rng=rng)

    if verbose:
        print(f"原子退相干：已应用 p_dephase={p_dephase:.4e}")
