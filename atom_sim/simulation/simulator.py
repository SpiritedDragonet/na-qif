# -*- coding: utf-8 -*-
"""
多轨迹统计模块

本模块提供运行多条轨迹并计算成功概率、保真度等统计估计的函数。
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from scipy.stats import sem

from .trajectory import TrajectoryRunner, TrajectoryResult, run_single_trajectory
from ..config import TimeGrid, EmitParams, QFCParams, FiberParams, DetParams


@dataclass
class SimulationResult:
    """
    运行多条轨迹的结果。

    Attributes
    ----------
    p_succ : float
        估计的成功概率
    p_succ_stderr : float
        成功概率的标准误差
    rho_cond : np.ndarray
        条件原子密度矩阵（在成功轨迹上平均）
    F_cond : float
        与目标贝尔态的条件保真度
    F_cond_stderr : float
        保真度估计的标准误差
    n_succ : int
        成功轨迹的数量
    n_traj : int
        运行的轨迹总数
    """
    p_succ: float
    p_succ_stderr: float
    rho_cond: np.ndarray
    F_cond: float
    F_cond_stderr: float
    n_succ: int
    n_traj: int


def run_simulation(
    time_grid: TimeGrid,
    emit_params: EmitParams,
    qfc_params: QFCParams,
    fiber_params: FiberParams,
    det_params: DetParams,
    n_traj: int = 1000,
    chi_max: int = 100,
    seed: Optional[int] = None,
    target_bell: Optional[np.ndarray] = None,
) -> SimulationResult:
    """
    运行多条轨迹并计算统计数据。

    Parameters
    ----------
    time_grid : TimeGrid
        时间离散化
    emit_params : EmitParams
        发射参数
    qfc_params : QFCParams
        QFC参数
    fiber_params : FiberParams
        光纤/琼斯/PMD参数
    det_params : DetParams
        探测参数
    n_traj : int
        要运行的轨迹数量
    chi_max : int
        MPS的最大键维度
    seed : int, optional
        用于可重复性的随机种子
    target_bell : np.ndarray, optional
        用于保真度计算的目标贝尔态（9x9矩阵）。
        若为None，使用 |Phi+> = (|00> + |11>) / sqrt(2)

    Returns
    -------
    SimulationResult
        统计结果，包括 p_succ ± stderr, F_cond ± stderr
    """
    # 设置默认目标贝尔态：|Phi+> = (|00> + |11>) / sqrt(2)
    # 在每个原子的 |0>, |1>, |e> 基下：
    # |00> 表示原子A处于 |0>，原子B处于 |0>
    # |11> 表���原子A处于 |1>，原子B处于 |1>
    if target_bell is None:
        target_bell = np.zeros((9, 9), dtype=complex)
        # 基顺序：|0_A0_B>, |0_A1_B>, |0_Ae_B>, |1_A0_B>, ...
        # |00> 位于索引 0*3 + 0 = 0
        # |11> 位于索引 1*3 + 1 = 4
        target_bell[0, 0] = 0.5
        target_bell[4, 4] = 0.5
        target_bell[0, 4] = 0.5
        target_bell[4, 0] = 0.5

    # 追踪结果
    success_count = 0
    rho_success_list = []
    fidelity_list = []

    # 运行轨迹
    for i in range(n_traj):
        # 为每条轨迹使用不同的种子
        traj_seed = None if seed is None else seed + i

        result = run_single_trajectory(
            time_grid=time_grid,
            emit_params=emit_params,
            qfc_params=qfc_params,
            fiber_params=fiber_params,
            det_params=det_params,
            chi_max=chi_max,
            seed=traj_seed,
        )

        if result.success:
            success_count += 1
            rho_success_list.append(result.rho_atom)

            # 计算保真度
            fidelity = np.real(np.vdot(target_bell.flatten(),
                                       result.rho_atom @ target_bell.flatten()))
            fidelity_list.append(fidelity)

    # 计算统计数据
    p_succ = success_count / n_traj
    p_succ_stderr = np.sqrt(p_succ * (1 - p_succ) / n_traj)

    if success_count > 0:
        rho_cond = np.mean(rho_success_list, axis=0)
        F_cond = np.mean(fidelity_list)
        F_cond_stderr = sem(fidelity_list) if len(fidelity_list) > 1 else 0.0
    else:
        rho_cond = np.zeros((9, 9), dtype=complex)
        F_cond = 0.0
        F_cond_stderr = 0.0

    return SimulationResult(
        p_succ=p_succ,
        p_succ_stderr=p_succ_stderr,
        rho_cond=rho_cond,
        F_cond=F_cond,
        F_cond_stderr=F_cond_stderr,
        n_succ=success_count,
        n_traj=n_traj,
    )


__all__ = [
    'SimulationResult',
    'run_simulation',
]
