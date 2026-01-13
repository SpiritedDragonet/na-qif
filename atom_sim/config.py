# -*- coding: utf-8 -*-
"""
参数配置类

本模块提供时间仓MPS仿真中使用的所有物理参数的数据类。
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple, List, Optional
import numpy as np


@dataclass
class TimeGrid:
    """
    时间离散化参数。

    Attributes
    ----------
    dt : float
        时间仓宽度（秒）
    N : int
        时间仓的数量
    """
    dt: float
    N: int

    @property
    def t(self) -> np.ndarray:
        """时间仓中心点数组：t[n] = n * dt"""
        return np.arange(self.N) * self.dt

    @property
    def total_time(self) -> float:
        """总时间长度：N * dt"""
        return self.N * self.dt


@dataclass
class EmitParams:
    """
    发射门参数。

    Attributes
    ----------
    gamma_A : float or Callable
        原子A的发射率（常数或时间函数）
    gamma_B : float or Callable
        原子B的发射率（常数或时间函数）
    Alpha_A : np.ndarray
        原子A的2x2偏振映射矩阵
        [[alpha_H+, alpha_H-], [alpha_V+, alpha_V-]]
    Alpha_B : np.ndarray
        原子B的2x2偏振映射矩阵
    phi_A : float
        原子A发射的整体相位
    phi_B : float
        原子B发射的整体相位
    """
    gamma_A: float = 0.1
    gamma_B: float = 0.1
    Alpha_A: np.ndarray = field(default_factory=lambda: np.eye(2))
    Alpha_B: np.ndarray = field(default_factory=lambda: np.eye(2))
    phi_A: float = 0.0
    phi_B: float = 0.0

    def get_gamma_A(self, t: float) -> float:
        """获取时刻t时原子A的发射率"""
        if callable(self.gamma_A):
            return self.gamma_A(t)
        return float(self.gamma_A)

    def get_gamma_B(self, t: float) -> float:
        """获取时刻t时原子B的发射率"""
        if callable(self.gamma_B):
            return self.gamma_B(t)
        return float(self.gamma_B)


@dataclass
class QFCParams:
    """
    量子频率转换参数。

    Attributes
    ----------
    theta_H : float
        H偏振的转换角（sin²(theta) = 转换概率）
    theta_V : float
        V偏振的转换角
    eta_ins_H : float
        H偏振的插入损耗
    eta_ins_V : float
        V偏振的插入损耗
    """
    theta_H: float = 0.0
    theta_V: float = 0.0
    eta_ins_H: float = 1.0
    eta_ins_V: float = 1.0


@dataclass
class FiberParams:
    """
    光纤/光学信道参数。

    Attributes
    ----------
    eta_fiber_A : float
        A臂的光纤透过率
    eta_fiber_B : float
        B臂的光纤透过率
    Jones_A : np.ndarray
        A臂的2x2琼斯矩阵
    Jones_B : np.ndarray
        B臂的2x2琼斯矩阵
    PMD_A : float
        A臂的PMD延迟（秒）
    PMD_B : float
        B臂的PMD延迟（秒）
    Rin_A : np.ndarray
        A臂的PSP输入旋转矩阵
    Rout_A : np.ndarray
        A臂的PSP输出旋转矩阵
    Rin_B : np.ndarray
        B臂的PSP输入旋转矩阵
    Rout_B : np.ndarray
        B臂的PSP输出旋转矩阵
    delta_bins : int
        两臂之间的相对仓延迟（B相对于A）
    """
    eta_fiber_A: float = 1.0
    eta_fiber_B: float = 1.0
    Jones_A: np.ndarray = field(default_factory=lambda: np.eye(2))
    Jones_B: np.ndarray = field(default_factory=lambda: np.eye(2))
    PMD_A: float = 0.0
    PMD_B: float = 0.0
    Rin_A: np.ndarray = field(default_factory=lambda: np.eye(2))
    Rout_A: np.ndarray = field(default_factory=lambda: np.eye(2))
    Rin_B: np.ndarray = field(default_factory=lambda: np.eye(2))
    Rout_B: np.ndarray = field(default_factory=lambda: np.eye(2))
    delta_bins: int = 0


@dataclass
class DetParams:
    """
    探测参数。

    Attributes
    ----------
    eta_det : float
        探测效率
    p_dark : float
        每个探测器每个仓的暗计数概率
    success_patterns : List[Tuple[int, int, int, int]]
        计为成功的探测器点击模式列表。
        每个元组为 (d1_H, d1_V, d2_H, d2_V)
    pattern_to_bell : Dict[Tuple[int, int, int, int], str]
        将每个成功模式映射到其投影的贝尔态。
        值：'phi_plus', 'phi_minus', 'psi_plus', 'psi_minus'
    pattern_to_correction : Dict[Tuple[int, int, int, int], str]
        将每个成功模式映射到所需的泡利校正。
        值：'I', 'X', 'Y', 'Z'

    Examples
    --------
    >>> # 部分BSM：在(1H,2V)或(1V,2H)点击时成功
    >>> params = DetParams(
    ...     success_patterns=[(1,0,0,1), (0,1,1,0)],
    ...     pattern_to_bell={(1,0,0,1): 'psi_minus', (0,1,1,0): 'psi_plus'},
    ...     pattern_to_correction={(1,0,0,1): 'I', (0,1,1,0): 'X'}
    ... )
    """
    eta_det: float = 1.0
    p_dark: float = 0.0
    success_patterns: List[Tuple[int, int, int, int]] = field(default_factory=list)
    pattern_to_bell: Dict[Tuple[int, int, int, int], str] = field(default_factory=dict)
    pattern_to_correction: Dict[Tuple[int, int, int, int], str] = field(default_factory=dict)

    def is_success(self, pattern: Tuple[int, int, int, int]) -> bool:
        """检查探测器模式是否为成功模式"""
        return pattern in self.success_patterns

    def get_bell_state(self, pattern: Tuple[int, int, int, int]) -> Optional[str]:
        """获取成功模式对应的贝尔态"""
        return self.pattern_to_bell.get(pattern)

    def get_correction(self, pattern: Tuple[int, int, int, int]) -> Optional[str]:
        """获取成功模式所需的泡利校正"""
        return self.pattern_to_correction.get(pattern)


@dataclass
class SimParams:
    """
    整体仿真参数。

    Attributes
    ----------
    n_traj : int
        运行的轨迹数量
    chi_max : int
        MPS的最大键维度
    svd_min : float
        SVD截断阈值
    seed : Optional[int]
        用于可重复性的随机种子
    """
    n_traj: int = 1000
    chi_max: int = 100
    svd_min: float = 1e-13
    seed: Optional[int] = None
