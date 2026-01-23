# -*- coding: utf-8 -*-
"""
时间离散化参数。

从旧 config.py 中拆出，避免保留未使用的配置类。
"""

from dataclasses import dataclass
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
        时间仓数量
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
