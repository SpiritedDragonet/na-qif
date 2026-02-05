# -*- coding: utf-8 -*-
"""
希尔伯特空间定义和张量积构造

本模块提供定义子空间与构造积空间的类和函数。
"""

from typing import Tuple, Optional
from functools import lru_cache
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class SubSpace:
    """
    单个希尔伯特子空间，带有名称和维度。

    Attributes
    ----------
    name : str
        此子空间的标识符（如 '780', '1517', 'atom'）
    dim : int
        此子空间的维度
    basis_labels : Optional[List[str]]
        基态的可选标签（如 ['vac', 'H', 'V']）
    """
    name: str
    dim: int
    basis_labels: Optional[Tuple[str, ...]] = None

    def __post_init__(self):
        if self.basis_labels is not None and len(self.basis_labels) != self.dim:
            raise ValueError(
                f"基标签数量 ({len(self.basis_labels)}) "
                f"必须与维度 ({self.dim}) 匹配"
            )


@dataclass(frozen=True)
class ProductSpace:
    """
    子空间的张量积。

    表示 H = H_1 ⊗ H_2 ⊗ ... ⊗ H_n

    Attributes
    ----------
    subspaces : Tuple[SubSpace, ...]
        张量积中的子空间
    """
    subspaces: Tuple[SubSpace, ...]

    def __post_init__(self):
        if not self.subspaces:
            raise ValueError("ProductSpace必须至少有一个子空间")

    @property
    def dim(self) -> int:
        """积空间的总维度。"""
        # 维度按张量积相乘：dim(H1⊗H2⊗...)=∏ dim(Hi)
        result = 1
        for s in self.subspaces:
            result *= s.dim
        return result

    @property
    def num_factors(self) -> int:
        """积空间中的子空间数量。"""
        return len(self.subspaces)

    def subspace_index(self, name: str) -> int:
        """通过名称获取子空间的索引。"""
        for i, s in enumerate(self.subspaces):
            if s.name == name:
                return i
        raise ValueError(f"未找到子空间 '{name}'")

    def subspace_dims(self) -> Tuple[int, ...]:
        """每个子空间的维度元组。"""
        return tuple(s.dim for s in self.subspaces)

    def subspace_dim(self, name: str) -> int:
        """通过名称获取特定子空间的维度。"""
        return self.subspaces[self.subspace_index(name)].dim


# 本项目的预定义子空间
# 这些与 README 中的物理模型对应

SUBSPACE_780 = SubSpace('780', 3, ('vac', 'H', 'V'))
SUBSPACE_1517 = SubSpace('1517', 6, ('vac', 'H', 'V', '2H', '2V', 'HV'))

# 5D bin 基序：vac, H780, V780, H1517, V1517
SUBSPACE_BIN5 = SubSpace('bin', 5, ('vac', 'H_780', 'V_780', 'H_1517', 'V_1517'))

# Bin格点：5D（单光子截断的 780/1517 账本）
BIN_SPACE = SUBSPACE_BIN5  # 5D

# ----------------------------------------------------------------------
# 约定：
#   - 780 子空间基序：|vac>, |H>, |V>
#   - 1517 子空间基序：|vac>, |H>, |V>, |2H>, |2V>, |HV>
#   - Bin 空间基序：|vac>, |H_780>, |V_780>, |H_1517>, |V_1517>
# 该约定贯穿 gates / channels / detection / visualization。
# ----------------------------------------------------------------------


@lru_cache(maxsize=4)
def proj_3_from_6() -> np.ndarray:
    """6D -> 3D 投影：取 {vac,H,V}。"""
    P = np.zeros((3, 6), dtype=complex)
    P[0, 0] = 1.0
    P[1, 1] = 1.0
    P[2, 2] = 1.0
    return P


@lru_cache(maxsize=4)
def proj_3_from_5() -> np.ndarray:
    """5D -> 3D 投影：取 telecom {vac,H_1517,V_1517}。"""
    P = np.zeros((3, 5), dtype=complex)
    P[0, 0] = 1.0
    P[1, 3] = 1.0
    P[2, 4] = 1.0
    return P


@lru_cache(maxsize=4)
def embed_5_from_3() -> np.ndarray:
    """3D -> 5D 嵌入：{vac,H,V} -> {vac,H_1517,V_1517}。"""
    P = np.zeros((5, 3), dtype=complex)
    P[0, 0] = 1.0
    P[3, 1] = 1.0
    P[4, 2] = 1.0
    return P


@lru_cache(maxsize=4)
def embed_6_from_3() -> np.ndarray:
    """3D -> 6D 嵌入：{vac,H,V} -> {vac,H,V,2H,2V,HV} 的单光子子块。"""
    P = np.zeros((6, 3), dtype=complex)
    P[0, 0] = 1.0
    P[1, 1] = 1.0
    P[2, 2] = 1.0
    return P


def project_6d_to_3d(op_6d: np.ndarray) -> np.ndarray:
    """将 36x36 双端口算符投影到 3D×3D (9x9)。"""
    op_6d = np.asarray(op_6d, dtype=complex)
    if op_6d.shape != (36, 36):
        raise ValueError(f"op_6d shape {op_6d.shape} != (36,36)")
    P = proj_3_from_6()
    Pi = np.kron(P, P)
    return Pi @ op_6d @ Pi.conj().T


def embed_3d_to_5d(op_3d: np.ndarray) -> np.ndarray:
    """将 9x9 双端口算符嵌入到 5D×5D (25x25)。"""
    op_3d = np.asarray(op_3d, dtype=complex)
    if op_3d.shape != (9, 9):
        raise ValueError(f"op_3d shape {op_3d.shape} != (9,9)")
    P = embed_5_from_3()
    Pi = np.kron(P, P)
    return Pi @ op_3d @ Pi.conj().T


def jones_3d(U_2x2: np.ndarray) -> np.ndarray:
    """把 2x2 琼斯矩阵嵌入到 3D：diag(1, U_2x2)。"""
    U = np.asarray(U_2x2, dtype=complex)
    if U.shape != (2, 2):
        raise ValueError(f"Jones matrix shape {U.shape} != (2,2)")
    U3 = np.eye(3, dtype=complex)
    U3[1:, 1:] = U
    return U3


