# -*- coding: utf-8 -*-
"""
希尔伯特空间定义和张量积构造

本模块提供定义子空间、构造积空间以及将门嵌入积空间的类和函数。
"""

from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np


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


def subspace_gate(
    full_space: ProductSpace,
    active_subspaces: Union[str, List[str]],
    gate_matrix: np.ndarray,
) -> np.ndarray:
    """
    将作用于特定子空间的门嵌入到完整的积空间中。

    给定一个作用于 active_subspaces 的门，将其嵌入为 I ⊗ ... ⊗ G ⊗ ... ⊗ I
    其中 G 作用于活动子空间，I 作用于其他子空间。

    Parameters
    ----------
    full_space : ProductSpace
        完整的积空间 H = H_1 ⊗ ... ⊗ H_n
    active_subspaces : Union[str, List[str]]
        门所作用的子空间名称
    gate_matrix : np.ndarray
        作用于活动子空间张量积的门矩阵。
        形状应为 (d_active, d_active)，其中 d_active 是活动
        子空间维度的乘积。

    Returns
    -------
    np.ndarray
        作用于完整空间的嵌入门矩阵。
        形状为 (full_space.dim, full_space.dim)

    Examples
    --------
    >>> space_780 = SubSpace('780', 3)
    >>> space_1517 = SubSpace('1517', 6)
    >>> bin_space = ProductSpace((space_780, space_1517))
    >>> gate_on_780 = np.eye(3)  # 780子空间上的单位门
    >>> full_gate = subspace_gate(bin_space, '780', gate_on_780)
    >>> full_gate.shape == (18, 18)
    True
    """
    if isinstance(active_subspaces, str):
        active_subspaces = [active_subspaces]

    # 验证活动子空间存在
    for name in active_subspaces:
        if name not in [s.name for s in full_space.subspaces]:
            raise ValueError(f"在 full_space 中未找到子空间 '{name}'")

    # 计算活动子空间的期望维度
    active_dim = 1
    for name in active_subspaces:
        active_dim *= full_space.subspace_dim(name)

    if gate_matrix.shape != (active_dim, active_dim):
        raise ValueError(
            f"门矩阵形状 {gate_matrix.shape} 与 "
            f"活动子空间维度 ({active_dim}, {active_dim}) 不匹配"
        )

    # 以张量积形式构建完整门
    # 从第一个子空间的单位门开始
    result = np.eye(1, dtype=complex)

    for i, subspace in enumerate(full_space.subspaces):
        if subspace.name in active_subspaces:
            # 此子空间受门作用
            # 需要从 gate_matrix 提取正确的分块
            # 并将其张量积入
            if len(active_subspaces) == 1:
                # 单个活动子空间 - 直接使用门
                result = np.kron(result, gate_matrix)
            else:
                # 多个活动子空间 - 需要分解门
                # 这更复杂；目前假设门已为
                # 活动子空间张量积格式化
                result = np.kron(result, gate_matrix)
        else:
            # 非活动子空间上的单位门
            result = np.kron(result, np.eye(subspace.dim, dtype=complex))

    return result


# 本项目的预定义子空间
# 这些与 README 中的物理模型对应

ATOM_3D = SubSpace('atom', 3, ('|0>', '|1>', '|e>'))
ATOM_A = SubSpace('atom_A', 3, ('|0>', '|1>', '|e>'))
ATOM_B = SubSpace('atom_B', 3, ('|0>', '|1>', '|e>'))

SUBSPACE_780 = SubSpace('780', 3, ('vac', 'H', 'V'))
SUBSPACE_1517 = SubSpace('1517', 6, ('vac', 'H', 'V', '2H', '2V', 'HV'))

# 系统格点：两个原子
SYSTEM_SPACE = ProductSpace((ATOM_A, ATOM_B))  # 9D

# Bin格点：780 x 1517
BIN_SPACE = ProductSpace((SUBSPACE_780, SUBSPACE_1517))  # 18D


def get_bin_space() -> ProductSpace:
    """获取标准的18D bin空间（780 x 1517）。"""
    return BIN_SPACE


def get_system_space() -> ProductSpace:
    """获取标准的9D系统空间（atom_A x atom_B）。"""
    return SYSTEM_SPACE
