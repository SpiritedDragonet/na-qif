# -*- coding: utf-8 -*-
"""
MPS态容器（TeNPy后端）

================================================================================
TeNPy的MPS封装，用于时间仓量子仿真。

核心设计：
-----------
- 两格点操作使用 get_theta + set_svd_theta 进行局域更新
- Kraus操作完全局域化（不需要 canonical_form() 扫描）
- 约化密度矩阵使用 get_rho_segment() 确保正确性

依赖：
-------------
pip install physics-tenpy

参考文献：
-----------
[1] TeNPy MPS: https://tenpy.readthedocs.io/en/stable/reference/tenpy.networks.mps.MPS.html
[2] TeNPy Array: https://tenpy.readthedocs.io/en/stable/reference/tenpy.linalg.np_conserved.html
================================================================================
"""

from typing import List, Optional, Union
import numpy as np

# TeNPy imports
from tenpy.networks.mps import MPS as TeNPy_MPS
from tenpy.networks.site import BosonSite
from tenpy.linalg.np_conserved import Array


class MPSState:
    """
    使用TeNPy的矩阵积态。

    Parameters
    ----------
    local_dims : List[int]
        局域希尔伯特空间维度
    init_state : Optional[Union[List[int], np.ndarray]]
        - None: 真空态 |0>...|0>
        - List[int]: 直积态
        - np.ndarray: 完整波函数（使用 MPS.from_full）
    max_bond : int
        截断的最大键维度
    """

    def __init__(
        self,
        local_dims: List[int],
        init_state: Optional[Union[List[int], np.ndarray]] = None,
        max_bond: int = 100,
    ):
        self.L = len(local_dims)
        self.d = local_dims
        self.max_bond = max_bond

        # 创建TeNPy格点（玻色型，无电荷守恒）
        sites = [BosonSite(dim - 1, None) for dim in self.d]

        # 根据init_state类型初始化MPS
        if init_state is None:
            # 真空态 |0>...|0>
            init_labels = ['0'] * self.L
            self._mps = TeNPy_MPS.from_product_state(sites, init_labels, bc='finite', form='B')
        elif isinstance(init_state, list):
            # 从基指标的直积态
            init_labels = [str(s) for s in init_state]
            self._mps = TeNPy_MPS.from_product_state(sites, init_labels, bc='finite', form='B')
        elif isinstance(init_state, np.ndarray):
            # 完整波函数 - 使用TeNPy的from_full
            psi_reshaped = init_state.reshape(self.d + [1] * (self.L - len(self.d)))
            psi_array = Array.from_ndarray_trivial(psi_reshaped, labels=[f'p{i}' for i in range(self.L)])
            self._mps = TeNPy.from_full(psi_array, sites, bc='finite', form='B')
        else:
            raise ValueError(f"Invalid init_state: {type(init_state)}")

        self._mps.chi_max = self.max_bond

    # ========================================================================
    # 底层局域更新（避免 canonical_form 扫描）
    # ========================================================================

    def _apply_two_site_op_local(
        self,
        i: int,
        op: Array,
        truncate: bool = True,
        normalize: bool = False,
    ) -> None:
        """
        使用局域更新应用两格点算符（get_theta + set_svd_theta）。

        通过只更新局域键来避免 canonical_form() 扫描。

        Parameters
        ----------
        i : int
            左格点索引
        op : Array
            带有标签 ['p0', 'p1', 'p0*', 'p1*'] 的TeNPy Array
        truncate : bool
            是否截断键维度
        normalize : bool
            若为True，应用后归一化（用于Kraus结果）
        """
        # 获取两格点的theta：腿为 (vL, p0, p1, vR)
        theta = self._mps.get_theta(i, n=2)

        # 转换为numpy进行收缩（避免LegCharge问题）
        theta_np = theta.to_ndarray()  # Shape: (chiL, d0, d1, chiR)
        op_np = op.to_ndarray()  # Shape: (d0, d1, d0, d1)

        # 收缩: op @ theta，其中 op[i,j,k,l] 作用于theta的物理腿
        # 结果: theta_new[a, i, j, b] = sum_{k,l} op[i, j, k, l] * theta[a, k, l, b]
        theta_new_np = np.einsum('ijkl,aklb->aijb', op_np, theta_np)

        # 转换回TeNPy Array
        theta_new = Array.from_ndarray_trivial(theta_new_np, labels=['vL', 'p0', 'p1', 'vR'])

        # 合并腿以进行SVD: (vL.p0) 和 (p1.vR)
        theta_combined = theta_new.combine_legs(
            [['vL', 'p0'], ['p1', 'vR']],
            new_axes=[0, 1],
            qconj=[+1, -1]
        )

        # 设置截断参数
        trunc_params = None
        if truncate:
            trunc_params = {'chi_max': self.max_bond, 'svd_min': 1e-13}

        # 通过SVD写回
        self._mps.set_svd_theta(i, theta_combined, trunc_par=trunc_params)

        # 若需要则归一化（用于Kraus分支结果）
        if normalize:
            self._mps.norm = 1.0

    # ========================================================================
    # 门操作
    # ========================================================================

    # ========================================================================
    # 核心API方法（仅局域，无 canonical_form 扫描）
    # ========================================================================

    def apply_bond_op(
        self,
        i: int,
        op: np.ndarray,
        truncate: bool = True,
    ) -> None:
        """
        通过局域更新应用两格点算符（幺正或非幺正）。

        使用 get_theta + set_svd_theta 来避免 canonical_form() 扫描。
        这是应用任何两格点门的主要方法。

        Parameters
        ----------
        i : int
            左格点索引（作用于格点i和i+1）
        op : np.ndarray
            算符矩阵，形状为 (d1*d2, d1*d2) 或 (d1, d2, d1, d2)
        truncate : bool
            是否截断键维度
        """
        d1, d2 = self.d[i], self.d[i + 1]

        # 重塑为4D: (d1, d2, d1, d2)
        op = np.asarray(op)
        if op.ndim == 2:
            op = op.reshape(d1 * d2, d1 * d2)
        op_4d = op.reshape(d1, d2, d1, d2)

        # 创建带正确标签的TeNPy Array
        op_arr = Array.from_ndarray_trivial(op_4d, labels=['p0', 'p1', 'p0*', 'p1*'])

        # 通过局域更新应用
        self._apply_two_site_op_local(i, op_arr, truncate=truncate, normalize=False)

    def apply_kraus_one_site(
        self,
        site: int,
        kraus_ops: List[np.ndarray],
        rng: Optional[np.random.Generator] = None,
    ) -> int:
        """
        通过量子轨迹应用单格点Kraus信道。

        采样一个Kraus算符并归一化后应用。

        Parameters
        ----------
        site : int
            格点索引
        kraus_ops : List[np.ndarray]
            Kraus算符列表，每个形状为 (d, d)
        rng : np.random.Generator, optional
            随机数生成器

        Returns
        -------
        int
            采样的Kraus算符索引
        """
        if rng is None:
            rng = np.random.default_rng()

        d = self.d[site]

        # 获取当前格点张量
        theta = self._mps.get_theta(site, n=1)  # Shape: (chiL, d, chiR)
        theta_np = theta.to_ndarray()

        # 计算每个Kraus算符的概率
        probs = []
        thetas_mu = []

        for K in kraus_ops:
            K = np.asarray(K).reshape(d, d)
            # 应用K: K @ theta（在物理指标上收缩）
            K_theta = np.einsum('ij,ajb->aib', K, theta_np)
            p_mu = np.linalg.norm(K_theta) ** 2
            probs.append(p_mu)
            thetas_mu.append(K_theta)

        # 归一化并采样
        probs = np.array(probs)
        p_total = np.sum(probs)

        if p_total < 1e-15:
            raise ValueError("总概率接近零 - Kraus算符可能无效")

        probs = probs / p_total
        mu = rng.choice(len(kraus_ops), p=probs)

        # 从选中分支创建归一化的theta
        theta_selected = thetas_mu[mu] / np.sqrt(probs[mu] * p_total)

        # 转换为TeNPy Array并直接写回格点张量
        # 对于单格点操作，直接设置B张量
        theta_arr = Array.from_ndarray_trivial(theta_selected, labels=['vL', 'p', 'vR'])

        # 直接设置格点张量（单格点不需要SVD）
        self._mps.set_B(site, theta_arr, form='Th')
        self._mps.canonical_form_finite(renormalize=True)

        return mu

    def apply_kraus_one_site_fixed(
        self,
        site: int,
        kraus_op: np.ndarray,
        eps: float = 1e-15,
        canonicalize: bool = True,
    ) -> float:
        """
        固定应用单个Kraus算符，并返回该分支概率。

        用于后选 no-loss 分支：不再采样，只走指定Kraus。
        """
        d = self.d[site]
        theta = self._mps.get_theta(site, n=1)  # Shape: (chiL, d, chiR)
        theta_np = theta.to_ndarray()

        K = np.asarray(kraus_op).reshape(d, d)
        K_theta = np.einsum('ij,ajb->aib', K, theta_np)
        p_mu = float(np.linalg.norm(K_theta) ** 2)
        if p_mu < eps:
            raise ValueError("固定Kraus分支概率过小，无法归一化")

        theta_selected = K_theta / np.sqrt(p_mu)
        theta_arr = Array.from_ndarray_trivial(theta_selected, labels=['vL', 'p', 'vR'])
        self._mps.set_B(site, theta_arr, form='Th')
        if canonicalize:
            self._mps.canonical_form_finite(renormalize=True)

        return p_mu

    def finalize_bin_pair(self, i: int) -> None:
        """
        冻结已测量的bin对以确保线性复杂度。

        测量后，bin不应再被访问。此方法确保该对右侧的键维度为1，
        从而有效地将它们与链的其余部分解耦。

        Parameters
        ----------
        i : int
            bin对的左格点索引（i, i+1）
        """
        # 获取格点i+1右侧键的当前theta
        # 若i+1是最后一个格点，无需操作
        if i + 1 >= self.L - 1:
            return

        # 强制SVD且chi_max=1以截断该键
        theta = self._mps.get_theta(i + 1, n=1)
        theta_np = theta.to_ndarray()

        # 重塑为矩阵并只保留1个奇异值的SVD
        chiL, d, chiR = theta_np.shape
        theta_mat = theta_np.reshape(chiL * d, chiR)

        # SVD并只保留最大奇异值
        U, s, Vh = np.linalg.svd(theta_mat, full_matrices=False)

        # 仅用第一个奇异值重构
        theta_trunc = np.outer(U[:, 0] * s[0], Vh[0, :])
        theta_trunc = theta_trunc.reshape(chiL, d, 1)

        # 转换回并设置
        theta_arr = Array.from_ndarray_trivial(theta_trunc, labels=['vL', 'p0', 'vR'])
        theta_combined = theta_arr.combine_legs(
            [['vL', 'p0'], ['vR']],
            new_axes=[0, 1],
            qconj=[+1, -1]
        )

        self._mps.set_svd_theta(i + 1, theta_combined, trunc_par={'chi_max': 1, 'svd_min': 1e-13})

    # ========================================================================
    # 便捷方法（向后兼容）
    # ========================================================================

    def apply_one_site_gate(self, site: int, gate: np.ndarray) -> None:
        """
        应用单格点幺正门。

        Parameters
        ----------
        site : int
            格点索引
        gate : np.ndarray
            幺正矩阵，形状为 (d, d)
        """
        d = self.d[site]
        gate = np.asarray(gate).reshape(d, d)

        # 创建带正确标签的TeNPy Array用于单格点门
        # 标签: ['p', 'p*'] 用于物理腿（输出，输入）
        gate_arr = Array.from_ndarray_trivial(gate, labels=['p', 'p*'])

        # 使用TeNPy的apply_local_op与Array对象
        self._mps.apply_local_op(site, gate_arr, unitary=True)

    def apply_two_site_kraus(
        self,
        site_left: int,
        kraus_ops: List[np.ndarray],
        rng: Optional[np.random.Generator] = None,
        probs_from_rho: bool = False,
        rho: Optional[np.ndarray] = None,
    ) -> int:
        """
        通过量子轨迹（蒙特卡洛采样）应用Kraus信道。

        对每个Kraus算符 K_mu：
        1. 计算 p_mu = ||K_mu @ theta||^2
        2. 根据概率 {p_mu} 采样 mu
        3. 应用 K_mu 并除以 sqrt(p_mu) 归一化

        这是完全局域的，不会触发 canonical_form()。

        Parameters
        ----------
        site_left : int
            左格点索引
        kraus_ops : List[np.ndarray]
            Kraus算符列表，每个形状为 (d1*d2, d1*d2) 或 (d1, d2, d1, d2)
        rng : np.random.Generator, optional
            随机数生成器

        probs_from_rho : bool
            若为True，则使用两站点约化密度矩阵计算Kraus概率，
            避免依赖正交中心位置。此模式下仍会用get_theta更新态。
        rho : Optional[np.ndarray]
            可选的两站点约化密度矩阵（site_left, site_left+1）。
            仅在 probs_from_rho=True 时使用。

        Returns
        -------
        int
            采样的Kraus算符索引
        """
        if rng is None:
            rng = np.random.default_rng()

        d1, d2 = self.d[site_left], self.d[site_left + 1]

        if probs_from_rho:
            if rho is None:
                rho = self.get_reduced_density([site_left, site_left + 1])

            if rho.ndim == 4:
                rho_mat = rho.reshape(d1 * d2, d1 * d2)
            else:
                rho_mat = rho.reshape(d1 * d2, d1 * d2)

            probs = np.zeros(len(kraus_ops), dtype=float)
            for idx, K in enumerate(kraus_ops):
                K_mat = np.asarray(K)
                if K_mat.ndim == 4:
                    K_mat = K_mat.reshape(d1 * d2, d1 * d2)
                else:
                    K_mat = K_mat.reshape(d1 * d2, d1 * d2)
                K_rho = K_mat @ rho_mat
                p_mu = np.trace(K_rho @ K_mat.conj().T).real
                probs[idx] = max(p_mu, 0.0)

            p_total = np.sum(probs)
            if p_total < 1e-15:
                raise ValueError("总概率接近零 - Kraus算符可能无效")

            probs = probs / p_total
            mu = rng.choice(len(kraus_ops), p=probs)

            # 获取当前theta并应用选中的Kraus
            theta = self._mps.get_theta(site_left, n=2)
            theta_np = theta.to_ndarray()  # Shape: (chiL, d1, d2, chiR)

            K_sel = np.asarray(kraus_ops[mu])
            if K_sel.ndim == 2:
                K_sel = K_sel.reshape(d1 * d2, d1 * d2)
            K_4d = K_sel.reshape(d1, d2, d1, d2)

            K_theta = np.einsum('ijkl,aklb->aijb', K_4d, theta_np)
            p_mu = probs[mu] * p_total
            if p_mu < 1e-15:
                raise ValueError("选中的Kraus概率过小，无法归一化")

            theta_selected = K_theta / np.sqrt(p_mu)

        else:
            # 获取当前theta
            theta = self._mps.get_theta(site_left, n=2)
            theta_np = theta.to_ndarray()  # Shape: (chiL, d1, d2, chiR)

            # 计算每个Kraus算符的概率和结果态
            probs = []
            thetas_mu = []

            for K in kraus_ops:
                K = np.asarray(K)
                if K.ndim == 2:
                    K = K.reshape(d1 * d2, d1 * d2)
                K_4d = K.reshape(d1, d2, d1, d2)

                # 应用K: 将K的输入腿与theta的物理腿收缩
                K_theta = np.einsum('ijkl,aklb->aijb', K_4d, theta_np)
                p_mu = np.linalg.norm(K_theta) ** 2
                probs.append(p_mu)
                thetas_mu.append(K_theta)

            # 归一化并采样
            probs = np.array(probs)
            p_total = np.sum(probs)

            if p_total < 1e-15:
                raise ValueError("总概率接近零 - Kraus算符可能无效")

            probs = probs / p_total
            mu = rng.choice(len(kraus_ops), p=probs)

            # 从选中分支创建归一化的theta
            theta_selected = thetas_mu[mu] / np.sqrt(probs[mu] * p_total)

        # 转换为TeNPy Array并写回
        theta_arr = Array.from_ndarray_trivial(theta_selected, labels=['vL', 'p0', 'p1', 'vR'])
        theta_combined = theta_arr.combine_legs(
            [['vL', 'p0'], ['p1', 'vR']],
            new_axes=[0, 1],
            qconj=[+1, -1]
        )

        trunc_params = {'chi_max': self.max_bond, 'svd_min': 1e-13}
        self._mps.set_svd_theta(site_left, theta_combined, trunc_par=trunc_params)
        self._mps.norm = 1.0

        return mu

    def swap_sites(self, i: int) -> None:
        """
        交换相邻格点i和i+1。

        这会同时交换张量索引和局域维度（self.d），
        使得后续的 apply_bond_op 调用使用正确的维度。

        Parameters
        ----------
        i : int
            左格点索引（交换i和i+1）
        """
        trunc_params = {'chi_max': self.max_bond, 'svd_min': 1e-13}
        self._mps.swap_sites(i, trunc_par=trunc_params)

        # 关键修复：同时更新self.d数组，交换两个位置的局域维度
        # 这是apply_bond_op正确计算维度所必需的
        self.d[i], self.d[i + 1] = self.d[i + 1], self.d[i]

    def find_sites_by_dim(self, target_dim: int) -> List[int]:
        """
        查找具有特定局域维度的所有格点。

        用于在SWAP后动态定位原子（3D）或bin（18D）。

        Parameters
        ----------
        target_dim : int
            要搜索的局域维度

        Returns
        -------
        List[int]
            具有目标维度的格点索引列表
        """
        return [i for i, d in enumerate(self.d) if d == target_dim]

    # ========================================================================
    # 态提取
    # ========================================================================

    def get_reduced_density(self, sites: List[int]) -> np.ndarray:
        """
        获取指定格点的约化密度矩阵。

        使用TeNPy的get_rho_segment()，它能正确处理
        施密特权重和规范条件。
        """
        rho_array = self._mps.get_rho_segment(sites)
        return rho_array.to_ndarray()

    def get_atom_state(self, system_site: int = 0) -> np.ndarray:
        """从系统格点获取原子密度矩阵。"""
        return self.get_reduced_density([system_site])

    def expectation_value(self, observable: np.ndarray, site: int) -> float:
        """
        计算某格点上可观测量期望值。

        Parameters
        ----------
        observable : np.ndarray
            可观测量矩阵 (d, d)
        site : int
            格点索引

        Returns
        -------
        float
            期望值 <O>
        """
        rho = self.get_reduced_density([site])
        return float(np.real(np.trace(observable @ rho)))

    # ========================================================================
    # 属性和工具方法
    # ========================================================================

    @property
    def chi(self) -> List[int]:
        """键维度。"""
        return self._mps.chi.copy()

    def norm(self) -> float:
        """获取态的模长。"""
        return float(self._mps.norm)

    def get_bond_dimensions(self) -> List[int]:
        """获取所有键维度列表，等价于 self.chi。"""
        return self.chi

    def test_sanity(self) -> bool:
        """运行TeNPy的健全性检查（验证规范形式）。"""
        return self._mps.test_sanity()

    def copy(self) -> 'MPSState':
        """创建深拷贝。"""
        new_state = MPSState(self.d.copy(), max_bond=self.max_bond)
        new_state._mps = self._mps.copy()
        return new_state

    def __repr__(self) -> str:
        """字符串表示。"""
        chi_str = str(self.get_bond_dimensions())
        return f"MPSState(L={self.L}, d={self.d}, chi={chi_str})"


# ========================================================================
# 工厂函数
# ========================================================================

def create_timebin_mps(
    n_bins: int,
    system_dim: int = 9,
    bin_dim: int = 5,
    max_bond: int = 100,
) -> MPSState:
    """
    创建时间仓仿真的MPS。

    结构：S - A1 - B1 - A2 - B2 - ... （系统，然后是原子-光子bin）

    Parameters
    ----------
    n_bins : int
        时间仓的数量
    system_dim : int
        系统（原子）希尔伯特空间维度
    bin_dim : int
        Bin（光子）希尔伯特空间维度
    max_bond : int
        最大键维度
    """
    local_dims = [system_dim] + [bin_dim] * (2 * n_bins)
    return MPSState(local_dims, max_bond=max_bond)
