下面给出一份“可实施的大图纸”：把我们定版的各模块（源端发射/偏振映射、QFC、滤波记忆、损耗与噪声、光纤偏振演化与 PMD、中站干涉与探测）按 **time-bin 离散 + MPS（TEBD/TDVP）** 的骨架拼成一条**可直接落地的核心算法流程**。我会尽量用纯文字与公式描述“程序上到底怎么串起来”，并把你上次提到的两个待细化点$1：$\Alpha^{(\mathrm{emit})}$ 构造；2：PMD 的 time-bin 移位网络$嵌入到整条流程里。

------

## 1. 总体结构：把整条链路写成“逐 bin 的局域量子线路”

我们把一次链路尝试（一次发射窗 + 一次中站符合窗）写成对离散 time bins 的迭代：

- 时间离散：$t_n = n\Delta t$, $n=1,\dots,N$
- 每个 bin 上有两臂（A/B）两偏振（H/V）两频段（780/1517）模式（或压缩后的等效局域空间）
- 整条链路在数值上就是在 MPS 上反复施加“局域门/局域信道”（collision/time-bin picture），再用 TEBD/TDVP 更新张量并做 SVD 截断。

把“每个 bin 的所有物理块”按顺序合成一个超算子（或一串 Kraus/幺正门）最清晰。对第 (n) 个 bin，我们把一次更新写成：

$$
\rho \ \xrightarrow{\ \mathcal{U}^{(\mathrm{emit})}*{n}\ }\
\rho \ \xrightarrow{\ \mathcal{U}^{(\mathrm{QFC})}*{n}\ }\
\rho \ \xrightarrow{\ \mathcal{U}^{(\mathrm{filt})}*{n}\ }\
\rho \ \xrightarrow{\ \mathcal{E}^{(\mathrm{loss/noise})}*{n}\ }\
\rho \ \xrightarrow{\ \mathcal{U}^{(\mathrm{pol/PMD})}*{n}\ }\
\rho \ \xrightarrow{\ \mathcal{U}^{(\mathrm{BS})}*{n}\ }\
\rho \ \xrightarrow{\ \mathcal{M}^{(\mathrm{det})}_{n}\ }\
\rho
$$

其中每一段都对应到**程序里的一个“局域操作对象”**（单站点门、两站点门或局域 Kraus 集），并且都能嵌入 MPS 的 TEBD 更新。

------

## 2. Hilbert 空间与数据结构：你需要哪些站点、每个站点多大

### 2.1 原子系统站点（system site）

每个节点 $j\in{A,B}$ 用 3 能级（两稳态 + 一个激发态）：
$$
\mathcal{H}*{\mathrm{atom},j}=\mathrm{span}{|0\rangle_j,|1\rangle_j,|e\rangle_j}.
$$
两端合起来 $\mathcal{H}*{\mathrm{atom}}=\mathcal{H}*{\mathrm{atom},A}\otimes\mathcal{H}*{\mathrm{atom},B}$。

如果你采用“滤波腔等效内部模”（有记忆），把它也并入 system（最常见做法）：每臂每偏振一个腔模 $a_{j,p}$$截断到 $n_{\max}^{(a)}$$。

因此 system 站点在程序上通常是一个 **“复合局域空间”**：
$$
\mathcal{H}*{S}= \mathcal{H}*{\mathrm{atom},A}\otimes \mathcal{H}*{\mathrm{atom},B}\otimes
\Big(\bigotimes*{j,p}\mathcal{H}*{a*{j,p}}\Big)\quad(\text{可选}).
$$

### 2.2 光场 time-bin 站点（field sites）

连续场离散为 time-bin 算符（对每臂、每频段、每偏振）：
$$
b^{p}*{j,n}=\frac{1}{\sqrt{\Delta t}}\int*{t_n}^{t_n+\Delta t}!dt, b^{p}*j(t),\quad
c^{p}*{j,n}=\frac{1}{\sqrt{\Delta t}}\int_{t_n}^{t_n+\Delta t}!dt, c^{p}*j(t),
$$
并满足 $[b*{j,n}^{p},b_{j,m}^{p'\dagger}]=\delta_{nm}\delta_{pp'}$，(c) 同理。

**局域维度怎么选（关键的工程化降维）：**

- 最实用第一版：每臂每 bin **单光子子空间 + 区分频段与偏振**
  $$
  {|{\rm vac}\rangle,\ |780,H\rangle,\ |780,V\rangle,\ |1517,H\rangle,\ |1517,V\rangle}
  $$
  即每臂每 bin 是 5 维。两臂合并成一个“super-site”（同一 (n) 的 A 与 B）就是 25 维，仍可控。
- 若要显式多光子污染（QFC 噪声、串扰）：再把每个模式的占据截断升到 $n_{\max}=2,3$，但成本会显著上升。

### 2.3 推荐的 MPS 站点排列（让所有关键门都是“局域”的）

为了让“同一 bin 的 A/B 干涉”是最近邻两站点门，同时又能让 system 顺序与每个 bin 相互作用，一种很稳的排列是：

$$
\underbrace{S}*{\text{system}};-;\underbrace{A_1}*{\text{arm A bin 1}};-;\underbrace{B_1}_{\text{arm B bin 1}};-;\underbrace{A_2};-;\underbrace{B_2};-;\cdots;-;\underbrace{A_N};-;\underbrace{B_N}.
$$

程序上你会做一个“conveyor belt”式的扫描：每一步把 (S) 往右移两格$经过 $A_n,B_n$$，让它依次与 $A_n$ 和 $B_n$ 相互作用，然后在 $A_n,B_n$ 上做中站 BS + 探测。该思路与“逐 bin 施加局域门”的端到端流程一致。

> 重要说明：这等价于在“retarded time”下工作——我们把 bin 标签理解为“到达中站的相对时间模式”。光纤的**绝对传播时延**不会影响 A/B 的相对对齐；若你要计入“等待时间导致的原子退相干”，单独对原子施加一段时间 $\tau$ 的退相干通道即可（见 §6.3）。

------

## 3. 每个模块怎么落成“局域门/局域信道”

下面每个模块我都用“作用在哪些自由度上”“数学表达是什么”“在 MPS 上是什么类型的门”来写。

### 3.1 源端发射门 $U^{(\mathrm{emit})}_{j,n}$

**原子跃迁通道$$\sigma^\pm$$与跳算符：**
$$
S_{j,+}=|0\rangle_j\langle e|,\qquad S_{j,-}=|1\rangle_j\langle e|.
$$

 把“几何/收集/偏振定义”压缩成一个 2×2 复矩阵 $\Alpha^{(\mathrm{emit})}$： 它把原子自然基$\sigma^\pm$映射到你数值里用的光路偏振基（H/V）。我们约定 

$$
 \Alpha^{$\mathrm{emit}$}
\begin{pmatrix}
\alpha_{H,+} & \alpha_{H,-}\
\alpha_{V,+} & \alpha_{V,-}
\end{pmatrix},
$$
并将其并入耦合算符
$$
L_{j,p}(t)=\sqrt{\gamma_j(t)}\Big(\alpha_{p,+} S_{j,+}+\alpha_{p,-} S_{j,-}\Big),\qquad p\in{H,V}.
$$

# **time-bin 离散后的局域发射门：** 

$$
U^{$\mathrm{emit}$}_{j,n}

 \exp!\left[ \sqrt{\Delta t}\sum_{p\in{H,V}} \Big( L_{j,p}(t_n), b^{p\dagger}_{j,n}

L^\dagger_{j,p}$t_n$, b^{p}_{j,n}
\Big)
\right].
$$
这就是“原子 ⊗ 当前 780 bin”的**两体幺正门**（在 MPS 上是 system–site 两站点门）。

#### （你上次说的“1”）$\Alpha^{(\mathrm{emit})}$ 怎么从几何算出来

这部分我直接给可实施步骤（纯公式）：

1. 选定坐标系：量子化轴 $\hat z$，收集方向单位矢量 $\mathbf{n}(\theta,\phi)$。
2. 给出两条偶极跃迁的偶极矩方向矢量 $\mathbf{d}*\mu$$$\mu\in{+,-}$ 对应球面基 $\mathbf{e}*{\pm 1}$$。
3. 远场辐射的横向电场方向（忽略标量包络）：
   $$
   \mathbf{E}*\mu(\mathbf{n})\propto \mathbf{n}\times(\mathbf{n}\times\mathbf{d}*\mu)
   = \mathbf{d}*\mu-\mathbf{n}(\mathbf{n}!\cdot!\mathbf{d}*\mu).
   $$
4. 在与 $\mathbf{n}$ 垂直的横向平面上选一组正交偏振基矢 ${\mathbf{e}*1,\mathbf{e}\*2}$$例如 $\mathbf{e}\*\theta,\mathbf{e}*\phi$，或你实验定义的 H/V 在横向平面中的方向$。
5. 得到每条跃迁在该偏振基下的 Jones 向量：
   $$
   |u_\mu\rangle =
   \begin{pmatrix}
   \mathbf{e}*1\cdot \mathbf{E}*\mu\
   \mathbf{e}*2\cdot \mathbf{E}*\mu
   \end{pmatrix},
   \quad \text{并作归一化（或把归一化吸收进收集效率）}.
   $$
6. 若你的“数值偏振基”是 H/V，而 ${\mathbf{e}*1,\mathbf{e}\*2}$ 不是 H/V，只要再乘一个 2×2 的基变换矩阵 (R)：
   $$
   \begin{pmatrix}\alpha\*{H,\mu}\ \alpha*{V,\mu}\end{pmatrix}
   = R,
   \frac{|u_\mu\rangle}{|u_\mu|}.
   $$
7. 将 Clebsch–Gordan 系数、两通道进入单模的比例 $\beta_\pm$、以及可能的相位差并入：
   $$
   \alpha_{p,\mu}\leftarrow \sqrt{\beta_\mu},e^{i\varphi_\mu},\alpha_{p,\mu}.
   $$
8. 最终得到 $\Alpha^{(\mathrm{emit})}$，并检查它的“可用纠缠性”：
   若两列向量 $|u_+\rangle,|u_-\rangle$ 近似正交，才有高质量偏振纠缠；偏离（尤其趋近 rank-1）会在源头直接毁纠缠。

一个很有用的诊断量是两通道的归一化重叠（某些自然基下可写成）：
$$
\langle u_+|u_-\rangle = \frac{\sin^2\theta}{1+\cos^2\theta},
$$
$\theta=0$ 时正交、$\theta=\pi/2$ 时完全不可区分（你说的“赤道面看线偏振”正对应这个极限）。

------

### 3.2 QFC 门 $U^{(\mathrm{QFC})}_{j,n}$

把 QFC 最小可信地写成“频域 beam splitter”幺正：对每偏振 (p), 
$$
U^{$\mathrm{QFC}$}_{j,n}

\exp!\left[
-i\sum_{p\in{H,V}}
\theta_p\Big(
b^{p}*{j,n}c^{p\dagger}*{j,n}+b^{p\dagger}*{j,n}c^{p}*{j,n}
\Big)
\right],
$$
其中 $\sin^2\theta_p$ 给出 $780\to1517$ 的转换概率。

**在 MPS 上怎么实现：**

- 若你把“同一臂同一 bin 的 780+1517+H/V”放在同一个 site（5 维压缩或更高维展开），那么 $U^{(\mathrm{QFC})}$ 是一个**单站点门**$矩阵维度 $d_{\rm bin}\times d_{\rm bin}$$。
- 若你把 780 与 1517 做成两个 site，则是同臂相邻两 site 的两体门；但通常不推荐（会增 swap/复杂度）。

**插入损耗 / PDL（偏振相关损耗）**：用 beamsplitter-to-vacuum 最干净：
$$
c^{p}*{j,n}\rightarrow \sqrt{\eta*{\rm ins}^{(p)}},c^{p}*{j,n}+\sqrt{1-\eta*{\rm ins}^{(p)}},e^{p}_{j,n},
$$
其中 (e) 是环境真空模。

在“纯态 MPS + 轨迹”实现里，你通常不用显式加入 (e) 模，而是用等价的幅度阻尼 Kraus（见 §5.2）。

**QFC 噪声注入**：第一版最省力、最不丢物理含义的实现是把它转化为“等效背景计数率/每 bin 的噪声点击概率”，因为噪声主要通过“假阳性符合”拉高 $p_{\rm succ}$ 并拉低 (F)。
若你要更物理地让噪声以“额外光子”进入，需要允许多光子截断（否则无法闭合）。

------

### 3.3 滤波器件（有记忆）门 $U^{(\mathrm{filt})}_{j,n}$

你定版方案里最值得显式进 MPS 的“非马尔可夫来源”之一就是滤波/窄带腔的记忆：把它做成系统内模 $a_f$ 顺序耦合 passing bins。

一个可直接离散实现的最小模型（单偏振示意，实际对 (p=H,V) 复制）：

- 腔自由哈密顿量：
  $$
  H_f=\hbar\Delta_f,a^\dagger a
  $$
- 腔与通过的 telecom bin 的耦合（碰撞模型形式）对应一个“beam-splitter 型”门：
  $$
  U^{(\mathrm{filt})}*{j,n}
  \approx
  \exp!\left[
  \sqrt{\Delta t}\sum*{p}
  \Big(
  \sqrt{\kappa_f},a_{j,p},c^{p\dagger}_{j,n}
  \sqrt{\kappa_f},a^\dagger_{j,p},c^{p}_{j,n}
  \Big)
  \right]
  \cdot
  \exp!\left[-\frac{i}{\hbar}H_f,\Delta t\right].
  $$
- 腔损耗（内损耗/出耦）同样用幅度阻尼通道或跳算符 $\sqrt{\kappa_{\rm loss}},a$。

**在 MPS 上：\**这是 system（含腔模）与当前 bin 的\**两站点门**，并且正是它会让相邻 bins 纠缠起来，从而需要 MPS bond dimension。

------

### 3.4 光纤偏振演化与 PMD

#### 3.4.1 无 PMD：Jones 矩阵（每 bin 局域 SU(2) 旋转）

把 QFC、光纤、补偿器件合并为一个总 Jones 矩阵 $U_j(\omega)$，窄带时取常数 $U_j$：
$$
\begin{pmatrix}
c^{H}*{j,n}\
c^{V}*{j,n}
\end{pmatrix}
\longrightarrow
U_j
\begin{pmatrix}
c^{H}*{j,n}\
c^{V}*{j,n}
\end{pmatrix}.
$$
两臂相对失配 $U_{\rm rel}=U_A^\dagger U_B$ 是干涉可见度的关键控制量。

在“单光子压缩基 ${|{\rm vac}\rangle,|H\rangle,|V\rangle}$”中，它对应一个 3×3 的局域门：
$$
U^{(\mathrm{pol})} = |{\rm vac}\rangle\langle{\rm vac}| + \sum_{p,q\in{H,V}}(U_j)_{pq},|p\rangle\langle q|.
$$
#### 3.4.2（你上次说的“2”）PMD：在 PSP 基的差分延迟 = 跨 bin 的移位门

PMD 的物理内核：存在两个主偏振态（PSP）$|\mathrm{ps}*1\rangle,|\mathrm{ps}\*2\rangle$，它们的群时延不同。频域写成
$$
U_j(\omega)=R\*{\rm out},
\begin{pmatrix}
e^{-i\omega\tau_1} & 0\
0 & e^{-i\omega\tau_2}
\end{pmatrix}
R*{\rm in},
\quad \Delta\tau=\tau_2-\tau_1.
$$
在时域（PSP 基）就是：
$$
c_{\mathrm{ps}1}(t)\to c_{\mathrm{ps}1}(t-\tau_1),\qquad
c_{\mathrm{ps}2}(t)\to c_{\mathrm{ps}2}(t-\tau_2).
$$
离散到 bins 后，令 $m=\mathrm{round}(\Delta\tau/\Delta t)$（整数延迟的第一版），则在 PSP 基可以写成“移位算符”：
$$
c_{\mathrm{ps}2,n}\ \to\ c_{\mathrm{ps}2,n-m},
\qquad
c_{\mathrm{ps}1,n}\ \to\ c_{\mathrm{ps}1,n}.
$$
更一般的分数 bin 延迟可用一个短 FIR（相邻 bins 的线性组合）逼近，但整数移位已足够构建可落地第一版。

**如何在 MPS 上做这个移位（局域两站点门序列）：**

1. 对每个 bin 先做一次偏振基变换 $R_{\rm in}$：把 ((H,V)) 旋到 $(\mathrm{ps}1,\mathrm{ps}2)$。这是每 bin 的单站点门。
2. 对 $\mathrm{ps}2$ 分量做“向后移动 (m) 个 bin”的置换。置换可以用最近邻 SWAP 链实现：
   对 $k = n-m,\dots,n-1$，依次对“bin (k) 与 bin (k+1)”的 $\mathrm{ps}2$ 子空间施加 SWAP。
   这在程序里就是一串**同臂相邻 bin 的两站点门**$每个门只作用在两站点的 $\mathrm{ps}2$ 子空间上，真空与 $\mathrm{ps}1$ 保持不动$。
3. 再对每个 bin 做 $R_{\rm out}$（回到 H/V 或到你想要的后续基）。

这一步就是你之前定义的“PMD：在 PSP 基上做差分延迟（跨 bin 的移位门）”。

**注意：**如果 PMD 很小$$\Delta\tau\ll\Delta t$$，整数移位会给出 (m=0)。此时你要么减小 $\Delta t$，要么升级到“分数延迟 FIR”，否则 PMD 在数值里等于被忽略。

------

### 3.5 光纤衰减等损耗（局域非幺正）

对单向链路，衰减可以作为量子信道作用在每个 bin 的 telecom 模式上：幅度衰减 $\eta=e^{-\alpha L}$，等效 beamsplitter-to-vacuum。

在“纯态轨迹 + 单光子截断”的实现里，单模式的损耗可用 Kraus：
$$
K_0 = |0\rangle\langle 0|+\sqrt{\eta},|1\rangle\langle 1|,
\qquad
K_1 = \sqrt{1-\eta},|0\rangle\langle 1|.
$$
对两偏振就是对 (H) 与 (V) 各自$或带 PDL 的 $\eta_H,\eta_V$$做一次。

------

### 3.6 中站干涉与探测（BS + POVM/Kraus 条件化）

对到达同一 bin (n) 的两臂 telecom 模式做 50/50 beamsplitter 干涉：对每偏振 (p), 
$$
\begin{pmatrix} d^{p}*{1,n}\ d^{p}*{2,n} \end{pmatrix}

\frac{1}{\sqrt{2}}
\begin{pmatrix}
1 & 1\
1 & -1
\end{pmatrix}
\begin{pmatrix}
c^{p}*{A,n}\
c^{p}*{B,n}
\end{pmatrix}.
$$
然后接 PBS/偏振分析把 $d^{H/V}_{k,n}$ 路由到不同探测器模式（可抽象为一个固定的线性光学路由门）。

探测用 on/off POVM（单光子截断下最简版）：效率 $\eta_d$、暗计数 $p_{\rm dc}$。成功事件集合 $\mathcal{S}$ 由你定义（例如“两个不同输出端、互补偏振的双击”对应部分 Bell 测量的一个成功签名）。条件化原子态：
$$
p_{\rm succ}=\Pr(\mathcal{S}),
\qquad
\rho_{AB|\rm succ}=
\frac{\operatorname{Tr}*{\rm field}!\big[M*{\rm succ}\rho_{\rm tot}M_{\rm succ}^\dagger\big]}
{p_{\rm succ}}.
$$
------

## 4. MPS 与“门如何更新张量”的执行细节（不写代码但写清楚算法学步骤）

### 4.1 MPS 记号与存储

把全纯态写为
$$
|\Psi\rangle=\sum_{{s_k}} A^{[1]s_1}A^{[2]s_2}\cdots A^{[L]s_L},|s_1s_2\cdots s_L\rangle,
$$
其中第 (k) 个站点的局域基标号是 $s_k$$维度 $d_k$$，张量 $A^{[k]}$ 的指标是 $(\alpha_{k-1},s_k,\alpha_k)$，$\alpha$ 的最大维度就是 bond dimension $\chi$。

### 4.2 施加两站点幺正门（TEBD 核心）

对相邻站点 (k,k+1) 的两体门 (U)$维度 $d_kd_{k+1}$$，TEBD 标准更新：

1. 取出相邻两张量与中间奇异值（若用 canonical form），形成两站点态张量 $\Theta$。
2. 用 (U) 作用在物理指标上：$\Theta'=(U\cdot \Theta)$。
3. 把 $\Theta'$ reshape 成矩阵，做 SVD：$\Theta' = USV^\dagger$。
4. 截断保留最大 $\chi$ 个奇异值（或按误差阈值），回写成新的 $A^{[k]},A^{[k+1]}$。

这一步就是“每一步只处理很小的局域 Hilbert 空间，SVD 截断控制成本”的实现本体。

### 4.3 施加局域非幺正信道（轨迹法的“门更新”）

对局域 Kraus 集 ${K_\mu}$（单站点或两站点），轨迹法每次只选取一个结果：

1. 对每个 $\mu$ 计算权重
$$
   p_\mu=\langle\Psi|K_\mu^\dagger K_\mu|\Psi\rangle.
$$
2. 抽样一个 $\mu$$或在你做“条件化成功事件”时强制选择某些 $\mu$ 并用权重补偿$。
3. 更新并归一化：
$$
   |\Psi\rangle \leftarrow \frac{K_\mu|\Psi\rangle}{\sqrt{p_\mu}}.
$$
4. 用和 §4.2 类似的张量更新$把 $K_\mu$ 当作一个“非幺正门”作用在局域物理指标上，再 SVD 截断$。

这套机制同时覆盖：损耗、腔损耗、原子退相干（跳算符）、以及探测点击/不点击等。

------

## 5. 端到端“核心算法流程”（一条可直接照着写程序的执行顺序）

下面我用“单次轨迹”的执行顺序写清楚“怎么组装”。统计量$$p_{\rm succ},F$ 等$通过多次轨迹平均得到。

### 5.1 预处理：把所有门工厂化

**输入：**参数集$发射 $\gamma(t)$、$\Alpha^{(\mathrm{emit})}$、QFC $\theta_p,\eta_{\rm ins}^{(p)}$、滤波 $\kappa_f,\Delta_f$、光纤 $\eta(L),U_j,\Delta\tau$、探测 $\eta_d,p_{\rm dc}$…$

在程序里对应的“静态数据结构”通常是：

- 时间网格：${t_n}_{n=1}^N$
- 每步门：
  - $U^{(\mathrm{emit})}*{A,n},U^{(\mathrm{emit})}*{B,n}$$依赖 $\gamma(t_n)$$
  - $U^{(\mathrm{QFC})}*{A},U^{(\mathrm{QFC})}*{B}$（常数或缓慢变）
  - $U^{(\mathrm{filt})}$（常数）
  - $U^{(\mathrm{pol/PMD})}$（可能依赖 bin 或随机抽样）
  - $U^{(\mathrm{BS})}$（常数）
  - 探测 Kraus（常数）
  - 损耗 Kraus（常数，可能偏振相关）

并明确每个门的“作用站点集合”（单站点/两站点）。

### 5.2 初始化：MPS 的站点与初态

- system 站点初态：两原子都在 $|e\rangle$ 或你定义的发射准备态（按你的发射序列），滤波腔模在真空。
- 所有 bin 站点初态：真空态 $|{\rm vac}\rangle$（在 5 维压缩基下就是基态）。

MPS 初始 bond dimension 通常为 1（完全乘积态）。

### 5.3 主循环：对 $n=1\ldots N$ 扫描

以下描述对应我们推荐的链排列
$S-A_1-B_1-A_2-B_2-\cdots$。

**Step n-A：让 system 与 $A_n$ 完成“发射→QFC→滤波/损耗/偏振”**

1. 在相邻站点 (S) 与 $A_n$ 上施加发射门 $U^{(\mathrm{emit})}_{A,n}$。
2. 在 $A_n$ 上施加 QFC 单站点门 $U^{(\mathrm{QFC})}_{A}$。
3. 在 (S) 与 $A_n$ 上施加滤波门 $U^{(\mathrm{filt})}_{A,n}$（若你采用等效腔模）。
4. 在 $A_n$ 上施加损耗/噪声信道（轨迹 Kraus）：插入损耗、（可选）把残留 780 分量视为立即丢失、光纤衰减、PDL 等。
5. 在 $A_n$ 上施加偏振 Jones 门 $U^{(\mathrm{pol})}_A$。必要时随后再施加 PMD 的跨 bin 移位网络（见 Step n-PMD）。

**Step n-Shift1：交换 $S\leftrightarrow A_n$**
这是“conveyor belt”的机械步骤：把 system 往右移动一格，使其接下来邻接 $B_n$。

**Step n-B：对 $B_n$ 重复同样流程**

对 (S) 与 $B_n$ 重复 1)–5)（把 (A) 换成 (B)）。

**Step n-Shift2：交换 $S\leftrightarrow B_n$**
此时链的局部顺序变成：
$$
\cdots - A_n - B_n - S - A_{n+1}-B_{n+1}-\cdots
$$
于是 $A_n$ 与 $B_n$ 现在相邻，可直接做中站干涉。

**Step n-PMD（若启用且需跨 bin）：**
如果你的 PMD 需要把某个偏振分量跨多个 bin 移位，你在每个 n 之后（或每隔若干步）对整条链上对应臂的 bins 执行一次“PSP 基变换 + SWAP 链置换 + 逆变换”。这在实现上是**一串相邻 bin 的两站点门**。

> 实务建议：第一版先把 PMD 限定为很短的记忆（(m=1) 或 (m=0)），这样你每步只需要对 $A_{n-1}\leftrightarrow A_n$（以及 B 臂同理）做一次局域 SWAP/混合门即可。

**Step n-BSM：在 $A_n,B_n$ 上做 BS 干涉与探测**

1. 在相邻站点 $A_n,B_n$ 上施加 50/50 BS 门（对 H、V 两偏振分别实现），得到输出模式 $d_{1,n}^p,d_{2,n}^p$。
2. 施加 PBS/偏振分析路由门（可并入探测模型或并入一个固定线性光学门）。
3. 对探测器模式施加 on/off 探测 Kraus（点击/不点击），得到本 bin 的点击记录 $\mathbf{r}*n$（比如 4 个探测器各 0/1）。同时累计整段时间窗的记录 $\mathbf{r}*{1:N}$。

**Step n-Discard：把 $A_n,B_n$ 从网络里“丢弃”以保持线性复杂度（可选但强烈建议）**

由于 $A_n,B_n$ 已经被测量（轨迹法下它们被投影到确定的后测态，通常是 vacuum 或已知局域态），你可以：

- 在张量网络层面把它们与 system 的纠缠切断（SVD 截断往往自动实现），
- 或在实现上直接把“处理完的 bins”从 MPS 左端弹出（滑动窗口），只保留还会与系统发生非马尔可夫耦合的那一段$比如滤波腔记忆导致只需保留最近 $N_{\rm mem}$ 个 bins$。

这一步不是物理模块，但决定你算得动的关键。

### 5.4 一条轨迹结束：从点击记录映射到“成功事件”与原子态

1. 判定是否成功：$\mathbf{r}_{1:N}\in\mathcal{S}$（你定义的符合模式集合）。
2. 若成功：取此轨迹终态的 system 部分（两原子 + 腔模），对腔模/内部自由度做偏迹或投影到真空（取决于你是否关心滤波腔末态），得到 $\rho_{AB}^{(\ell)}$$第 $\ell$ 条成功轨迹的原子态$。
3. 若失败：丢弃。

### 5.5 统计输出（多轨迹平均）

跑 $N_{\rm traj}$ 条轨迹：

- 成功概率估计：
$$
  p_{\rm succ}\approx \frac{N_{\rm succ}}{N_{\rm traj}}.
$$
- 条件原子态：
$$
  \rho_{AB|\rm succ}\approx \frac{1}{N_{\rm succ}}\sum_{\ell\in\mathrm{succ}}\rho_{AB}^{(\ell)}.
$$
- 保真度：
$$
  F=\langle\Phi_{\rm target}|\rho_{AB|\rm succ}|\Phi_{\rm target}\rangle.
$$
并建议同时输出诊断量：模式重叠/不可区分性
$$
\mathcal{M}=\left|\sum_n \xi_{A,n}^\dagger \xi_{B,n}\right|^2
$$
（在“单光子纯态波包”近似下），用于区分“偏振/波包失配毁了你”还是“噪声假阳性毁了你”。

------

## 6. 容易被遗漏但必须在“大图纸”里明确放置的仿真项

### 6.1 原子退相干（建议用轨迹或等效通道）

常见三类跳算符：自由空间散射、纯退相干、比特翻转等。
在本大图纸里它的位置是：

- 在“发射完成后到符合判定前”的等待时间上（包括光纤传播、电子学延迟），对两原子施加一段时间 $\tau$ 的噪声通道 $\mathcal{E}_{\rm atom}(\tau)$。
- 如果你用轨迹法：在每个 $\Delta t$ 步都对 system 施加一次原子跳算符抽样（与 bins 无关，是 system 的局域噪声门）。

### 6.2 绝对传播时延与“是否需要真的把 10–50 km 做成 1000 sites”

在我们当前的 A/C 版本里，光纤主要作为**线性信道**作用在场上：衰减、偏振旋转、（可选）色散/PMD。对于单向无反射链路，这些都可以作为信道直接作用于每个 bin，而不必把光纤空间段显式塞进哈密顿量。

“绝对时延”只通过原子等待退相干体现（见 §6.1）。如果你未来做时间复用/多轮并行（你之前说的方向 B），才需要显式维护更长的 time-bin 缓冲与更复杂的时间判定。

### 6.3 探测门宽、符合窗与“点击记录到成功事件”的映射

你在程序里需要把下面三件事显式参数化，否则结果会有歧义：

- bin 宽度 $\Delta t$（探测时间分辨率/数值分辨率）
- 探测门宽 $T_{\rm gate}=N\Delta t$
- 符合判据$例如“两个点击必须落在同一 bin”或“允许 $|n_1-n_2|\le w$”$

这些直接决定 $\mathcal{S}$ 的定义，从而决定 $p_{\rm succ}$ 与 (F)。

------

## 7. 数据结构草图（不写代码，但把“对象与字段”写清楚）

为了能按上面流程实现，程序层面至少要有以下几类结构（你可以用 class/struct/字典实现）：

### 7.1 参数配置类（只存参数）

- **EmissionParams**：$\Delta t$（dt\_ns）、(N)（n\_bins）、$t_n$
- **EmitParams**：$\gamma_j(t_n)$, $\Alpha^{(\mathrm{emit})}$, $\beta_\pm$, $\phi_j$
- **QFCParams**：$\theta_H,\theta_V$, $\eta_{\rm ins}^{(H/V)}$,（可选）噪声等效计数率
- **FilterParams**：$\kappa_f,\Delta_f,n_{\max}^{(a)}$
- **FiberParams**：$\eta(L)$, $U_A,U_B$, PMD 参数 $\Delta\tau$, PSP 旋转 $R_{\rm in/out}$
- **DetParams**：$\eta_d,p_{\rm dc}$, 成功事件集合 $\mathcal{S}$, 符合窗参数

### 7.2 局域算符/门工厂（把参数变成矩阵或 Kraus 集）

- **LocalBasisSpec**：每类站点的基（例如 5 维压缩基的基标号与物理含义）
- **OpLibrary**：在该基下的 $b^p,c^p,S_{j,\pm}$ 等矩阵表示
- **GateFactory**：
  - 生成 $U^{(\mathrm{emit})}*{j,n}$, $U^{(\mathrm{QFC})}*{j}$, $U^{(\mathrm{filt})}$
  - 生成 polarization 门、BS 门
  - 生成损耗/探测 Kraus 集

### 7.3 MPSState（状态容器）

- 站点张量列表 ${A^{[k]}}_{k=1}^L$
- bond dimension 管理$$\chi_{\max}$ 或截断阈值$
- 基本操作：
  - apply_one_site_gate
  - apply_two_site_gate
  - swap_neighbor_sites（用于 conveyor belt 和 PMD shift）

### 7.4 TrajectoryRecord（轨迹记录）

- 每个 (n) 的点击向量 $\mathbf{r}_n$
- 判定函数：$\mathbf{r}_{1:N}\in\mathcal{S}$ ?
- 若成功：存原子末态 $\rho_{AB}^{(\ell)}$

------

## 8. “考虑是否全面”的核对清单（按物理块逐项对齐）

你可以用下面 checklist 对照实现是否漏项：

1. 源端是否包含：
   - $S_{j,\pm}$ 两通道；$\gamma(t)$ 波包整形；
   - $\Alpha^{(\mathrm{emit})}$（几何/对准/模式匹配）；$\beta_\pm$ 通道不平衡；相位 $\phi_j$。
2. QFC 是否包含：
   - 转换角 $\theta_p$（可偏振相关）；插入损耗/PDL；噪声等效背景。
3. 滤波是否包含：
   - 等效腔模 + $\kappa_f,\Delta_f$ 的记忆；腔损耗。
4. 光纤是否包含：
   - 衰减（幅度阻尼）；Jones 旋转；PMD（跨 bin 移位网络）。
5. 中站是否包含：
   - A/B 干涉 BS；PBS/分析；探测效率与暗计数；成功事件集合 $\mathcal{S}$；输出 $p_{\rm succ},\rho_{AB|\rm succ},F$。
6. 非幺正过程实现路径是否一致：
   - 你选的是“纯态 MPS + 轨迹”还是“显式环境纯化”或“密度矩阵 MPO”；三者不要混乱。我们之前建议“损耗 beamsplitter 纯化 + 原子退相干轨迹”是平衡组合，但若你把探测也做成轨迹，通常顺带把损耗也做成轨迹会更统一。

------

## 9. 你接下来最应该先落地的“最小可跑版本”（不丢主物理）

为了你能尽快把这张图纸变成可运行程序，我建议最小可跑版本的取舍是：

- bin 维度：每臂 5 维压缩基（vac + 4 单光子态）
- QFC 噪声：先折算为等效背景点击概率$并入 $p_{\rm dc}$ 或单独统计$
- PMD：先做整数 bin 移位 $m\in{0,1}$ 的版本（确保你“跨 bin 门”的框架跑通）
- 滤波：一个等效腔模（每臂每偏振一个），截断到 $n_{\max}^{(a)}=1$ 或 2（看你是否需要腔内多光子）
- 统计：用轨迹法估 $p_{\rm succ}$ 与 (F)（先让端到端闭环再谈收敛优化）

这样你就能完整跑通：**发射$含 $\Alpha^{(\mathrm{emit})}$$→ QFC → 滤波记忆 → 光纤偏振/PMD → 中站 BSM → 探测条件化 → 输出 $p_{\rm succ},F$**，并且每一步都对应到上面定义的局域门/信道，没有“口头上说了但程序里找不到落点”的部分。

------

如果你希望我把这份“图纸”再进一步压成一张“步骤-站点-门类型”的执行表（例如每个 (n) 具体要调用哪些门、哪些是 one-site、哪些是 two-site、哪些是跨 bin SWAP 序列，且把 A/B 两臂并行流程画成严格的时序），我也可以在不写代码的前提下给出那种“实现者视角”的逐行流水表。
