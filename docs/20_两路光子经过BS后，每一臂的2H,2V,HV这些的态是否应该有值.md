Q：话说两路光子经过BS后，每一臂的2H，2V HV这些的态应该有值嘛？

A：一般情况下，**有值，而且你需要把它们保留下来**。原因很简单：即使每个输入臂在某个 time‑bin 里最多只有 1 个光子$A_bin m 里 0/1 个，B_bin m 里 0/1 个$，经过 50:50 分束器后，**“一进一出”并不成立**，因为分束器把输入模式做了线性叠加，二光子项会产生“bunching”（两光子同端口）分量——这正是 HOM 干涉的本质。

下面我用你最关心的“每个 bin m”来写清楚。

一、分束器对算符的变换（同一个 bin m，偏振不耦合）

对每个偏振 $P\in{H,V}$ 和同一个时间 bin (m)，取最常用的 50:50 BS 约定：

$$
a^\dagger_{P,m} ;\mapsto; \frac{c^\dagger_{P,m}+d^\dagger_{P,m}}{\sqrt 2},\qquad
b^\dagger_{P,m} ;\mapsto; \frac{c^\dagger_{P,m}-d^\dagger_{P,m}}{\sqrt 2},
$$

其中 (a,b) 是输入臂$你原来的 A_bin m 与 B_bin m$，(c,d) 是输出端口$你说的 port1_bin m 与 port2_bin m；之后再通过 PBS 对应到 H1,V1,H2,V2$。

二、为什么会出现 (2H,2V,HV)：直接算给你看

1）同偏振输入：必然产生 (2H) 或 (2V)（HOM 完全 bunching）

以 (H) 为例（同一个 bin）：

$$
a^\dagger_{H,m} b^\dagger_{H,m},|0\rangle
;\mapsto;
\frac{1}{2}\Big[(c^\dagger_{H,m})^2-(d^\dagger_{H,m})^2\Big]|0\rangle.
$$

注意归一化二光子 Fock 态定义
$$
|2H\rangle_{c,m}=\frac{(c^\dagger_{H,m})^2}{\sqrt 2}|0\rangle,
$$
所以
$$
a^\dagger_{H,m} b^\dagger_{H,m}|0\rangle
;\mapsto;
\frac{1}{\sqrt 2}\Big(|2H\rangle_{c,m}|0\rangle_{d}-|0\rangle_{c}|2H\rangle_{d,m}\Big).
$$

结论：**输出端口 1 或 2 里出现 “2H” 的振幅是非零的**；而“一边一个”的 $c^\dagger d^\dagger$ 项在同偏振情况下会严格相消（HOM dip）。

同理 (V) 偏振会给出 (2V)。

2）正交偏振输入：会同时出现 “同端口 HV” 和 “分端口 HV（符合）”

例如 $a_H^\dagger b_V^\dagger$：

$$
a^\dagger_{H,m} b^\dagger_{V,m}|0\rangle
\mapsto
\frac{1}{2}\Big(
c^\dagger_{H,m}c^\dagger_{V,m}

- c^\dagger_{H,m}d^\dagger_{V,m}

- d^\dagger_{H,m}c^\dagger_{V,m}

- d^\dagger_{H,m}d^\dagger_{V,m}
  \Big)|0\rangle.
$$

把它翻译成你用的“局域基底语言”：

- $c^\dagger_H c^\dagger_V|0\rangle$ 是 **port1_bin m 的 (HV)**；
- $d^\dagger_H d^\dagger_V|0\rangle$ 是 **port2_bin m 的 (HV)**；
- $c^\dagger_H d^\dagger_V|0\rangle$ 是 **H1 与 V2 的跨端口符合**；
- $c^\dagger_V d^\dagger_H|0\rangle$ 是 **V1 与 H2 的跨端口符合**。

结论：**(HV) 在同一输出臂（同一 bin）当然可能出现，而且它和“跨端口符合”是同一个输入态展开出来的不同项**。

三、用 Bell 态看得更直观：哪些情况下只剩下 (HV) 或只剩下跨端口符合

考虑两光子偏振 Bell 态（同一时空模式、同一 bin 重叠时才有意义）：

$$
|\Psi^\pm\rangle = \frac{1}{\sqrt2}\left(a_H^\dagger b_V^\dagger \pm a_V^\dagger b_H^\dagger\right)|0\rangle.
$$

代入上面的 BS 变换可以得到标准结果：

- 反对称态 $ |\Psi^-\rangle $ 输出只含“跨端口一边一个”：
  $$
  |\Psi^-\rangle \mapsto \frac{1}{\sqrt2}\left(c_V^\dagger d_H^\dagger - c_H^\dagger d_V^\dagger\right)|0\rangle.
  $$
  也就是说只会给出你说的 **H1V2 或 V1H2** 这种跨端口符合（理想情况）。
- 对称态 $ |\Psi^+\rangle $ 输出只含“同端口 HV”：
  $$
  |\Psi^+\rangle \mapsto \frac{1}{\sqrt2}\left(c_H^\dagger c_V^\dagger - d_H^\dagger d_V^\dagger\right)|0\rangle,
  $$
  即只会给出 **H1V1 或 H2V2** 这种同端口（但不同探测器）符合。

这两条恰好对应你之前提到的两类点击组合。它也顺便解释了：为什么你必须保留 **每一臂的 (HV)** 这样的双光子局域基底。

四、那 (2H,2V) 在你的探测里“看得到吗”？（SNSPD 桶式的后果）

你说 SNSPD 是桶式、一个探测器不太可能对“双光子同模”给出两次 click。那：

- 如果两个光子都落到 **同一端口同一偏振**$比如 port1_bin m 的 (2H)$，经过 PBS 后它们会一起去 **H1**（同一个 SNSPD）。
  桶式探测器一般只给出 **一次 click**，于是你在“需要两路 coincidence”这类筛选中会把它当作 **不成对事件 / 失败事件 / 丢弃**。
- 但这不代表态里没有 (2H/2V)。它们在量子态层面是**真实存在的概率幅**，只是你的探测记录规则（桶式 + coincidence 条件）让它们多数不会被计入成功符合。

因此，从仿真角度总结就是：

1. **BS 后 (2H,2V,HV) 一般都应该是非零的**（只要存在同 bin 的两光子重叠并干涉，或态里含有对应分量）。
2. **你是否“在最终统计里看到它们”取决于探测器是否数分辨、以及你怎么做 coincidence / post‑selection。**
3. 如果你在 MPS 里把这些态砍掉，会直接破坏 HOM/bunching 物理，从而让 BSM 的成功/失败概率结构不对。

五、一个很实用的自检准则（看你代码是否 BS 做对了）

选一个最简单输入，在单一 bin 上测试：

- 输入 $|1_H\rangle_A|1_H\rangle_B$（同偏振、同 bin、完全重叠）
- 理论上输出应该 **完全没有** “一边一个”的项（例如 H1H2 的符合应当为 0），而是只剩下 $|2H\rangle$ 在 port1 或 port2 的叠加。

如果你数值上看到输出仍有明显的 $|H\rangle_{port1}\otimes|H\rangle_{port2}$（一边一个）概率，那通常说明：

- 你的 BS 变换在截断 Hilbert 空间里实现不对；或
- 你在实现时把 boson 的“同一模式两光子归一化 $\sqrt2$”处理错了；或
- 你把不同路径应该相加的振幅当成了概率去加（会杀掉干涉）。

结论：回答你的原问题

两路光子经过 BS 后，**每一臂（port1、port2）的 (2H)、(2V)、(HV) 这些局域态通常应该有非零振幅**。尤其是 (HV) 对应你能用 H/V 分析器在同一端口形成 (H1,V1) 或 (H2,V2) 的符合；而 (2H,2V) 则是 HOM bunching 的典型分量，只是桶式 SNSPD 往往不会把它们记录成“双击符合”，所以在“只看两探测器 coincidence”的统计里它们更像“丢失/失败”通道。

如果你愿意把你 BS 门的局域矩阵（你对 6×6 局域基底怎么写的）贴一小段，我可以直接帮你核对：在你的 ${\text{vac},H,V,2H,2V,HV}$ 截断下，它是否是一个正确的幺正、以及对 $|H\rangle_A|H\rangle_B$、$|H\rangle_A|V\rangle_B$ 等基准输入是否给出上面这些解析结果。