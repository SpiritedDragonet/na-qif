# 第一章国内外研究现状补充设计

### 设计目标

评阅意见要求补充本方向国内外研究现状。当前第一章已经包含量子计算、模块化体系结构、中性原子量子接口和端到端链路瓶颈的背景材料，但这些内容分散在绪论开头、可扩展性压力、中性原子链路选型和性能瓶颈分析中。修改目标是让评审在第一章中清楚看到一个显式的“国内外研究现状”小节，并且让该小节自然导向本文选择的中性原子远程纠缠链路与端到端建模问题。

新增内容应服务论文主线。写法上采用完整段落和连贯散文，正文中少用项目式罗列。需要呈现的研究现状包括可扩展量子计算的系统压力、模块化/分布式体系结构、原子--光子量子接口、量子频率转换、量子存储网络、远程宣告纠缠和端到端误差预算。国内研究部分需要从综述性判断推进到代表性实验，体现国内在量子通信网络、量子存储网络、原子--光子关联和中性原子控制方向的具体进展。

### 当前第一章诊断

第一章开头目前以量子计算与量子接口的重要性切入，随后进入“量子计算简介”。这种结构呈现了背景介绍，显式研究现状标题仍需补上。`Li2025QIT` 与 `Li2023QNS` 已经提供国内综述支撑，`vanleent2020longdistance`、`vanleent2022entangling`、`zhou2024longlived`、`Hartung2024QuantumNetworkRegisterTweezerCavity` 等文献支撑国际原子接口链路，国内代表性实验结果适合在第一章中集中补出。

`figures/quantum_figs_pack` 原本服务第一章图一“可扩展量子计算的系统压力”。包内材料覆盖控制扇出、低延迟纠错、逻辑纠错运行、闭环标定和光子互连等方向。当前图一正文主要引用综述和体系结构文献，包内这些更具体的论文适合补入图一附近叙述。它们可作为“系统压力已经在近期硬件、控制和纠错工作中具体显现”的证据链。

### 正文结构方案

建议在绪论开头三段之后、现有 `\section{量子计算简介}` 之前插入新的 `\section{国内外研究现状}`。新增小节篇幅控制在约 1200 到 1800 个中文字符，采用四到五个自然段。这样既回应评阅意见，也保持后续“量子计算简介”“中性原子量子接口链路”“端到端性能瓶颈分析”的章节功能。

第一段写国际可扩展量子计算研究从单器件指标走向系统运行指标。这里应把控制线规模、读出吞吐、实时译码、逻辑纠错运行和闭环标定写成一条工程演进脉络，并在段末引出图一。该段可以引用 Zhao、Ahmad、Dai、Caune、Google Quantum AI 与 Vepsäläinen 等工作，说明图一的四个压力维度来自真实硬件研究中的具体问题。

第二段写模块化与分布式体系结构。该段应从“单处理器扩展压力”过渡到“多个可维护处理模块通过可宣告光子互连协同”。这里可引用 Monroe、Jiang、Wehner、Covey、Sinclair 和 Main 等工作。Main 2025 的两模块光网络量子计算实验适合放在段末，作为远程纠缠资源进入分布式计算流程的近期代表。

第三段写国际原子--光子量子接口。该段应围绕腔增强原子节点、量子频率转换、电信光纤和远程 BSM 展开，保留与本文链路直接相关的 van Leent 2020、van Leent 2022、Zhou 2024、Hartung 2024 和 Reiserer 2022。这里的重点是交代本文建模对象来自一条已有实验基础且仍存在端到端误差预算需求的链路。

第四段写国内研究现状。该段应先用 `Li2023QNS` 和 `Li2025QIT` 概括国内量子信息与量子网络研究版图，再用 Chen 2021 的 4600 km 空天地一体量子通信网络说明国内量子通信工程化基础。随后转向与本文更贴近的量子存储和原子--光子接口：Yu 2020 的几十公里光纤量子存储纠缠、Liu 2024 的城域多节点 memory--memory entanglement、Zhang 2024 的 12 km 光纤 heralded atom-photon quantum correlation 与多模复用。最后可用国内中性原子阵列控制工作作为计算节点侧的补充，形成“网络基础、存储节点、接口链路、处理节点”四个层面的国内现状。

第五段作为收束段，直接落到论文问题。该段应写成事实推进：已有实验展示了链路组件和节点能力，系统级评估需要把发射、QFC、光纤传播、中继站干涉、探测噪声和条件化测量统一到端到端模型中。这个收束自然引出后续第一章已有的中性原子量子接口链路和端到端性能瓶颈分析。

### 图一包文献补入计划

图一附近的正文建议补入以下文献。每篇文献只承担一个清晰角色，避免让图一变成泛泛文献堆叠。

| 图一语义位置 | 建议补入文献 | 正文作用 |
| --- | --- | --- |
| 物理比特扩容与控制扇出 | Zhao 2024 row-column addressing；Ahmad 2022 scalable cryoelectronics | 说明大规模处理器面临控制线数量、低温布线、控制电子学下沉和共享寻址等系统问题。 |
| 串扰与标定成本 | Dai 2021 flux crosstalk | 说明规模化控制带来串扰矩阵和自动化标定需求。 |
| 并行读出反馈与译码延迟 | Caune 2024 low-latency QEC | 说明纠错循环要求 syndrome 数据、经典译码和反馈路径达到微秒量级吞吐。 |
| 长时纠错运行 | Google Quantum AI 2025 below-threshold surface code | 说明逻辑错误率随码距扩展而下降已进入实验验证阶段，长期运行与实时解码成为核心指标。 |
| 标定闭环维护 | Vepsäläinen 2022 closed-loop feedback | 说明参数漂移可以通过连续探测、估计和回写形成闭环维护问题。 |
| 模块化处理节点与光子互连 | Main 2025 distributed quantum computing across an optical network link | 说明可宣告远程纠缠可以被调度进跨模块量子计算流程。 |

图一正文可在现有第 34 行附近加强一段，把“控制扇出、综合征吞吐、逻辑寿命预算、漂移追踪”分别绑定到这些文献。图题保持当前解释性功能，正文承担引用和叙事功能。

### 国内研究补充计划

国内研究现状建议补入以下代表线索。它们和本文的关系由远到近推进，最终落在原子--光子链路和端到端建模。

| 研究线索 | 建议文献 | 正文作用 |
| --- | --- | --- |
| 国内量子信息与量子网络综述 | `Li2025QIT`；`Li2023QNS` | 提供国内领域版图和关键技术分类。 |
| 大尺度量子通信工程基础 | Chen 2021 integrated space-to-ground quantum communication network | 说明国内已形成长距离量子通信网络和实地链路运行基础。 |
| 量子存储远程纠缠 | Yu 2020 entanglement of two quantum memories via fibres over dozens of kilometres | 说明国内原子系综量子存储、QFC 和城域光纤链路的早期代表结果。 |
| 城域多节点量子存储网络 | `Liu2024MemoryMemoryEntanglementMetropolitanNetwork` | 说明国内 memory--memory entanglement 已推进到城域多节点网络。 |
| 原子--光子关联与多模复用 | Zhang 2024 heralded atom-photon quantum correlation over 12 km fiber | 说明国内在 Rb 原子系综、时间分箱多模复用、QFC 和 heralding 速率方面已有直接相关进展。 |
| 中性原子阵列控制 | Li 2025 fiber array architecture for atom quantum computing | 说明国内处理节点侧也在推进中性原子阵列的独立寻址和并行控制。 |

Zhang 2024 与 Yu 2020 很适合进入正文，因为它们都同时触及量子存储、光纤距离、QFC 或 heralding。Chen 2021 更适合作为国内量子通信网络工程化背景，篇幅控制在一句。Li 2025 的定位应放在中性原子计算节点控制侧，并且需要先核对现有 `reference.bib` 条目的题名、作者和 DOI。

### 参考文献处理

新增或核对的 bib 条目建议采用稳定、可读的 key。图一包相关条目可命名为 `Zhao2024RowColumnAddressing`、`Ahmad2022ScalableCryoelectronics`、`Dai2021FluxCrosstalk`、`Caune2024LowLatencyQEC`、`GoogleQuantumAI2025BelowThreshold`、`Vepsalainen2022ClosedLoopFeedback` 和 `Main2025DistributedQuantumComputing`。国内研究相关条目可命名为 `Chen2021SpaceGroundQuantumNetwork`、`Yu2020EntanglementQuantumMemoriesFiber`、`Zhang2024HeraldedAtomPhoton12km`，并复用已有的 `Liu2024MemoryMemoryEntanglementMetropolitanNetwork`、`Li2023QNS` 和 `Li2025QIT`。

现有 `Li2025FiberArrayArchitectureAtomQC` 条目需要核对。已查到的 Nature Communications 页面显示题名为 “A fiber array architecture for atom quantum computing”，DOI 为 `10.1038/s41467-025-64738-8`，作者列表以 Xiao Li、Jia-Yi Hou、Jia-Chao Wang 等开头。当前 `reference.bib` 中同 key 的题名、作者和 DOI 与该页面存在差异，执行正文修改前应先决定采用哪一篇中性原子阵列控制文献。

### 写作约束

正文新增段落遵循 AGENTS.md 的 Writing Style。段落应以完整句子推进，少用加粗、斜体和列表。正文中只使用必要的 `\cite{}`、行内术语和公式符号。每句话都应能独立成立，避免通过先否定一个说法再提出另一个说法来完成表达。涉及论文主线时，优先直接陈述研究对象和建模需求，减少“本文要做什么”的自指性表述。

建议正文措辞保持事实链条。例如可以写“这些实验进展表明，量子接口链路的评价已经从单个器件效率推进到端到端资源质量、宣告统计和条件态保真度的联合评估。” 这种句式能直接连接研究现状与论文问题。

### 验收标准

完成修改后，第一章应出现明确的 `国内外研究现状` 小节。该小节应同时包含国际与国内进展，并且国内部分包含至少三条代表性实验线索。图一附近正文应吸收 `quantum_figs_pack` 中与图一含义对应的论文，使图一的四个压力维度具备具体文献支撑。新增文字应与后文中性原子接口链路、端到端性能瓶颈和本文研究内容形成连续叙事。

编译后需要检查参考文献条目、引用格式和图表位置。新增内容完成后，应再次对照 `评阅意见修改建议汇总.md` 中第一项，确认该项可以被标记为已处理。

### 已核对来源

本设计参考了本地 `figures/quantum_figs_pack/notes.md`、当前 `body/chapter1.tex` 和 `reference.bib`。外部来源包括 Nature 的 Google Quantum AI 2025 below-threshold surface code 页面、Main 2025 distributed quantum computing 页面、Zhang 2024 heralded atom-photon quantum correlation 页面、Yu 2020 quantum memories 页面、Liu 2024 metropolitan quantum network 页面、Chen 2021 space-to-ground quantum network 页面、Vepsäläinen 2022 closed-loop feedback 页面，以及 Zhao 2024、Dai 2021、Caune 2024、Ahmad 2022 的论文页面或本地 PDF。
