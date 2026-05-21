量子架构替换图下载包

我按你那张总图的五个外框，逐一做了视觉匹配检查。这里给出最接近的原始图文件，以及我建议你在排版时优先取用的子图。

1. 左上：物理比特扩容 / 控制扇出
   首选：zhao_2024_fig1.png
   这张图直接包含：
   - 室温控制电子学到稀释制冷机的层级连接
   - 独立控制 O(N) 与行列寻址 O(sqrt(N)) 的对比
   - 右侧有很明确的 row/column addressing 结构
   视觉判断：和你左上框的“控制扇出 + 规模化布线”最贴近。

   备选：ahmad_2022_scalable_cryoelectronics.pdf
   建议查看 Figure 1 的 panel (c)。
   视觉判断：更像“集成控制/读出下沉到低温端”的体系结构图，适合替换你左上框里偏“架构级”而不是“矩阵级”的部分。

   串扰矩阵 inset：
   dai_2021_flux_crosstalk.pdf 的 Figure 6（PDF 第 12 页附近）。
   视觉判断：这张最像你左上框右侧的小热图矩阵。

2. 右上：并行读出反馈 / syndrome throughput / decoder latency
   首选：caune_2024_fig1.png
   建议使用 panel (a) 的左半部分。
   这张图直接包含：
   - 实时 decoder
   - readout RX/TX 与 low-latency crossbar
   - 明确的纳秒级链路延迟标注
   视觉判断：语义非常准，尤其适合替掉“decoder + latency”这一块。

   补充：caune_2024_fig4.png
   这张图更适合做“full decoding response time”的补图或 inset。

3. 左下：长时纠错运行 / logical survival / correlated tail
   主图：google_2025_fig1.png
   建议使用 panel (c)。
   视觉判断：它就是逻辑错误概率随量子纠错持续时间增长的曲线，非常接近你左下框的主图角色。

   相关噪声尾部：google_2025_fig3.png
   建议使用 panel (b)。
   视觉判断：它正好展示 burst event 后 detection probability 的抬升与衰减，最接近你左下框“correlated tail”的意思。

   说明：
   这两张图是最贴语义的一组，但它们分别承担“主曲线”和“相关尾部”两个功能。
   如果你只替一张，优先用 google_2025_fig1.png。
   如果你允许主图 + inset 组合，这一组是最合适的。

4. 右下：标定闭环维护 / 漂移追踪 / writeback
   首选：vepsalainen_2022_fig1.png
   建议优先使用 panel (c)，必要时参考 panel (b)。
   视觉判断：
   - panel (c) 是很标准的反馈回路图
   - panel (b) 展示 probing / computation / correction 的循环
   它们和你右下框“estimate → update → writeback”非常接近。

5. 底部中间：模块化处理节点 + 可宣告光子互连
   首选：main_2025_fig1.png
   建议优先使用 panel (a)。
   视觉判断：这是所有候选里和你底部框最像的一张。
   它直接给出 photonically interconnected modules，并且是 heralded entanglement 的叙述方式。
   如果你想要模块内部网络/电路量子比特的结构说明，可再参考 panel (b)。

许可与使用提示

- Vepsäläinen 2022 与 Ahmad 2022 是开放获取里相对更容易复用的来源。
- Main 2025 也是开放获取来源，适合作为直接替换候选。
- Google 2025 这篇文章适合作为“原图参考”或整体引用；如果你后续要做大幅改版或重绘，建议再单独核对其 CC BY-NC-ND 许可边界。
- Zhao 2024 与 Caune 2024 目前更适合作为重绘参考，或者在你自行确认预印本许可之后再决定是否直接嵌入。

建议你现在的替换策略

- 左上用 Zhao Fig.1 主体 + Dai Fig.6 小热图 inset
- 右上用 Caune Fig.1(a)
- 左下用 Google Fig.1(c) 主图 + Google Fig.3(b) inset
- 右下用 Vepsäläinen Fig.1(c)
- 底部中间用 Main Fig.1(a)
