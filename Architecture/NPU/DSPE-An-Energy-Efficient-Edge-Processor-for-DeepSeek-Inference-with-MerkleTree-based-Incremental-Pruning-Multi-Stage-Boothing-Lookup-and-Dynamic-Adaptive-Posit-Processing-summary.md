# 深度分析：《DSPE: An Energy-Efficient Edge Processor for DeepSeek Inference with MerkleTree-based Incremental Pruning, Multi-Stage Boothing Lookup and Dynamic Adaptive Posit Processing》

## 基本信息

- **标题**：DSPE: An Energy-Efficient Edge Processor for DeepSeek Inference with MerkleTree-based Incremental Pruning, Multi-Stage Boothing Lookup and Dynamic Adaptive Posit Processing
- **文档类型**：论文（DAC'26 会议论文，7 页）
- **作者**：Yuhan Zhang、Zhou Wang（共同一作）、Zhou Shu、Jiuren Zhou、Yanqing Xu、Xiaonan Tang、Shushan Qiao、Tianchun Ye、Yang Liu、Anil A. Bharath、Emm Mic Drakakis
- **机构**：Northeastern University、Imperial College London、Imperial Global Singapore、Xidian University、CUHK Shenzhen、Wisemaytech、Institute of Microelectronics CAS、UCAS、Nanyang Technological University
- **发表 venue**：63rd ACM/IEEE Design Automation Conference（DAC '26），2026 年 7 月 26–29 日，Long Beach, CA, USA
- **年份**：2026
- **链接**：DOI 10.1145/3770743.3805813；arXiv:2605.08615v1 [cs.AR]

## 一句话总结

> DSPE 用三项技术（Merkle 树增量剪枝、多级 Booth 查找、动态自适应 Posit）在 TSMC 28 nm 上做了一个 8.23 mm²、122–345 mW 的 DeepSeek 边缘推理处理器，报告 109.4 TFLOPS/W@POSIT8 的峰值能效，是 H100@FP8 的 19.35×。

## 研究动机与问题定义

- **要解决的核心问题**：DeepSeek-V3 的推理效率虽好，但参数规模与 MoE 分布特性使它难以部署在能耗受限的边缘设备上。
- **现有方法的不足**：论文把边缘部署的困难归为三类（第 3 页 Figure 3）：（1）层间计算冗余——相邻 token 产生高度相似的 Q/K 向量与中间表示，造成重复的相似度评估与 KV 访问；（2）线性变换的能耗开销——大量重复的大规模线性变换、注意力计算与路由决策造成信号翻转与存储功耗；（3）大规模矩阵运算的高延迟——同一权重乘以多组激活时，各组的 Booth 编码差异大，导致大量位翻转与能耗上升。
- **本文的切入角度**：不做静态稀疏假设，而是按输入数据特征、层类型与分布式负载实时调整计算过程（第 1 页）。三项技术分别对应上述三类问题。

## 核心方法

### 方法概述

DSPE 的整体架构包含 4 个 Attention Core，每个含 64 个 PE core，数据通路中有 iRouter 与 oRouter 负责把激活与权重分配到目标 PE（第 3 页 Figure 4）。片上还包括 Input Buffer、Output Buffer、24 KB Parameter Buffer、48 KB Weight Buffer，以及两个 48 KB 的 Q/K 专用 SRAM 与一个 48 KB 的 V 专用 SRAM。Top Controller 集中调度数据流与访存，并按调度策略分块加载不同层的参数以做流水化权重复用，Weight Buffer 配合 iRouter 在不同 Attention Core 之间广播共享权重张量。

### 关键技术细节

**MIPS：MerkleTree-based Incremental Pruning Scheme**（第 3 页 Figure 5）

三级流水线：相似度重排 → Merkle 树早期判定 → 动态复用。

- 软件层维护一个 Sequential Incremental Sorter，新生成的向量按与当前序列中已有向量的余弦相似度插入到增量有序位置，所有相似度分数缓存在 Cos-SRAM 中。
- 重排后的向量经轻量 MAC 投影到紧凑语义空间（$V_{low} = MAC(V_{reordered})$），再用局部敏感哈希计算叶节点，自底向上构建 Merkle 树。第 $i$ 层的中间节点哈希记为 $H_{cur}(i)$，与同一专家处理过的历史输入在同层的哈希 $H_{ref,j}(i)$ 比较，差值 $\Delta H_j(i) = |H_{cur}(i) - H_{ref,j}(i)|$。
- 两个阈值 $T_{zero}$ 与 $S_{th}$ 决定三种决策：**Early-Skip**（$\Delta H_j(i) \le T_{zero}$，直接复用对应 KV Cache 索引与中间结果，跳过本轮）、**Diff-Reuse**（$T_{zero} < \Delta H_j(i) \le S_{th}$ 且在 History-LUT 中命中，直接查表复用结果并终止更高层构建）、**Full-Compute**（超出 $S_{th}$ 或未命中，视为新模式，继续构建到根哈希并把 $\Delta H$ 与结果写回 History-LUT）。
- 论文提到在硬件侧预留统计接口，系统软件可在离线后台模式重算完整根哈希，统计各类早期决策与根节点判定的一致性。

**MBLM：Multi-Stage Boothing Lookup Method**（第 4 页 Figure 6）

面向「多乘数 × 同一被乘数」的时序局部性，由三部分组成：

- **无效计算检测器**：一次取 8 个乘法操作数，设置近零阈值 $R_{zero\_wgt}$ 与 $R_{zero\_act}$，凡 $|w| < R_{zero\_wgt}$ 或 $|a| < R_{zero\_act}$ 的乘法对直接标记为无效并跳过。
- **Booth 贝叶斯网络**：用相邻 token 乘法请求的位相似度 BS 与重复片段长度（Repeat Length）构成冗余特征，$BS = 1 - BV/8$（$BV$ 为相邻乘法请求之间的位翻转数）。BN 输出 $P(R=\text{Low})$ 与 $P(R=\text{High})$，冗余分数为 $r_L \cdot P(R=\text{Low}) + r_H \cdot P(R=\text{High})$（式 5）。分数低于 0.8 走 radix-4 Booth 常规路径，高于 0.8 切到 radix-8 Booth 扩展路径以减少部分积。
- **统计表简化与查找复用**：累积任意两个被乘数之间的 $BV$ 统计构建 8×8 Bit-Variation Matrix，剔除交换对重复计数（A/B 与 B/A）与自比较无效计数（A 与 A），有效统计区简化为 Variation Simplified Triangle（VST）。每个 VST 配一个小的 Booth-LUT 记录上次执行的位翻转模式与序号，新乘法对若位翻转数低于完全匹配阈值则跳过 Booth 编码与部分积生成，由比较器选择执行顺序。

**DAPPM：Dynamic Adaptive Posit Processing Mechanism**（第 5 页 Figure 7）

- **DA-Posit 数值格式**：借鉴 Posit 的 regime（编码有效指数范围），把指数与尾数视为可重构的 dynamic precision field（Dyn-field）。低位高度相关或冗余时做结构化合并压缩，例如 8-bit Posit 在指数与尾数末 2 位或末 1 位相同时折叠为共享位；超低精度配置下还可通过末位折叠再省 1 bit。论文称这些操作只影响低位、可无损还原，且复用极少数边界 regime 码扩展为「scale + mode」联合映射，区分「不压缩 / 1-bit 压缩 / 2-bit 压缩」三种模式而不增加编码位宽。
- **硬件路径**：DA-Posit decoder 解出 sign、regime、$(k,e)$ 与 Dyn-field，$(k,e)$ 合成为复合指数 $E = k \cdot 2^{es} + e$（式 6）。随后按 mode 值（0/1/2）选择 16、9 或 4 个 Array Multiplier PE 的计算配置，部分积经多级 CSA Tree 与 CPA 得到结果并归一化。scale 路径先检查归一化结果是否落在预设范围 (0, 2) 内，不在则做补偿修正。

### 核心创新点

1. 把 Merkle 树从数据完整性校验用途转用到推理冗余判定，并给出 Early-Skip / Diff-Reuse / Full-Compute 三级决策。
2. 用贝叶斯网络按位翻转特征动态在 radix-4 与 radix-8 Booth 之间切换，并配 Booth-LUT 做重复检测。
3. 提出 DA-Posit 格式，把 Posit 的 exponent 与 fraction 边界做成可重构字段以实现低位结构化压缩。

### 与现有方法的关键区别

论文强调自身相对静态权重剪枝的差别：MBLM 做的是权重与激活的联合近似计算（joint weight–activation approximate computing），通过冗余分数分类器指导操作数重排、自适应 Booth 编码与表重放，在降低位翻转能耗与冗余计算上优于静态 weight-only pruning（第 6 页）。

## 实验与结果

### 实验设置

- **工作负载**：DeepSeek-V3 在 MMLU 数据集上的推理。
- **实现流程**：Verilog 实现 → Synopsys Design Compiler 综合 → 28 nm CMOS 工艺下用 IC Compiler 做布局布线。
- **评估指标**：面积、电压范围、功耗、频率、性能（TFLOPS）、能效（TFLOPS/W）。

### 主要结果

**实现结果**（第 6 页）：4 个 PE 阵列（Attention Core），每个阵列 64 个 PE core；总面积 8.23 mm²；供电 0.6–1.10 V；功耗 122–345 mW；最高频率 710 MHz（1.10 V），此时达到 22.8 TFLOPS@POSIT8；在 0.6 V、200 MHz 下达到峰值能效 109.4 TFLOPS/W@POSIT8。

**与其它处理器对比**（Table 1，第 6 页）：

| 项 | GPU H100 | ISSCC'23 [6] | ISSCC'23 [7] | VLSI'24 [8] | 本文 |
| --- | --- | --- | --- | --- | --- |
| 工艺 (nm) | 4 | 12 | 28 | 22 | 28 |
| 面积 (mm²) | 814 | 4.6 | 14.36 | 6.4 | 8.23 |
| 电压 (V) | NA | 0.62–1.0 | 0.6–1.0 | 0.6–1.0 | 0.6–1.1 |
| 频率 (MHz) | 1620 | 77–717 | 85–275 | 115–495 | 200–710 |
| 精度 | FP64/32/8, INT32/8/4 | FP8/4 | INT16/8 | BF16/FP8 | POSIT8, INT8/4 |
| 功耗 (mW) | 700000 | 10–122@FP8, 9–111@FP4 | 29.83–152.75 | 49.2–451.2 | 122–345 |
| 性能 | 1978.9@FP16, 3957.8@FP8 | 0.367@FP8, 0.734@FP4 | 0.89@INT16, 3.55@INT8 | 2.38@BF16, 5.69@FP8 | 22.8 TFLOPS@POSIT8 |
| 能效 | 2.827@FP16, 5.654@FP8 | 8.24@FP8, 18.1@FP4 | 60.3@INT16, 101.1@INT8 | 20.58@BF16, 54.94@FP8 | 109.4 TFLOPS/W@POSIT8 |

论文据此宣称能效比 GPU-H100@FP8 高 19.35×。

**三项技术的独立贡献**（均在 DeepSeek-V3 / MMLU 上评估）：

| 技术 | 报告的效果 |
| --- | --- |
| MIPS | 节省 33.5% 的 DRAM 访存与 36.2% 的 SRAM 访存 |
| MBLM | 计算量减少 39.1% |
| DAPPM | 计算速度提升 1.47×，并称保持相同的计算精度 |

### 消融实验要点

论文没有消融章节，三项技术各自的效果数字分散在对应小节，没有给出组合后的总体收益，也没有给出各技术在总收益中的占比。Table 1 的性能与能效数字是整颗处理器（含全部三项技术）的结果，无法分离单项贡献。

## 局限性与未来方向

- **没有任何精度评估数据**。这是最严重的问题。MIPS 的 Diff-Reuse 是按阈值 $S_{th}$ 近似复用历史结果，MBLM 跳过近零乘法并做位翻转级别的表重放，DAPPM 对低位做结构化合并——三者都是近似计算。论文两次声称"保持相同的计算精度"（第 5 页 DAPPM、第 6 页结论），但**全文没有给出任何 MMLU 准确率数字**，既没有 baseline 也没有开/关各技术的对比。因此"精度无损"目前只是声明。
- **22.8 TFLOPS 的来源不透明**。架构是 4 个 Attention Core × 64 个 PE core = 256 个 PE。若每个 PE 是单 MAC，则 710 MHz 下只有约 0.36 TOPS，与 22.8 TFLOPS 相差约 63×。论文没有说明单个 PE core 的位宽与并行度，也没有说明 TFLOPS 的计数方式（是否把 POSIT8 的位级操作计入、是否含稀疏/近似路径的等效算力）。这个数字无法从论文验证。
- **对比基准的口径问题**：Table 1 引用编号与参考文献表不对应——表中两列分别标注 ISSCC'23 [6] 与 ISSCC'23 [7]，但文献 [6] 是 2017 年的 TPU 论文，[7] 与 [8] 是同一条 Keller 等人的 VLSI 2022 记录（重复条目）。同时 H100 一列用的是稀疏（sparse）峰值（3957.8 TFLOPS@FP8、1978.9@FP16），与边缘芯片的 dense 峰值直接相比。峰值能效的对比也未说明工作点差异：DSPE 的 109.4 TFLOPS/W 取自 0.6 V/200 MHz，而 710 MHz 下的能效没有报告。
- **对比对象不完全可比**：H100 是数据中心 GPU，其余三项是加速器芯片；本文的 8.23 mm² 与 122–345 mW 属于边缘量级，与 H100 的 700 W 相差三个数量级，峰值能效的倍数关系需要结合工作点理解。
- **未报告的项**：没有说明 48 KB SRAM 与 24/48 KB buffer 的具体组织（bank 数、端口数）；没有给出布局后的时序余量；没有给出 MIPS 的 History-LUT 容量与命中率；没有给出 BN 分类器的训练方式与误判率。

## 个人点评

- **亮点**：三项技术各自指向一个明确的问题，而且都落在具体的硬件结构上（Merkle 树 PE、Booth-LUT、DA-Posit 编码器与 Array Multiplier 模式选择），不是泛泛的算法主张。其中把 Merkle 树用作近似复用的判定结构是个有意思的跨界用法：它的价值在于中间节点哈希可以提前判定，从而在构建完整棵树之前就终止计算；同时哈希链提供了可校验性，论文保留的离线根哈希复核接口是负责任的设计。DA-Posit 复用边界 regime 码来编码压缩模式、不增加位宽，也是干净的做法。
- **不足**：这是一篇结论无法自证的论文。近似计算方案的核心代价就是精度，而全文没有一个精度数字；"保持相同精度"被当作前提而不是结论。22.8 TFLOPS 与 256 个 PE 之间存在无法解释的约 63× 差距，能效的 19.35× 倍数建立在这个数字与 H100 稀疏峰值的比较上。Table 1 的引用编号错位与重复条目进一步降低了可信度。此外三项技术各自的百分比（33.5%、36.2%、39.1%、1.47×）没有统一基线说明，无法判断是相对理想情况还是相对朴素实现。
- **启发**：对做边缘推理加速器的人，可迁移的思路有两条：一是把"冗余判定"做成硬件可提前终止的层级结构（Merkle 树只是其中一种实现），让复用决策在数据通路内部完成；二是把 Booth 编码的位翻转特征当作一个可在线测量的信号，用它驱动编码基数与部分积策略的切换。但这两条都需要配套的精度验证方法，否则收益无法进入产品。对评估这类论文的人，这份材料也提供了一个提醒：能效的倍数只有在工作点、精度口径与计费方式一致时才具有可比性。

## 工程化三问总结

### 1. 它解决了什么瓶颈？

- **应用场景与核心瓶颈**：DeepSeek-V3 在能耗受限的边缘设备上推理。核心瓶颈是三类冗余与开销：层间计算冗余（相邻 token 的相似 Q/K 与中间表示）、大规模线性变换的信号翻转与存储功耗、以及共享权重乘法中因 Booth 编码差异造成的位翻转能耗（第 3 页）。
- **现有方法为何不足**：算法侧方法（蒸馏、低秩分解、稀疏化、MoE）减少参数与复杂度但不针对边缘能耗；硬件侧通用方案依赖静态稀疏或统一操作数假设，无法按输入数据特征实时调整。论文特别指出常规矩阵乘法阵列缺乏对时序邻近带来的复用机会的感知（第 4 页）。
- **论文证据**：MIPS 节省 33.5% DRAM 与 36.2% SRAM 访存、MBLM 减少 39.1% 计算、DAPPM 提速 1.47×，均在 DeepSeek-V3/MMLU 上评估；实现结果为 TSMC 28 nm、8.23 mm²、122–345 mW、22.8 TFLOPS@POSIT8、峰值 109.4 TFLOPS/W。这些是论文的直接报告值，但**全部缺少精度验证**，且峰值能效取自 0.6 V/200 MHz 的工作点。因此"瓶颈被缓解"在访存与计算量指标上有数字支撑，在"不损失精度"这一前提上没有证据，在能效对比上存在口径问题。

### 2. 用了什么结构或训练方法？

- **整体结构与数据流**：4 个 Attention Core（各 64 PE core），配 iRouter/oRouter 做激活与权重分发；片上含 Input/Output Buffer、24 KB Parameter Buffer、48 KB Weight Buffer、两个 48 KB Q/K SRAM 与一个 48 KB V SRAM；Top Controller 集中调度并按层分块加载参数以实现流水化权重复用，Weight Buffer 配合 iRouter 跨 Attention Core 广播权重。推理数据流上，MIPS 在 token 级 decode 时先做相似度重排与 Merkle 树判定，命中则走 Early-Skip 或 Diff-Reuse 路径直接复用 KV Cache 索引与中间结果；未命中才进入完整计算，计算中由 MBLM 做无效计算检测、Booth 基数切换与表重放，算术部分由 DA-Posit 编解码与可变 Array Multiplier 配置完成。
- **关键模块/结构**：Sequential Incremental Sorter + Cos-SRAM、MerkleTree PE + History-LUT、无效计算检测器、Booth 贝叶斯网络、8×8 BVM 简化为 VST、Booth-LUT、DA-Posit Decoder/Encoder、可变规模 Array Multiplier（16/9/4 PE）、多级 CSA Tree 与 CPA。
- **训练目标、损失函数与数据策略**：不适用（论文不训练模型）。Booth 贝叶斯网络的冗余分类是基于位翻转特征的在线推断，论文未说明其参数如何获得（离线训练还是在线统计）。
- **数据与评估策略**：以 DeepSeek-V3 在 MMLU 上的推理作为评估负载，三项技术的效果数字均在此设置下给出；无精度指标。

### 3. 对芯片架构和 RTL 有什么启发？

- **芯片架构**：三条可引用的取向。第一，把「是否可复用」的判断放进数据通路（Merkle 树 PE + History-LUT），让近似复用由硬件在层内提前终止，而不是由软件在调度层决定——这是边缘加速器在延迟与能耗上的一个可行方向。第二，把 Booth 编码的位翻转特征当作在线可测信号来切换算术策略（radix-4/radix-8）与查表复用，属于算术层与调度层联动的设计。第三，为近似计算引入可校验性：保留离线根哈希复核接口，使近似决策可以被统计审计，这一思路在需要正确性保证的场景里比单纯提高复用率更有价值。另外，8.23 mm²、122–345 mW、0.6–1.1 V 的电压频率缩放范围说明该设计把功耗弹性作为一等目标，但其峰值能效只在最低工作点取得，架构评估时需要连同频率一起看。
- **RTL**：可落在 RTL 上的模块包括：Merkle 树的哈希 PE 与逐层比较逻辑、History-LUT 的查找与写回通路、Sequential Incremental Sorter 的插入逻辑、无效计算检测器的阈值比较、Booth BN 的分类逻辑与分数计算（式 4、式 5）、8×8 BVM 的统计与 VST 简化、Booth-LUT 的重放与比较器、DA-Posit 编解码器与 Dyn-field 折叠/还原、可变规模 Array Multiplier 的模式选择（16/9/4 个 PE）以及多级 CSA Tree 与 CPA 的位宽配置。这些都属工程推断的模块划分，论文只给出结构框图，未提供位宽、时序、流水级数或面积分解。其中 DA-Posit 的无损还原依赖 regime 提供的尺度信息，这在 RTL 上要求编码器与解码器对折叠位有严格一致的解释规则，是设计风险点（推测）。论文未给出任何 RTL 代码、综合报告或时序余量。
- **推断边界**：第 1、2 问基于论文正文与 Table 1，属论文报告值，且如局限性所述存在精度缺失与对比口径问题；第 3 问的芯片架构结论是对论文结构的归纳，RTL 部分全部为工程推断。22.8 TFLOPS 的计数方式、各模块的面积/功耗分解、精度影响、History-LUT 命中率与容量均 `TBD`。
