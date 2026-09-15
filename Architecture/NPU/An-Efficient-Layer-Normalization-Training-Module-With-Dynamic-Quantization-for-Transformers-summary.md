# 深度分析：《An Efficient Layer Normalization Training Module With Dynamic Quantization for Transformers》

## 基本信息

- **标题**：An Efficient Layer Normalization Training Module With Dynamic Quantization for Transformers
- **文档类型**：论文
- **作者**：Haikuo Shao、Aotao Wang、Zhongfeng Wang
- **机构**：南京大学、中山大学
- **发表 venue**：IEEE Transactions on Circuits and Systems II: Express Briefs，Vol. 72，No. 9
- **年份**：2025
- **链接**：[DOI:10.1109/TCSII.2025.3591633](https://doi.org/10.1109/TCSII.2025.3591633)
- **阅读范围**：全文 5 页，包括算法、架构图、数据流图和全部实验表格

## 一句话总结

> 论文用双 power-of-two scale 的动态量化和分段线性逆平方根近似，将 Transformer 训练中的 LayerNorm 前向与反向运算压缩到整数数据通路，并通过可复用的 SCU/ATU 与向量级流水线显著降低 FPGA/ASIC 资源和延迟。

## 研究动机与问题定义

- **要解决的核心问题**：端侧 Transformer 训练需要低延迟和数据隐私，但层归一化（Layer Normalization，LN）既包含量化敏感的极端离群值，又需要平方根和除法等硬件不友好运算。线性层量化后，BERT 中 LN 的运行时间占比会从约 6% 上升到 15%（第 1 页）。
- **现有方法的不足**：整数 LN 推理方案常使用 19 bit 以上甚至 32 bit 数据；LUT 近似工作主要面向推理；现有训练硬件集中于卷积网络的批归一化（Batch Normalization，BN），而 LN 的归一化区域、参数共享方式及反向数据流与 BN 不同，不能直接套用。
- **本文的切入角度**：先针对前向输入 `x` 与反向梯度 `dy` 的离群值提出动态 power-of-two 量化（DPTQ），再把 LN 的统计计算与仿射变换映射到可重构整数硬件，并让两阶段在相邻向量之间重叠执行。

## 核心方法

### 方法概述

LN 对每个 token 的隐藏维向量计算均值和方差，再执行归一化与可学习仿射变换。训练还需反向计算 `dx`、`dγ` 和 `dβ`。论文观察到，当 Transformer 的隐藏维 `M > 512` 时，完整 `dx` 公式中的第四项比其他项小三个数量级以上，因此删除该项得到较简洁的反向表达式（公式 5、6，第 2 页）。

动态 power-of-two 量化（Dynamic Power-of-Two Quantization，DPTQ）为普通值和离群值分别使用较小 scale `s(1)` 和较大 scale `s(2)`。阈值 `tau` 由位宽与 `s(1)` 决定，`s(1)` 通过 calibration batch 覆盖普通值动态范围；两个 scale 都限制为 2 的整数次幂，因此两组累加结果只需移位和加法即可对齐。`x` 和 `dy` 使用 DPTQ，其他无明显离群值的变量使用普通 power-of-two 量化（算法 1，第 3 页）。

方差改写为 `E[x^2] - E[x]^2`，使均值与二阶矩可以并行累加。量化逆平方根（Quantized Inverse Square Root，QISR）先借助移位把输入缩放到 `[0.25, 1)`，用 12 段线性函数逼近逆平方根，再按原指数移位恢复。硬件把统计计算单元（Statistics Computation Unit，SCU）和仿射变换单元（Affine Transformation Unit，ATU）组成 Layer Norm Unit（LNU），通过多路器复用前向与反向数据通路（图 3、4，第 3 页）。

### 关键技术细节

- **DPTQ**：每个元素携带 1 bit 离群标志，选择 `s(1)` 或 `s(2)`；两路寄存器分别累加，Align Unit 根据 scale 指数差做移位相加。
- **定点位宽**：实验中乘法器操作数为 12 bit，加法器为 32 bit；普通 int16 对照不使用 DPTQ，精度明显恶化（第 4 页）。
- **QISR**：Leading-One Detector（LOD）确定缩放指数，12 段线性近似完成核心计算，之后重新缩放；Inverse Square Root Unit（ISRU）仅在前向阶段使用。
- **反向复用**：乘法器 `M0` 在前向计算均值平方，在反向计算 `γ_i * dy_i`；ATU 通过多路器在前向 `y` 和反向 `dx` 运算间复用。
- **局部存储**：`x_i`、`γ_i * dy_i` 等中间量暂存在 local RAM，供 ATU 第二阶段使用；前向保存 `γ`、归一化输出和 `γ*x_hat` 供反向复用。
- **向量级流水线**：向量 `x1` 进入 ATU 时，`x2` 同时进入 SCU。两阶段延迟接近，理论上将整体吞吐提高约 2 倍（图 5，第 4 页）。
- **可选冻结参数**：由于 `γ/β` 在微调中的更新较小，实验评估冻结 `γ/β`，从而避免跨所有 normalization area 累加 `dγ/dβ` 的额外开销。

### 核心创新点

1. 用两个 power-of-two scale 同时保留普通值精度和离群值幅度，使动态量化后的统计量仍能通过简单移位合并。
2. 将方差分解、QISR 和简化的 LN 反向公式联合映射为整数 SCU/ATU 数据通路，而非只优化推理阶段的非线性函数。
3. 通过前后向运算复用和跨向量的 SCU/ATU 流水重叠，在较小面积增量下支持完整训练数据流。

### 与现有方法的关键区别

BN 训练加速器依赖 batch/feature-map 的分布与 min-max 近似，不能适配 LN 沿 token 隐藏维的归一化区域。LN 推理工作则没有 `dx`、梯度中间量和前后向复用。本文同时约束量化算法、平方根近似、数据位宽、存储和流水计划，目标是作为训练加速器 PE array 旁的专用功能模块。

## 实验与结果

### 实验设置

- **算法任务**：BERT-base 在 GLUE 上微调；GPT-2 117M 在 PTB、WikiText-2 上微调；ViT-110M 在 CIFAR-10、CIFAR-100、ImageNet 上微调。
- **训练配置**：BERT/GPT-2 训练 3 epochs、学习率 `2e-5`；ViT 训练 5 epochs、学习率 `1e-5`；均使用 Adam。
- **算法基线**：fp32、普通 int16、QLN、冻结 `γ/β` 的 QLN。
- **FPGA**：32 个并行 LNU，Xilinx ZCU102，250 MHz，Vivado 实现。
- **ASIC**：单 LNU，TSMC 28 nm，1 GHz，Synopsys Design Compiler 与 PrimeTime-PX 评估。
- **硬件基线**：Xilinx fp32/fp16 IP、BN 训练模块 RBN/ACBN/PBN，以及 LN 推理模块 NN-LUT、GQA-LUT 和文献 [14]。

### 主要结果

| 对比 | 主要结果 | 证据位置 |
| --- | --- | --- |
| 训练精度 | 冻结 `γ/β` 的 QLN 在 BERT/ViT 上相对 fp32 精度下降小于 1%，GPT-2 perplexity 仅轻微增加；普通 int16 显著退化 | 表 II，第 4 页 |
| FPGA vs. Xilinx fp32 | LUT 减少 90%，FF 减少 91%，功耗减少 80%，前向延迟减少 83%，反向延迟减少 79% | 图 6，第 4 页 |
| FPGA vs. Xilinx fp16 | LUT 减少 76%，FF 减少 78%；三者均使用 160 个 DSP | 第 4 页 |
| BERT-base 系统影响 | 在 `32x32` int8 PE array 假设下，前向、反向和总训练延迟分别降低 13.0%、7.7%、7.4% | 第 4 页 |
| 对比 RBN | 吞吐提高 2.5 倍，能效提高 8.5 倍 | 表 III，第 5 页 |
| 对比 ACBN/PBN | 吞吐最高提高 1.3 倍，能效相对 ACBN 提高约 2.5 倍 | 表 III，第 5 页 |
| 对比 LN 推理模块 | 吞吐提高 2.0-5.4 倍，面积延迟积（ADP）效率提高 4.4-23.8 倍 | 表 IV，第 5 页 |

表 III 中本文 FPGA 实现为 17,996 LUT、11,768 FF、160 DSP、96 BRAM、1.659 W、4 ns latency，能效 4.82 GinS/W。表 IV 将不同实现缩放到 28 nm 后，本文单 LNU 面积为 3,684 `um^2`、周期 1 ns、功耗 2.08 mW、ADP 为 0.037 `mm^2*ns`。摘要给出的峰值吞吐为 FPGA 0.25 GinS、ASIC 1.0 GinS（第 1、5 页）。

### 消融实验要点

- **LN 分块近似**：把 BN 的 block-wise min/max 思路直接用于 LN 后，BERT GLUE 多项任务明显下降，说明 BN 的高斯分布假设和归一化区域不能迁移到 LN（表 I，第 2 页）。
- **DPTQ vs. 普通 int16**：12-bit QLN 能维持多个模型的微调性能，而 16-bit 普通量化仍严重退化，证明关键不是单纯增加位宽，而是单独处理离群值（表 II，第 4 页）。
- **冻结 `γ/β`**：fp32 预实验显示冻结参数影响很小；QLN 与 QLN(frozen-`γβ`) 的 BERT 平均值分别约为 88.40 和 88.46，支持微调场景下省去相关梯度计算，但不代表从头训练也成立（表 I、II，第 2、4 页）。
- **训练支持面积代价**：相对不含反向支持的 vanilla inference-only LN，训练模块面积只增加 12.7%；仅前向使用的 ISRU 占 LNU 面积 16.0%（第 5 页）。

## 局限性与未来方向

- **训练场景边界**：实验主要是预训练模型的短周期微调；冻结 `γ/β` 与简化梯度是否适用于从头训练、长周期持续学习或小隐藏维模型未得到证明。
- **公式近似边界**：删除的反向项在 `M > 512` 时被观察为小三个数量级，但没有给出跨模型、跨训练阶段的误差分布；较小 `M` 需要重新验证。
- **校准依赖**：`s(1)` 和离群阈值来自 calibration batch。数据分布漂移、任务切换或梯度爆炸时，静态阈值可能失效，论文未给出在线重校准机制。
- **硬件成熟度**：FPGA 有实现结果，ASIC 仅基于综合与功耗工具评估，没有布局布线、时序收敛、IR drop、存储宏和流片测量。
- **基线可比性**：表 III 同时比较不同 FPGA、不同归一化类型和不同频率；表 IV 将其他设计缩放到 28 nm，且多项基线只支持推理，倍数结果应视为参考而非严格同条件竞赛。
- **系统评估限制**：BERT 总训练延迟改善来自 `32x32` PE array 的系统模型，论文未展示完整加速器 RTL、片上网络、DMA 或外存调度的端到端实现。
- **潜在改进方向**：加入运行时 scale 监控；系统评估中覆盖真实 PE/LNU backpressure；开展 post-layout 与硅后验证；评估更大 LLM、不同 hidden size、长训练和混合精度优化器状态。

## 个人点评

- **亮点**：算法中的每个选择都能在硬件上找到直接对应：power-of-two scale 对应移位对齐，双 scale 对应两路累加寄存器，QISR 对应 LOD 和分段线性单元，方差分解对应并行统计。这是较完整的算法-硬件协同设计。
- **不足**：论文篇幅短，数值误差、累加器溢出、QISR 各分段误差和训练稳定性展开不足；跨论文硬件比较存在工艺、任务和功能范围差异。
- **启发**：非线性模块在主矩阵乘法量化后会成为新的 Amdahl 瓶颈。相比做一个通用浮点单元，为训练常用算子建立可重构、与 PE dataflow 融合的短位宽功能单元，可能更有效。

## 工程化三问总结

### 1. 它解决了什么瓶颈？

- **应用场景与核心瓶颈**：端侧 Transformer 微调中的 LN 因离群值、平方根/除法以及反向梯度计算而难以低位宽实现；线性层优化后，LN 的 BERT 运行占比从约 6% 增到 15%。
- **现有方法为何不足**：普通统一量化即使用 16 bit 仍被离群值拖累；BN 训练架构依赖不同的数据分布；已有 LN 硬件大多只支持推理。
- **论文证据**：QLN 在多类 Transformer 微调任务上将精度损失控制在 1% 内；相对 Xilinx fp32 模块减少 90% LUT、91% FF、80% 功耗，并使建模的 BERT 总训练延迟下降 7.4%。前者有训练实验支持，后者的系统级 7.4% 是论文基于特定 PE array 的估算。

### 2. 用了什么结构或训练方法？

- **整体结构与数据流**：PE array 输出 `x/dy` -> DPTQ 与双路统计累加 -> Align Unit 移位合并 -> QISR/ISRU -> ATU 仿射变换或反向 `dx` -> 下一向量在 SCU 与当前向量在 ATU 重叠执行。
- **关键模块/结构**：并行 LNU、SCU、ATU、双 Reg 累加、1 bit outlier flag、Align Unit、LOD、12 段线性 ISRU、local RAM、前后向复用乘法器和多路器。
- **训练目标、损失函数或优化方法**：不改变任务损失函数，使用 Adam 微调；通过 12-bit DPTQ、32-bit 累加、QISR 和可选冻结 `γ/β` 改变 LN 数值实现。
- **数据与训练策略**：BERT/GPT-2 微调 3 epochs，ViT 微调 5 epochs；`s(1)` 用 calibration batch 决定，离群值使用更大 `s(2)`；论文未给出在线 scale 更新策略。

### 3. 对芯片架构、RTL、验证有什么启发？

- **芯片架构**：在 PE array 旁布置可配置数量的 LNU，并融合 PE-LNU 数据传输，可避免增加 off-chip bandwidth；SCU 与 ATU 的向量级 producer-consumer pipeline 能将功能单元吞吐匹配到矩阵阵列。真实 NoC、buffer 容量与 backpressure 仍为 `TBD`。
- **RTL**：可直接拆为 DPTQ/flag、双 accumulator、shift-align、LOD、piecewise-linear QISR、SCU、ATU、local RAM 和调度控制模块；位宽、hidden size、LNU 并行数、scale 指数及 12 段系数应参数化。需要明确 accumulator guard bits、饱和/舍入规则和 FP/BP 模式切换时序。
- **验证**：建立 fp32 与定点 bit-accurate 双参考模型；覆盖 `|x| = tau` 两侧、scale 最大差、全零/常数向量、极端离群值、累加溢出、QISR 12 个分段边界、FP/BP 数据冒险、local RAM 对齐、流水线 flush/backpressure，以及 `dx/dγ/dβ` 数值梯度检查。冻结与不冻结 `γ/β` 应分别验证。
- **推断边界**：模块结构与 FPGA/综合数据来自论文；完整训练加速器的互连、DMA、缓存一致性和硅后 PPA 结论均是待验证的工程扩展。
