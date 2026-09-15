# 深度分析：《DSPE: An Energy-Efficient Edge Processor for DeepSeek Inference with MerkleTree-based Incremental Pruning, Multi-Stage Boothing Lookup and Dynamic Adaptive Posit Processing》

## 基本信息

- **标题**：DSPE: An Energy-Efficient Edge Processor for DeepSeek Inference with MerkleTree-based Incremental Pruning, Multi-Stage Boothing Lookup and Dynamic Adaptive Posit Processing
- **文档类型**：论文
- **作者**：Yuhan Zhang、Zhou Wang、Zhou Shu、Jiuren Zhou、Yanqing Xu、Xiaonan Tang、Shushan Qiao、Tianchun Ye、Yang Liu、Anil A. Bharath、Emm Mic Drakakis
- **机构**：东北大学、Imperial College London、西安电子科技大学等
- **发表venue**：DAC 2026
- **年份**：2026
- **链接**：[DOI:10.1145/3770743.3805813](https://doi.org/10.1145/3770743.3805813)

## 一句话总结

> DSPE 面向边缘 DeepSeek 推理，用 MerkleTree 增量剪枝、分阶段 Booth 查找和动态自适应 Posit 三种机制减少冗余计算、位翻转和数据宽度，并在 TSMC 28nm 论文实现中报告 109.4 TFLOPS/W 的峰值能效。

## 研究动机与问题定义

- **要解决的核心问题**：DeepSeek 的 MLA、MoE、KV-cache 和大规模线性变换给边缘设备带来存储、带宽、信号翻转和算力压力。
- **现有方法的不足**：静态稀疏、仅权重剪枝和统一精度不能利用 token/激活的时空相似性；通用 Booth 乘法对重复操作和 bit flip 不敏感；固定 Posit 的 exponent/fraction 划分存在低位冗余。
- **本文的切入角度**：在处理单元内部同时做相似向量复用、近似乘法路径选择和动态数值编码，形成面向 DeepSeek 解码阶段的数据流。

## 核心方法

### 方法概述

DSPE 包含 4 个 Attention Core，每个 Core 由 iRouter/oRouter、输入/输出 buffer、参数和权重 buffer、Query/Key/Value SRAM 以及 PE 阵列组成。Top Controller 负责不同层的参数装载、权重广播和流水调度。

MIPS 先按余弦相似度对新向量重排，再用 Cos-SRAM 和 MerkleTree PE 逐层比较当前/历史 hash：完全相同则 Early-Skip，中等差异且命中 History-LUT 则 Diff-Reuse，否则 Full-Compute 并回写历史表。MBLM 对 8 组乘法操作做近零检测、位变化统计和 Bayesian Network 分类，在 radix-4 与 radix-8 Booth 路径之间切换。DAPPM 用 regime 携带尺度和 mode，将 Posit 的低位 exponent/fraction 合并成 0/1/2-bit 压缩模式，选择 16、9 或 4 个阵列乘法 PE。

### 关键技术细节

- **MIPS**：MerkleTree 层间差值阈值 `Tzero`、`Sth` 决定 skip/reuse/full-compute；历史结果存于 History-LUT（第 3–4 页图 5）。
- **MBLM**：依据 bit similarity、repeat length 和 bit-flip matrix 构造 Variation Simplified Triangle；冗余分数低于 0.8 走 radix-4，高于 0.8 走 radix-8（第 4–5 页图 6）。
- **DAPPM**：DA-Posit 通过 regime 的边界 code 编码压缩 mode，动态选择 16/9/4 个 Array Multiplier PE，并用 CSA/CPA 树完成累加（第 5–6 页图 7）。
- **实现流程**：Verilog → Synopsys Design Compiler 综合 → IC Compiler 28nm CMOS 布局布线（第 6–7 页）。

### 核心创新点

1. 用 MerkleTree 的中间节点作为推理冗余的早期判定点，并保留后台完整 root hash 校验接口。
2. 将输入/权重联合 bit-flip 统计用于自适应 Booth 编码，而不是只做静态权重剪枝。
3. 用无额外位宽的 DA-Posit mode 实现精度/面积/功耗可调的乘法阵列。

### 与现有方法的关键区别

相对于静态剪枝、固定 Booth 和固定 Posit，DSPE 让“是否计算、采用何种乘法、使用多少乘法 PE”均由当前数据和历史统计决定，目标是同时减少 DRAM/SRAM 访问、部分积和低位编码冗余。

## 实验与结果

### 实验设置

- **数据集**：DeepSeek-V3 在 MMLU 上进行方法分析。
- **基线方法**：GPU H100、ISSCC’23 FP8/FP4 与 INT16/INT8 设计、VLSI’24 BF16/FP8 设计。
- **评估指标**：DRAM/SRAM 访问、计算量、推理速度、面积、频率、功耗、峰值性能和能效。

### 主要结果

| 方法/指标 | DSPE 结果 | 证据与边界 |
| --- | ---: | --- |
| MIPS DRAM 访问减少 | 33.5% | MMLU 分析，第 4 页；算法模拟结果 |
| MIPS SRAM 访问减少 | 36.2% | MMLU 分析，第 4 页；算法模拟结果 |
| MBLM 计算量减少 | 39.1% | DeepSeek-V3/MMLU，第 5 页 |
| DAPPM 速度 | 1.47×，保持相同计算精度 | DeepSeek-V3/MMLU，第 6 页 |
| 工艺 / 面积 | TSMC 28nm / 8.23 mm² | 第 6–7 页表 1 |
| 频率 / 功耗 | 710 MHz / 122–345 mW | 第 7 页 |
| 峰值性能 / 能效 | 22.8 TFLOPS@POSIT8 / 109.4 TFLOPS/W | 第 7 页表 1 |

### 消融实验要点

- MIPS、MBLM、DAPPM 分别对应存储访问、乘法冗余和数据表示三类开销，论文没有提供去掉每一项后的完整端到端精度/能效矩阵。
- DA-Posit 的 0/1/2-bit 压缩模式和 16/9/4 PE 选择是结构内部的模式比较；“保持相同计算精度”来自论文描述，缺少公开误差分布和任务级准确率表。

## 局限性与未来方向

- **作者提到的局限**：论文主要展示 DSPE 架构和综合/布局布线结果，未报告流片后的硅测量。
- **潜在改进方向**：需要对 MIPS hash 碰撞、近似 skip 的误判、Bayesian 阈值迁移和 DA-Posit 在不同层/模型上的精度做系统验证。
- **证据边界**：109.4 TFLOPS/W 是峰值 POSIT8 设计指标，不能直接与 H100 的 FP8 实际应用能效作等价比较；论文表 1 的跨工艺、跨精度比较应降级理解。

## 个人点评

- **亮点**：把时空相似性、位翻转和数值格式放在同一个 edge data path 中，硬件模块边界清晰。
- **不足**：MIPS 的安全性与近似性目标并存，但误判率、hash 质量和最终任务准确率披露不足；MBLM 的 Bayesian Network 训练/部署成本也未展开。
- **启发**：边缘 NPU 可以将可恢复的近似路径做成带统计回退的旁路，而不是把所有稀疏性固化为离线剪枝。

## 工程化三问总结

### 1. 它解决了什么瓶颈？

- **应用场景与核心瓶颈**：低功耗边缘设备上的 DeepSeek 解码，瓶颈包括重复向量、重复乘法、bit flip 能耗和多精度转换。
- **现有方法为何不足**：静态剪枝不适应输入时序局部性；统一 radix Booth 和固定 Posit 无法随数据分布调整；大型缓存访问限制边缘带宽。
- **论文或文档证据**：MIPS 访问减少 33.5%/36.2%、MBLM 计算量减少 39.1%、DAPPM 速度 1.47×；这些是 MMLU/架构模型结果，峰值能效来自综合后实现。

### 2. 用了什么结构或训练方法？

- **整体结构与数据流**：输入向量 → 相似度重排 → MerkleTree 判定/复用 → MBLM 乘法路径 → DA-Posit 解码/乘法/归一化 → 输出。
- **关键模块/结构**：Cos-SRAM、MerkleTree PE、History-LUT、invalid detector、Bayesian Network、Booth-LUT、DA-Posit decoder、可变 PE 阵列。
- **训练目标、损失函数或优化方法**：没有模型训练目标；Bayesian Network 的训练数据和离线校准流程为 `TBD`。
- **数据与训练策略**：以 DeepSeek-V3/MMLU 做算法分析；硬件使用 Verilog 综合和布局布线。

### 3. 对芯片架构、RTL、验证有什么启发？

- **芯片架构**：为相似性检测、历史复用和近似计算预留旁路；用小型 LUT/统计阵列换取大规模 DRAM/SRAM 访问削减；支持可配置 PE 阵列和 Posit/整数混合格式。
- **RTL**：需要 hash/tree 流水、阈值比较与回退 FSM、History-LUT 一致性接口、radix-4/8 Booth 双路径、动态 mode 解码和 CSA/CPA 累加树；阈值和表项深度为 `TBD`。
- **验证**：覆盖 hash 碰撞和误判回退、历史表更新一致性、近零阈值边界、radix 路径选择、DA-Posit 无损恢复、溢出/归一化和端到端准确率回归。
- **推断边界**：论文未给出 RTL 仿真、形式验证、流片测试或详细软件 ISA；上述接口与验证项为工程推断。
