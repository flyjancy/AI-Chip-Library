# 深度分析：《The Eighth Generation TPU Family: Two Chips Optimized for the Agentic Era》

## 基本信息

- **标题**：The Eighth Generation TPU Family: Two Chips Optimized for the Agentic Era
- **文档类型**：产品与系统架构演讲（Hot Chips 2026）
- **作者**：Norman P. Jouppi、Sridhar Lakshmanamurthy（含多位贡献者）
- **机构**：Google
- **发表 venue**：Hot Chips 2026
- **年份**：2026
- **链接**：Google TPU8 产品资料（演讲未提供独立论文链接）

## 一句话总结

> Google 同时推出面向训练的 TPU 8t 与面向推理的 TPU 8i，以不同 ICI 拓扑、HBM/SRAM 配比、Collective Acceleration Engine、SparseCore、OCS 共享内存和 Virgo 全球 fabric 覆盖 agentic AI 的训练到服务全链路。

## 研究动机与问题定义

- **要解决的核心问题**：agentic 模型需要长上下文、复杂逻辑、MoE、采样和低延迟推理；训练和推理对 compute、HBM bandwidth、SRAM、网络 topology 与服务器连接的比例不同。
- **现有方法的不足**：单一 TPU 设计难以同时优化 pretraining、distillation、sampling、MoE 和 agent serving；通用 3D torus 在 MoE all-to-all 上 hop 数和延迟较高。
- **本文的切入角度**：以 TPU8i/8t 分化、Boardfly/OCS/ICI、CAE、SparseCore、Pallas/Mosaic 和大规模 RAS 构成“ML-first”平台。

## 核心方法

### 方法概述

TPU 8i 追求 zero-wait、reasoning 和 RL sampling 的低延迟，配置 384 MB SRAM、288 GB 12-high HBM3E、约 150--200 TB/s SRAM bandwidth、10--15 TB/s HBM bandwidth，采用 Boardfly topology。TPU 8t 为 trillion-parameter training 设计，单 superpod 9,600 chips、2 PB shared HBM、121 exaflops，使用更高 ICI 和 OCS 共享内存。

### 关键技术细节

- **8i memory hierarchy**：SRAM 约 1--2 ns、HBM 约 150--200 ns；SRAM bandwidth 约 15--20× HBM、energy/bit 约 10--20× 优势（第 8 页，规格区间）。
- **8i Boardfly**：每 tray 4 TPU fully connected，36 groups、总 1,152 chips/pod，最大 7 hops；演讲将其与 3D torus（示例最大 16 hops）比较，目标是降低 MoE all-to-all latency。
- **CAE**：位于 ICI I/O die，做 in-network collective，避免 HBM access 并减少 package 内距离，演讲宣称 on-chip latency 约 5× 降低（第 11 页）。
- **8i package**：TensorCore、XLU/MXU、VPU/Vmem、HBM controller、ICI router、SC-CAE、PCIe Gen5 x16/Gen2 x1 等 chiplet（第 12 页）。
- **8t superpod**：2.4 TB/s I/O per chip、121 exaflops、2 PB shared HBM、9,600 chips，约为 Ironwood 3× compute、2× perf/W（第 13 页）。
- **Virgo network**：全局 non-blocking cluster fabric，134,400 TPUs、1.6 YottaFLOPS、47 Pb/s（第 15 页）。
- **RAS 与软件**：HBM CRC/retry、interconnect parity、V/T/droop/aging sensors、in-field unit test、fleet health monitoring；Pallas/Mosaic 支持 Python 硬件感知 kernel，XLA 自动处理 Boardfly topology/CAE synchronization（第 16、21 页）。
- **AI-assisted design**：演讲称 AI designer/optimizer 为约 100 个 TPU 或 CPU 运行一周，带来 TPU8t TensorCore 5.8% area/6% power reduction、SparseCore 13% area reduction；TPU8i MXU area 5.3% reduction、相同 thermal budget 下 FLOPS +5%（第 20 页）。

### 核心创新点

1. 用训练/推理双芯片与不同 topology/memory ratio 匹配 agentic AI 阶段差异。
2. 以 CAE、Boardfly、OCS 和 Virgo 将 collective、共享内存和大规模网络纳入 TPU 架构。
3. 以编译器/XLA 和 AI-assisted RTL/physical design 作为 ML-first co-design 的核心。

### 与现有方法的关键区别

TPU8 不是单代通用加速器，而是训练 TPU8t、推理 TPU8i 与多级网络/共享内存协同；软件抽象隐藏 topology，但由编译器、CAE 和硬件保证可预测数据流。

## 证据、案例与论证

### 证据设置

- **工作负载**：trillion-parameter pretraining、MoE inference、RL sampling、long-context agent serving。
- **基线**：前代 TPU/Ironwood、Boardfly vs 3D torus、常规网络 collective。
- **评估指标**：compute/IO/HBM/SRAM bandwidth、latency、perf/W、机架规模、RAS 和设计 PPA。

### 主要结果

| 指标 | 演讲结果 | 证据位置与强度 |
| --- | ---: | --- |
| TPU 8i | 384 MB SRAM、288 GB HBM3E、约 150--200 TB/s SRAM、10--15 TB/s HBM | 第 8 页；规格区间 |
| TPU 8i pod | Boardfly 1,152 chips、最大 7 hops | 第 10 页；拓扑规格 |
| CAE | I/O die in-network collective，on-chip latency 约降 5× | 第 11 页；架构主张 |
| TPU 8t superpod | 9,600 chips、2 PB shared HBM、121 exaflops、2.4 TB/s I/O/chip | 第 13 页；平台规格 |
| Virgo cluster | 134,400 TPUs、1.6 YottaFLOPS、47 Pb/s | 第 15 页；平台目标 |
| AI-assisted design | TensorCore/SparseCore/MXU area/power/FLOPS 改进约 5%--13% | 第 20 页；演讲案例 |

### 消融/案例要点

- TPU8i/8t 的差异本身是架构 ablation：推理增加 SRAM/CAE/Boardfly，训练增加 I/O、SparseCore/OCS 和共享内存；但没有公开同一 workload 的完整隔离实验。
- AI-assisted design 的面积/功耗数字是设计流程案例，未披露 baseline RTL、综合 corner、时钟或验证缺陷率。
- 8i 的带宽与延迟以区间表达，8t/Virgo 规模是平台规格，不能直接视为客户可获得 tokens/s。

## 局限性与未来方向

- **演讲范围明确的局限**：产品演讲未给出完整模型 benchmark、芯片面积/功耗拆分、编译器开销或第三方验证；部分数据为规划/目标。
- **方法限制**：不同 topology 和 shared-memory slicing 增加 compiler/XLA、CAE 和 RAS 复杂度；超大规模系统的故障、热和网络一致性难度很高。
- **潜在改进方向**：公开 MoE all-to-all、long-context TTFT/TPOT、CAE 利用率、OCS slice reconfiguration、AI RTL/PD 的可复现实验和 fleet RAS 数据。
- **证据边界**：Virgo 1.6 YottaFLOPS、121 exaflops、2× perf/W 等属于平台/代际主张，不能等同于端到端模型结果。

## 个人点评

- **亮点**：把训练/推理分化、SRAM/HBM 层次、collective accelerator、拓扑和编译器放进一套可扩展系统设计，逻辑完整。
- **不足**：大部分性能是规格或目标，缺少模型层面吞吐、能耗和 tail latency；AI 设计改进案例也缺少可验证细节。
- **启发**：TPU 类架构的价值在于用编译器把 topology、同步和内存切片编译掉，硬件则专注于可预测的 collective 和 RAS。

## 工程化三问总结

### 1. 它解决了什么瓶颈？

- **应用场景与核心瓶颈**：agentic AI 的训练规模、MoE all-to-all、长上下文推理、RL sampling 延迟和大规模系统 RAS。
- **现有方法为何不足**：单一 chip/topology 无法同时满足训练高 I/O 与推理高 SRAM/低延迟；3D torus hop 数和 HBM access 会限制 MoE。
- **论文或文档证据**：8i Boardfly 最大 7 hops、CAE latency 约 5×、8t 9,600-chip/2 PB、Virgo 134,400 TPU（第 10、11、13、15 页）。

### 2. 用了什么结构或训练方法？

- **整体结构与数据流**：8i 以 SRAM/HBM/CAE/Boardfly 服务 inference，8t 以 HBM/ICI/OCS/SparseCore 服务 training；XLA/Pallas/Mosaic 负责 topology-aware kernel 与同步。
- **关键模块/结构**：TensorCore/MXU、XLU/VPU/Vmem、SparseCore、CAE、ICI router、HBM controller、Boardfly/OCS/Virgo network、RAS sensors。
- **训练目标、损失函数或优化方法**：不适用；支持 pretraining、distillation、sampling、MoE 和 agent serving，但演讲不描述损失函数。
- **数据与训练策略**：面向 long-context/agentic workloads；具体训练数据、模型和 batch 未披露。

### 3. 对芯片架构、RTL、验证有什么启发？

- **芯片架构**：按 training/inference 分配 SRAM/HBM/ICI/OCS，加入 in-network collective 和可重配置拓扑；用编译器静态消除部分同步/布局开销。
- **RTL**：需要 CAE reduce engine、ICI router、HBM CRC/retry、SRAM/HBM DMA、SparseCore、OCS slice control、sensor/RAS 和 parity paths；具体接口为 `TBD`。
- **验证**：覆盖 collective 数值与顺序、Boardfly/torus 路由、HBM error retry、OCS slice、clock/link fault、thermal telemetry、XLA/Pallas 映射和 AI-generated RTL 回归。
- **推断边界**：演讲没有公开 TPU8 RTL、综合、形式化和硅后全量数据；工程实现与验证项需后续规格化。
