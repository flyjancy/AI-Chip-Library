# 深度分析：《Jalapeño ASIC + System》

## 基本信息

- **标题**：Jalapeño ASIC + System
- **文档类型**：产品与系统架构演讲（Hot Chips 2026）
- **作者**：Richard Ho、Ravi Narayanaswami、Chris Leary
- **机构**：OpenAI（与 Broadcom、Celestica 等合作）
- **发表 venue**：Hot Chips 2026
- **年份**：2026
- **链接**：[OpenAI](https://openai.com/)

## 一句话总结

> Jalapeño 是面向 OpenAI 推理工作负载的 700 W 空间架构 ASIC：通过每个 core slice 的本地 HBM、专用 collective 网络、显式张量布局和 AI 辅助 PPA/内核优化，在 9 个月内从初始 RTL 走到 tapeout，并以低延迟 perf/W 为主要目标。

## 研究动机与问题定义

- **要解决的核心问题**：frontier 模型推理同时包含 prefill、draft/speculate 和 verify/decode 三种硬件特征；仅比较峰值 FLOPS 或单卡吞吐无法满足低延迟和能耗 SLA。
- **现有方法的不足**：统一的内存、网络和同步路径导致操作数晚到、全局 reduce 等待以及跨阶段 KV 搬运；孤立的专用加速器还会在 workload 比例变化时闲置并持续耗电。
- **本文的切入角度**：将计算、HBM、collective 网络和软件映射设计成空间架构；保持常用数据在本地，将全局路径留给必要通信，并用 AI 加速 RTL、PPA 和 kernel 收敛。

## 核心方法

### 方法概述

Jalapeño 将芯片划分为 64 个 core slice，每个 slice 具有 tensor/SIMD/scalar 引擎、L1 和对应 HBM slice 的低延迟本地视图。专用 collective 网络负责高带宽低延迟的常见通信，通用 NoC 保留更灵活但较慢的访问。128 颗 ASIC 构成 local domain，通过 Broadcom TH6 交换芯片组成 2,048 颗 ASIC 的 global domain；带宽从本地 600 GB/s 降到全局 200 GB/s，以匹配流量和功耗。

软件层使用 Gluon/TensorInfo 显式表达张量布局、物理放置和通信，AI 系统从功能正确的 kernel 开始，在芯片/模拟器上测量、验证和优化。演讲强调“measure → verify → learn → change → repeat”的持续收敛，而不是先写完一份静态规格。

### 关键技术细节

- **矩阵与数值格式**：支持 MXFP8×MXFP8 3.4 PFLOP/s、MXFP8×MXFP4 6.7 PFLOP/s、MXFP4×MXFP4 13.4 PFLOP/s（单 ASIC）。
- **内存系统**：每 ASIC 216 GiB HBM4、15.4 TB/s；演讲把 128 ASIC 的聚合 HBM 带宽概括为 1+ PB/s，并明确指出这是带宽上限而非实际 token 速率。
- **网络层次**：本地 128 ASIC @600 GB/s，全球 2,048 ASIC @200 GB/s；“half-flattened”两级 Clos 提供 tensor parallel 的高带宽和 expert parallel 的较低带宽路径。
- **空间编程**：Gluon 将物理 core 类比为 thread block，TensorInfo 记录布局与放置，specialized collectives 协调 core；局部路径优先、全局内存显式访问。
- **三种推理阶段**：prefill 是高 FLOP/低内存带宽，draft 是超低 batch、网络延迟敏感，spec-verify 同时需要 attention、HBM 带宽和突发通信；设计目标是单一芯片内部按阶段门控不同单元。
- **AI 辅助设计**：内部 AI 模型与 XLS 硬件语言、快速 QoR/验证工具配合，优化 BF16 multiply、FP4 dot、FP32 accumulator 和矩阵/SIMD 单元面积。
- **开发周期**：2024-10 架构概念，2025-02 初始 RTL，2025-07 RTL freeze，2025-11 tapeout，2026-05 首片硅；从初始 RTL 到 tapeout 约 9 个月。

### 核心创新点

1. 以端到端 request latency、TBT 和 tokens/Joule 作为主要指标，而非 peak FLOPS。
2. 通过本地 HBM slice、专用 collective 和显式布局降低数据与同步延迟。
3. 把 AI 代理纳入 RTL/QoR/内核优化闭环，缩短 late-stage 设计迭代。

### 与现有方法的关键区别

Jalapeño 不是把更多通用计算单元堆在统一 NoC 上，而是用空间划分和本地内存视图暴露通信成本，让编译器/AI 生成映射。与多种异构加速器跨系统搬运 KV 相比，它把 compute、memory、network 的异构性放在单芯片内部，并用门控适配 phase 比例。

## 证据、案例与论证

### 证据设置

- **工作负载**：GPT-OSS 120B、DeepSeek R1 670B、Kimi K2.5 1T；标称 ISL/OSL 8K/1K、权重 MXFP4/f4。
- **基线方法**：SemiAnalysis InferenceX 2026-07 公共 Pareto 数据，GB200/GB300；Jalapeño 通常为 STP，部分对照为 MTP。
- **评估指标**：峰值 mixed TPS/kW、端到端 latency、最小 TBT、在基线最佳 TBT 下的 throughput/perf/W。

### 主要结果

| 模型/比较 | 峰值 TPS/kW | 端到端延迟 | 最小 TBT | 基线最佳 TBT 下吞吐 |
| --- | ---: | ---: | ---: | ---: |
| GPT-OSS 120B：Jalapeño vs GB200 | 约 1.9× | 约 1.7× 低 | 约 2.7× 低 | 约 53.7× |
| DeepSeek R1 670B：Jalapeño vs GB300 | 约 1.7× | 约 3.6× 低 | 约 4.1× 低 | 约 104.3× |
| Kimi K2.5 1T：Jalapeño vs GB300 | 约 1.5× | 约 3.4× 低 | 约 3.8× 低 | 约 56.1× |

以上数字来自第 10--21 页的 InferenceX 图表，属于公开、功率归一化的厂商引用结果；例如 GPT-OSS 峰值为 85,448 vs 44,960 mixed/kW，DeepSeek 为 19,641 vs 11,781，Kimi 为 18,195 vs 11,862。

### 消融/案例要点

- AI 辅助设计相对“optimized human baseline”报告 BF16 multiply +56%、FP4 dot +21%、FP32 accumulator +10%，矩阵单元面积 -10%、SIMD 单元面积 -8%（第 25 页）；这是 block-level PPA 结果，不是整芯片收益。
- 内部 AI 优化的 attention/MoE kernel 相对已有专家实现约 1.5--1.8×，但模型、编译器和硅上执行配置未完整披露。
- Jalapeño STP 对 GB300 MTP 的附录比较显示收益缩小到约 1.5× perf/W、2.2× latency、2.1× TBT、8.6× matched throughput，说明多 token prediction 基线会显著影响结论。
- “带宽-only ceiling”估算 1,000--2,000 token/s/user（无 speculative decoding）或 5,000--10,000（有 speculative decoding）明确标注为理论上限，现实远未达到。

## 局限性与未来方向

- **演讲范围明确的局限**：InferenceX 是厂商引用的第三方数据，功耗采用 package TDP 归一化；Jalapeño 使用 STP、部分基线使用 MTP，比较不是完全同口径。
- **方法限制**：空间架构和显式放置增加编程/编译负担；工作负载比例、speculative acceptance、上下文长度变化可能导致局部资源失衡。
- **潜在改进方向**：公开完整 InferenceX run configuration、芯片/系统实际功耗、P99/TBT 分布、编译时间和故障数据；统一 STP/MTP、模型量化和服务 SLA 后再做第三方复现。
- **证据边界**：首片硅和 PPA block 数字是演讲陈述；未提供完整 post-layout、量产良率、RTL 覆盖率或长期可靠性。

## 个人点评

- **亮点**：明确指出“架构暴露的数据移动延迟”比原始 HBM 带宽更可能主导观察到的延迟，也把设计流程和真实 workload 绑定。
- **不足**：STP/MTP 不同基线会放大或缩小优势；AI 辅助 PPA 的搜索空间、失败率和可维护性没有量化。
- **启发**：自研 ASIC 的价值不只是更高 TOPS，而是把软件可见的局部性、collective 和 phase gating 固化成可优化的执行模型。

## 工程化三问总结

### 1. 它解决了什么瓶颈？

- **应用场景与核心瓶颈**：OpenAI frontier 模型的低延迟、多芯片推理；瓶颈是 HBM/网络/同步路径上的操作数到达延迟和阶段性资源闲置。
- **现有方法为何不足**：统一内存/NoC 产生争用，独立异构加速器需要搬运 KV；峰值 FLOPS 或理论 HBM 带宽不能代表 request-level SLA。
- **论文或文档证据**：三阶段瓶颈图、local/global 网络带宽设计、InferenceX 的 1.5--1.9× perf/W 和 1.7--4.1× 延迟指标；比较口径和第三方方法需保留限定。

### 2. 用了什么结构或训练方法？

- **整体结构与数据流**：64 个 core slice 各自连接 HBM slice，本地路径优先；collective 网络承担常见跨 core 通信，通用 NoC 和两级 Clos 处理其余流量。
- **关键模块/结构**：tensor/SIMD/scalar engine、L1、HBM4、specialized collectives、Gluon/TensorInfo、分布式控制、phase-aware clock/power gating。
- **训练目标、损失函数或优化方法**：不适用；AI 用于 RTL/QoR 和 kernel 优化，不是模型训练方法。
- **数据与训练策略**：不适用；使用 GPT-OSS、DeepSeek、Kimi 的推理 workload，具体权重映射和编译数据未公开。

### 3. 对芯片架构、RTL、验证有什么启发？

- **芯片架构**：将本地内存、collective、全局 NoC 和 phase gating 作为可组合层次；按 tensor/expert 流量配置不同网络带宽。
- **RTL**：可实现为 core slice/HBM slice 接口、collective engine、Gluon 映射支持、低延迟 reduction、NoC/Clos 路由和门控电源遥测；协议、时钟域和 TDP 控制细节为 `TBD`。
- **验证**：覆盖布局/放置与地址一致性、collective 顺序与数值误差、局部/全局拥塞、STP/MTP 执行一致性、phase gating 唤醒、低 TBT/P99 latency 以及 AI 生成 RTL 的回归和形式性质。
- **推断边界**：演讲没有给出完整 RTL、门级时序、验证覆盖率或 silicon measurement；上述项是工程推断。
