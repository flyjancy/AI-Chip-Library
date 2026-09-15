# 深度分析：《Dataflow at Scale: the SN50 RDU》

## 基本信息

- **标题**：Dataflow at Scale: the SN50 RDU
- **文档类型**：产品与系统架构演讲（Hot Chips 2026）
- **作者**：Raghu Prabhakar
- **机构**：SambaNova Systems
- **发表 venue**：Hot Chips 2026
- **年份**：2026
- **链接**：[SambaNova Systems](https://sambanova.ai/)

## 一句话总结

> SN50 RDU 以显式数据流、持久化 Transformer 解码内核、片上 SRAM/外部 HBM 分层和可重叠的通信路径，把 agentic inference 的主要问题从峰值算力转化为可持续的模型带宽利用率（MBU）。

## 研究动机与问题定义

- **要解决的核心问题**：多轮 agent 推理的 decode 阶段占比高，权重和 KV cache 读取使低批量请求受内存带宽限制；增加 GPU 数量往往只增加峰值 HBM 带宽，不能同比提高模型实际使用的带宽。
- **现有方法的不足**：通用 GPU 将 Transformer 拆成许多 kernel，存在 kernel launch、同步、数据局部性和跨芯片通信开销。演讲用 B300 InferenceX 点说明 16 到 64 张 GPU 提供约 4 倍峰值 HBM 带宽时，token 速度只增加约 4%--25%，MBU 从约 69% 降到约 74%（第 3--7 页）。
- **本文的切入角度**：以 RDU 的空间数据流结构、persistent decoder kernel、PMU/PCU 协同和片内/片间流式通信，提升模型带宽利用率并在 scale-up 时保持计算与通信重叠。

## 核心方法

### 方法概述

SN50 是采用 5 nm CoWoS-L 的数据流加速器，单 RDU 宣称 3,200 TFLOPS FP8、432 MB 片上 SRAM、64 GB HBM（1.84 TB/s）以及最高 2 TB/s 的以太网 scale-up。芯片由 864 个 PCU（计算单元）和 PMU（存储单元）组成，AGCU 连接 HBM、DDR/CXL 和网络。片上 SRAM 保存热点权重和中间数据，索引加载可访问 SRAM、HBM 或 DDR。

Transformer 解码被映射为一次持久化 kernel：多个 decoder 层在同一 kernel 调用中循环，消除逐算子 launch 边界，并通过访问模式完成转置、融合 RMSNorm/激活/门控等操作。PMU 双缓冲和独立数据流让 HBM 读取、PCU 计算与网络传输并行推进。

### 关键技术细节

- **显式数据流与 PCU/PMU**：计算和存储单元通过 mesh switch 连接，数据流由编译器和运行时安排；PMU 负责片上/片外存储访问，PCU 执行 GEMM、逐元素和归约。
- **Persistent decoder kernel**：把 embedding、attention、RMSNorm、MoE/FFN、classifier 等阶段放进单次 kernel 调用，减少 launch 和同步开销，并提高跨层数据局部性（第 15--17 页）。
- **计算-通信重叠**：在 RDU 内将 HBM/DDR 访问、PCU 计算和 AGCU 数据流流水化；跨 RDU 支持 tensor、pipeline、expert、data parallel 等通信原语。
- **MoE 数据流**：支持 token dispatch/combine 的 all-to-all 和 broadcast+filter 两种形式。all-to-all 由源端分组并过滤，broadcast 以网络规则流量换取目的端过滤；索引权重加载支持将常用 expert 预取到 SRAM。
- **Scale-up/scale-out**：每个 RDU 集成 10 个 800G Ethernet 端口，文档给出 1,000 GB/s 双向带宽；定制 RDU NIC 用于 scale-up，RoCEv2 NIC 用于跨 scale-up 域的 scale-out。
- **异构解耦服务**：Dynamo、vLLM/SGLang、LMCache/NIXL 将 prefill、KV appliance 和 decode 放在不同 GPU/RDU 资源上，使 KV 迁移与解码资源可独立扩展。

### 核心创新点

1. 用 MBU（模型实际用于权重和 KV 的 HBM 带宽占比）而不是峰值 HBM 带宽作为 decode 效率的核心指标。
2. 用持久化解码 kernel 和显式数据流消除通用 GPU 的算子边界，持续复用片上 SRAM。
3. 将 TP/PP/EP/DP、KV 解耦和网络流控统一到同一套 RDU 数据流路径中，面向数百至数千颗器件扩展。

### 与现有方法的关键区别

SN50 的主要差异不是再增加一个矩阵阵列，而是让编译器预先确定操作、存储和通信的时间表，利用片上存储和独立流把 Transformer 的多个阶段融合。GPU 基线依赖运行时 kernel 调度和全局同步，SN50 则把跨层循环和通信显式暴露给数据流硬件。

## 证据、案例与论证

### 证据设置

- **工作负载**：DeepSeek-R1/V3、Kimi K3、GLM-5.2、MiniMax-M3/M2.7 等 frontier 模型；图表多采用 8K 输入/1K 输出、低批量 decode。
- **对比对象**：NVIDIA B300 FP4 InferenceX 测量点、DGX B300 与 NVL72 的固定功率比较，以及不同规模 SN50。
- **评估指标**：输出 token/s/user、MBU、TFLOPS 利用率、compute/communication overlap、功耗和规模扩展。

### 主要结果

| 指标/论点 | 结果 | 证据位置与强度 |
| --- | ---: | --- |
| 单 RDU 规格 | 3,200 TFLOPS FP8、432 MB SRAM、64 GB HBM@1.84 TB/s | 第 11 页；厂商规格 |
| 空冷机架 | 16 RDUs；25.6 PFLOPS BF16、51.2 PFLOPS FP8、1 TB HBM；典型 15--20 kW，最高 34 kW | 第 12 页；产品规格 |
| TP GEMM | 8/16/32 sockets 均报告 70% 以上 TFLOPS 利用率；32-SN50 图示为计算与通信完全重叠，图中峰值约 78.9% | 第 29--30 页；厂商 on-silicon benchmark |
| MoE/EP | 64-socket expert 加载图示超过 70% 带宽利用率 | 第 37 页；厂商图示 |
| SN50 扩展 | 64→128→256 SN50：约 200→300→500 tok/s/user，MBU 约 51%→44%→45% | 第 39 页；厂商/InferenceX 图示 |
| MiniMax M2.7 | SN50 解耦私有端点 763 tok/s，上一代 SN40 482 tok/s；公开服务商约 51--187 tok/s | 第 44 页；Artificial Analysis 2026-07-07 benchmark，第三方但由厂商引用 |

### 消融/案例要点

- 演讲没有论文式逐模块消融；“GPU kernel 边界→persistent decoder kernel→compute/communication overlap”是架构案例链，不代表严格控制变量实验。
- 规模从 64 到 256 RDU 时 token 速度增加而 MBU 仍保持约 44%--51%，支持“可扩展的有效带宽”论点，但配置、并发、软件版本和服务 SLA 未完整公开。
- MBU 公式把 token 速度写成 HBM 带宽、MBU、芯片数与参数/KV 字节的函数；它解释了为何仅增加芯片数会降低利用率，但不等同于端到端成本模型。

## 局限性与未来方向

- **演讲范围明确的局限**：大多数性能图为厂商演示或引用的 InferenceX 点，缺少完整请求分布、尾延迟、准确率、编译器版本和功耗测量方法。
- **方法限制**：显式数据流和 persistent kernel 依赖编译器能提前确定算子、expert 路由和内存布局；动态序列长度、稀疏路由或新算子可能降低收益。
- **潜在改进方向**：公开可复现的 kernel trace、TP/EP 拓扑和功耗数据；补充故障恢复、长上下文 KV 解耦、不同 batch/并发下的 MBU 与 P99 延迟。
- **证据边界**：763 tok/s 是 Artificial Analysis 服务端 benchmark，SN50 规模图和 MBU 是演讲材料；不能直接推导芯片面积、硅后 PPA 或所有模型上的收益。

## 个人点评

- **亮点**：用 MBU 把“硬件能搬多少数据”和“模型真正用掉多少带宽”分开，指标比单纯峰值 FLOPS 更贴近 decode。
- **不足**：持久化 kernel 的编译复杂度、动态 MoE 路由的最坏情况和大规模网络拥塞处理披露有限；第三方结果的配置不够完整。
- **启发**：AI 加速器评审应同时记录算子边界、有效带宽、KV/权重驻留、通信重叠和功耗，而非只比较单芯片 TOPS。

## 工程化三问总结

### 1. 它解决了什么瓶颈？

- **应用场景与核心瓶颈**：agentic inference 的 decode 阶段受权重/KV 带宽、kernel launch、同步和跨 RDU token 通信限制。
- **现有方法为何不足**：GPU 增加 HBM 峰值带宽时，模型带宽利用率可能下降；通用运行时还会引入算子边界和全局同步。
- **论文或文档证据**：B300 16→64 GPU 图、SN50 64→256 规模图、32-socket overlap 结果和 763 tok/s 案例；均为演讲或厂商引用证据。

### 2. 用了什么结构或训练方法？

- **整体结构与数据流**：PCU 计算、PMU 存储和 AGCU 网络端口组成显式数据流；片上 SRAM 保存热点数据，HBM/DDR 承担容量，persistent kernel 循环执行 Transformer decoder。
- **关键模块/结构**：864 PCU/PMU、mesh switch、432 MB SRAM、64 GB HBM、双缓冲 PMU、索引加载、RDU NIC/RoCE NIC、TP/PP/EP/DP 通信原语。
- **训练目标、损失函数或优化方法**：不适用；这是推理硬件和运行时演讲，优化目标是 MBU、token 速率和通信重叠。
- **数据与训练策略**：不适用；使用 DeepSeek、Kimi、GLM、MiniMax 等模型的推理工作负载，具体模型权重和编译配置未披露。

### 3. 对芯片架构、RTL、验证有什么启发？

- **芯片架构**：把 persistent kernel、SRAM/HBM 分层、expert 路由和通信重叠作为一个调度闭环；为大规模 TP/EP 提供可组合的流式网络。
- **RTL**：可落地为 PCU/PMU 数据流 FIFO、双缓冲 DMA、索引寻址、all-to-all/broadcast 流控、网络包重排和性能计数器；具体时序、协议和一致性规则为 `TBD`。
- **验证**：覆盖多 decoder 融合的数据依赖、动态 MoE token 路由、FIFO 满/空和乱序、HBM/DDR 回退、通信-计算同时 backpressure，以及 MBU/吞吐计数与软件模型一致性。
- **推断边界**：演讲未公开 RTL、门级 PPA、形式验证或硅后故障数据；以上 RTL 与验证项是基于架构图的工程推断。
