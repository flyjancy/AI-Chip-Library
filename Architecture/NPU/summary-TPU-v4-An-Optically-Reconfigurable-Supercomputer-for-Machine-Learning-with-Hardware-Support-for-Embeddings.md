# 深度分析：《TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings》

## 基本信息

- **标题**：TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings
- **文档类型**：论文（工业产品论文）
- **作者**：Norman P. Jouppi、George Kurian、Sheng Li、Peter Ma、Rahul Nagarajan 等
- **机构**：Google、University of California Berkeley
- **发表venue**：2023 ACM/IEEE ISCA
- **年份**：2023
- **链接**：[DOI:10.1145/3579371.3589350](https://doi.org/10.1145/3579371.3589350)

## 一句话总结

> TPU v4 将 4096 芯片 3D torus、可重构光路交换（OCS）和专用于 embedding 的 SparseCore 结合，在提高故障可用性和 all-to-all 带宽的同时，将 TPUv3 性能提升 2.1×、性能/瓦提升 2.7×。

## 研究动机与问题定义

- **要解决的核心问题**：LLM 和推荐模型同时带来大规模 dense 计算、embedding 的不规则 gather/scatter、all-to-all 通信和长时间训练的可靠性问题。
- **现有方法的不足**：静态 2D/3D torus 难以绕过失效节点和适配不同 slice；TensorCore 不适合低算术强度 embedding，CPU DRAM 又成为 Amdahl 瓶颈。
- **本文的切入角度**：用 OCS 动态重连拓扑与故障旁路，用 SparseCore 将 HBM/ICI 组织成全局 embedding 内存，并用 PA-NAS 共同优化模型和拓扑。

## 核心方法

### 方法概述

TPU v4 supercomputer 使用 64 个 4×4×4 building block、48 个 136×136 端口 OCS 连接 4096 个 TPU 芯片。OCS 用 MEMS 镜面在毫秒级重配置光路，支持 regular、twisted 3D torus 等 slice 拓扑，并允许在 CPU host 失效时重新分配资源。

单颗 TPU v4 包含两个 TensorCore，每个 TensorCore 有四个 128×128 MXU、128-lane VPU 和 16 MiB VMEM；两者共享 128 MiB CMEM，封装带 4 个 HBM。SparseCore 是独立的 16-tile dataflow 处理器，每个 tile 连接一个 HBM channel，配有 Fetch、8-wide SIMD scVPU、Flush 和 Cross-Channel Units，用于 embedding 的 gather/scatter、去重和 all-to-all。

### 关键技术细节

- **OCS**：光路成本低于系统成本 5%、功耗低于 3%；光纤带宽可跨 4096 芯片切换，绕过故障节点（第 2–4 页）。
- **SparseCore**：约占 die 面积和功耗各 5%；在 TPU v4 中形成 128 TiB 全局可寻址 embedding 内存（第 5–6 页）。
- **3D torus**：bisection bandwidth 随芯片数按 `N^(2/3)` 扩展，相比 2D torus 更适合 embedding all-to-all。
- **PA-NAS**：平台感知 NAS 同时选择 DNN sparse/dense 层，使模型质量、吞吐和 topology 形成 Pareto 优化（第 6–8 页）。

### 核心创新点

1. 首个大规模生产部署的 OCS 可重构 ML supercomputer 互连。
2. 将 embedding lookup 从 TensorCore/CPU 中剥离为 SparseCore，并用全局 HBM/ICI 处理不规则访问。
3. 用 ML 搜索模型结构与拓扑，适配快速变化的生产 workload。

### 与现有方法的关键区别

TPU v4 不仅增加矩阵算力，而是针对 all-to-all、故障、embedding 和多租户重新设计系统级资源；相对于 NVSwitch/InfiniBand，OCS 以 circuit switching 低功耗提供大规模拓扑可编程性。

## 实验与结果

### 实验设置

- **数据集/工作负载**：Google 生产 DNN，包括 DLRM、Transformer/LLM、BERT、CNN；embedding 使用 DLRM0。
- **基线方法**：TPUv3、Graphcore IPU Bow、NVIDIA A100，以及 CPU/外部 variable server 放置 embedding 的配置。
- **评估指标**：系统吞吐、性能/瓦、功耗、拓扑 bisection bandwidth、embedding lookup、MLPerf Training 和 CO2e。

### 主要结果

| 指标 | 结果 | 证据与边界 |
| --- | ---: | --- |
| TPU v4 相对 TPUv3 | 性能 2.1×，性能/瓦 2.7× | 摘要、第 10 页 |
| 系统规模 | 4096 chips，约为 TPUv3 的 4×，整体近 10× 快 | 摘要、第 2–4 页 |
| SparseCore | embedding 加速 5×–7×，约占 5% die/功耗 | 摘要；生产结果第 6 页 |
| Embedding | TPU v4 比 TPUv3 最高 3.1×、比 CPU 30.1×；放 CPU 会慢 5×–7× | 第 6 页图 9 |
| LLM 训练利用率 | 约 60% 峰值 FLOPS | 摘要 |
| 对比 A100/IPU | A100 速度 1.2×–1.7×、TPU v4 功耗低 1.3×–1.9×；IPU 速度约 4.3×–4.5× | 摘要/第 9–10 页 |
| OCS 成本/功耗 | <5% 系统成本、<3% 系统功耗 | 第 4 页 |

### 消融实验要点

- TPU v3/v4 3D/2D torus 的 bisection bandwidth 比较显示，embedding 性能随拓扑规模明显改善（第 6 页图 8）。
- PA-NAS 在 CNN1 上报告约 1.6× 性能提升且精度相近；在 DLRM0 上通过平衡 SparseCore/TensorCore 消除约 25% SC 空闲（第 7–8 页）。
- CMEM 从 0 增至 128 MiB 时，生产应用平均性能从 69% 上升到 98%；MLPerf server 从 76% 上升到 100%（第 10 页图 13）。

## 局限性与未来方向

- **作者提到的局限**：OCS 采用电路交换，连接必须 1:1；与 IB 的 apples-to-apples 比较复杂；MLPerf DLRM 与生产 DLRM 规模和特征分布不同。
- **潜在改进方向**：更高端口数/波分复用、scale-up/scale-out 融合、可编程拓扑和对不断变化的 LLM/embedding workload 的持续 co-design。
- **证据边界**：部分 MLPerf 结果标注为 unverified，A100/IPU 对比来自相近时期和不同系统配置，不能直接视为芯片级公平基准。

## 个人点评

- **亮点**：把网络可靠性、拓扑、embedding 稀疏访存和模型搜索作为一个生产 DSA 的必要组成部分。
- **不足**：OCS 的调度软件、故障切换时延和 SparseCore 的详细微架构/RTL 未充分公开。
- **启发**：面向推荐模型和 MoE 的 NPU 不应只扩展 dense MXU，应为不规则 memory access 和 all-to-all 设计独立数据流单元。

## 工程化三问总结

### 1. 它解决了什么瓶颈？

- **应用场景与核心瓶颈**：4096 芯片 ML 训练、embedding gather/scatter、all-to-all bisection bandwidth、节点故障和多租户部署。
- **现有方法为何不足**：静态 torus 难以绕过失效设备；TensorCore/CPU 处理 embedding 会受算术强度、HBM/DRAM 和网络带宽限制。
- **论文或文档证据**：OCS <5% 成本/<3% 功耗、SparseCore 5×–7×、TPU v4 2.1×/2.7×、embedding 放 CPU 慢 5×–7×；部分对比为生产或 unverified MLPerf 证据。

### 2. 用了什么结构或训练方法？

- **整体结构与数据流**：TPU v4 TensorCore 处理 dense 算子；SparseCore 从 HBM fetch embedding，经过 scVPU/Cross-Channel Units，再 flush 更新；ICI/OCS 负责跨芯片交换。
- **关键模块/结构**：OCS、3D/twisted torus、TensorCore MXU/VPU/VMEM/CMEM、16-tile SparseCore、PA-NAS。
- **训练目标、损失函数或优化方法**：不提出新的模型损失；PA-NAS 做平台感知结构搜索，兼顾质量和性能。
- **数据与训练策略**：生产 DLRM、Transformer、CNN/LLM；按照 slice 形状和 workload 动态选择拓扑。

### 3. 对芯片架构、RTL、验证有什么启发？

- **芯片架构**：为稀疏 embedding 建立独立 dataflow core、全局地址空间和高 bisection interconnect；拓扑重配置应纳入系统 RAS 和调度。
- **RTL**：实现 Fetch/Flush、可变长度 gather/scatter、dedup、HBM channel partition、ICI router、OCS 控制与故障旁路；具体端口协议为 `TBD`。
- **验证**：覆盖不规则地址、重复特征去重、跨芯片 all-to-all、拓扑重配置、节点失效恢复、CMEM/HBM 一致性、租户隔离和 P99 延迟。
- **推断边界**：论文未给出 SparseCore RTL、OCS 控制器实现或形式验证；工程建议来自公开方框图和系统行为。
