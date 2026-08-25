# 深度分析：《Patterns behind Chaos: Forecasting Data Movement for Efficient Large-Scale MoE LLM Inference》

## 基本信息

- **标题**：Patterns behind Chaos: Forecasting Data Movement for Efficient Large-Scale MoE LLM Inference
- **文档类型**：论文（ISCA 2026 接收论文的预印本）
- **作者**：Zhongkai Yu、Yue Guan、Zihao Yu、Chenyang Zhou、Zhengding Hu、Shuyi Pei、Yangwook Kang、Yufei Ding、Po-An Tsai
- **机构**：UC San Diego、Indiana University Bloomington、Columbia University、Samsung Semiconductor、NVIDIA
- **发表 venue**：ISCA 2026（论文首页注明 Accepted to ISCA 2026；当前版本为 arXiv v5）
- **年份**：2026
- **链接**：[arXiv:2510.05497](https://arxiv.org/abs/2510.05497)；专家选择 trace：[Hugging Face 数据集](https://huggingface.co/datasets/core12345/MoE_expert_selection_trace)

## 一句话总结

> 论文对 235B--1000B 规模 MoE 模型的专家选择进行跨模型、跨任务的数据搬运画像，证明看似随机的路由具有时间和空间结构，并据此设计 Wafer-scale GPU 的任务分配/本地 HBM 预测机制，以及真实多 GPU 集群的 prefill-aware 专家放置策略。

## 研究动机与问题定义

- **要解决的核心问题**：MoE 只激活少量专家，但 token 到专家的动态路由会造成专家权重搬运、GPU 间 All-to-All、远端 HBM 访问和单元负载不均衡。在 DeepSeek-V3 的 4K 序列示例中，MoE All-to-All 与 MoE 权重搬运占不同服务配置总延迟的约 60%--90%（图 2，论文证据）。
- **现有方法的不足**：既有工作多从 CPU/GPU offload、多 GPU 通信或某种加速器出发，属于系统中心（system-centric）优化，难以区分平台因素与模型路由本身的规律。若专家选择完全随机，DeepSeek-V3 的 256 个专家中选 8 个会有 \(\binom{256}{8}=4,426,165,368\) 种组合，预取、缓存、复制和负载均衡都难以进行。
- **本文的切入角度**：改用模型中心（model-centric）、系统无关的专家选择 trace 分析，分别研究 layer-level、token-level、prefill/decode-level 时间关系，以及 single-expert imbalance、expert-pair co-activation 空间关系，再将规律映射到两类硬件/软件系统。

## 核心方法

### 方法概述

论文在 2025 年发布的四个大规模 MoE 模型上收集所有层、所有 token 的专家选择记录：DeepSeek-V3（671B）、Llama 4 Maverick 128E（402B）、Qwen3-235B（235B）和 Kimi K2（1000B）。数据覆盖超过 24,000 个请求、超过 2,000 GPU 小时，形成超过 70,000 条 trace、总计超过 150 GB 的 JSON 数据。请求来自 MMLU、MMLU-Pro（含中文）、ChineseSimpleQA 和 LiveCodeBench 等不同任务与语言。

分析框架将专家选择关系分成两类。时间关系用于预测未来专家并指导预取、缓存、迁移；空间关系用于估计专家负载和共激活关系，并指导专家放置、任务分配与复制。作者从五个观察（Ob1--Ob5）归纳出六条系统设计 insight。

### 关键技术细节

- **三种时间关系**：Ob1 比较相邻 MoE 层的专家选择；Ob2 比较同一层相邻 token；Ob3 比较 prefill 与 decode 阶段的专家热度和专家对热图。三者分别对应短 reuse distance、较长 reuse distance 和 decode 初期的冷启动预测。
- **两种空间关系**：Ob4 统计单个专家在每层的激活频率及任务/语言影响；Ob5 统计专家对的共激活概率。Llama4 每层只选一个专家，因此不适用专家对共激活分析。
- **Prefill-data-driven prediction**：利用 prefill 期间已获得的专家选择历史，预测 decode 初期的热点专家；论文以条件 CDF、热图相似度、Spearman 相关和 top-k overlap 验证其可行性。
- **Wafer-scale GPU 设计**：在单 GPU 视图下增加 Global Command Processor（Global CP）与每 die 的 Local CP；在 D2D controller 中加入 Address Translation Unit（ATU）和 Prediction Unit（PDU）。Global CP 保存专家分布表与 cross-token heatmap，PDU 维护 `cp_en`/`is_local` 预测表，控制远端专家是否复制到本地 HBM。
- **任务分配算法**：对每个专家先生成包含存储 die 及相邻 die 的候选集合，再按负载排序，以 50 个请求为块使用包含 DRAM 访问、计算和 D2D 通信的成本模型选择目标 die；这是对 NP-hard 分配问题的启发式近似。
- **Prefill-guided placement**：在 8×H100 上针对 Qwen3-235B 实现 Remap-based 和 Duplication-based 两种放置。前者保持每 GPU 专家数不变并重映射热点专家，后者为每 GPU 增加一个专家槽并复制热点专家，均使用 prefill 频率和 roofline cost 估计负载。

### 核心创新点

1. 在 235B--1000B 规模、四个最新 MoE 模型和多任务 trace 上，给出面向数据搬运的系统无关画像，而不是只分析单个平台。
2. 将专家选择的时间相关性和空间偏斜归纳为六条可执行的 serving insight，并明确将不同时间尺度映射到不同内存层级。
3. 用一个 Wafer-scale GPU 的硬件/软件协同案例和一个真实 8×H100 案例验证这些 insight 的可迁移性；前者改变极少硬件模块，后者不改模型参数或训练过程。

### 与现有方法的关键区别

既有方法通常围绕 CPU--GPU offload、GPU 间通信或特定加速器做部署优化；本文先从模型 trace 中提取可复用的路由规律，再将同一组规律分别实现为硬件托管的本地 HBM 管理、Wafer 上的任务分配和 GPU 集群的专家放置。因此论文的主贡献是 profiling 与设计原则，两个系统实现是验证这些原则的 case study，而不是新的 MoE 训练算法。

## 实验与结果

### 实验设置

- **数据集与请求**：MMLU、MMLU-Pro（含中文版本）、ChineseSimpleQA、LiveCodeBench；总计超过 24,000 个请求。附录说明 trace 由 MMLU 预先记录并公开，实验脚本自动下载。
- **模型**：DeepSeek-V3、Llama4-Maverick-128E、Qwen3-235B、Kimi K2；参数规模 235B--1000B。
- **Trace 收集**：SGLang 部署在 8×H100 DGX 与 8×H200 AWS 实例上，收集所有层和 token 的专家选择。
- **Wafer-scale 仿真**：自研 Python event-driven、多 chiplet simulator，建模 LLC、HBM、计算单元和 D2D 链路；验证误差在 8×H100 单 GPU 与 P2P 测试中均不超过 5%（图 13）。配置包括 5×5 Dojo mesh 和 8×3 TSMC-SoW mesh；每 die 为 H100-like（FP16 1,000 TFLOPS、80 GB HBM、本地 3.35 TB/s、相邻 D2D 1.7 TB/s）。
- **指标**：Case Study 1 测量 decode 阶段 MoE layer throughput、Manhattan hop count 和 DRAM 访问；Case Study 2 测量三层专家线性层的 MoE computation time，不计 attention、All-to-All 和 top-k 时间。

### 主要结果

| 方法或观察 | 结果 | 证据位置与强度 |
|---|---:|---|
| Ob1：跨层关系 | 下一层 top 20% 候选覆盖条件概率质量：DeepSeek-V3 50%、Qwen3 65%、Llama4 77%、Kimi K2 56% | 图 4；论文统计证据 |
| Ob2：跨 token 关系 | 下一 token top 20% 候选覆盖：DeepSeek-V3 47%、Qwen3 62%、Llama4 80%、Kimi K2 53% | 图 5；论文统计证据 |
| Ob3：prefill/decode 关系 | 大多数层的热图 Spearman 相关较强（通常 \(\rho\ge 0.7\)）；prefill top-5 约覆盖 decode top-5 的 60%，top-10 约 75%，top-20 约 90% | 图 6--7；论文统计证据 |
| Ob4：单专家偏斜 | Llama4 第 7 层部分专家激活频率超过平均值 16 倍；热点随任务和语言明显变化 | 图 8；单层/特定模型示例，不能外推为所有层的统一倍数 |
| Ob5：专家对共激活 | 部分专家对的概率为随机基线的 20--40 倍；最高频的 10% 专家对贡献约 60%--80% 共激活 | 图 9；DeepSeek/Qwen 统计证据 |
| Case Study 1：Allo+Pred | 相对 Base，DeepSeek-V3、Kimi K2、Llama4、Qwen3 throughput 分别提升 7.0×、8.2×、7.3×、4.1×；Dojo 与 TSMC-SoW 分别约提升 6.0×、7.5×，摘要报告平均 6.6× | 图 12；event-driven 仿真结果 |
| Case Study 1：相对 EP | batch=16,384 时 Allo+Pred 比 expert parallelism 快 1.44×；batch=4,096 时二者接近 | 图 12/正文；仿真结果 |
| Case Study 2：prefill-aware placement | Remap 和 Dup 相对 Default 分别提升 15.5% 和 12.5%，均在不可用的 oracle Best 的 10% 以内，并比 Worst 快超过 2× | 图 17；8×H100 实测 |

### 消融实验要点

- 在 Wafer-scale 仿真中，Pred Only 将 hop count 降低 4.5×、性能提升约 3.0×；Allo Only 将 hop count 降低 142×、性能提升 6.3×；Allo+Pred 将 hop count 降低超过 213×、性能提升 6.63×，但相对 Allo Only 的平均性能增益只有约 1.1×。这说明通信消除后瓶颈转向任务负载分布，而不是 hop count 本身。
- Case Study 1 在 batch=4,096 时接近 EP，batch 增大到 16,384 后才体现跨 die 拆分一个专家的收益，说明方法依赖每专家 token 数和计算/搬运比例。
- Remap 与 Dup 在 Case Study 2 的收益相近：Remap 节省额外 HBM 容量，Dup 以一个额外专家槽换取更直接的热点复制，适合不同内存约束。

## 局限性与未来方向

- **作者或实验范围明确的局限**：只分析四个 2025 年发布的模型；Case Study 2 仅为 EP8，默认布局的最大/最小执行时间比约 1.3×，作者预计在更大 EP 规模下收益更高。Case Study 1 依赖 Wafer-scale GPU 的单 GPU 编程模型和仿真平台。
- **证据强度限制**：Wafer-scale 结果来自 event-driven simulator；面积/功耗来自 Yosys、CACTI 和 ARM core 数据的 5 nm 缩放估算，表 II 给出的总面积和功耗开销均约 0.04%，不是完整 RTL 综合、流片或 silicon measurement。
- **方法限制**：任务分配是带候选 die、固定块大小 50 的启发式近似；预测表和热图面向当前模型/工作负载，路由分布变化或预测错误时的缓存污染、回退开销和尾延迟没有被充分量化。
- **泛化限制**：MMLU 等离线 trace 能覆盖多任务和中英文差异，但不能证明所有领域、语言、长上下文和未来路由器都具有同样强度的相关性；“系统无关”应理解为设计原则可迁移，而不是性能增益无需重新标定。
- **未来方向**：扩大模型、任务和 EP 规模，加入在线热图更新、置信度/错误预测处理、尾延迟与能耗指标，并在真实多 chiplet 硬件或更完整 RTL 流程上验证 ATU/PDU 和 Global/Local CP 的时序、面积与功耗。

## 个人点评

- **亮点**：论文把“专家路由看似随机”转化为可量化的条件概率、热度偏斜和共激活结构，尤其是将 layer-level 与 token-level 的不同 reuse distance 对应到 LLC 与 DRAM，是连接模型行为和存储层次的清晰抽象。两组 case study 也验证了同一 profiling 结论可落到未来 Wafer-scale 硬件和现有 GPU 软件。
- **不足**：六条 insight 主要是相关性统计，尚未系统报告预测错误率、预取误命中代价、缓存容量敏感性和端到端 p99；四模型 trace 与两类硬件配置仍不足以证明所有 MoE 部署都获得相同收益。面积/功耗表更适合方案级估算，不应解读为已实现硬件成本。
- **启发**：MoE router trace 应成为 serving 架构的输入信号，而不只是运行时日志；可以按时间尺度设计分层预测器，按任务/语言建立热点专家目录，并把专家放置、通信调度、缓存复制和验证参考模型统一起来。

## 工程化三问总结

### 1. 它解决了什么瓶颈？

- **应用场景与核心瓶颈**：大规模 MoE LLM 的 decode serving。核心瓶颈是专家权重与 token 在 GPU、chiplet、HBM 或远端内存之间移动，以及热门专家造成的单元负载倾斜。图 2 显示 DeepSeek-V3 4K 请求中 MoE 数据搬运占 60%--90% 总延迟；图 4--9 证明路由不是独立同分布的随机选择。
- **现有方法为何不足**：部署专用的 offload、All-to-All 或专家均匀放置无法利用模型固有的时间相关性和任务相关的空间偏斜；统一 HBM 地址空间也会隐藏本地/远端访问代价。
- **论文或文档证据**：四模型、超过 24,000 请求的 trace 给出 Ob1--Ob5；Allo+Pred 在仿真中平均约 6.6×，prefill-aware placement 在 8×H100 上提升 12.5%--15.5%。这些是论文实验/仿真证据，不等于所有硬件部署的保证。

### 2. 用了什么结构或训练方法？

- **整体结构与数据流**：模型仍是 decoder-only Transformer + MoE gating，不改权重和训练流程。prefill 阶段产生专家选择 trace；Global CP 聚合专家分布和 cross-token heatmap，生成下一 token 的热点预测与任务分配；Local CP 将子 kernel 发给各 die，D2D controller 根据 PDU/ATU 将远端专家复制或重定向到本地 HBM。真实 GPU 案例则把 prefill 统计转换成 decode 初期的每层专家放置。
- **关键模块/结构**：Global CP/Local CP 两级命令处理器、Expert Distribution Table、Cross-token Heatmap、PDU Prediction Table、ATU、本地 LLC/HBM、D2D controller；软件侧为候选 die + 成本模型的任务分配器，以及 Remap/Dup placement 算法。
- **训练目标、损失函数或优化方法**：不适用。论文没有训练新的路由器或预测神经网络；预测来自离线/运行时专家选择频率和条件热图，分配与放置是启发式优化。
- **数据与训练策略**：使用 SGLang 采集的多任务、多语言专家选择 trace；附录称 artifact 使用预记录 MMLU trace，Case Study 1 为 CPU 可运行 simulator，Case Study 2 需要 8×H100。没有额外训练数据或微调步骤。

### 3. 对芯片架构、RTL、验证有什么启发？

- **芯片架构**：将 layer-level 短 reuse distance 映射到更快的 LLC，将 token-level 长 reuse distance 映射到本地 DRAM；用 Global/Local CP 感知专家位置和负载，用 PDU/ATU 做本地 HBM 复制与地址重定向；对热门专家复制/分散，对高共激活专家分置以换取并行度。D2D 拓扑、HBM 带宽和复制容量需要按 batch 与任务分布重新权衡。
- **RTL**：可拆成 Global CP 调度器、Local CP 分发接口、专家分布表、热图缓存、预测表、ATU 地址转换、PDU 命中/复制控制、D2D 请求路由和本地 HBM 写回路径。应参数化 layer/专家数和 die 数，明确远端 miss、重复请求、复制冲突、表更新与回退路径；论文中的块大小 50、候选距离 1 和固定表容量属于实现假设，不是通用 RTL 规格。
- **验证**：建立软件路由 trace 参考模型，检查专家请求到 die 的分配、复制后地址等价性、`is_local` 状态更新、热图预测和远端回退的一致性；覆盖空热图、全远端、热点突发、任务/语言分布切换、复制空间耗尽、D2D 拥塞和异常/乱序响应。性能验证应同时检查 hop count、远端 DRAM 读比例、吞吐、p99 延迟和复制带宽，不能只验证平均 speedup。
- **推断边界**：Global/Local CP、ATU、PDU 的功能和表结构来自论文架构图与正文；具体时钟域、协议、缓存一致性、NoC 流控、RTL 面积/时序和形式化性质均未被论文覆盖，属于工程推测，需标记为 TBD 并通过完整微架构规格和综合验证确认。
