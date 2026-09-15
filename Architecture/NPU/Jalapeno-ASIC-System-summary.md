# 深度分析：《You Can Just Build Things … Chips》（Jalapeño ASIC + System）

## 基本信息

- **标题**：You Can Just Build Things … Chips（Jalapeño ASIC + System）
- **文档类型**：厂商技术幻灯片（Hot Chips，34 页含附录）
- **讲者**：Richard Ho（VP Hardware）、Ravi Narayanaswami、Chris Leary（Member of Technical Staff）
- **机构**：OpenAI（合作方：**Broadcom** 与 **Celestica**）
- **发表 venue**：Hot Chips（具体年份未在文中标注；按时间线中 2026 年 7 月为最后节点，判断为 2026 年）
- **年份**：2026
- **链接**：性能数据引自 **SemiAnalysis InferenceX**（公开的功耗归一化对比）

## 一句话总结

> OpenAI 用 **9 个月从初始 RTL 到 tapeout**做出第一颗自研推理芯片 Jalapeño：单芯片 **700 W、mxfp4×mxfp4 13.4 PFLOP/s、15.4 TB/s + 216 GiB HBM4**，配合 128 芯片本地域（600 GB/s）与 2048 芯片全局域（200 GB/s）的"半扁平化"两级 Clos；在 InferenceX 的功耗归一化对比中宣称相对 GB200/GB300 **交互性高 2.1–4.1×、端到端延迟低 1.7–3.6×、在基线原本最佳 TBT 点上每瓦吞吐高 8.6–104.3×**，而支撑这些数字的两个关键前提是**用 STP 对比基线的 MTP**、以及**只按 package TDP 归一化**。

## 研究动机与问题定义

- **要解决的核心问题**：推理与 agentic 负载的经济性——**"requests/second/watt at the required SLA latency"**。幻灯片把用户关心的指标拆成两组：

| | 用户体验 | 推理效率 |
| --- | --- | --- |
| 指标 | 请求延迟：TTLT（time to last token）、TBT | 每请求能耗：energy/token、tokens/joule、tokens/s/kW |

  核心权衡被明确写出：**单用户延迟 ↔ 每 token 能耗；更低的延迟通常要付出更高的每 token 能耗**。因此评价方式必须是"跨整条 Pareto 前沿比较"，而非单点对比。

- **明确列出的 non-goals**：**芯片数量、每芯片吞吐、TTFT**。幻灯片写道 "Compare efficiency @ matched user experience"——即不看单芯片吞吐，而看在相同用户体验下的效率。
- **评价基准的选择**：采用 **SemiAnalysis InferenceX**（公开、功耗归一化），理由是它覆盖多个 OSS 模型而非单一手挑内核、覆盖完整的 prefill/decode Pareto、测量端到端请求路径而非峰值 FLOPS 或孤立内核，且对硬件专属优化开放（各厂商可用自己的完整软硬件栈调优）。工况固定为 **nominal ISL/OSL = 8k/1k、权重 dtype f4、按 package TDP 归一化**：**Jalapeño 700 W、GB200 1.2 kW、GB300 1.4 kW、MI355X 1.4 kW**。
- **agentic 负载的三个不同硬件区间**（第 13 页）：

| 阶段 | 瓶颈 | 特征 |
| --- | --- | --- |
| **01 Prefill（编码上下文）** | FLOPs + attention | 计算重、内存带宽需求低、通信易于平滑调度（Compute HIGH / Memory BW LOW / Comms SMOOTH） |
| **02 Draft model（推测）** | **网络延迟** | 小模型、超低 batch；网络带宽低但**延迟极端敏感**（Model SMALL / Batch ULTRA-LOW / Comms LATENCY BOUND） |
| **03 Spec-verify（解码）** | Attention + HBM 带宽 | 用 GQA 与 spec token 后 attention 不再是内存瓶颈，**MoE 在低延迟下是 HBM 带宽饥渴的**，通信呈突发（Comms BURSTY） |

- **由此推出的架构论点：把异构性放进芯片内部，而不是放在机架上**。幻灯片用一个对比说明：
  - **专用异构集群**：prefill 芯片活跃时 draft 芯片与 verify 芯片闲置，但**闲置加速器仍要付出封装、HBM、I/O、网络与制冷的基础功耗**；而且**KV 状态必须在系统之间搬移**。
  - **单芯片内部异构**：按阶段激活 compute / memory / network 的正确配比，未用的块"熄灭"（gated）。
  - 结论句：**"dark silicon is cheaper than idle accelerators"** 与 **"Locality is king: keep KV local and activate the right resources"**。
- **对 HBM 带宽天花板的量化**（第 16 页）：128 芯片聚合 **1+ PB/s** ÷ 0.5 TB（1T 参数 × 4 bit）**= 2,000+ 次完整模型权重读取/秒**。理论上限：**不含投机解码 1,000–2,000 tokens/s/user；含投机解码 5,000–10,000**。幻灯片的评述是"**Reality: we are nowhere close to either rate**"，因此"raw HBM bandwidth is not the sole limiting resource"——这直接引出下一节的架构论点。
- **为什么延迟能被架构主导**：幻灯片列出三类"长延迟路径"——长延迟路径（网络、内存系统、全局归约）、操作数到得太晚（需要时不在寄存器里）、计算单元被阻塞等操作数。根因被归为：**统一内存子系统倾向于形成高争用路径；独立内核不同步使全局内存 fence 昂贵；集中式资源中介网络访问**。结论句：**"→ NOT hardware speed of light"**，即这些延迟不是物理极限。

## 核心结构（架构与规格）

### 片上架构：core slice 与 HBM slice 配对

- **把每个 core slice 与一个 HBM slice 配对**，形成一个低延迟、高带宽的本地视图（幻灯片画的是 64 个 core slice 与 64 个 HBM slice）。
- 三层通信结构：
  1. **本地路径（最快）**：core slice ↔ 配对 HBM slice 的局部视图。
  2. **专用跨核集合通信网络**：为常见通信模式优化带宽与延迟。
  3. **共享 NoC（较慢但更灵活）**：通用通信与访问 scale-up 网络，另接 **Scale-up Ethernet bridge**。
- 设计原则：**常用操作数保持本地，只在必要时才走共享路径**；**显式放置与分布式控制**防止数据搬移与同步主导执行时间。
- **编程模型**：Jalapeño 是**空间架构（spatial architecture）**。**Gluon** 把每个 core 当作一个 thread block 编程；多个引擎共享 L1，并有一个快速的本地内存视图；**专用集合通信**跨物理核协调；**TensorInfo** 显式捕获张量的布局与物理放置。
- **人机分工**（第 29 页）：
  - 人负责推理的部分：**简单抽象**（本地张量、显式通信、可预测的同步）。
  - 前沿 AI 负责搜索的部分：**映射（mapping）、放置（placement）、调度与流水（scheduling + pipelining）、集合通信编排（collective orchestration）**。
  - 硬件负责暴露的部分：清晰的层次、快速本地路径、较慢的全局路径、分布式控制。
  - 结论句：**"Spatial programming is onerous for humans. It is easy for frontier AI."**

### 系统与网络

- **本地域：128 颗 Jalapeño**，用 Broadcom **TH6** 交换；**全局域：2048 颗 Jalapeño**，跨 8 条 rail（rail 0–7），同样是 TH6。
- 拓扑被描述为**"half flattened" 两级 Clos**。
- 网络带宽分配**按并行类型差异化**：**Tensor Parallel 用更高的带宽，Expert Parallel 用较低的带宽，所有流量都要低延迟**。
- 幻灯片给出的量化目标：**核到核延迟**的度量口径是"从生产核结果可用到消费核使用结果"，并要求低。

### 规格（第 32 页，全部数据来源）

| | Jalapeño ASIC |
| --- | --- |
| 矩阵算力 | mxfp8 × mxfp8 = **3.4 PFLOP/s**<br>mxfp8 × mxfp4 = **6.7 PFLOP/s**<br>mxfp4 × mxfp4 = **13.4 PFLOP/s** |
| 内存系统 | **15.4 TB/s，216 GiB** |
| Scale-up 网络 | **本地 = 128 ASIC @ 600 GB/s**<br>**全局 = 2048 ASIC @ 200 GB/s** |
| 功耗 | **700 W** |
| | **Jalapeño 系统（2048 ASIC）** |
| 矩阵算力 | mxfp4 × mxfp4 = **27 EFLOP/s** |
| 内存系统 | **32 PB/s，432 TiB** |

**封装平面图**：**compute die** 居中，上方一个 **I/O chiplet**，两侧各有 HBM4 堆叠——从图中可见 **6 个 HBM4**（左 3 + 右 3）。这与容量自洽：$216\ \text{GiB} \div 6 = 36\ \text{GiB}$，正好对应 HBM4 12-Hi 的单堆容量。**注意：幻灯片正文并未给出堆叠数量，这是从平面图读出并与容量交叉验证的推断。**

幻灯片最后一行是全文的定位声明：**"Most important spec: perf/W on end-to-end workloads"**。

### 从 RTL 到硅的时间线（第 3 页）

| 时间 | 里程碑 |
| --- | --- |
| Oct'24 | 架构概念（Architecture Concept） |
| Feb'25 | 初始 RTL |
| Apr'25 | Codex CLI |
| Jul'25 | RTL Freeze |
| Nov'25 | Tapeout |
| Feb'26 | Codex Desktop |
| May'26 | 首颗硅回来；Codex 在 Jalapeño 上运行 |
| Jul'26 | ChatGPT Work |

幻灯片标注 **"9 months"**（初始 RTL → tapeout），首页称 **"From initial RTL to tapeout in 9 months"**，并强调 **"AI accelerated HW+SW co-design"**。执行范式被写成五步闭环：**Vision → Workload → Simulate → RTL + QoR → Convergence**，口号是 **"MEASURE → VERIFY → LEARN → CHANGE → REPEAT"** 与 **"Win by shortening the loop and running it relentlessly"**；幻灯片明确说 **"The specification becomes precise through the loop—not before it"**。

### AI 辅助设计（第 26–27、31 页）

用**内部 AI 模型 + XLS 硬件描述语言**搭建了一个"可控的优化面"（清晰语义、足够的可控性、快速 QoR 反馈、稳健验证），流程为 **Generate → Verify → Measure PPA**。声明的两类收益：

- **Impact 1 · late-stage velocity**：**"Major changes landed up until the day of RTL freeze"**。
- **Impact 2 · Measured PPA wins vs optimized human baseline**：
  - 改进的基本单元：**BF16 乘法 56%**、**FP4 dot 21%**、**FP32 累加 10%**；
  - 块级 PPA：**Matrix Unit 面积 10%**、**SIMD Unit 面积 8%**。
- **AI 优化内核**（第 31 页）：从功能正确的基线出发，内部 AI 系统驱动优化。结果：**GPT-OSS 的块比既有的专家手写实现在 attention 与 MoE 块上快 1.5–1.8×**，且优化后的内核在芯片上端到端验证过。

## 证据、案例与论证

### 基准设置

- **三个模型**用于展示通用性：**GPT-OSS 120B**（窄小模型，压测延迟与窄并行的极限，兼作大模型的 draft model 代理）、**DeepSeek R1 670B**（大模型、跨平台高度优化，验证向更多设备扩展）、**Kimi K2.5 1T**（1T 稀疏 MoE）。
- 幻灯片明确标注三条前提：**模型并未针对 Jalapeño 协同设计**；**Jalapeño 结果使用 STP，而选定的基线包含 MTP**；工况为 nominal ISL/OSL = 8k/1k、权重 dtype = f4。
- 速度声明：**"We got this all running between when A0 came back to the lab and now"**——即全部结果是在首颗硅回片（2026 年 5 月）之后到报告时点（约 2026 年 8 月）之间完成的。

### STP 与 MTP 的差别（第 7 页）

| | STP（单 token 预测） | MTP = 7（多 token 预测） |
| --- | --- | --- |
| 结构 | 8 次串行的大 trunk 前向 | 7 次廉价串行 draft + **1 次批量的大 trunk 前向** |
| 权重读取 | **8 次昂贵的权重读取、8 个串行 trunk 步** | **1 次昂贵的 trunk pass，最多输出 8 个 token** |

幻灯片自己的结论：**"Up to 8× fewer large-model passes → lower latency + higher throughput"**，并标注 **"Benchmark context: Jalapeño uses STP; selected comparison baselines use MTP"** 与 **"Actual speedup depends on acceptance rate, draft overhead, and verification efficiency"**。

### 主要性能结果

**GPT-OSS 120B，Jalapeño STP vs GB200（700 W vs 1,200 W）**

| 指标 | 结果 |
| --- | --- |
| 峰值混合 TPS/kW | 约 **1.9×** 更高（**85,448 vs 44,960**） |
| 端到端延迟 | 约 **1.7×** 更低（**1.03 s vs 1.80 s**） |
| 最低 TBT | 约 **2.7×** 更低（**0.69 vs 1.87 ms**，即 1,459 vs 535 tok/s/user） |
| 在基线原最佳 TBT 点的吞吐 | 约 **53.7×** 更高（**22,935 vs 427** mixed/kW，在 535.28 tok/s/user） |

**DeepSeek R1 670B MXFP4，Jalapeño STP vs GB300（700 W vs 1,400 W）**

| 指标 | 结果 |
| --- | --- |
| 峰值混合 TPS/kW | 约 **1.7×** 更高（**19,641 vs 11,781**） |
| 端到端延迟 | 约 **3.6×** 更低（**1.65 s vs 5.99 s**） |
| 最低 TBT | 约 **4.1×** 更低（**1.43 vs 5.90 ms**，700 vs 169 tok/s/user） |
| 在基线原最佳 TBT 点的吞吐 | 约 **104.3×** 更高（**12,258 vs 118**，在 169.41 tok/s/user） |

**DeepSeek R1 MXFP4，Jalapeño STP vs GB300 MTP（同一位点的"公平"对照）**

| 指标 | 结果 |
| --- | --- |
| 峰值混合 TPS/kW | 约 **1.5×** 更高（**19,641 vs 12,951**） |
| 端到端延迟 | 约 **2.2×** 更低（**1.65 s vs 3.69 s**） |
| 最低 TBT | 约 **2.1×** 更低（**1.43 vs 3.04 ms**，700 vs 329 tok/s/user） |
| 在基线原最佳 TBT 点的吞吐 | 约 **8.6×** 更高（**1,940 vs 225**，在 328.95 tok/s/user） |

**Kimi K2.5 1T MXFP4，Jalapeño STP vs GB300（700 W vs 1,400 W）**

| 指标 | 结果 |
| --- | --- |
| 峰值混合 TPS/kW | 约 **1.5×** 更高（**18,195 vs 11,862**） |
| 端到端延迟 | 约 **3.4×** 更低（**1.56 s vs 5.31 s**） |
| 最低 TBT | 约 **3.8×** 更低（**1.44 vs 5.48 ms**，694 vs 182 tok/s/user） |
| 在基线原最佳 TBT 点的吞吐 | 约 **56.1×** 更高（**6,744 vs 120**，在 182.46 tok/s/user） |

**汇总（Key Takeaways，第 12 页）**

- 交互性高 **2.1×–4.1×**；
- 端到端延迟低 **1.7×–3.6×**；
- 在 baseline 原本最佳 TBT 点的 perf/W 高 **8.6×–104.3×**；
- 峰值吞吐点的 perf/W 高 **1.5×–1.9×**。
- 另两条定性结论：优势对**内部模型更明显**；**多 token 预测能在等效率下把延迟再降 3–5×**；**前沿模型上可实现 <1 ms 的 TBT 且维持在经济的吞吐水平**。

### 附录

两页附录给出另外两个口径的对照：**用 all-in utility MW 计的 Jalapeño STP vs GB300 MTP**，以及**在同一速度（matched speed）下的对照**。这两页在文本中只有标题，图表内容需要看原页，因此本报告不引用其数值。

## 局限性与未来方向

- **STP 与 MTP 的对照口径是本材料最重要的限定**：幻灯片三处明确标注 **"Jalapeño uses STP; selected comparison baselines use MTP"**，而第 7 页论证 MTP 可以带来"最多 8× 更少的大模型前向"。也就是说，在 GPT-OSS、DeepSeek R1（vs GB300）、Kimi K2.5 三组对照中，**Jalapeño 是在不使用自己承认有效的加速技术的情况下与使用该技术的对手比较的**。这既让结果更保守（Jalapeño 未用 MTP 仍胜出），也意味着 **8.6×–104.3× 这类数字混合了架构优势与基线 MTP 的收益**。唯一干净的同技术对照是那一页专门的 "Jalapeño STP vs GB300 MTP"，其中峰值 perf/W 优势降到 **1.5×**——这才是纯架构差异的量级。
- **按 package TDP 归一化，而非实测功耗**：所有对比都用 package TDP（700 W vs 1,200/1,400 W）作分母。TDP 是散热设计点而非实际耗电，在低利用率区间与实测功耗的偏离可能很大。附录提供了 "all-in utility MW" 的替代口径，但正文未给其数值。
- **极端交互性点上的倍数需要谨慎解读**：53.7×（GPT-OSS）、104.3×（DeepSeek R1）、56.1×（Kimi K2.5）这三个大数都是**在基线自己的最低 TBT 点**（即基线最快但吞吐崩塌的位置，如 535.28、169.41、182.46 tok/s/user）上测的。在同一点上，基线每瓦只有 427/118/120 mixed，因此倍数极大。这个比较方式本身是对的（对应"matched user experience"原则），但读者应当意识到**它衡量的是"在对手最不利的工况下"的差距**，而不是典型工况下的差距。
- **结果来自早期硅的短窗口**：幻灯片自述"全部跑通是在 A0 回片到现在的这段时间"（约 3 个月）。没有给出多次运行的方差、热稳态条件或软件版本冻结情况；三个模型中可能只有部分完成了端到端优化。
- **关键的实现规格缺失**：**没有工艺节点、die 面积、晶体管数**；216 GiB 对应的 HBM4 堆叠数需要从平面图推断（6 个）；**没有任何精度/量化评估**（mxfp4×mxfp4 13.4 PFLOP/s 隐含原生 MXFP4 支持，但没有对模型精度影响的任何数据）；没有给出 HBM 容量与 15.4 TB/s 的理由（如 6 堆 × 2.57 TB/s）。此外 **"most important spec: perf/W on end-to-end workloads" 这一行下面没有给出任何绝对 perf/W 数值**。
- **架构论点的量化支撑较薄**：本地/远程分层（core slice 配 HBM slice、专用集合通信网络、共享 NoC）的收益只有定性论述（"延迟不是物理极限"、"dark silicon is cheaper than idle accelerators"），**没有给本地与共享路径的延迟/带宽对比数字**。"核到核延迟"只给了度量口径而没有数值。
- **AI 辅助设计的收益是相对值**：BF16 乘法 56%、FP4 dot 21%、FP32 累加 10%、Matrix Unit 面积 10%、SIMD Unit 面积 8%，全部是"vs optimized human baseline"的相对改进，**没有绝对面积或功耗**，也没有说明这些基本单元在总芯片面积中的占比，因此无法换算成整芯片收益。内核侧的 1.5–1.8× 是实测且端到端验证过，这部分证据较强。
- **未来方向（材料给出）**：**多代路线图**——Gen 1 为 Jalapeño，**Gen 2 已接近 tapeout，Gen 3 在规划中**；三条"Jalapeño 解锁的能力"是：**经济的低延迟服务**（前沿模型在交互式 SLA 延迟下变得实用）、**更好的性能/瓦**（优势延伸到非交互式推理）、**架构解锁 HBM**（"主要瓶颈是暴露聚合 HBM 带宽，而不是增加更多原始带宽"）。最后一条是全文对下一代硬件方向最明确的表态。

## 个人点评

- **这份材料最有价值的不是那些倍数，而是它把"比较什么"重新定义了一遍**。幻灯片明确列出 non-goals（芯片数量、每芯片吞吐、TTFT），并坚持"在匹配的用户体验下比效率"。这对推理芯片是正确的方法论——因为 TBT 与每 token 能耗是此消彼长的，单点对比没有意义。第 12 页那张汇总表把结论压缩成四个维度（交互性、端到端延迟、基线原最佳 TBT 点的 perf/W、峰值吞吐点的 perf/W），其中前两个是用户体验、后两个是效率，结构清晰。
- **STP vs MTP 的处理方式值得单独肯定，也值得单独警惕**。Jalapeño 在自己的所有主对照里用 STP，而基线包含 MTP——幻灯片在三个地方标注了这一点，并且**专门做了一页 "Jalapeño STP vs GB300 MTP"** 的对照（峰值 perf/W 优势 1.5×，而不是 1.7–1.9×）。这种做法比只挑有利口径诚实。但读者必须自己完成这个换算：**8.6×–104.3× 要打掉 MTP 的贡献才能得到架构本身的差异，而那个差异大约是 1.5×**。
- **架构上的核心主张是"把异构性放进芯片内部"**，论证的重心落在 locality：专用异构集群在某个阶段总有闲置芯片，而闲置加速器仍要付封装、HBM、I/O、网络与制冷的基础功耗，还要搬移 KV；单芯片内部异构则按阶段激活 compute/memory/network 的配比、未用块熄灭，"dark silicon is cheaper than idle accelerators"。这条论点在逻辑上是成立的，但**材料没有给出这一策略的量化收益**（例如同一请求在两种组织方式下的能耗对比）。相比之下，"1+ PB/s ÷ 0.5 TB = 2,000+ 次权重读取/秒"这个算法很干净——它用理论天花板证明瓶颈不在原始带宽，从而为"架构暴露带宽"的论点提供起点。
- **"You can just build things" 这句话配 9 个月 RTL-to-tapeout 和 AI 辅助设计的数字，是这份材料最容易被误读的地方**。BF16 乘法 56%、Matrix Unit 面积 10% 这类改进都是相对"optimized human baseline"的，没有绝对数；但第 31 页的"GPT-OSS 的 attention 与 MoE 块比既有专家手写实现快 1.5–1.8×"是实测且端到端验证过的，这一条证据强度高得多。另外幻灯片自述"major changes landed up until the day of RTL freeze"——这既是 AI 加速设计的收益，也是风险（临近冻结日仍在做重大改动）。
- **与仓库里其他材料的对照**：Jalapeño 的"本地 core slice + HBM slice 配对"与 Google TPU 第八代的"384 MB 片上 SRAM 破内存墙"是同一问题的两种解法（前者靠物理就近，后者靠容量与带宽）；而它"把专用集合通信网络与通用 NoC 分开"的做法，与 TPU 的 CAE 放在 ICI I/O die、Meta 用独立 ME 阵列是同一类选择。三家的共识是：**集合通信不该由通用计算资源承担**。Jalapeño 独特之处在于把"异构"推到**单芯片内部按阶段切换**，这是我在本仓库其他材料里没有见过的表述。

## 工程化三问总结

### 1. 它解决了什么瓶颈？

- **应用场景与核心瓶颈**：OpenAI 的推理与 agentic 生产负载。瓶颈被定义为**"在要求的 SLA 延迟下的 requests/second/watt"**，而不是峰值算力或每芯片吞吐（后者被明确列为 non-goal）。三个 agentic 阶段的瓶颈各不相同：prefill 是 FLOPs/attention、draft 是**网络延迟**、spec-verify 是 attention + **HBM 带宽（MoE 在低延迟下带宽饥渴）**。
- **现有方法为何不足**：专用异构集群在任一阶段都有闲置加速器，而闲置加速器仍付出封装、HBM、I/O、网络与制冷的基础功耗，并且**KV 状态必须在系统间搬移**；长延迟来自"统一内存子系统的高争用路径、不同步内核导致的昂贵内存 fence、集中式的网络访问中介"，而不是硬件速度极限。此外用理论 HBM 带宽算出的上限（不含投机解码 1,000–2,000 tokens/s/user，含则 5,000–10,000）与实际差距巨大，说明**原始带宽不是唯一限制资源**。
- **论文证据的分层**：
  - **规格级（可引用）**：第 32 页全部数字——mxfp8×mxfp8 3.4 / mxfp8×mxfp4 6.7 / mxfp4×mxfp4 13.4 PFLOP/s；15.4 TB/s、216 GiB；本地 128 ASIC @ 600 GB/s、全局 2048 ASIC @ 200 GB/s；700 W；系统级 27 EFLOP/s、32 PB/s、432 TiB；封装含 compute die + I/O chiplet + HBM4 堆叠（平面图可见 6 个，与 216 GiB ÷ 36 GiB 自洽，但正文未标注堆叠数）。
  - **实测性能（第三方基准 + 自述工况）**：InferenceX 上的四组对照，逐个给出绝对数值（85,448 vs 44,960；19,641 vs 11,781；19,641 vs 12,951；18,195 vs 11,862 mixed/kW）与延迟/TBT 数值。因为给出绝对值，可比单独看倍数更可信。
  - **AI 辅助设计的实测收益**：GPT-OSS 的 attention 与 MoE 块比专家手写实现快 1.5–1.8×（端到端验证过）；基本单元 PPA 与块级面积的相对改进（56%/21%/10%、10%/8%，均 vs human baseline，无绝对数）。
  - **需要降级的**：主对照中 **Jalapeño 用 STP 而基线含 MTP**（材料自己标注）；**按 package TDP 归一化**而非实测功耗（附录另给 all-in utility MW 但正文无数值）；53.7×/104.3×/56.1× 三个大数取自**基线自身最不利的最低 TBT 点**；结果为**首颗硅回片后约 3 个月**内取得，无方差或热稳态说明；**无工艺节点、die 面积、晶体管数、精度评估**。

### 2. 用了什么结构或方法？

- **整体结构**：单颗 ASIC = compute die + I/O chiplet + HBM4 堆叠（平面图 6 个）；**每个 core slice 与一个 HBM slice 配对**形成低延迟本地视图（图中为 64 对）。三级通信：最快为本地 core↔HBM slice 路径；中间为**专用跨核集合通信网络**；最慢但最灵活为**共享 NoC**（通用通信 + scale-up 接入，含 Scale-up Ethernet bridge）。
- **关键结构与机制**：
  1. **本地/远程分层与显式放置**：常用操作数保持本地，共享路径仅按需使用；显式放置 + 分布式控制防止数据搬移与同步主导执行时间。
  2. **"半扁平化"两级 Clos**：本地域 128 芯片（TH6）、全局域 2048 芯片跨 8 rail（TH6）；TP 给高带宽、EP 给较低带宽、所有流量要低延迟。
  3. **空间架构与 Gluon 编程模型**：Gluon 把每个 core 当作 thread block；TensorInfo 显式捕获张量布局与物理放置；专用集合通信跨物理核协调。
  4. **人机分工的编程范式**：人给出简单抽象（本地张量、显式通信、可预测同步），前沿 AI 搜索映射/放置/调度流水/集合通信编排。
  5. **按阶段切换片内异构资源**：prefill / draft / verify 三阶段激活不同的 compute·memory·network 配比，未用块熄灭，KV 保持本地。
- **数值与量化策略**：原生 **MXFP4 / MXFP8** 支持（权重 dtype 用 f4 做基准）；算力按三种乘加组合分别给出。

### 3. 对芯片架构和 RTL 有什么启发？

- **芯片架构**：五条。第一，**"把异构性放进芯片内部而非机架之间"是一个可执行的架构立场**，其论证核心是 locality：闲置加速器仍付基础功耗、KV 必须搬移、跨阶段都要网络与同步；片内则按阶段改活跃配比并熄灭未用块。这条论点对任何面向多阶段 agentic 负载的架构都适用，但需要用能耗对比来验证（材料未提供）。第二，**core slice 与 HBM slice 一一配对**用物理就近换取低延迟本地视图，是"用布局解决延迟"的直接手段；代价是容量与算力的绑定粒度变粗（每个 slice 只能看到自己的 HBM 切片），因此需要一层共享 NoC 兜底——**"常用本地、例外走全局"的分层是这套设计的骨架**。第三，**集合通信应该独立成网**：专用跨核网络与通用 NoC 分开，与 TPU 的 CAE（放 ICI I/O die）和 Meta 的 ME 阵列是同一类选择，三家独立收敛到同一结论。第四，**给出理论天花板是定位瓶颈的好方法**：1+ PB/s ÷ 0.5 TB = 2,000+ 次权重读取/秒，与实测差距巨大，从而证明"瓶颈是暴露聚合带宽而非原始带宽"——这个论证结构值得照搬。第五，**算力规格应按乘加组合分列而非只报一个峰值**：3.4（mxfp8×mxfp8）/ 6.7（mxfp8×mxfp4）/ 13.4（mxfp4×mxfp4）PFLOP/s 明确显示了低精度在何时才真正翻倍，这比单一峰值数字有用得多。
- **RTL**：可落到实现层的项目包括：**core slice 与其配对 HBM slice 之间的本地通路**（需要独立于共享 NoC 的地址映射与仲裁，且要保证本地访问不被全局流量阻塞）；**专用集合通信网络的 RTL**（常见的跨核通信模式要被做成硬件原语，而非软件循环）；**共享 NoC 与 scale-up Ethernet bridge** 的接入逻辑；**按阶段熄灭未用块的电源门控与唤醒时序**（"dark silicon"策略要求动态电源域管理，且切换点必须与请求流水线对齐，避免在阶段切换时引入新的延迟）；**TensorInfo 对应的物理放置元数据通路**（张量布局与物理位置显式编码，需要硬件侧有对应的地址生成支持）；**AI 生成 RTL 的验证与 PPA 反馈闭环**（XLS 语言 + Generate→Verify→Measure PPA 流程，这属于设计方法学而非芯片 RTL）。上述均基于幻灯片的架构与流程描述；**材料没有给出任何 RTL 细节、面积分解、时序或功耗数据**。
- **推断边界**：第 1 问中规格（第 32 页）与 InferenceX 对照数值属材料直接给出（对照口径的限定已在上文列出）；第 2 问的结构描述来自第 16–19、29–30、32 页。第 3 问的芯片架构与 RTL 内容为工程推断；HBM4 堆叠数（6）是从平面图读出并用容量交叉验证的推断，正文未标。材料中完全缺失的量值——工艺节点、die 面积、晶体管数、绝对 perf/W、精度/量化评估、本地与共享路径的延迟带宽对比、核到核延迟的实测值、AI 辅助设计收益的绝对值换算——以及两页附录（all-in utility MW 与 matched speed）的图表数值，均 `TBD`。
