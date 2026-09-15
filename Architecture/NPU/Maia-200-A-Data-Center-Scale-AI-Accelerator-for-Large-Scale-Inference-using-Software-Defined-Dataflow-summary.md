# 深度分析：《Maia 200: A Data Center Scale AI Accelerator for Large Scale Inference using Software Defined Dataflow》

## 基本信息

- **标题**：Maia 200: A Data Center Scale AI Accelerator for Large Scale Inference using Software Defined Dataflow
- **文档类型**：厂商技术幻灯片（Hot Chips 2026，35 页）
- **作者**：Prashant Ranjan、Jackson Peng、Torsten Hoefler
- **机构**：Microsoft
- **发表 venue**：Hot Chips 2026
- **年份**：2026
- **链接**：未提供论文或公开报告链接

## 一句话总结

> 微软第二代为 Azure 推理设计的自研加速器 Maia 200，围绕一种叫 **SDLA（Software Defined Local Access Dataflow）** 的执行模型构建：控制流与数据流分离、由 DMA 与 PE 组成的**预条件/后条件信号量驱动**的数据流指令、默认**本地访问**数据；规格上单芯片 **TSMC 3nm、~820 mm²、750 W、FP4 10,145 TOPS / FP8 5,072 TOPS / BF16 1,268 TOPS、6 堆 HBM 共 216 GB @ 7 TB/s、272 MB 片上 SRAM @ 80 TB/s**，并以**全以太网统一 scale-up 网络**（自研 ANC NIC + ATL 传输）从 FCQ 的 4 卡直连一路扩到 6k 加速器集群；截至 2026 年 6 月的内核实测为 FP8 GEMM 4.5 PFlop/s、FP4 GEMM 8.6 PFlop/s、attention 1.65 PFlop/s、AllReduce 1,298 GB/s、AllToAll 655 GB/s。

## 研究动机与问题定义

- **要解决的核心问题**：在**最小 $/token 与 W/token** 的前提下同时满足三类压力（第 5 页）：
  - **负载多样**：prefill 与 decode、Mixture of Experts、agentic 交互、可编程性；
  - **应用多样**：交互式对话、摘要、深度推理、深度研究；
  - **严苛的 SLA**：低延迟 token、高吞吐、高可用、高可靠。
- **现有方案的不足**：幻灯片没有直接批评具体竞品，而是用"多样性 + SLA"这组约束推出对**架构可编程性**的要求——即固定数据流的加速器无法同时应付 prefill/decode/MoE/agentic 这几种瓶颈迥异的阶段。
- **切入角度**：提出 SDLA 这一"新类别架构"，其特征被列为四条（第 8 页）：
  1. **显式软件编排**（explicit software orchestration）；
  2. **控制流与数据流相互独立**（independent control and data streams）；
  3. **数据无关性**（data obliviousness）；
  4. **本地访问**（locally accessed）——这也是 "Local Access" 的来源。
- **目标定位**：幻灯片的自我描述是 **"The Most Efficient Inference Engine on Azure"**。

## 核心结构（架构与规格）

### SoC 与内存层级

**规格表（第 13 页"Putting It All Together: Maia 200 SOC"）**

| 项目 | 数值 |
| --- | --- |
| 峰值稠密 Tensor TOPS | **FP4 10,145** / **FP8 5,072** / **BF16 1,268** |
| 峰值 HBM 带宽 | **7 TB/s** |
| HBM 堆叠数 | **6** |
| HBM 容量 | **216 GB** |
| **SRAM 容量** | **272 MB** |
| **SRAM 带宽** | **80 TB/s** |
| Host PCIe 带宽 | 64 GB/s（**Gen6 x8**） |
| 后端网络带宽（单向） | **1,400 GB/s** |
| 后端网络配置 | **28 × 400**（Gbps） |
| SoC die 面积 | **~820 mm²** |
| 封装尺寸 | 75 × 75 |
| SoC TDP（provision） | **750 W** |
| 工艺 | **TSMC 3nm** |
| 封装技术 | **CoWoS-S** |

**内存层级（第 11 页）**

- **L1 = TSRAM，位于 Tile 内；L2 = CSRAM，位于 Cluster 内**，用来促进数据复用、减少数据搬移。
- 数据搬移是**分层的 DMA 引擎**：**Tile（TSRAM）↔ Cluster（CSRAM）↔ Chip（HBM）**，另支持 **Cluster 到 Cluster 的广播**。
- 该页给出的原则是"Local Data Access"与"Hierarchical Data Storage & Movement"。

**计算单元（第 9–10 页）**

- **TTU（张量单元）**：面向低精度块缩放数据类型（**FP8/FP6/FP4**），通过 **TDMA** 支持硬件数据类型的**线速转换**。
- **TVP（向量单元）**：为模型与算法变化保留灵活性，同样通过 TDMA 支持线速数据类型转换。
- **TU / 控制单元**：负责编排 SDLA 数据流——协调 TTU、TVP 与 DMA 任务，并控制**硬件信号量**以实现数据搬移与执行之间的细粒度同步。

**异步控制与数据通路（第 9 页）**

数据通路由 **DMA 与 Processing Elements（PE）** 执行；**CP 负责控制**。核心机制是**数据流指令**：

```
Pre0  Pre1  Pre2        →  [ DMA command | kernel invocation | Network tx/rx ]  →  Post0  Post1  Post2
(最多 3 个信号量的预条件)                                                          (最多 3 个信号量的后条件)
      wait                                                                              signal
```

即：指令等待最多 3 个信号量就绪（precondition），执行 DMA 命令、内核调用或网络收发，然后向最多 3 个信号量发出通知（postcondition）。这套机制把**同步做成了指令的一部分**，而不是靠额外的屏障指令——这是 SDLA "显式编排"与"控制/数据流独立"的具体落地形式。

### I/O：自研 NIC 与传输协议

- **ANC（Custom Embedded AI NIC）** 与 **ATL（Custom Scale Up Fabric Transport）**：
  - **基于以太网**；
  - **端点控制的多路径（endpoint-controlled multipathing）**，支持 **packet spray、负载均衡、乱序接收端（OOO Receiver）**；
  - 支持**多层网络**；
  - 面向可靠性：**硬件级快速故障检测与恢复**；
  - **端到端加密**。
- 声明的功耗/延迟/面积指标：**< 8 pJ/b**、**延迟 < 1 µs（P2P、单向、mem2mem）**、"small size"。
- 幻灯片称 **ATL 对 Ultra Ethernet Transport（UET）与 Multipath Reliable Connection（MRC）两个标准的制定有重要影响**。

### 系统与网络拓扑

- **Tray 内：FCQ（Fully Connected Quad）**——**4 个加速器通过固定以太网链路直连，不经交换机**，为 TP 密集的操作提供最近邻之间的最大带宽，Quad 内可高效做 AllGather 与 AllReduce。
- **Tray 以上：统一的以太网 scale-up 网络**——"One Network, One Protocol, from Chip to 6k Accelerator Cluster"。幻灯片给出的三条理由：**单一以太网 fabric、处处同一套 ATL 栈、标准交换机生态**。
- **拓扑组合**：**固定（FCQ）+ 交换（两级统一 Clos）**。
- **负载均衡分两层**：**片内**用分层的 NoC 与计算调度，支持**片内 multicast/broadcast**；**片间**通过 **ANC 分片**并利用 **FCQ**，网络内部也做负载均衡（第 12 页）。
- **I/O 栈分层**（第 22 页）：MCCL（Microsoft Collective Communication Library，非阻塞 MPI、隐藏同步开销、减少 incast、最大化重叠）→ E2E 负载均衡（片内分片、SOC 到系统的 I/O 均衡）→ 网络（RDMA、传输卸载、端到端拥塞控制、可靠性）→ 连接（集成 ANC、ATL、面积/功耗/延迟优化）。

### 内核协同设计

- **SDLA 驱动的设计**：全异步编程模型；分离数据流、数据计算与控制流以实现完全重叠；**在硅开发的早期阶段就能得到可预测的数据搬移开销**。
- **分布式 Batch GEMM 的 gather 式实现**：激活值通过共享的 broadcast NoC 在网络上做 gather（广播），每颗芯片对收到的激活分块各自跑（Batch）GEMM；收益是**细粒度地重叠集合通信与 GEMM**；同时"**固化高精度输出、只广播低精度激活**"可以减少权重搬移从而省电。
- **集合通信的自适应逻辑拓扑**（第 29–30 页）：按消息大小选实现——
  - **Broadcast 式**：适合**小传输、低延迟**；
  - **Hierarchical 式**：适合**中等传输、吞吐与延迟平衡**；
  - **Ring 式**：适合**大传输、峰值吞吐**。

## 证据、案例与论证

### 基准设置

- 幻灯片标注所有结果的时间点为 **"Kernel Performance (as of June 2026)"**，即**内核级微基准**，非端到端模型服务。
- 三个基准的配置：
  - **GEMM**：Float8 × Float8 与 Float4 × Float4，**TP = 1**；画的是算术强度（Tensor Ops / I/O）对实测吞吐的 **roofline vs achieved** 图。
  - **Attention**：融合的 scaled-dot-product（**FlashAttention 2** 实现），**Tensor 用 Float8、SIMD 用 Float32**，**GQA = 4**；计算量按 $QHead \times Context^2$ 计。
  - **Collective**：**AllReduce（BF16）TP8** 与 **AllToAll TP8**；画的是传输大小对实测吞吐的 roofline vs achieved 图。

### 主要结果

| 基准 | 实测峰值 | 对应规格峰值 | 达成率 |
| --- | ---: | ---: | ---: |
| GEMM FP8 × FP8（TP=1） | **4.5 PFlop/s** | 5,072 TOPS | **约 89%** |
| GEMM FP4 × FP4（TP=1） | **8.6 PFlop/s** | 10,145 TOPS | **约 85%** |
| Attention（FP8 tensor / FP32 SIMD，GQA=4，FA2） | **1.65 PFlop/s**（标为 "Eff Peak"） | — | — |
| AllReduce（BF16，TP8） | **1,298 GB/s** | 后端网络 1,400 GB/s | **约 93%** |
| AllToAll（TP8） | **655 GB/s** | 后端网络 1,400 GB/s | **约 47%** |

（达成率一栏为本次分析用第 13 页的规格峰值自行计算，幻灯片本身未给出百分比。）

其他从图上可读出的信息：

- **GEMM 的 roofline 拐点在算术强度约 800 FLOPs/byte**（图中以虚线标出 800，横轴从 100 到 6400），说明内核对带宽的敏感区间边界在 800 FLOP/byte 附近。
- **Attention 的实测点散布很宽**（在 $10^7$ 到 $10^{10}$ 的总计算量范围内，吞吐从数百 TFlop/s 一直到 1.65 PFlop/s 以上），且其 roofline 上限（图中绿线）明显低于 GEMM 的 roofline——这与 attention 含 SIMD float32 的 softmax 等非张量操作一致。
- **Collective 的三种实现各有适用区间**：AllReduce 图上标出了 Broadcast 实现（小消息）、Broadcast & Hierarchical 实现（中消息）、Ring 实现（大消息，右侧峰值 1,298 GB/s）；AllToAll 图则标出 Broadcast 与 Hierarchical 两段，峰值 655 GB/s。

### 论证结构

幻灯片把整套设计压成一条因果链（第 34 页）：

**垂直协同设计的 SoC 与系统 → SDLA 架构（数据搬移中心的架构新类别）→ 优化过的系统设计（统一全以太网）→ 内核协同设计（计算 + 通信内核设计优化）→ 结果：更低的 TCO、更好的能效、更高的性能/$**。

其中各层的对应关系被明确列出：TTU、DMA 层级、内存分层、片内 NoC、集成 NIC 与软件 SDK；集合通信库 MCCL 与端到端负载均衡；内核侧的 gather 式 GEMM 与自适应集合通信拓扑。

## 局限性与未来方向

- **没有任何竞品对照，也没有端到端模型结果**：整份材料**没有出现一次与 GPU 或其他加速器的性能比较**，也没有任何模型的端到端指标（tokens/s、延迟、$/token、W/token）。这与第 5 页设定的目标（"Minimum $/Token & W/Token"）之间有一个明显缺口——**目标被写成了 $/token 与 W/token，结果却只有内核级吞吐**。
- **结果是内核微基准，且时间点明确限定为 2026 年 6 月**：GEMM 与 attention 都是 **TP=1**（单芯片内核），集合通信是 **TP=8**。这些不是部署形态的指标。幻灯片自己没有给出多芯片 GEMM、端到端 attention 或 MoE 的结果。
- **无精度/量化评估**：规格与 TDMA 都强调 **FP8/FP6/FP4 块缩放数据类型**与线速类型转换，但**没有任何模型精度数据**。对于一颗把 FP4 作为最高算力点的推理芯片，缺少精度影响评估是一个显著缺口。
- **AllToAll 的达成率偏低未作解释**：按后端网络 1,400 GB/s 单向带宽计算，AllReduce 达到约 93%，而 **AllToAll 只有约 47%**。幻灯片把 AllToAll 标为"Hierarchical 实现"，但没有说明这个差值是算法固有（all-to-all 的 incast 特性）、拓扑限制（FCQ 之外的交换层）还是软件尚未优化。考虑到 MoE 推理高度依赖 all-to-all，这条值得追问。
- **关键规格缺少口径说明**：**272 MB SRAM 的 80 TB/s** 没有说明统计口径（各层聚合还是某个具体层）；**750 W 被写成 "SoC TDP (provision)"**，即供给能力而非实测功耗；**7 TB/s HBM 带宽对应 6 堆 HBM**（推得每堆约 1.17 TB/s），但未标注 HBM 代际；~820 mm² 是 SoC die 面积，未给封装内总硅面积。
- **ATL 的延迟与功耗是点指标**：**< 1 µs（P2P、单向、mem2mem）**是链路层的点对点指标，不代表在集合通信或跨层拓扑下的端到端延迟；**< 8 pJ/b** 也没有说明测量条件。
- **"新类别架构"的论证偏定性**：SDLA 的四条特征（显式编排、控制/数据流独立、数据无关、本地访问）与预条件/后条件信号量的机制描述是清楚的，但**材料没有给出这套模型相对常规加速器编程模型的可量化收益**（例如"可预测的数据搬移开销"具体是多少、相比什么）。
- **未来方向（材料给出）**：幻灯片自身没有 future work 章节。从第 12、22 页的结构看，方向是继续沿"统一以太网 + 端到端负载均衡"扩展（当前声明到 6k 加速器集群），以及把 SDLA 的数据搬移可预测性用于更早的硅开发阶段。ATL 对 UET 与 MRC 标准的影响也提示微软在推动这套传输协议走向标准化，这可以理解为后续更大规模部署的技术铺垫。

## 个人点评

- **"控制流与数据流分离 + 预条件/后条件信号量"是这份材料里最具体的技术贡献**。数据流指令的形态（最多 3 个信号量等待 → DMA/内核/网络操作 → 最多 3 个信号量通知）把同步从"额外的屏障指令"变成"指令的一部分"，这是让异步执行与数据搬移可预测的机制基础。与此配套的是"数据无关性"（data obliviousness）——即执行时间不依赖数据内容，从而可以在硅回来之前就算出数据搬移开销。这两条合起来，是 SDLA 与常规"靠运行时动态调度"的加速器最本质的区别，也直接对应幻灯片说的"在硅开发早期阶段就能得到可预测的数据搬移开销"。
- **规格表的完整性在本仓库的材料里属于中上**：给了 die 面积（~820 mm²）、封装尺寸（75×75）、工艺（TSMC 3nm）、封装技术（CoWoS-S）、TDP（750 W，虽是 provision）、SRAM 容量与带宽、HBM 堆叠数与容量带宽、后端网络配置（28×400）——这些恰恰是 TPU 第八代那份材料完全缺失的。更重要的是**它给了内核实测值**，因此可以自行算出达成率：**FP8 GEMM 约 89%、FP4 GEMM 约 85%、AllReduce 约 93%**。这些百分比是我从第 13 页规格与第 31–33 页实测反推的，幻灯片没给，但它是判断"规格是否可用"的关键信息。
- **两处需要特别留意**。第一，**没有任何竞品对照与端到端模型数据**。目标写的是 $/token 与 W/token，交付的是 TP=1 的内核吞吐与 TP=8 的集合通信带宽。这在 Hot Chips 的厂商宣讲里并不罕见，但对读者来说意味着**无法判断这块芯片在实际推理服务中的位置**——相比之下 Meta MTIA 那份至少给了 150B DLRM 上的 GPU 持平结论，OpenAI Jalapeño 那份给了 InferenceX 的功耗归一化端到端对比。第二，**AllToAll 只达到后端带宽的约 47%**，而 AllReduce 达到约 93%。在 MoE 推理里 all-to-all 是核心通信模式（本仓库 DeepSeek-V3 那篇专门为此做了 Node-Limited Routing），这个差距值得追问：如果 all-to-all 只能跑到 655 GB/s，那么 28×400 的 1.4 TB/s 后端网络在 MoE 场景下的实际价值就要打折。
- **网络设计的两个选择值得对照**：**FCQ 用固定以太网链路直接把 4 颗芯片全互连、不经交换机**，这与 Google TPU 第八代的 BoardFly（4 TPU/tray 全互连 → 8 tray 一组 → 36 组）在同一层做了同类选择；而"从芯片到 6k 集群统一一套以太网与 ATL 栈"则与 Meta MTIA 400 的"RoCE scale-up/scale-out + 72 ASIC scale-up 域"思路接近。三家的共识是：**scale-up 域内不用通用交换机，尽量直连；跨域扩展用标准以太网生态**。微软的独特之处是走了**统一协议**而非"专用 scale-up 协议 + 通用 scale-out 协议"的双栈路线，代价是 FCQ 里的固定链路无法像 OCS 那样重配置。
- **一个可以量化的交叉验证**：803 mm² 的 SoC + 6 堆 HBM + 272 MB SRAM，在 750 W 的 provision 下给出 10,145 FP4 TOPS，即 **约 13.5 TOPS/W（FP4，按 provisioning TDP 计）**。这个数字可以与仓库里其他材料对照（例如 TPUv4i 是 138 TFlops / 175 W ≈ 0.79 TFlops/W bf16）。但要注意口径差异大：FP4 vs BF16、provision TDP vs 实测、峰值 vs 达成率——因此只能作为量级参考，不能直接比较。

## 工程化三问总结

### 1. 它解决了什么瓶颈？

- **应用场景与核心瓶颈**：Azure 上的大规模 LLM 推理。目标被明确写成 **"Minimum $/Token & W/Token"**，约束是多样的负载（prefill/decode、MoE、agentic）与多样应用（对话、摘要、深度推理）叠加严苛 SLA（低延迟、高吞吐、高可用、高可靠）。核心瓶颈被定位为**数据搬移**：SDLA 的四条特征与分层内存（Tile TSRAM → Cluster CSRAM → Chip HBM）、分层 DMA、片内 broadcast NoC 都是围绕减少与优化数据搬移组织的。
- **现有方法为何不足**：材料没有直接批评竞品，而是用"负载多样性 + SLA"推出对可编程性的要求——固定数据流的加速器无法同时覆盖瓶颈迥异的多个阶段。因此提出 SDLA 这一"新类别"：显式软件编排、控制/数据流独立、数据无关性、本地访问。
- **论文证据的分层**：
  - **规格级（可引用）**：第 13 页全部数字——FP4 10,145 / FP8 5,072 / BF16 1,268 TOPS；7 TB/s、6 堆 HBM、216 GB；272 MB SRAM @ 80 TB/s；PCIe Gen6 x8 / 64 GB/s；后端 1,400 GB/s 单向、28×400；~820 mm²；75×75 封装；750 W provision；TSMC 3nm；CoWoS-S。
  - **机制级（描述具体）**：数据流指令的预条件/后条件信号量结构（最多各 3 个）；分层 DMA（Tile↔Cluster↔Chip）与 cluster 间广播；ANC/ATL 的多路径特性（packet spray、OOO receiver、硬件快速故障检测与恢复、端到端加密）；FCQ 的 4 卡固定直连；自适应集合通信拓扑（broadcast / hierarchical / ring）。
  - **内核实测（截至 2026-06）**：FP8 GEMM 4.5 PFlop/s（约 89% 规格峰值）、FP4 GEMM 8.6 PFlop/s（约 85%）、attention 1.65 PFlop/s（FP8 tensor + FP32 SIMD，GQA=4，FA2）、AllReduce 1,298 GB/s（约 93% 后端带宽）、AllToAll 655 GB/s（约 47%）。
  - **需要降级的**：**无任何竞品对照、无端到端模型指标、无精度/量化评估**；所有结果为 TP=1 或 TP=8 的内核微基准；750 W 是 provision 而非实测功耗；< 8 pJ/b 与 < 1 µs 未给测量条件；"Most Efficient Inference Engine on Azure" 为自我定位主张。

### 2. 用了什么结构或方法？

- **整体结构**：SoC 由 Tile（含 TSRAM = L1）与 Cluster（含 CSRAM = L2）经 GNOC 组织，外围 6 堆 HBM；计算单元为 TTU（张量，支持 FP8/FP6/FP4 块缩放）、TVP（向量，保灵活性）与 TU（控制，编排数据流并管理硬件信号量）。数据通路由 DMA 与 PE 执行，CP 负责控制。
- **关键机制**：
  1. **数据流指令 = 预条件（最多 3 个信号量）+ 操作（DMA 命令 / 内核调用 / 网络收发）+ 后条件（最多 3 个信号量）**，把同步内建到指令中。
  2. **分层数据搬移**：Tile（TSRAM）↔ Cluster（CSRAM）↔ Chip（HBM），加 Cluster 间广播；片内 multicast/broadcast 由 NoC 支持。
  3. **自研 ANC NIC + ATL 传输**：以太网为基础、端点控制多路径（packet spray、负载均衡、乱序接收）、多层网络、硬件快速故障检测与恢复、端到端加密。
  4. **两级网络拓扑**：tray 内 FCQ（4 卡固定直连、无交换机）＋两级统一 Clos（固定 + 交换），统一到 6k 加速器集群。
  5. **内核协同设计**：gather 式分布式 batch GEMM（激活广播、各芯片跑分块 GEMM、高精度输出固化、只广播低精度激活）；集合通信按消息大小在 broadcast / hierarchical / ring 三种逻辑拓扑间自适应。
- **量化与数值策略**：TTU 面向低精度块缩放数据类型（FP8/FP6/FP4），通过 TDMA 做线速类型转换；attention 基准的配置是 FP8 tensor + FP32 SIMD。**但材料未给出任何精度评估**。

### 3. 对芯片架构和 RTL 有什么启发？

- **芯片架构**：五条。第一，**把同步做成指令的一部分而非额外的屏障操作**，是让异步数据流可预测的关键——数据流指令的"最多 3 个信号量等待 + 操作 + 最多 3 个信号量通知"结构，等价于把生产者-消费者依赖编译成显式的硬件信号量操作。这套机制值得与 Groq TSP 的"完全没有硬件同步、只在程序开始同步一次"以及 TPU 的编译器静态调度做对照——三家都在解同一个问题（如何让数据搬移可预测），但选择了三种不同强度的硬件支持。第二，**分层内存 + 分层 DMA 是降低搬移成本的标准骨架**：Tile TSRAM（L1）→ Cluster CSRAM（L2）→ HBM，且层级之间由专门的 DMA 引擎连接，另有 cluster 间广播。这与 OpenAI Jalapeño 的"core slice 配 HBM slice"、TPU v4i 的 CMEM 是同一族思路。第三，**"只广播低精度激活、固化高精度输出"是一个值得记录的具体优化**：它同时减少了网络带宽占用与权重搬移，直接对应功耗。第四，**scale-up 域内的直连与跨域的标准协议**：FCQ 的 4 卡固定以太网直连（无交换机）保证了最近邻的最大带宽用于 TP 密集操作，而对外统一到 ATL/以太网栈并声明影响 UET/MRC 标准——这是"局部极致 + 全局标准"的组合。第五，**多路径与乱序接收是规模化的必要条件**：ATL 把 packet spray、端点控制负载均衡与 OOO receiver 放在一起，说明**在链路层做多路径就必须在接收端解决乱序**——这与 DeepSeek-V3 那篇提出的"多端口 NIC 需要原生支持乱序放置"是同一个技术判断，两家独立得出同一结论。
- **RTL**：可落到实现层的模块包括：**硬件信号量阵列与其与数据流指令译码的耦合**（预条件/后条件的等待与通知逻辑，且要支持"最多 3 个"的并行条件判定）；**分层 DMA 引擎**（Tile↔Cluster↔Chip 三级，含 cluster 间广播的扇出逻辑）；**TTU 与 TVP 的 TDMA 线速类型转换通路**（FP8/FP6/FP4 之间以及到 FP32 的转换，且要在流水线内完成不损失吞吐）；**NoC 的 multicast/broadcast 支持与片内分层负载均衡**；**GNOC 与 CSRAM/TSRAM 的接口仲裁**；**ANC NIC 的多路径发送（packet spray）、负载均衡与乱序接收重排缓冲**；**硬件级快速故障检测与恢复状态机**；**端到端加密引擎**；**MCCL 对应的硬件加速集合通信原语**（broadcast / hierarchical / ring 三种模式的硬件支持）；以及**16 条 PAM4 类高速 SerDes 接口**（第 13 页平面图标注了 PAM4）。上述模块的功能动机来自材料描述，但**材料没有给出任何 RTL 细节、面积分解、时序或功耗数据**。
- **推断边界**：第 1 问中规格来自第 13 页（可引用，但 750 W 是 provision、SRAM 带宽口径未标），机制来自第 9–12、22 页，内核实测来自第 31–33 页且**限定为 2026 年 6 月的 TP=1/TP=8 微基准**；达成率百分比（89%/85%/93%/47%）为本次分析用规格峰值自行计算。第 2 问的结构描述全部来自材料。第 3 问的芯片架构与 RTL 内容为工程推断。材料中完全缺失的量值——die 面积分解、实测功耗、HBM 代际、精度/量化评估、端到端模型指标、竞品对照、"可预测数据搬移开销"的具体数值、AllToAll 达成率偏低的成因——均 `TBD`。
