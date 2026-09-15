# 深度分析：《AMD Instinct MI400 Series GPU Architecture》

## 基本信息

- **标题**：AMD Instinct MI400 Series GPU Architecture — Powering the next frontier of AI and High Performance Computing
- **文档类型**：技术演示文稿（Hot Chips 2026 厂商报告，共 18 页，文档标注 AMD Confidential）
- **作者**：Alan Smith（Corporate Fellow, Graphics Architecture）、Maiyuran Subramaniam（Senior Fellow, Graphics Architecture）
- **机构**：AMD
- **发表 venue**：Hot Chips 2026（August 2026）
- **年份**：2026
- **链接**：不适用（厂商演示文稿，随 Hot Chips 发布）

## 一句话总结

> MI455X 把 GPU 拆成 8 颗 N2 计算 die、2 颗 N3P 缓存/fabric die 和 2 颗 N3P I/O die，用 CoWoS-L 与 3D 混合键合封装 12 栈 HBM4，得到 432 GB / 23.3 TB/s、192 MB 全局 L2 和 256 个 WGP；Helios 机柜以 72 颗 GPU 组成单一 scale-up 域。

## 研究动机与问题定义

这不是研究型论文，而是一份面向 Hot Chips 的产品架构发布。它给出的动机链条很直接（第 2 页）：AI 负载从 2017 年的 Transformer（约 6500 万参数）演进到 2023 年的 GPT-4（约 1 万亿）再到 2026 年的 Agentic + Reasoning 模型（约 10 万亿以上），同时使用模式从「大规模预训练」扩展到「常开推理服务」「企业微调」和「Agentic AI」。结论是基础设施的数据搬运速度必须跟上模型规模的扩张，因此这一代架构的重心放在缓存/内存层次、封装与互连，而不是单纯堆算力。

## 核心结构（技术规格与机制）

### 系统级：Helios 机柜与计算托盘

**Helios 机柜**（第 3 页）：2.9 EF「AI 算力」，31 TB HBM4 容量，1.7 PB/s HBM4 带宽，72 颗 GPU/机柜，260 TB/s scale-up 带宽，43 TB/s scale-out 带宽。演示文稿没有标明 2.9 EF 对应的精度，这一点在使用该数字时必须注明。

**计算托盘**（第 4 页）是机柜的构建单元：

| 部件 | 规格 |
| --- | --- |
| GPU | 4 个 Instinct MI455X EAM，每个 EAM 36 条 UALoE 链路（x2） |
| CPU Host | 单路 EPYC 9006 SP7，与 MI455X 之间为缓存一致 Infinity Fabric（x16），128 GB/s/dir per GPU |
| Scale-up 网络 | UALoE，1.8 TB/s/dir per GPU |
| Scale-out 网络 | 每个 EAM 最多 3 个 Pensando Vulcano 800 AI NIC，UALink128（x8），128 GB/s/dir per NIC |
| 前端网络 | Pensando Salina 400 DPU PCIe CEM NIC，800GbE |

### 芯片级：MI455X 的 chiplet 划分

MI455X 由三类 die 组成（第 5 页、第 8 页）：

- **2× Fabric and Cache Die（FCD，N3P）**：192 通道 HBM4 接口，192 MB 全局 L2。
- **2× I/O Die（IOD，N3P）**：2 路 PCIe Gen6，或通过 UALink 接 3 个 AI NIC；Infinity Fabric 256 GB/s（双向）；72 条 UALoE lane，3.6 TB/s（双向）。
- **8× Accelerator Complex Die（XCD，N2）**：合计 256 个 active WGP（按此推算每个 XCD 32 个 WGP）。

第 8 页的概念框图显示：8 个 XCD 上方各有 Shared Resources，下方接 GPU L2，再经 Infinity Fabric die 连到 12 栈 HBM4，对外一侧是 36×2 条 UALoE 链路。封装上采用 CoWoS-L 与 3D 混合键合（hybrid bonding）堆叠 XCD，官方给出的收益是更高计算密度与更好的性能/瓦。

### 缓存与内存层次的重构（第 9 页）

| 项 | MI355X | MI455X | 变化 |
| --- | --- | --- | --- |
| Vector Register | 128 KB | 128 KB | 每 SIMD 2.0× |
| Scalar Register | 3.2 KB | 8 KB | — |
| LDS | 160 KB | 384 KB | 每 WGP/CU 2.0× |
| L1 Data Cache | 32 KB | 与 Vector Data/LDS 合并为 384 KB | — |
| L1 Constant Cache | 16 KB | 16 KB（WGP Constant Cache） | — |
| L2 | 4 MB | 96 MB | 每 L2 1.5×，配合 Broadcast Arbitrator 最高 4× 带宽放大 |
| Infinity Cache | 256 MB | 未列出 | 该级缓存在这一代未出现 |
| HBM | 288 GB HBM3E | 432 GB HBM4 | 总容量 2.9× |

第 5 页另有「192 MB 全局 L2」的表述。96 MB 与 192 MB 的差别应该是每 die 与全芯片的口径差异（2 颗 FCD × 96 MB），但演示文稿没有明确标注单位基准，引用时需注明。

### 计算与低精度支持

- **Wave32 原生执行**：32 宽向量机上单周期执行，官方给出的影响是降低指令延迟、分支发散惩罚与寄存器压力（第 10 页）。
- **MX 格式**：新增 MXFP8、MXFP6、MXFP4，支持 block scale 16/32；MXFP4 增加 fractional scaling。第 11 页的格式表列出 mxfp8（两种 E5M2/E4M3 变体）、mxfp6（两种变体）、mxfp4（共享指数 8/5/4、每值指数 0/3，即存在无每值指数的模式）。另外新增 4-bit tensor LUT 指令，允许以 4-bit 存储并转换到 4/6/8-bit MX 计算格式。
- **Transcendental Engine**：新增 tanh 指令并提升超越函数吞吐，目标是加速 Softmax、激活与注意力流水线（第 10 页）。
- **峰值算力**（第 10 页，单位为 MI455X 单卡）：

| 精度 | 峰值 | 相对 MI355X |
| --- | ---: | --- |
| OCP MXFP4 | 40.26 PF | 最高 4× |
| OCP MXFP6 | 20.13 PF | 最高 2× |
| OCP MXFP8 / FP8 | 20.13 PF | 最高 4× |
| Matrix FP16/BF16 | 5.03 PF | 最高 2× |
| Vector FP16 | 315 TF | 最高 2× |
| Matrix/Vector FP32 | 315 TF | 最高 2× |

### 效率机制（第 12–13 页）

- **Tensor Data Mover（TDM）**：Global 到 LDS 的异步直传，不经寄存器堆暂存，目的是让访存与计算重叠并降低寄存器压力。
- **Work group Cluster / L2 Multicast / 数据预取**：多个 WGP 组成 cluster，L2 数据返回多播，并可预取到 L2 隐藏访存延迟；官方称这些机制共同降低冗余流量并支撑 GEMM、Flash Attention 以及 wave-specialized kernel。
- **Split / Named Barriers**：更细的同步粒度，减少工作阶段之间的等待。
- **更低 dispatch 延迟**：新的 command processor 与调度，减少短 kernel 之间的空泡。
- **DMA 引擎**：每 GPU 独立的 DMA 引擎，可在 WGP 之外执行数据传输，支持本地与远端 GPU 内存互传；**Topology-Aware DMA** 自动把流量亲和到 UALoE 链路上，让软件不必感知数据位置；跨 72 颗 GPU 自动分摊流量以降低 fabric 拥塞。

### 软件栈

第 14 页介绍 ROCm.ai 开发平台，分三层：AI Skills（让 Claude Code、Codex、Cursor、Gemini 等 agent 成为 ROCm 的熟练用户）、AI Assisted Optimization（Hyperloom，端到端负载调优）、ROCm Core（编译器、工具、库、框架）。

## 证据、案例与论证（厂商测量数据）

### 厂商测量数据（第 15 页）

| 指标 | MI455X 实测 | 相对 MI355X |
| --- | ---: | ---: |
| Memory（MLA Decode，FP8） | 20 TB/s | 3.8× |
| Compute（FP4，AITER GEMM kernel） | 20 PF | 3.3× |
| Scale-up 网络带宽 | 3.2 TB/s | 3.5× |
| Scale-out 网络带宽 | 190 GB/s | 2× |

同页给出「依据估算，MI455X 相对 MI355X 的 AI 能效提升约 2.4×」，并重申 2030 年实现机柜级能效 20× 提升的目标。

## 局限性与未来方向

- **证据强度必须先降级**：这是厂商演示文稿，不是经同行评审的论文。第 10 页的峰值算力由 endnote `MI400-006` 说明是「peak theoretical precision performance」的推算值，依据是与前几代产品公开规格的对比。第 15 页的「实测」数字来自 AMD Performance Labs 2026 年 6–7 月的内部测试，方法与负载细节只有一条 endnote。
- **对比口径不对称**：`MI400-028`（FP4 计算）是「实测 vs 对手公开规格」，即用自己的测量值对比对方的数据手册值；`MI400-029`（scale-up 带宽）用 4× MI455X 对 8× MI355X，两侧 GPU 数量不同。这两处的倍数因此不能直接当作架构效率的提升。
- **文档自身的口径问题**：L2 容量在第 5 页写 192 MB、第 9 页写 96 MB，未标注是每 die 还是全芯片；第 3 页的 2.9 EF 未标注精度；MI355X 的 256 MB Infinity Cache 在 MI455X 一列消失，演示文稿没有解释这一变化。
- **缺失项**：没有功耗、面积、成本、良率或封装尺寸数据；没有与竞品的第三方对比；没有吞吐/延迟的端到端负载结果（只有 kernel 级和带宽级数字）；没有说明 12 栈 HBM4 在热与机械应力上的处理方式。
- **未来方向**：第 16 页列出六条产品方向（先进封装、吞吐与利用率优化、硬件机密计算、内存子系统、72 GPU 机柜架构、容错 scale-up 设计），其中**硬件机密计算**是相对新的能力，演示文稿在这一页只有一句话，没有技术细节。

## 个人点评

- **亮点**：把容量从「末级大缓存」搬到「每 die 的 L2」是这一代最值得关注的结构选择：L2 从 4 MB 涨到 96 MB（全芯片 192 MB），同时取消 MI355X 上的 256 MB Infinity Cache，配合 broadcast arbitrator 宣称 4× 带宽放大。这条路径说明 AMD 判断 KV cache 与 attention 的数据复用可以在更靠近计算单元的层次上解决，而不是依赖 HBM 带宽。TDM 与 Topology-Aware DMA 是同一思路的延伸：减少数据经过寄存器堆和经过 CPU 的绕行。
- **不足**：几乎所有关键数字都缺少可比口径。peak theoretical 与 measured 混排在同一页的宣传语境里，「最高 4×」这类倍数没有说明基准精度的具体配置。L2 容量两个口径并存、Infinity Cache 的去向没有交代，这类含糊会让做架构对比的人无法复用。此外低精度叙事集中在格式支持（MXFP4/MXFP6/MXFP8），但没有任何精度-吞吐权衡的实测结果，第 10 页的表格只给了峰值。
- **启发**：对做 AI 加速器的人，这份材料有用的是三件事——一是把「大 L2 + 多播」作为降低 HBM 压力的替代方案，二是把 DMA 做成拓扑感知的硬件引擎让软件不必感知数据位置，三是 MX 格式向更细粒度的块缩放（16/32 元素共享 scale，MXFP4 再加 fractional scale）演进。这些方向与架构无关，可以独立评估。但对任何具体数字，使用前都要回到 endnote 确认它是 theoretical 还是 measured。

## 工程化三问总结

### 1. 它解决了什么瓶颈？

- **应用场景与核心瓶颈**：面向 10 万亿参数级 Agentic + Reasoning 模型的长上下文、连续推理与大规模训练，瓶颈从算力转向数据搬运：模型与 KV cache 必须留在本地内存，互连带宽与延迟决定实际可交付性能（第 2 页、第 6 页）。
- **现有方法为何不足**：演示文稿没有直接批评既有方案，其隐含论点是上一代以 HBM 带宽加末级大缓存（Infinity Cache）为核心的层次结构不足以同时满足容量、带宽与能效；演进路径是把容量向计算 die 侧移动、把搬运交给专用引擎（第 6–7 页、第 12–13 页）。
- **论文证据**：可引用的是官方给出的三类数字——容量与带宽（432 GB、23.3 TB/s、机柜 31 TB / 1.7 PB/s）、缓存层次（L2 4 MB → 96 MB 每 die、384 KB LDS、Scalar Register 3.2 → 8 KB）、以及 AMD 自测的四项结果（MLA Decode FP8 20 TB/s、FP4 20 PF、scale-up 3.2 TB/s、scale-out 190 GB/s）。这些是厂商宣称的直接证据，但其中峰值算力属理论推算，部分对比采用实测对规格的不对称口径，因此「瓶颈得到缓解」这一判断的证据强度只能算中等，且没有独立复现。

### 2. 用了什么结构或训练方法？

此处按结构与架构组织，演示文稿不涉及训练方法。

- **整体结构与数据流**：一颗 MI455X 由 8 颗 N2 计算 die（XCD，共 256 WGP）、2 颗 N3P 缓存/fabric die（FCD，各含 GPU L2 与 192 通道 HBM4 接口）和 2 颗 N3P I/O die（IOD，提供 PCIe Gen6、UALoE 与 Infinity Fabric）组成；12 栈 HBM4 分布在封装两侧，通过 CoWoS-L 与 3D 混合键合集成。计算 die 之间经 Infinity Fabric die 互通，对外经由 36×2 条 UALoE 链路进入 scale-up 域。数据路径上新增 TDM 用于 Global→LDS 直传，DMA 引擎负责跨 GPU 搬运并按拓扑亲和。
- **关键模块/结构**：FCD（HBM4 控制器 + 192 MB 全局 L2）、IOD（UALoE 72 lane / 3.6 TB/s、IF 256 GB/s 双向、PCIe Gen6）、XCD（32 WGP/颗）、TDM、Broadcast Arbitrator、Work group Cluster、Split/Named Barriers、Topology-Aware DMA、Transcendental Engine。
- **训练目标、损失函数与数据策略**：不适用。低精度方面给出的是格式支持（MXFP8/6/4、block scale 16/32、MXFP4 fractional scale、4-bit tensor LUT）与峰值算力，而非训练方法论。

### 3. 对芯片架构和 RTL 有什么启发？

- **芯片架构**：三条可以直接引用的结构判断。第一，把容量从末级大缓存移到每 die 的 L2（4 MB → 96 MB，全芯片 192 MB）并取消上一代 256 MB 的 Infinity Cache，同时用 Broadcast Arbitrator 追求 4× 带宽放大，说明 KV cache 与 attention 的复用被判断为可以在 L2 层次解决。第二，把数据搬运从计算单元中剥离：TDM 让 Global→LDS 直传不经寄存器堆，DMA 引擎独立于 WGP 执行，两者都指向「减少数据经过计算流水线次数」的架构取向。第三，互连分层已经非常明确：封装内 Infinity Fabric（256 GB/s 双向）、scale-up UALoE（1.8 TB/s/dir）、scale-out UALink128 接 NIC（128 GB/s/dir/NIC）、CPU 侧 IF128 GB/s/dir，并靠 Topology-Aware DMA 让软件不必感知数据位置。封装上，8 颗 N2 计算 die 采用 3D 混合键合堆叠，官方收益是计算密度与性能/瓦——这与本仓库中 AMD 十年 exascale 回顾里「封装面积是被低估的约束」这条经验互为印证。
- **RTL**：演示文稿不提供任何 RTL 实现细节，以下均为工程推断。TDM 需要独立于 WGP 的地址生成、LDS 写端口仲裁与完成跟踪逻辑。Broadcast Arbitrator 与 L2 multicast 需要在 L2 侧实现多播标签、返回路径合并与ordering 保证。Split/Named Barriers 意味着 WGP 内部要新增 barrier 资源与相应状态机。Wave32 单周期执行对寄存器堆端口数与调度器时序提出更高要求。4-bit tensor LUT 指令需要把 4-bit 存储解包并转换到 4/6/8-bit MX 计算格式的转换通路。UALoE 72 lane / 3.6 TB/s 与 12 栈 HBM4 的 192 通道接口都需要相应的 PHY、流控与 ECC 逻辑，这部分规模在演示文稿中完全没有涉及。
- **推断边界**：第 1、2 问的规格数据来自演示文稿，属厂商宣称；其中峰值算力（40.26 PF MXFP4 等）按 endnote `MI400-006` 属理论推算，FP4 与 scale-up 带宽两项对比采用实测对规格或 GPU 数不对等的口径。第 3 问的芯片架构结论是对演示文稿结构选择的归纳，属原文证据的整理；RTL 相关问题演示文稿完全未涉及，全部为工程推断。功耗、面积、成本、端到端负载性能 `TBD`。
