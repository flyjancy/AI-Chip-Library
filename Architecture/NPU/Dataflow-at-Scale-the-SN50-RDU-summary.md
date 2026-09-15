# 深度分析：《Dataflow at Scale: the SN50 RDU》

## 基本信息

- **标题**：Dataflow at Scale: the SN50 RDU
- **文档类型**：Hot Chips 2026 演讲幻灯片（45 页）
- **作者**：Raghu Prabhakar（Chief Architect）
- **机构**：SambaNova Systems Inc.
- **发表 venue**：Hot Chips 2026（文件属性 `Title: HotChips 2026`、`Creator: Google`；正文标注 July 2026 与 Palo Alto 地址）
- **年份**：2026
- **来源**：sambanova.ai（幻灯片页脚）
- **重要标记**：**部分是 `Copyright © 2026 SambaNova Inc. | Confidential & Proprietary` 的保密标记幻灯片**（第 26、39 页带有该页脚）；第 39 页页脚却写 `© 2025`，与其余页的 2026 不一致
- **文档结构**：问题陈述（agentic 推理以 decode 为主、decode 受带宽限制）→ MBU 概念与 GPU 对照 → SN50 芯片/机架/节点/扩展规格 → 数据流机制（持久化 decoder kernel、算通重叠）→ 模型并行（TP/EP）与实测 → 功率-容量-带宽对比 → 异构分离式部署与端到端基准

## 一句话总结

> 这份幻灯片提出一个可检验的指标 **MBU（Model Bandwidth Utilization）**——"峰值 HBM 带宽中真正被权重与 KV cache 用掉的比例"——并据此论证：**前沿模型要跑到高 tok/s/user，所需的 MBU 会超过 100%（单域物理不可能），因此必须靠规模；而 GPU 扩规模时"功率买的是容量不是带宽"（16→64 张 GPU 峰值带宽 4×，速度只 +4%–+25%，MBU 暴跌），SN50 则声称在高芯片数下 MBU 仍守在 40%–51%**，其手段是 **dataflow + 持久化 decoder kernel（一个 kernel 跑完所有 decoder 层、零 kernel 启动开销、无全局同步）+ 算通/HBM 重叠（collective 终止在 SRAM）+ 片内 800G 以太网扩展**。

## 研究动机与问题定义

### 三条前提论证（幻灯片的前 10 页几乎全在建立这个论证链）

1. **Agentic 推理是以 decode 为主的**（第 2 页，来源标注 `AgentX, Cao4, et. al.; decode = (OSL−1) × TPOT`）：

| 模型 | decode 占比 |
| --- | ---: |
| DeepSeek V3 8K/1K（参考点，B300 FP4） | **97%** |
| MiniMax M3 | 94% |
| DeepSeek V4 Pro | 90% |
| Kimi K3 | 82% |
| Qwen 3.5 | 82% |
| GLM 5.2 | 75% |

2. **decode 受带宽限制**（第 3 页）：用 DeepSeek-R1 8K/1K 在 B300 FP4 上、来自 InferenceX 的实测数据画 roofline，标注 **HBM 带宽 128 TB/s**、**FP4 算力屋顶 240 PFLOP/s**、**ridge point = 1,875 FLOP/byte**，右侧为 compute bound、左侧为 memory bound。每个数据点是一个批大小（标注 C×2 到 C×128）；MTP3 的口径是 `batch8/width4`，并注明"**NVIDIA R1 平均接受长度 2.82 仅用于推算 target-forward 速率**"。

3. **关键重新定义（第 4 页）**——这是全篇的分析支点：

   ```
   MBU = 峰值 HBM 带宽中被权重与 KV cache 用掉的占比
   tokens/s = HBM 带宽/芯片 × MBU × 芯片数 ÷ (参数字节 + 每 token KV 字节)
   ```

   幻灯片用两行标注强调：**"峰值 HBM 带宽 = 硬件能搬多少"**、**"MBU = 模型实际用掉多少"**、**"token 速度跟随 MBU，而不是跟随芯片数"**。来源标注为 Databricks 的 LLM inference 性能工程博文。

### 由此推出的核心痛点

- **GPU 扩规模时 MBU 反而下降**（第 6–7 页）：把 26 个 InferenceX GPU 点按交付 tok/s/user 排序，16 张与 64 张 GPU 在同一并发下对照，幻灯片标注：**峰值 HBM 带宽 4×，速度只提升 +4% 到 +25%，MBU ↓69% 到 ↓74%**（图中 16 GPU 的青绿柱落在约 69%–74%，64 GPU 的红点降到很低）。即"**更高的 token 速度、更低的 MBU**"。
- **前沿模型要求的 MBU 超过 100%**（第 8 页）：把 4 个前沿模型（DeepSeek-R1、Kimi K3、GLM-5.2、MiniMax-M3）的全部工作点画在（所需 MBU，所需权重+KV 容量）平面上，**1000 tok/s/user 的红色点落在所需 MBU 约 1,000%–3,000% 的位置**——单域物理上不可能，所以"Frontier Models Demand Scale"。
- **500 kW 预算下的 GPU 困境**（第 9 页）：标题即结论 **"Power Buys Capacity, not Bandwidth"**。同图上标注 **`32 × DGX B300 / 25% MBU / 16 TB/s effective`** 与 **`4× NVL72 / 5% MBU / 28.8 TB/s effective`**（这两个标注的内部一致性问题见"局限性"）。

## 核心结构（架构与规格）

### SN50 芯片（第 11 页）

| 项目 | 数值 |
| --- | --- |
| 算力 | **3,200 TFLOPS（FP8）**，标注 **5× SN40** |
| 工艺/封装 | **5nm CoWoS-L** |
| 片上 SRAM | **432 MB** |
| HBM | **64 GB @ 1.84 TB/s** |
| Prompt caching | **CXL-based DDR，100 GB/s** |
| Scale-up 网络 | **2 TB/s Ethernet**，scale-up domain **256× SN50** |
| Scale-out 网络 | **400G** |

### 芯片内部结构（第 14 页）

- **864 个 PCU 与 PMU**（PCU = Compute unit，PMU = Memory unit），通过 **S（mesh switch）** 组成二维网格；**AGCU = 门户（portal），通向片外内存与 I/O**（图上左侧 AGCU 标 `HBM 500 GB/s`、`PCIe 64 GB/s`、`host I/O 64 GB/s`、`DDR 100 GB/s`；右侧 AGCU 标 `HBM`、`DDR`）。
- **同一块硅片在空间上按算子分区**：图上用颜色区分 **RMSNorm（紫）、QKV（洋红）、Attention（蓝）、Projection（深色）、FFN（绿）** 五个区域，**每个区域各有一组 PCU+PMU**，数据在区域间流动。这意味着**"哪个区域算什么"是一次软件映射（kernel 配置），而不是固定硬件**——这是"dataflow at scale"最具体的体现。

### 数据流机制（第 15–20 页）

- **GPU 的问题（第 16 页）**：同一张 decoder 结构图上叠加约 10 个 `Sync` 与 10 段 kernel（`K1`–`K10`），标注三个病症：**低算子融合（Low Operator Fusion）、低数据局部性（Low Data Locality）、高启动与同步开销（High Launch and Synchronization Overheads）**。
- **SN50 的做法（第 17 页）**：标题即 **"Persistent Decoder Kernel, No Overheads!"**——整张图变成 **一个 `K0`**："**High Operation Fusion: One Kernel Call for All Decoders!**"，配三条标语：**零 kernel 启动开销、高数据局部性**。
- **算通重叠（第 18–20 页）**：
  - 第 18 页把 attention、`Q GEMM`、`K GEMM`、`V GEMM`、`QK Matmul`、`PV matmul` 等分别画到不同区域，并说明 **transpose 是"作为访问模式（access pattern）实现的"**——不需要专用转置硬件。
  - 第 19 页标注 **PMU 双重缓冲（double-buffered PMUs for pipeline parallelism）**，并给出关键推论：**"PMU 存一块权重与激活的 tile；SRAM 容量取决于 tile 大小；模型越大 → 需要越多 SRAM"**。
  - 第 20 页标题 **"No Global Synchronization!"**，说明执行由两条驱动：**数据的流动（Flow of data）与控制 token（Control Tokens）**。
- **overlap 的具体收益（第 30 页）**：**Pipeline All Reduce**——"与计算重叠、无 HBM 流量"（`Embedding with Compute, no HBM traffic!`）、"**collective 终止在 SRAM，不浪费 HBM 带宽**"。第 29 页用第一原理对比"Without Overlap (GPU)"与"With Overlap (RDU)"在 8/16/32 socket 下的差异，结论是**分离的通信阶段会造成瓶颈并拉长总时间，重叠则保持工作流动**。

### 芯片与节点扩展（第 22–25 页）

- **每个 RDU 有 10 个集成 800G 以太网口**，**进出带宽各 1,000 GB/s**（第 23 页）：
  - **7 个口**用于 **8 卡全互联节点内 scale-up**（第 23 页图标注"每 RDU 200 GB/s 用于节点间 scale-up"）
  - **2 个口**用于 scale-up 网络（跨节点）
  - **1 个 RoCEv2 NIC** 用于 scale-up 域之间的 scale-out（**每 RDU 50 GBps 以太网**，400G NIC）
- **8 卡节点**：全互联拓扑（第 23 页图给出 RDU 环形互联示意）。
- **64 卡 scale-up**：**用 2 台 64 端口 800G 交换机**即可实现，每个 RDU 向每台交换机各连一条 800G 链路（第 24 页）——"高带宽单层交换架构"。
- **512 卡 scale-out**：**rail-optimized 两级网络**，每 SN50 RDU 一个 400G RoCE NIC，512S scale-out 域用 **12 台 800G 64 端口交换机**（第 25 页）。
- **机架（第 12 页，风冷）**：

| 项目 | 数值 |
| --- | --- |
| 配置 | **16 RDU / 2 节点** |
| 算力 | **25.6 PFLOPS BF16 / 51.2 PFLOPS FP8** |
| SRAM / HBM | **6.9 GB / 1 TB** |
| RDU DDR | **256 GB 至 2 TB** |
| Host | 2× 64 核 CPU + 1 TB DDR5 |
| Host 存储 | 启动 4× 960 GB NVMe；数据 6× 7.6 TB NVMe |
| 网络 | scale-up **3.2 TB/s**；scale-out **800 GB/s**；host 4×100G |
| 功耗 | **典型 15–20 kW，最大至 34 kW** |
| 尺寸 | 高 2026 mm × 宽 600 mm × 深 1272 mm（含门） |
| 冷却 | **风冷** |

  **一致性核对（自行计算）**：16 × 3,200 TFLOPS = **51.2 PFLOPS FP8** ✓；16 × 64 GB = **1,024 GB ≈ 1 TB** ✓；16 × 432 MB = **6.9 GB** ✓。三项均自洽。

### 模型并行支持（第 27、31、33 页）

- **TP / PP / EP / DP 全部支持**；对应通信原语：TP 与 DP → reduce-scatter + all-gather（或 all-reduce），PP → send-receive，EP → all-to-all。幻灯片特别指出 **"TP 与 DP 归结为同一对 collective，同一条 fabric 路径服务两者，因此扩展其一不需要第二套互连策略"**。
- **MoE 映射**：第 31 页说明 MoE **常用 TP-8/TP-16/TP-32 映射到 RDU**；**"kernel looping 自然扩展到 MoE，配合索引化权重加载（indexed weight loads）"**；索引加载**同时支持 SRAM 与片外内存（HBM、DRAM）**；为捕捉局部性，**同一层的专家经常被预取进 SRAM 并从那里索引**。
- **EP 的两种 token dispatch/combine（第 33 页）**，这是一个明确的工程权衡，值得记录：
  - **Broadcast + filter**：网络流量均匀，但有额外流量，过滤发生在目的端；
  - **All-to-all**：网络流量不均匀，无额外流量，过滤发生在源端。
  - 第 38 页补充：broadcast 方案中"**这个传输可以与 router 并行启动**"，且 **EP+TP 时只广播到同 TP rank 的芯片**。
  - 第 36–37 页说明 all-to-all 路径用 **bins 对 token 与专家分组、然后路由到不同 RDU 目的地**，且是**独立流控的流（independently flow-controlled streams）**，从而"处理动态性、不会缓冲区溢出"。

## 证据、案例与论证

### 证据 1：TP GEMM 片上实测（第 28 页）

| socket 数 | 8 | 16 | 32 |
| --- | ---: | ---: | ---: |
| 加速比 | 1.00 | **1.97** | **4.39** |
| TFLOPs 利用率 | **71.9%** | **70.9%** | **78.9%** |

- 幻灯片的关键结论：**"8、16、32 socket 上一致测得 70% 或更高的 TFLOPs 利用率"**；**"32×SN50 实现算力与通信的完全重叠"**。
- **自行核对**：以 3,200 TFLOPS/芯片为基准，8 socket 聚合算力 = 8×3.2×0.719 = **18.4 PFLOP/s**，32 socket = 32×3.2×0.789 = **80.8 PFLOP/s**，比值 **4.39** ✓——与柱状图的 4.39 完全一致。**也就是 32 socket 的 4.39× 略超线性的 4×，原因是利用率从 71.9% 提升到 78.9%，而非测量噪声。** 这是一个干净、自洽的实测点。

### 证据 2：MoE 层的 MBU 轨迹（第 32 页）

- 标题：**"DeepSeek TP-32 MoE Layers Achieve >70% MBU"**。
- 图为单 RDU 上约 600 µs 的"**HBM and Compute Activity**"双轴图：左轴 HBM 带宽 0–2000 GB/s，右轴 compute activity 0–100%。
- 渲染核对后的读数：紫色实线（HBM 带宽）在预热后平台上界触及约 **1,750–1,840 GB/s**（即接近规格的 1.84 TB/s），**紫色虚线（预热后平均 HBM 带宽）约 1,420 GB/s**；**绿色虚线为 "Target: 85% Utilization"（1,750 GB/s）**；灰线（compute activity）约 **40%**。
- 图上标注两段：**"Shared Experts"** 与 **"Routed Experts"**，以及 **"256× routed expert weight load"**。
- **我的核算**：1,420 / 1,840 = **77%**，与标题的 ">70% MBU" 一致。**但需要同时记录两点**：① **没有达到幻灯片自己画的 85% 目标线**；② 这段窗口内的 **compute activity 仅约 40%**，即 MBU 与算力利用率并不同步。

### 证据 3：EP 专家装载带宽（第 39 页）

- 标题：**"EP on 64-socket SN50: >70% Bandwidth To Load Experts"**（该页带 `Confidential & Proprietary` 标记，页脚年份写 2025）。
- 图为 GB/s 随 cycle（0 至 3M）的轨迹：**平台接近 1.85 TB/s（即该 RDU 的 HBM 峰值）**，周期性下探到 **600–900 GB/s**，首次下探深至约 **120 GB/s**（约 200K cycles 处）。
- **口径注意**：标题说 64-socket，但纵轴是**单个 RDU 的 HBM 带宽**——这是板级实验而不是 64 卡聚合带宽。

### 证据 4：功率 → 容量与可实现带宽（第 42 页，SN50 侧）

| 规模 | 功耗 | MBU | 可实现带宽 | 容量（权重+KV） |
| --- | ---: | ---: | ---: | ---: |
| 128 RDUs | 120 kW | **43.7%** | **103 TB/s** | 8.2 TB |
| 256 RDUs | 240 kW | **45.1%** | **212 TB/s** | 16.4 TB |
| 512 RDUs | 480 kW | **40.0%** | **377 TB/s** | 32.8 TB |

- **自行核对（这是本文最有价值的一处自洽性验证）**：128 × 1.84 TB/s = 235 TB/s 峰值，103/235 = **43.8%** ✓；256 × 1.84 = 471，212/471 = **45.0%** ✓；512 × 1.84 = 942，377/942 = **40.0%** ✓。三项全部对上，**证明 MBU 确实是按"每芯片 HBM 峰值带宽 × 芯片数"定义的**，且可实现带宽 = MBU × 聚合峰值。
- 缩放行为：**功耗 4×、容量 4×、可实现带宽 3.66×**，MBU 从 43.7% 到 40.0%（不随规模衰减）。

### 证据 5：高芯片数下 MBU 不衰减（第 40 页）

在 GPU 的 26 个 InferenceX 点上按交付速度插入 SN50 的三个点：**64 → 128 → 256 张 SN50 对应 200 → 300 → 500 tok/s/user，MBU 为 51% → 44% → 45%**，结论是 **"MBU Stays High with Higher Chip Counts!"**。

**与 GPU 的对照（第 7、9 页）**：GPU 从 16 到 64 张（4× 峰值带宽）只换来 +4% 到 +25% 速度、MBU 大幅下降到个位数百分比。**这是整份幻灯片的核心对照**：同样是扩规模，GPU 的 MBU 崩塌而 SN50 维持。

### 证据 6：端到端基准（第 44 页）

- 指标：**MiniMax-M2.7、10,000 输入 token、每次运行最多 32,768 输出 token 的 output tokens/s**；来源标注 **Artificial Analysis Serverless API benchmarking（07-07-26）**。
- 柱状：

| 服务 | tok/s |
| --- | ---: |
| **SambaNova SN50 Private Endpoint（Disagg）** | **763** |
| SambaNova Public Endpoint（SN40） | 482 |
| Together AI | 187 |
| Fireworks | 98 |
| Novita（FP8） | 53 |
| MiniMax（第一方） | 51 |

- 底部配置块：**部署 = 异构分离式服务**；`1× NVL/H200` → `16× SambaNova SN50 · TF16`；**引擎 = vLLM**。
- 左侧另有一张 SemiAnalysis 图表：`Token Throughput per GPU vs Interactivity`（MiniMax-M2.7 2026-07-08），同图例含 B200（SGLang Custom, MTP）、H200-MBRD、GB200、B300 等多个配置，并**在该图上标注 "3.3x Faster"**。

## 局限性与未来方向

### 需降级或明确限定口径的地方

- **端到端基准不是同类对比**：SN50 的点是 **`Private Endpoint`（专用端点）**，而 Together AI / Fireworks / Novita / MiniMax 都是 **serverless 共享端点**（Artificial Analysis 量的是公共 API）。专用硬件与共享服务不在同一口径上，**763 vs 187/98/53/51 的差距不能直接读作"硬件效率差距"**。另外 SN50 配置标 **TF16** 而 Novita 标 **FP8**，数值精度口径也不同。
- **"3.3x Faster" 的归属需要小心**：该标注画在左侧 **SemiAnalysis 的 per-GPU 吞吐图上**（图中同一模型不同配置的对比），**不是 SN50 对 SN40 的倍数**（763/482 = **1.58×**）。幻灯片未明确说明这个 3.3× 的比较对象，**不应读作 SN50 的加速比**。
- **第 9 页的两个 GPU 标注在算术上无法互相对齐**。按第 42 页已验证的定义（可实现带宽 = MBU × 聚合峰值带宽）反推：`16 TB/s ÷ 25% = 64 TB/s` 峰值，正好等于 **8 张** 8 TB/s GPU；`28.8 TB/s ÷ 5% = 576 TB/s` 峰值，正好等于 **72 张** 8 TB/s GPU。若按标注文字理解成"32 个 DGX B300"（256 张）与"4× NVL72"（288 张），则两者都相差约 4×，且 **"5% MBU 配 28.8 TB/s" 与 "25% MBU 配 16 TB/s" 出现"百分比更低但绝对值更高"的倒置**。**这两个标注要么归属单位与文字不符，要么"effective"不是 MBU × 峰值——我无法用任何合理的峰值带宽把两组数字同时对上**，因此"GPU @500 kW 只有 16–28.8 TB/s"这个对照的**绝对数值应按存疑处理**（其"扩规模 MBU 崩塌"的定性趋势另有第 6–7 页的实测支撑）。
- **第 7 页的百分比标注语义不明**：文字为 **"↓69% 到 ↓74% MBU"**，但图中 16 GPU 的青绿柱本身就在约 69%–74% 处、64 GPU 的红点在个位数百分比。因此这组数字**可以读成"MBU 降到 69%–74%"（与图矛盾）或"MBU 下降 69%–74%"（与图大致相符）**；应以后者为准，但标注写法本身有歧义。
- **第 32 页的 ">70% MBU" 是 kernel 级测量，不是端到端 MBU**：图的时间窗只有约 600 µs、内容是 MoE 层内的 HBM 与 compute activity；且**预热后平均（约 77%）低于幻灯片自画的 85% 目标线**，**同窗口 compute activity 仅约 40%**。把 ">70% MBU" 当作"整模型推理的 MBU"会高估。
- **第 39 页的 EP 结果同样不是端到端**：纵轴是单 RDU 的 HBM 带宽，且曲线有明显的周期性下探（最低约 120 GB/s），幻灯片只给了 ">70%" 的下界而没有给平均值或吞吐损失。
- **关键规格缺失**：**没有 die 面积、没有 TDP（只有机架级 15–20 kW 典型 / 34 kW 最大）、没有 SRAM 带宽数字**（第 42 页的可实现带宽是 HBM 口径）、没有 scale-up 网络的实测延迟、**没有与任何 GPU 在同一模型、同一端点类型、同一精度下的逐点对照**。
- **MBU 定义本身的适用范围**：MBU 用"权重 + KV cache 的字节"作分子，因此它**只对 memory-bound 的 decode 阶段有意义**。幻灯片第 2 页论证了 agentic 推理 75%–97% 是 decode，所以这个指标适用于主场景，但它**不能衡量 prefill 与训练**，而第 44 页的 benchmark（10,000 输入 token）实际上包含相当比例的 prefill 工作。
- **来源性质**：这是**厂商幻灯片**，多页带 `Confidential & Proprietary` 标记；benchmark 数字包含厂商自测（第 28、32、39 页）与第三方（Artificial Analysis、SemiAnalysis、InferenceX）两类，**报告中已分别标注**。

### 幻灯片给出的未来方向

- **异构分离式部署（Heterogeneous Disaggregation，第 43 页）**：控制面用 Dynamo 做 frontend/router/planner；prefill/midfill 可由 GPU 或 RDU 承担；中间设 **KV Appliance**；decode 侧为 RDU。运行时栈为 **vLLM / SGLang + LMCache + NIXL + SambaBaseSDK**，存储层级贯穿 DDR / HBM / Host / local NVMe / SSD，跨节点用 scale-out NIC。**即 GPU 与 RDU 不是替代关系，而是按阶段分工。**

## 个人点评

- **这份材料最有价值的东西是那个指标，而不是那些规格。** MBU = "峰值 HBM 带宽中真正被权重与 KV cache 用掉的比例"，并把 tokens/s 改写成 `HBM 带宽 × MBU × 芯片数 ÷ 每 token 字节数`。这一步把"堆芯片"这个动作的收益从"线性"降级为"取决于 MBU 是否守住"，**而第 6–7 页的 GPU 数据正是这个论点的实测支撑**：16→64 张 GPU，峰值带宽 4×，速度只 +4% 到 +25%。**这条论证即使把 SN50 的所有数字都去掉也成立**，是全文最可迁移的部分。
- **第 8 页那张图是全文论证的核心**：把前沿模型的工作点画在（所需 MBU，所需容量）平面上，**1000 tok/s/user 的点落在所需 MBU 1,000%–3,000% 的位置**——超过 100% 就是物理不可能，所以"必须靠规模"不是营销话术而是算术结论。这比"我们的芯片更快"有信息量得多。
- **我复核了三组数字，全对，这是这份材料可信度的重要加分**：① 第 42 页 128/256/512 RDU 的 MBU（43.7/45.1/40.0%）与可实现带宽（103/212/377 TB/s）**全部与"芯片数 × 1.84 TB/s"精确吻合**；② 第 28 页 32 socket 的 **4.39×** 加速与利用率 71.9%→78.9% **完全自洽**（也就是略超线性的 4× 来自利用率提升，不是噪声）；③ 机架的 51.2 PFLOPS FP8、1 TB HBM、6.9 GB SRAM **三项都由 16 卡的规格推出**。**厂商幻灯片里能这样对上的不多**，所以第 42 与第 28 页可以较放心地引用。
- **同一份材料里也有一处对不上，值得单独记下**：第 9 页的 `32× DGX B300 / 25% MBU / 16 TB/s` 与 `4× NVL72 / 5% MBU / 28.8 TB/s`。用第 42 页刚验证过的定义反推，前者要求 64 TB/s 峰值（= 8 张 B300），后者要求 576 TB/s（= 72 张 B300）——**正好是"一个 DGX B300"和"一个 NVL72"的量，而不是标注写的 32 个和 4 个**，而且出现了"MBU 百分比更低、绝对带宽反而更高"的倒置。**这是本篇最需要降级的数字。** 好的做法是：引用"扩规模 MBU 崩塌"的**趋势**（第 6–7 页有实测图支撑），不引用这两个标注的**绝对值**。
- **第 17 页那张对比图把"dataflow"讲得比任何文字都清楚**：同一张 decoder 结构图，GPU 版有约 10 个 `Sync` 和 `K1`–`K10` 十个 kernel，并标注"低算子融合 / 低数据局部性 / 高启动与同步开销"；SN50 版把整张图收成**一个 `K0`**。而第 14 页的芯片图显示**同一块硅片在空间上按 RMSNorm/QKV/Attention/Projection/FFN 分区、每个分区各有自己的 PCU/PMU**——这两页合起来才是完整的机制：**把"逐层执行的算子序列"变成"空间上并置、由数据流与控制 token 驱动的持久化流水线"**。这是本篇对其他架构最有借鉴意义的部分。
- **与仓库其他材料的对照**：
  1. **Deck 的"持久化 kernel + 零启动开销"与 B3 的 Groq TSP 是同一诊断**（kernel/同步开销吃掉有效算力），但处方不同：Groq 走全 SRAM + 静态调度 + 低 batch，SambaNova 走 dataflow + 大 HBM + 算通重叠。
  2. **第 30 页"collective 终止在 SRAM、不浪费 HBM 带宽"与本仓库 Designing AI Chip 一文的主张完全同向**——后者明确要求"支持 SRAM–network–SRAM DMA，绕过 HBM"，理由是"否则网络原语会消耗大量 HBM 带宽与功耗"。**两个不同来源独立指向同一个 RTL 需求，这条很值得作为设计要点记录。**
  3. **"transpose 作为访问模式实现、不需要专用转置硬件"** 与 Designing AI Chip 一文的论述一致（后者指出推理可以靠 `A×(Wᵀ)ᵀ` 离线转置权重、或用 `A*Bᵀ` 模式规避转置硬件，但保留 permute 更保险）。
  4. **第 33 页 broadcast vs all-to-all 的权衡**与本仓库 Patterns behind Chaos 的 MoE 结论呼应：后者指出 MoE 的多跳是 torus 的软肋（"小 torus 可以、大 torus 不行"），而 SambaNova 选择在以太网 fabric 上同时提供 broadcast（流量均匀但有冗余）与 all-to-all（流量不均但零冗余）两种原语——**这正是"用可选的冗余换流量均匀性"的工程折中**，与 torus/router 的讨论是同一个设计空间。

## 工程化三问总结

### 1. 它解决了什么瓶颈？

- **应用场景**：数据中心规模的**前沿模型推理服务，且以 agentic 场景为主**（幻灯片第 2 页论证 decode 占 75%–97%）。
- **被识别的瓶颈是"带宽利用率"而不是"带宽总量"**。幻灯片第 4 页给出明确定义：`MBU = 峰值 HBM 带宽中被权重与 KV cache 用掉的占比`，并把 tokens/s 改写成 `HBM 带宽/芯片 × MBU × 芯片数 ÷ (参数字节 + 每 token KV 字节)`，据此宣称"**token 速度跟随 MBU，而不是跟随芯片数**"。
- **对这个瓶颈的证据**：
  - **GPU 扩规模时 MBU 崩塌**（第 6–7 页，InferenceX 实测的 26 个 B300 FP4 点）：16→64 张 GPU 在同一并发下，峰值 HBM 带宽 4×，**速度只提升 +4% 到 +25%，MBU 大幅下降到个位数百分比**。
  - **前沿模型要求的 MBU 超过 100%**（第 8 页）：1000 tok/s/user 的工作点落在所需 MBU 约 **1,000%–3,000%**，因此"必须靠规模"。
  - **500 kW 下的 GPU 困境**（第 9 页）："Power Buys Capacity, not Bandwidth"，标注 16–28.8 TB/s 可实现带宽（**但这组绝对值与自身定义的算术不一致，见局限性，只能采信其定性趋势**）。
- **SN50 侧的对照证据（厂商自测 + 第三方混合）**：
  - **TP GEMM 片上实测**（第 28 页）：8/16/32 socket 加速比 **1.00/1.97/4.39**，TFLOPs 利用率 **71.9%/70.9%/78.9%**；32 socket 实现完全重叠。**此组数字内部自洽（已复核）。**
  - **MoE 层 MBU 轨迹**（第 32 页）：DeepSeek TP-32，预热后平均 HBM 带宽约 **1,420 GB/s**（≈ 1,840 GB/s 峰值的 **77%**），标题称 >70% MBU；**但未达自画的 85% 目标线，且同窗口 compute activity 仅约 40%**。
  - **EP 专家装载**（第 39 页）：64 socket 上 **>70%** 带宽用于装载专家；曲线平台接近单 RDU HBM 峰值 1.85 TB/s，周期性下探至 600–900 GB/s（最低约 120 GB/s）。
  - **功率-容量-带宽**（第 42 页）：128/256/512 RDU @ 120/240/480 kW → MBU **43.7%/45.1%/40.0%**，可实现带宽 **103/212/377 TB/s**，容量 **8.2/16.4/32.8 TB**。**三项 MBU 与"芯片数 × 1.84 TB/s"精确吻合（已复核）**，即功耗 4×、容量 4×、带宽 3.66×，MBU 不随规模衰减。
  - **高芯片数下 MBU 不衰减**（第 40 页）：64→128→256 张 SN50 对应 200→300→500 tok/s/user，MBU **51%→44%→45%**。
  - **端到端**（第 44 页）：MiniMax-M2.7、10,000 输入 token 下 SN50 **763 tok/s**（专用端点、分离式）对 SN40 482、Together AI 187、Fireworks 98、Novita(FP8) 53、MiniMax 51。**口径不同类（专用 vs serverless），需降级。**

### 2. 用了什么结构或方法？

- **Dataflow 架构 + 持久化 decoder kernel**：把整张 transformer 图（Embedding → 32 个 Decoder → Classifier → Sampling）收成**一个 kernel `K0`**，实现"所有 decoder 只用一次 kernel 调用"，消除 kernel 启动开销与全局同步（第 17、20 页）。
- **片内空间分区 + 网格互连**：**864 个 PCU/PMU + mesh switch（S）+ AGCU 门户**，按算子（RMSNorm / QKV / Attention / Projection / FFN）在**同一块硅片上分区**，由**数据流与控制 token** 驱动（第 14、20 页）。
- **算子内联而非专用硬件**：**transpose 实现为访问模式**，不占专用转置单元（第 18 页）。
- **算通与 HBM 重叠**：**PMU 双重缓冲**支撑流水并行；**collective 终止在 SRAM（不进 HBM）**；"Pipeline All Reduce"与计算、HBM 访问并行；与计算重叠以避免额外 HBM 流量（第 19、29、30 页）。
- **索引化权重加载支持 MoE**：kernel looping 自然扩展、索引加载同时支持 SRAM 与片外内存、**同层专家预取进 SRAM 再索引**（第 31 页）。
- **EP 的两种原语**：broadcast+filter（流量均匀、有冗余、目的端过滤、可与 router 并行启动）与 all-to-all（流量不均、零冗余、源端过滤、独立流控防止缓冲溢出）（第 33、36–38 页）。
- **网络设计**：**每 RDU 10 个集成 800G 以太网口**（7 口做 8 卡全互联、2 口做跨节点 scale-up、1 个 400G RoCE 做 scale-out）；**64 卡单层交换（2 台 64p 800G）**、**512 卡 rail-optimized 两级 scale-out**；TP 与 DP 共用同一对 collective 因而 **scale-up fabric 一条路径服务两类并行**（第 23–25、27 页）。
- **存储层级**：432 MB 片内 SRAM / 64 GB HBM @1.84 TB/s / **CXL DDR 做 prompt caching @100 GB/s** / 机架级 RDU DDR 256 GB–2 TB / host NVMe（第 11、12 页）。
- **部署方法**：**异构分离式服务**，Dynamo 做控制面，prefill/midfill 可用 GPU 或 RDU，中间设 KV Appliance，decode 用 RDU，运行时栈 vLLM/SGLang + LMCache + NIXL（第 43 页）。
- **量化策略**：本材料**不涉及把模型量化到具体 bit 宽度的实验**；只出现部署侧的精度标签（SN50 配置标 **TF16**、对比方 Novita 标 **FP8**）与硬件侧的 FP8 峰值算力口径（3,200 TFLOPS FP8）。

### 3. 对芯片架构和 RTL 有什么启发？

- **芯片架构层面**：
  1. **把"带宽利用率"作为一等设计指标，而不是只报峰值带宽**。MBU 的定义（峰值 HBM 带宽中被权重与 KV cache 用掉的比例）可以直接搬进架构评估：`tokens/s = HBM 带宽 × MBU × 芯片数 ÷ 每 token 字节数`。**这个式子的实用价值在于它把"堆芯片"的收益变成了一个必须论证的量**——堆芯片只有在 MBU 守得住时才有线性收益，而第 6–7 页的 GPU 实测显示 MBU 会随规模崩塌。
  2. **把算子序列空间化，用数据流代替逐 kernel 启动**。第 17 页的对比（GPU 的 `K1`–`K10` + 约 10 个 Sync vs 单个 `K0`）与第 14 页的分区图给出了具体做法：**同一块硅片上按算子划分区域、每个区域自带 PCU/PMU 组，由数据流与控制 token 驱动**。对应的硬件需求是**区域间的片上互连与流控**（本材料用 mesh switch + 独立的流控流），而不是更高的单点带宽。
  3. **collective 要终止在 SRAM，不要经 HBM**（第 30 页明确写"Collective terminate in SRAM, no wasted HBM bandwidth"）。这条与 Designing AI Chip 一文的"S​​RAM–network–SRAM DMA、否则网络原语会消耗大量 HBM 带宽与功耗"**来自两个独立来源、指向同一个硬件需求**，是本仓库内最值得采信的设计要点之一。
  4. **大规模并行的收益取决于 fabric 是否让 TP 与 DP 共用同一路径**。第 27 点的设计选择（TP 与 DP 都归结为 reduce-scatter + all-gather，因此同一条 fabric 服务两者）意味着**扩展其中一种并行不需要第二套互连**——这是一个可以被其他架构直接采纳的拓扑简化。
  5. **用"可选冗余"换流量均匀性**：EP 同时提供 broadcast+filter（均匀但有冗余）与 all-to-all（零冗余但不均）两种原语，并说明 broadcast 可**与 router 计算并行启动**、EP+TP 时**只广播到同 TP rank**。这类"把两种语义都做进 fabric"的设计，比赌单一原语更稳健。
  6. **片上网络用标准以太网而非专有互连**（每 RDU 10× 800G、64 卡只需 2 台 64p 800G 交换机、512 卡用 12 台做 rail-optimized 两级）。这是一个明确的成本立场：**用可采购的成熟交换机与协议，而不是自研 fabric**。
  7. **CXL DDR 专门用于 prompt caching（100 GB/s）**体现了"KV 相关数据按用途分层"的思路——与 B4 Memory 批次讨论的"HBF/HBM 分层"是同一问题的不同解法。
- **RTL 层面**：可落到实现层的模块包括：
  - **PCU（Compute unit）与 PMU（Memory unit）两种可复用的 tile**，以 **864 个**规模组成二维网格，配 **mesh switch（S）** 做路由——即"少数几种 tile + 规则网格"的 RTL 策略（有利于后端与验证）。
  - **AGCU（门户）**：通向片外 HBM（图上标 500 GB/s）、PCIe（64 GB/s）、host I/O（64 GB/s）、DDR（100 GB/s）的统一出入口。
  - **PMU 的双重缓冲**（支撑流水并行）与**权重/激活 tile 的 SRAM 暂存**——幻灯片明确给出容量推导关系：**"SRAM 容量取决于 tile 大小；模型越大需要越多 SRAM"**。
  - **硬件流控的独立传输流**（用于 all-to-all 的 token dispatch，宣称"处理动态性、无缓冲溢出"）。
  - **索引化权重加载通路**，同时支持 SRAM 与片外内存（HBM/DRAM），以服务 MoE 的专家选择。
  - **transpose 作为访问模式**的地址生成/访存模式支持（而非专用转置单元）。
  - **控制 token 机制**（执行由"数据流动 + 控制 token"驱动，第 20 页）——这要求在 RTL 里实现一套基于 token 的依赖/触发逻辑。
  - **拓扑构建的同步原语**：reduce-scatter / all-gather / all-reduce / send-receive / all-to-all。
  - **网络接口**：每 RDU 10× 800G 以太网 MAC/PHY 逻辑（含 RoCEv2 与自定义 RDU NIC 两条路径，第 22 页区分了"Custom RDU NIC"用于 scale-up、"Standard RoCE NIC"用于 scale-out）。
  上述模块的动机均来自幻灯片描述，**但材料没有给出任何面积、功耗、时序、SRAM 带宽或队列深度的量化预算**（尤其**没有 die 面积、没有芯片级 TDP、没有 SRAM 带宽**）。
- **推断边界**：第 1 问中第 6–9 页的 GPU MBU 趋势来自 InferenceX 实测数据；第 28、32、39、42、40、44 页为厂商自测或第三方基准（口径已在正文逐条标注，其中第 9 页的绝对值与第 32 页的 kernel 级 MBU 已明确降级）。第 2 问的结构描述全部来自幻灯片。第 3 问的架构与 RTL 内容为工程推断。**材料中未提供的量值**——die 面积、芯片级 TDP、SRAM 带宽、网络实测延迟、与 GPU 在同一模型/同一端点类型/同一精度下的逐点对照、第 9 页两组标注的正确归属与单位、以及"3.3× Faster"的确切比较对象——均为 `TBD`。
