# 深度分析：《TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings》

## 基本信息

- **标题**：TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings
- **文档类型**：论文（ISCA '23 Industry Track，14 页）
- **作者**：Norman P. Jouppi、George Kurian、Sheng Li、Peter Ma、Rahul Nagarajan、Lifeng Nai、Nishant Patil、Suvinay Subramanian、Andy Swing、Brian Towles、Cliff Young、Xiang Zhou、Zongwei Zhou、David Patterson
- **机构**：Google（David Patterson 兼 UC Berkeley）
- **发表 venue**：The 50th Annual International Symposium on Computer Architecture (ISCA '23)，June 17–21, 2023, Orlando, FL, USA
- **年份**：2023
- **链接**：DOI 10.1145/3579371.3589350

## 一句话总结

> TPU v4 用 **48 台 MEMS 光路交换（OCS，136×136 端口，毫秒级切换）**把 4096 颗芯片连成可重构的 3D torus，并公开了自 TPU v2 起就存在的 **SparseCore**（占 die 面积与功耗各约 5%，把 embedding 加速 5–7×）；OCS 与光器件合计不到系统成本的 5%、功耗的 3%，却换来可绕开 1%–1.1% 不可用 CPU 主机的可用性、可分片调度、可扭曲 torus（all-to-all 吞吐 1.31–1.63×），使 540B 的 PaLM 在 50 天训练中稳定在峰值算力的 57.8%。

## 研究动机与问题定义

- **要解决的核心问题**：ML 模型同时朝两个方向演化——规模（LLM）与算法结构（推荐系统的 embedding、Transformer/BERT）。单机规模从 TPU v2 的 256 颗扩到 4096 颗时出现三个具体障碍：
  1. **规模与可靠性**：DNN 训练采用 HPC 式的 checkpoint/restore、"所有部件必须工作"的范式，与 Google 主线分布式系统的软件可靠性水平相去甚远。系统里有 1K 台 CPU 主机（每台带 4 颗 TPU v4），主机在 0.1%–1.0% 的时间里不可用。
  2. **embedding 的硬件支持**：DLRM 是 Google ML 负载的四分之一（Table 1：2022 年 10 月 TPU v4 训练负载中 MLP/DLRM 占 24%），其 embedding 查找是低算术强度的小粒度 gather/scatter，与 TensorCore 优化的稠密运算不匹配。
  3. **all-to-all 通信**：embedding 引入 all-to-all 模式，它不像反向传播里的 all-reduce 那样适合 2D/3D torus，会挤压 bisection bandwidth。
- **现有方法的不足**：
  - **电互连的物理限制**：TPU v3 的部分 2D torus 环绕链路因距离过远必须用光互连，而**光链路比电链路贵 10 倍以上**（第 2 页）。规模再翻 4 倍只会产生更多光链路。
  - **2D torus 的 bisection bandwidth** 在 4096 颗规模下不足。
  - **静态拓扑无法容忍故障**：TPU v3 系统要等全部 1024 颗芯片与线缆安装测试完才能使用，任何一个部件的交付延迟会拖住整个超算。
  - **embedding 放在哪都别扭**：放 TensorCore 上因为 gather/scatter 粒度小而次优；放主机 CPU 上会因 CPU DRAM 接口成为 Amdahl 瓶颈（TPU 与主机是 4:1），还受数据中心网络的尾延迟与带宽限制。

## 核心方法

### OCS：可重构光交换

**物理与时序结构**

- **Palomar OCS** 基于 3D MEMS 反射镜，**毫秒级切换**；端口数 **136×136（128 个端口 + 8 个备用，供链路测试与修复）**。
- 采用**环行器（circulator）**让光在同一根光纤中双向传输，**把所需端口数与线缆数减半**。

**拓扑构造**

- 基本积木选 **4³ = 64 颗芯片**（64 颗与 16 台 CPU 主机正好放进一个机架）；512 颗需要多机架，所以选 4³ 作 building block。
- 每个 4³ 块有 6 个"面"，**每面 16 条链路，共 96 条光链路**接 OCS。为提供 3D torus 的环绕链路，**相对两侧的链路必须接到同一台 OCS**，因此每个 4³ 块连 **6 × 16 ÷ 2 = 48 台 OCS**。
- 48 台 OCS 连接 48 对来自 64 个 4³ 块的线缆（每块 64 颗），得到 **4096 颗 TPU v4**。**48 台 OCS 把 8 排机架连成完整系统**（一个超算共 64 个机架）。

**芯片与板级结构（第 3 页）**

- 每颗 TPU v4 含 **2 个 TensorCore（TC）**；每个 TC 含 **4 个 128×128 MXU** 与一个 **128 lane 的 VPU（每 lane 16 个 ALU）** 加 **16 MiB VMEM**。两个 TC 共享 **128 MiB CMEM**。
- 板内嵌 **4 条 ICI 链路**接成 **2×2 mesh**；**16 条外部 ICI 链路**接其它 tray 以构造 3D torus。
- 机架内用无源电缆构成 **4×4×4 的 3D mesh**；电光转换发生在 TPU tray 的光纤连接器处，**中间不再有其它转换**，直到目的 tray 的连接器做光电转换。

**OCS 的八项收益（第 12 页总结）**

1. **可扩展性**：规模到 4096 颗。
2. **可用性**：OCS 可以绕开故障单元。图 4 给出在不同 CPU 主机可用率（99.0%–99.9%）与不同 slice 尺寸下的 goodput：**没有 OCS 时主机可用率必须达到 99.9%**。论文解释了 goodput 在大 slice 下的反直觉行为：3 个 slice 占满 4K 芯片的 3/4，每个 slice 的 goodput 就是 75%；从 4K 里切一个 2K 的 slice 会剩 50% 备用（goodput 50%），切一个 3K 的 slice 只剩 25% 备用（goodput 75%）。
3. **模块化**：能给出 3D torus 拓扑（相对 mesh 使 bisection bandwidth 与 all-reduce 带宽翻倍），并能做到 **163（4096）颗规模**。
4. **性能**：用户可按负载选拓扑。
5. **功耗**：MEMS 光交换比电子包交换更省电。
6. **调度简化**：TPU v3 的 256 颗 slice 要求调度器找到 256 颗连续空闲芯片；TPU v4 可以从超算任意位置取 4³ 块。slice 也不必是 2 的幂，可以是 **4i×4j×4k（0 < i ≤ j ≤ k）**，例如 4×4×12 = 192 颗。
7. **部署速度**：TPU v3 要等 1024 颗芯片与全部线缆就绪才可用；TPU v4 每个 4³ 块（64 颗芯片）装好、测好即可投产。
8. **安全**：OCS 可以在不同 slice 之间做**物理隔离（air-gapped network isolation）**，便于多客户共享同一超算。

**成本**：OCS 与整个光 fabric（光模块、光纤、OCS 基础设施）合计 **<5% 的资本成本、<3% 的功耗**（第 5 页）。

### 扭曲 torus（Twisted Torus）

- 对完美立方体的 slice，对称 torus 最小化延迟并最大化 bisection bandwidth；对非立方 slice，可构造矩形 torus，各维芯片数不同。
- **门限条件**：只有形如 **n×n×2n 或 n×2n×2n（n ≥ 4）**的 slice 才可扭曲。
- 实现方式：由于 4³ 块之间本就由 OCS 连接，"重新布线"**主要是改 OCS 的路由表**，不需要物理重新接线（图 5）。TPU v4 采用 Camarero、Martinez、Beivide 的 **k×k×2k** 配置。
- **实测收益（图 6）**：以大批量 all-to-all 为微基准（单次 DMA 4 KiB，稳态大聚合传输），扭曲 torus 相对规则 torus 的吞吐提升 **4×4×8 上 1.63×、4×8×8 上 1.31×**。

### SparseCore：embedding 的硬件支持

**动机**（第 5 页）

embedding 是通过查找表把大类空间（词表）映射到稠密向量的标准手段。一个模型可以有很多不同大小的表；一个训练样本只触及极少行（univalent 查一行，multivalent 查少量行并求和）。查找操作主要是小粒度 gather/scatter，**算术强度低**，性能瓶颈在内存带宽、内存容量、VPU 性能与 ICI 互连，而不是 FLOPS。

**结构（图 7）**

- SC 是**从 TPU v2 起就存在**的 DSA，TPU v3/v4 逐步改进，**合计只占约 5% 的 die 面积与约 5% 的功耗**。
- 以 **sea-of-cores** 方式工作，把超算规模的 HBM 与 ICI 组合成**一个平坦的全局可寻址内存空间（TPU v4 上为 128 TiB）**。
- 最通用的单元是 **16 个 compute tile**（每 tile 绑一个 HBM channel，支持多个在途访存），每 tile 含 **Fetch Unit**（把激活与参数读进 tile 的 2.5 MiB Sparse Vector Memory）、**8-wide SIMD 的 scVPU**（复用 TC 的 VPU ALU）、**Flush Unit**（反向传播时写回更新后的参数）。
- 另有 **5 个 Cross-Channel Unit** 执行特定的 embedding 操作，跨 Spmem 的全部 16 个 bank 工作。
- 与 TPU v1 类似，SC 执行 **CISC 式指令**、处理变长输入，**每条指令的运行时间依赖数据**。
- 论文把 SC 称为"dataflow 架构"，理由是数据从内存流向一组直接相连的专用计算单元。

**性能机制**

- 端到端 embedding 查找性能**基本上正比于 bisection bandwidth**（因为要 all-to-all 传小的 embedding 向量）。TPU v2/v3 的 2D torus 的该带宽按 $N^{1/2}$ 缩放，**TPU v4 的 3D torus 按 $N^{2/3}$ 缩放**。
- 图 8：在给定芯片数下 **TPU v4/v3 的 bisection bandwidth 比值为 2–4×**，把 embedding 加速 **1.1×–2.0×**；到 1024 颗时 SC 自身开销开始占主导，bisection bandwidth 不再那么关键。

### 用 ML 共同优化 DNN、拓扑与 SparseCore

- **PA-NAS（platform-aware NAS）**用于把 DNN 与 TPU v4 超算自动对齐。PA-NAS 设计的 CNN1 相对通用 NAS 的设计有 **约 1.6× 的性能提升（QPS 与延迟）**且精度相当。
- **DLRM 同时用到 SC 与 TC**，PA-NAS 可以在稀疏层（跑在 SC）与稠密层（跑在 TC）之间转移计算负载。图 10：原始 DLRM0 虽然经过手工与通用 NAS 优化，仍因 SC 与 TC 的负载不均而**让 SC 闲置约 25% 的执行时间**；PA-NAS 使其接近完美的 SC-TC 负载均衡，端到端性能提升 **>10%**。论文称这个提升相当于 10 人以上专家团队约半年的优化成果。
- **拓扑也交给搜索**（Table 3）：

| 案例 | 版本 | 吞吐（seqs/sec） | 超参（拓扑、partition spec、1D/2D 划分） |
| --- | --- | ---: | --- |
| LLM | 新手选择 | 17.9 (1.0×) | 4×8×16, [1, 1, 16, 32], 2D/2D |
| LLM | 最优 | **41.3 (2.3×)** | 8×8×8, [1, 1, 64, 8], 1D/2D |
| GPT-3 预训练 | 专家选择 | 21.0 (1.0×) | 8×8×8, [8, 1, 8, 8], 2D/2D |
| GPT-3 预训练 | 最优 | **25.0 (1.2×)** | 4×8×16, [16, 4, 1, 8], 1D/1D |

## 实验与结果

### 实验设置

- **对照**：TPU v3（同 slice 尺寸，同芯片数）、NVIDIA A100、Graphcore MK2 IPU Bow、以及 CPU 方案（576 个 Skylake socket：400 个 learner + 176 个 variable server）。
- **负载**：8 个生产应用（DLRM0/1、CNN0/1、RNN0/1、BERT0/1）加 MLPerf Training 2.0 的 BERT 与 ResNet。
- **功耗测量**：TPU v4 侧在 Google 数据中心以 64 颗规模跑 MLPerf 2.0 代码；A100 侧用 `nvidia-smi` 在 Azure `Standard_ND96amsr_A100_v4` VM 上重跑 NVIDIA 的 64 芯片提交来取值。

### 主要结果

**芯片级特性对比（Table 4）**

| | TPU v4 | TPU v3 |
| --- | ---: | ---: |
| 生产部署 | 2020 | 2018 |
| 峰值 TFLOPS | **275（bf16 或 int8）** | 123（bf16） |
| 时钟 | 1050 MHz | 940 MHz |
| 工艺 / die 面积 | 7 nm / <600 mm² | 16 nm / <700 mm² |
| 晶体管数 | **22 B** | 10 B |
| 每 CPU 主机的芯片数 | 4 | 8 |
| 空闲功耗 min/mean/max | 90 / 121 / 170 / 192 W | 123 / 175 / 220 / 262 W |
| 片间互连 | **6 links @ 50 GB/s** | 4 links @ 70 GB/s |
| 最大规模 | **4096 颗** | 1024 颗 |
| Processor Style | 单指令 2D 数据 | 单指令 2D 数据 |
| SparseCore / 芯片 | **4** | 2 |
| 片上内存 | **128（CMEM）+ 32（VMEM）+ 10 MiB（spMEM）** | 32（VMEM）+ 5 MiB（spMEM） |
| HBM2 容量 / 带宽 | 32 GiB / **1200 GB/s** | 32 GiB / 900 GB/s |

- 7 nm 替代 16 nm 使矩阵乘法器数量翻倍、时钟快 11%，共同驱动 **2.2× 的峰值性能增益**。
- **perf/Watt 提升 2.7×，其中约 40% 来自工艺，其余来自设计**（平衡流水线、时钟门控等）。
- HBM 带宽高 1.3×；bisection bandwidth 依 slice 尺寸高 **2–4×**；另有 TPU v3 没有的 128 MB CMEM。

**相对 TPU v3 的生产应用表现**

- 图 12（同 slice 尺寸）：多数应用 **1.5×–2.0×**；**DLRM0 3.0–3.5×、DLRM1 2.8×（512 颗）**，原因是 TPU v4 的 SC 数量翻倍且时钟更快；**RNN1 3.3×** 是意外项——它权重小、batch 小，**显著受益于 CMEM 相对 HBM 的带宽**。
- 图 13：**per-chip 性能与封装级 perf/W 相对 TPU v3 的几何平均为 2.1× 与 2.7×**；关闭 CMEM 后整体性能损失 **1.2×**，而 **RNN1 损失 2×**。
- §7.5 另给一组口径：把 CMEM 打开使片上 SRAM 从 32 MB 增加到 160 MB，**性能提升 1.18×、perf/W 提升 1.24×**。注意正文中 1.2×（§5、§7.5 口径）与图 13 几何平均柱（带/不带 CMEM）不是同一口径。

**SparseCore 的端到端收益（图 9）**

内部生产推荐模型 DLRM0，128 颗规模：

- **TPU v3 比 CPU 快 9.8×；TPU v4 比 TPU v3 快 3.1×、比 CPU 快 30.1×**。
- 把 embedding 放到 CPU 内存（不使用 SC）时，**TPU v4 的性能掉 5×–7×**，瓶颈在 CPU 内存带宽。

**生产负载的扩展性（图 11）**

8 个生产负载在 TPU v4 上的扩展情况：**一半负载（CNN0、RNN0、RNN1、BERT1）能良好扩展到 3K 颗芯片**；其余负载的扩展上限已被生产团队识别并正在做方案，但尚未实现，因此无法测到完整 3K 规模。

**MLPerf 对比（Table 5、图 14–15）**

Table 5 的对照芯片特性：

| | NVIDIA A100 | Graphcore MK2 IPU |
| --- | ---: | ---: |
| 生产部署 | 2020 | 2021 |
| 峰值 TFLOPS | 312（bf16）/ 624（i8） | 250（bf16） |
| 时钟 base/boost | 1095 / 1410 MHz | 1850 MHz |
| 工艺 / die 面积 | 7 nm / 826 mm² | 7 nm / 832 mm² |
| 晶体管数 | 54 B | 59 B |
| TDP | 400 W | 300 W |
| 片间互连 | 12 links @ 25 GB/s | 3 links @ 64 GB/s |
| MLPerf 2.0 最大规模 | 4216 颗 | 256 颗 |
| Processor Style | 单指令多线程 | 多指令多数据 |
| 处理器 / 芯片 | 108 | 1472 |
| 线程 / 核 | 32 | 6 |
| 片上内存 | 40 MiB | **900 MiB** |
| 寄存器堆 | **27 MiB** | 1.40 MiB |
| HBM 容量 / 带宽 | 80 GiB / 2039 GB/s | **0** |

结果：

- **同规模下，TPU v4 对 A100：BERT 1.15×、ResNet 1.67×**；对 IPU Bow：**BERT 约 4.3×、ResNet 约 4.5×**。（注：摘要把这些写成 "1.2×–1.7×"，是把 1.15× 向上取整。）
- 峰值 FLOPS 并不预测实际性能（§7.1）：TPU v4 对 IPU Bow 在同等规模下快 4.3–4.5×，而峰值 FLOPS 只领先 **1.10×**；A100 的峰值 FLOPS 是 TPU v4 的 **1.13×**，而 TPU v4 在同芯片数下快 1.15–1.67×。
- **功耗（Table 6，DSA + HBM，64 芯片）**：BERT 上 A100 380 W vs TPU v4 197 W（比值 1.93）；ResNet 上 273 W vs 206 W（比值 1.33）。即 **A100 平均多用 1.3×–1.9× 的电**。
- §7.5 补充一条对 A100 的解释：TPU v4 的片上 SRAM 是 A100 的 4×（160 MB vs 40 MB），可让 DRAM 传输以更大块进行从而更省能；此外 GPU 的多线程支持带来 **100× 大的寄存器堆（27 MiB vs 0.25 MiB）**，而功耗大致随内存容量的平方根增长；TPU v4 的 128×128 MXU 每个输入重用 128 次，而 A100 的 4×4 FP16 阵列只重用 4 次，导致更多片上 SRAM 访问。

**规模化的整体收益**

- 摘要口径：TPU v4 **比 TPU v3 快 2.1×、perf/Watt 高 2.7×**；超算规模大 4×（4096 颗）因而**整体快近 10×**；**LLM 训练平均达到峰值 FLOPS/s 的约 60%**。
- 具体实例：**540B 参数的 PaLM 在 TPU v4 超算上训练 50 天，稳定在峰值浮点性能的 57.8%**（§9）。

### 消融与敏感性实验要点

- **CMEM 开关**（图 13）：整体 1.2×，RNN1 达 2×；L2 类小权重、小 batch 的负载受益最大。
- **bisection bandwidth**（图 8）：TPU v3/v4 比值 2–4×，embedding 加速 1.1–2.0×；1024 颗后 SC 开销主导。
- **拓扑搜索**（Table 3）：换几何从 4×8×16 到 8×8×8 使 LLM 吞吐提升 **2.3×**；对 GPT-3 预训练专家配置再提升 **1.2×**。
- **扭曲 torus**（图 6）：all-to-all 吞吐 1.63×（4×4×8）与 1.31×（4×8×8）。
- **如果改用 InfiniBand**（§7.3）：按 NVIDIA 的指引用三层 fat-tree 组混合 IB/ICI 网络，替换 48 台 128 端口 OCS 需要 **568 台 IB 交换机**（每台约 1.5–1.8 万美元）外加 4096 个 NIC；内部事件驱动模拟器（忽略 CPU 协议处理）给出**优化后的 all-reduce 慢 1.8×–2.4×、all-to-all 慢 1.2×–2.4×**，折算到整个 DNN 可能只慢约 10%，但会失去 OCS 带来的可用性、规模、利用率、模块化、功耗效率与可部署性。

**OCS 拓扑使用分布（Table 2，2022 年 11 月某日）**

- **29% 的 slice 小于 4³**（只能用 2D mesh），其中 1×2×2（4 颗）占 6.7%、2×4×4（32 颗）占 8.9%、4×4×4（64 颗）占 13.9%。
- 剩余 71% 中，只有形如 n×n×2n 或 n×2n×2n 的可以扭曲，占 **33%**（即 71% 的 48%）；**实际使用扭曲 torus 的占 28%**（可扭曲中的 86%）。换言之，**4³ 或更大的拓扑中有 40% 使用扭曲 torus**。
- 最大规模的使用：8×16×16_T（2K）1.4%、12×16×16（3K）占 **5.7%**（3K 是最大的一类）。

**工作负载类型的快速迁移（Table 1）**

| 模型类型 | TPU v1 7/2016（推理） | TPU v3 4/2019（训练+推理） | TPU v4 Lite 2/2020（推理） | TPU v4 10/2022（训练） |
| --- | ---: | ---: | ---: | ---: |
| MLP/DLRM | 61% | 27% | 25% | 24% |
| RNN | 29% | 21% | 29% | **2%** |
| CNN | 5% | 24% | 18% | 12% |
| Transformer | — | 21% | 28%（BERT 28%） | **57%**（BERT 26% / LLM 31%） |

Transformer 从 2019 年的 21% 升到 2022 年的 57%，其中 LLM 独占 31%；RNN 从 29% 跌到 2%。图 17 另给出 DLRM0 从 2017 到 2022 的变化：**权重增长 4.2×、embedding 增长 3.8×**，五年间约每 6 周发布一个新版本（共 43 个），并且它跑遍了全部五代 TPU 产品。

## 局限性与未来方向

- **本文为厂商自评，多个关键对照是自测而非第三方认证**：Table 6 的功耗测量中，A100 侧是论文自己在 Azure VM 上重跑 NVIDIA 的 MLPerf 2.0 提交代码得到，TPU v4 侧是在 Google 数据中心跑；论文坦承 "MLPerf 3.0 may add power measurements to performance in the October 2023 round"。§7.3 的 IB 对比来自**内部事件驱动模拟器**，且论文明确写了"忽略 CPU 上的协议处理，而这部分可能很显著"。
- **H100 未被对照，作者给出了理由但也承认了口径问题**：§7.4 说明 2022 年做研究、2023 年提交 camera-ready 时，700 W 的 H100 在 AWS/Azure/Google Cloud 都不可得，并认为"H100 的合适对照是与 TPU v4 在同期同工艺（如 2023 年、4 nm）广泛部署的后续芯片"。这个理由成立，但意味着**本文的对外对比停留在 2020–2021 一代的 A100/IPU 上**。
- **摘要与正文的数字口径不完全一致**：正文给出对 A100 的 BERT 加速是 **1.15×**，摘要写作 "1.2x–1.7x"；CMEM 的整体增益在 §5/§7.5 是 1.2×/1.18×，而图 13 的两组几何平均柱是 1.9 与 2.1，两者不是同一口径。引用时应落到具体图表而不是摘要。
- **生产负载的扩展性证据不完整**：图 11 中 8 个负载只有一半能良好扩展到 3K 颗，其余"解决方案尚未实现，因此无法测到完整 3K 规模"。同时 LLM 被排除在 TPU v3 的对比之外（占 31% 负载），理由是 TPU v3 的 2D 固定拓扑阻碍了 LLM 需要的模型切分、且芯片容量不足以在合理时间内收敛。**这意味着最重的一类负载没有跨代对照。**
- **SparseCore 的收益高度依赖模型与规模**：100× 量级的 embedding 加速（RNN1 2.22× 类）来自"bisection bandwidth 正比于端到端 embedding 性能"这一链条，而该链条在 1024 颗之后被 SC 自身开销削弱。图 9 的 30.1× vs CPU 用的是**内部生产模型 DLRM0**，与 MLPerf 的 DLRM 不同（论文在第 9 页脚注明确 "The MLPerf DLRM is not representative of production DLRMs"）。
- **OCS 的抽象收益缺少独立验证**：可用性（图 4 的 goodput 曲线）与调度收益是论文自己构造的模型；扭曲 torus 的 1.63×/1.31× 是稳态大批量 all-to-all 微基准，不是端到端训练时间。
- **未来方向**：论文未设独立 future work 章节。从 §2 与 §9 看，后续重点是**把 OCS 的拓扑灵活性与并行策略、DNN 结构进一步协同**（PA-NAS 的方向），以及 OCS 在训练与推理之外的更广部署。

## 个人点评

- **这是我所知的第一篇把光交换当作量产超算核心部件的论文**，而它的论证方式值得学习：作者没有只讲"光交换更快"，而是把 OCS 的收益拆成八条（可扩展性、可用性、模块化、性能、功耗、调度、部署速度、安全），其中**可用性、调度与部署速度这三条与带宽完全无关**。图 4 的 goodput 反直觉分析尤其清楚：从 4K 芯片里切 3 个 slice，每个只有 75% goodput，因为 3 个 slice 占了 3/4 的芯片；切一个 3K 的 slice 反而有 75% 的 goodput。这种"备用芯片就是 goodput"的算术，把"可重构"与"可用性"直接联系了起来。
- **成本账是全文最有说服力的一页**。OCS 加整个光 fabric 不到系统成本 5%、功耗 3%，而 §7.3 给出改用 IB 的代价：替换 48 台 128 端口 OCS 需要 568 台 IB 交换机（每台 1.5–1.8 万美元）加 4096 个 NIC。作者还补了一句很关键的物理对比：**OCS 只是用微镜反射光源编码的光，是无源的，而 IB 交换机要做主动包处理**，两者功耗不在一个量级。这条把"光交换省电"从口号变成了机制解释。
- **SparseCore 是本文另一个值得记录的设计**。它只占约 5% 面积与功耗，却把 embedding 加速 5–7×；机制上不是靠更快的算力，而是靠**把超算的全部 HBM 与 ICI 组织成一个 128 TiB 的平坦可寻址空间**，让 embedding 表可以放在任何地方。论文还给了一个干净的对照：把 embedding 放回 CPU 内存，TPU v4 性能立刻掉 5–7×。这说明"加速 embedding"的关键不是加计算单元，而是消除 CPU DRAM 接口这个 Amdahl 瓶颈。SC 的 3D torus bisection bandwidth 按 $N^{2/3}$ 而非 $N^{1/2}$ 缩放，也是把它和网络拓扑绑在一起设计的证据。
- **两处需要注意**。第一，Table 6 的功耗是论文自测（A100 侧在 Azure VM 上重跑 NVIDIA 提交），且论文承认当时 MLPerf 还没有功耗项；1.3×–1.9× 这个结论方向可信、精度有限。第二，摘要与正文的数字口径有偏差（1.15× 写成 1.2×；CMEM 增益的 1.2× 与图 13 的 1.9/2.1 不是同一口径），引用时应以图表为准。此外 §7.3 的 IB 对比是模拟结果且明确忽略了 CPU 协议处理开销——文中用"整体 DNN 可能只慢约 10%"来软化这个结论，但同时也承认失去的可用性/调度/部署收益无法用性能指标衡量。
- **一个数字值得单独记住**：PaLM 540B 在 TPU v4 上训练 50 天，稳定在峰值浮点性能的 **57.8%**。这是"大规模超算能否被长期喂饱"的一个少见的公开实测值，比任何峰值 TFLOPS 都更能说明 OCS + 3D torus + ICI 这套组合的实际价值。

## 工程化三问总结

### 1. 它解决了什么瓶颈？

- **应用场景与核心瓶颈**：4K 芯片规模的 ML 超算。三个具体瓶颈：(1) 规模带来的可靠性问题——1K 台 CPU 主机在 0.1%–1.0% 时间不可用，而 DNN 训练要求"所有部件必须工作"；(2) DLRM 的 embedding 查找是低算术强度的小粒度 gather/scatter，与 TensorCore 的稠密优化不匹配；(3) embedding 引入 all-to-all 通信，挤压 bisection bandwidth，而 all-reduce 本来就难映射到 2D torus。
- **现有方法为何不足**：光链路比电链路贵 10× 以上，且 TPU v3 的部分 2D torus 环绕链路已被迫用光互连，规模再 4× 只会更多；静态拓扑无法绕开故障（TPU v3 要等全部 1024 颗芯片与线缆就绪才可用）；2D torus 的 all-to-all 带宽按 $N^{1/2}$ 缩放而 3D torus 按 $N^{2/3}$；embedding 放 TC 上因粒度小而次优，放主机 CPU 上则撞上 4:1 的 CPU DRAM 接口瓶颈（实测掉 5–7×）。
- **论文证据的分层**：
  - **机制与结构级（可引用）**：Palomar OCS 的 136×136 端口、MEMS 毫秒级切换、环行器减半端口；4³ 块与 48 台 OCS 的拓扑算术（6×16÷2 = 48；48 对 × 64 块 = 4096 颗）；Table 4/5 的芯片特性；SC 的结构（16 个 compute tile、每 tile 绑一个 HBM channel、2.5 MiB Spmem、5 个 Cross-Channel Unit、CISC 式变长指令）。
  - **实测性能**：图 12/13 的同 slice 加速（多数 1.5–2.0×，DLRM0 3.0–3.5×，RNN1 3.3×，几何平均 2.1× 性能与 2.7× perf/W）；图 9 的 DLRM0 加速（TPU v4 相对 TPU v3 3.1×、相对 CPU 30.1×；embedding 放 CPU 内存掉 5–7×）；图 6 的扭曲 torus all-to-all 1.63×/1.31×；Table 6 的功耗（A100 多用 1.3–1.9×）；PaLM 57.8% 峰值利用率的 50 天实测。
  - **需要降级的**：Table 6 为论文自测（第三方口径未成熟）；§7.3 的 IB 对比为内部模拟器结果且忽略 CPU 协议开销；摘要与正文存在 1.15× vs "1.2×"、CMEM 1.2× vs 图 13 几何平均柱的口径差异；图 4 的 goodput 模型与图 11 的扩展性（半数负载未测到 3K）均为论文自建。

### 2. 用了什么结构或训练方法？

- **整体结构**：Palomar OCS（3D MEMS 反射镜，136×136 端口，毫秒级切换，环行器双向传输）连接 48 个 4³ 块的 96 条光链路，构成 4096 颗芯片的 3D torus 超算。每颗 TPU v4 含 2 个 TensorCore，每 TC 含 4 个 128×128 MXU + 128-lane VPU（每 lane 16 ALU）+ 16 MiB VMEM，两 TC 共享 128 MiB CMEM；板内 4 条 ICI 接成 2×2 mesh，16 条外部 ICI 接其它 tray。
- **关键结构与机制**：
  1. **OCS 作可编程插线板**：相对两侧链路接同一台 OCS 以提供 torus 环绕；拓扑在毫秒级内改变，主要是改路由表。
  2. **扭曲 torus（k×k×2k）**：改善最坏情况延迟，提升 all-to-all 吞吐 1.31–1.63×。
  3. **SparseCore（sea-of-cores）**：16 个 compute tile，各绑一个 HBM channel，含 Fetch Unit、8-wide SIMD scVPU、Flush Unit，加 5 个 Cross-Channel Unit；把全超算 HBM + ICI 组合成 128 TiB 平坦可寻址空间。
  4. **三种 embedding 分区策略**：列切分（按表宽）、行切分（按词表大小）、表切分（不同表放不同芯片）；小表用数据并行复制更好。
  5. **PA-NAS**：在稀疏层（SC）与稠密层（TC）之间转移计算负载，使 DLRM0 的 SC-TC 负载接近完美均衡，端到端提升 >10%；也用搜索选择拓扑与并行超参（LLM 提升 2.3×、GPT-3 预训练提升 1.2×）。
- **训练与量化策略**：本文不涉及量化。训练范式层面值得注意的是：**每颗芯片维持数万个在途访存请求**（对本地与远端内存），远程内存**只通过异步 DMA 写**访问，逻辑共享地址空间由软件显式控制访问与数据搬移。

### 3. 对芯片架构和 RTL 有什么启发？

- **芯片架构**：五条。第一，**用可重构光互连替代包交换网络，能在同等成本/功耗下同时买到拓扑灵活性与可用性**：OCS + 光 fabric <5% 成本、<3% 功耗；而替换方案的代价可以量化（4096 颗需要 568 台 IB 交换机 + 4096 个 NIC；all-reduce 慢 1.8–2.4×、all-to-all 慢 1.2–2.4×）。第二，**拓扑从固定改为每任务可配，会把"并行策略选择"从软件调参变成硬件能力**：Table 3 显示仅换几何就使 LLM 吞吐提升 2.3×，这类收益不需要任何芯片改动，纯靠互连重配。第三，**互连的 bisection bandwidth 缩放规律决定是否值得为特定算子加硬件**：2D torus 的 $N^{1/2}$ 对 3D torus 的 $N^{2/3}$ 直接决定了 SparseCore 的收益上限，也解释了为什么它在 1024 颗后被自身开销压住。第四，**对低算术强度算子的加速不该靠加算力，而该靠消除瓶颈接口**：SC 只花约 5% 面积/功耗，把 embedding 加速 5–7×，靠的是把全机的 HBM 与 ICI 组织成平坦地址空间；一旦 embedding 回到 CPU 内存，性能立刻掉 5–7×。第五，**片上 SRAM 容量既是性能参数也是功耗参数**：TPU v4 的 160 MB 片上 SRAM 是 A100 的 4×，让 DRAM 传输得以成块进行；论文把 A100 多耗的 30%–90% 电部分归因于 27 MiB 寄存器堆（多线程支持导致，是 TPU v4 的 100×）与 4×4 FP16 阵列的输入重用仅 4 次（TPU v4 的 128×128 重用 128 次）。
- **RTL**：可落到实现层的项目包括：OCS 的路由表配置接口与 slice 建立/拆除流程；ICI 收发（6 links @ 50 GB/s）与板内 2×2 mesh、板上 3D torus 的路由逻辑；扭曲 torus 的地址映射与路由表生成；SparseCore 的 16 个 tile（Fetch Unit、8-wide SIMD scVPU、Flush Unit）、5 个 Cross-Channel Unit、以及 Spmem 16 bank 的跨通道访存；变长 CISC 式指令的译码与数据相关的执行时长控制；数万个在途访存的请求队列管理（本地与远端）；远端内存**只写**的异步 DMA 通路；以及嵌入 128 MB CMEM 的地址空间与访问仲裁。需要强调：**远程内存仅支持异步 DMA 写**（不支持远程读/load-store），这简化了 RTL 但把一致性责任完全推给软件。论文未给出任何 RTL 面积、时序或功耗分解。
- **推断边界**：第 1 问的机制与实测数据来自第 2–6 节的正文与图表（可核算），系统级结论按"厂商自测 + 部分模拟"降级；第 2 问的结构描述全部来自论文正文与图 1/2/3/5/7。第 3 问的芯片架构与 RTL 内容为工程推断。论文中未提供的量值——OCS 的切换时延对训练吞吐的实际影响、SC 的具体指令集与面积分解、CMEM 的访问仲裁细节、ICI 路由实现、以及 128 TiB 平坦地址空间的页表/地址翻译机制——均 `TBD`。
