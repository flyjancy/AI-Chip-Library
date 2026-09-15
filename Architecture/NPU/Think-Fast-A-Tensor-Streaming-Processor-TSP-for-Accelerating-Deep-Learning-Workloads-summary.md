# 深度分析：《Think Fast: A Tensor Streaming Processor (TSP) for Accelerating Deep Learning Workloads》

## 基本信息

- **标题**：Think Fast: A Tensor Streaming Processor (TSP) for Accelerating Deep Learning Workloads
- **文档类型**：论文（ISCA 2020，14 页）
- **作者**：Dennis Abts、Jonathan Ross、Jonathan Sparling、Mark Wong-VanHaren、Max Baker、Tom Hawkins、Andrew Bell、John Thompson、Temesghen Kahsai、Garrin Kimmell、Jennifer Hwang、Rebekah Leslie-Hurd、Michael Bye、E.R. Creswick、Matthew Boyd、Mahitha Venigalla、Evan Laforge、Jon Purdy、Purushotham Kamath、Dinesh Maheshwari、Michael Beidler、Geert Rosseel、Omar Ahmad、Gleb Gagarin、Richard Czekalski、Ashay Rane、Sahil Parmar、Jeff Werner、Jim Sproch、Adrian Macias、Brian Kurtz
- **机构**：Groq, Inc. · Mountain View, California
- **发表 venue**：2020 ACM/IEEE 47th Annual International Symposium on Computer Architecture (ISCA)，pp. 145–157
- **年份**：2020
- **链接**：DOI 10.1109/ISCA45697.2020.00021

> 这是 Groq TSP 的**首篇架构论文**，讲的是**单芯片微架构**；本仓库另有后续的《A Software-defined Tensor Streaming Multiprocessor for Large-scale Machine Learning》（ISCA'22），讲的是**多芯片 Dragonfly 组网**。两篇是同一架构的前后半部分。

## 一句话总结

> TSP 把传统 2D mesh of cores 重组成「功能切片 + 交错流寄存器」（MEM / VXM / MXM / SXM / C2C / ICU 各占垂直一列），**取消所有仲裁器与 cache 等反应式部件**，只靠编译期静态调度的 producer-consumer 流在片内传递 320 字节向量；结果为 14 nm、25×29 mm、26.8B 晶体管、900 MHz 标称的芯片上做到 820 TeraOps/s 峰值、**>1 TeraOp/s/mm²**，ResNet50 在 batch=1 下 20.4K IPS（单图延迟 49 µs），代价是没有 cache、没有仲裁器、全靠编译器把时空关系算准。

## 研究动机与问题定义

- **要解决的核心问题**：硬件复杂度使运行时的 stall 难以推断。cache、分支预测、预取器能大幅提升平均性能，但**无法界定最坏情况性能**（第 1 页）。
- **现有方法的不足**：传统 CPU/GPU 依赖内存层次与动态仲裁，这些"反应式"部件在数据通路中引入非确定性——cache 层次把内存事务伪装成顺序一致，代价是不可预测性。
- **切入角度**：两条基本观察。(1) ML 工作负载有丰富的数据并行，可以天然映射到硬件里的张量；(2) **简单且确定性的处理器 + producer-consumer 流编程模型**，使硬件行为可以被精确推理与控制，从而取得好的性能与能效。具体做法是**消除硬件中所有反应式部件（仲裁器、cache）**，把调度责任整体移交编译器。

## 核心方法

### 微架构：功能切片（functional slicing）

传统 manycore 芯片是 2D mesh of cores（图 1a）；TSP 将其重组成**功能切片**（图 1b）——每种功能单元占据一列（slice），slab 之间通过流寄存器通信。图 2 的结构自上而下是：

- **20 个 tile 组成一个 slice**，每个 tile 对 16 个元素做 16-way SIMD。指令**逐周期错开（staggered）执行**：在时刻 $t$ 发到 slice 最底部的 tile（对应第一个 16 元素 superlane），下一周期传播到北面一个 tile，**指令流过 20 个 tile 需 20 个周期**（图 6）。
- **minVL = 16**（一个 superlane），**maxVL = 320**（全部 20 个 superlane）。未使用的 superlane 可通过 `Config` 指令断电，实现 energy-proportional 设计。
- 五个功能区域（Table I 给出完整指令清单）：

| 单元 | 结构 | 关键指令 |
| --- | --- | --- |
| **ICU**（指令控制） | — | `NOP N`、`Ifetch`、`Sync`、`Notify`、`Config`、`Repeat n, d` |
| **MEM**（内存） | 东/西半球各 **44 个并行 SRAM slice**，每个 slice 对 16 字节字做 13-bit 物理寻址，**每字节映射一个 lane**，合计 **220 MiB** 片上 SRAM | `Read`、`Write`、`Gather`、`Scatter` |
| **VXM**（向量） | 每 lane 一个 **4×4 ALU mesh**（即每 lane 16 个 ALU，全芯片 **5,120 个向量 ALU**），支持 32-bit 定点与浮点 | unary/binary op、类型转换、`ReLU`、`TanH`、`Exp`、`RSqrt` |
| **MXM**（矩阵） | **4 个独立的 320×320 二维 MACC 阵列**，支持 int8 或 fp16 | `LW`（load weights）、`IW`（install weights）、`ABC`、`ACC`（int32 或 fp32 累加） |
| **SXM**（交换） | 北/南 lane shifter、distributor（重映射 superlane 内的 16 个 lane）、置换与转置 | `Shift up/down N`、`Permute map`、`Distribute map`、`Rotate stream`（n=3 或 4）、`Transpose sg16` |
| **C2C**（片间） | 16 × 4 条链路，每条 30 Gbps | `Deskew`、`Send`（发 320 B 向量）、`Receive` |

- **片外 pin 带宽**：$16 \times 4 \times 30\ \text{Gb/s} \times 2\ \text{方向} = 3.84\ \text{Tb/s}$，可灵活划分以支持高基数互连（这就是 ISCA'22 那篇 Dragonfly 组网的物理基础）。PCIe Gen4 主机接口也在此模块。

### 带宽账（第 3 页，式 1–2）

- **流寄存器带宽**：$B = 2\ \text{方向} \times 32\ \frac{\text{bytes}}{\text{lane}} \times 320\ \text{lanes} = \mathbf{20\ TiB/s}$（读操作数 + 写结果合计）。
- **SRAM 带宽**：$M = 2\ \text{半球} \times 44\ \frac{\text{slices}}{\text{hem}} \times 2\ \frac{\text{banks}}{\text{slice}} \times 320\ \frac{\text{bytes}}{\text{cycle}} = \mathbf{55\ TiB/s}$（每半球 27.5 TiB/s）。
- **指令取指**最大消耗 $144 \times 16 = \mathbf{2.25\ TiB/s}$。
- 于是每半球 27.5 TiB/s 的 SRAM 带宽中，2.25 TiB/s 用于取指，剩余 25 TiB/s 服务 20 TiB/s 的流寄存器带宽——**留有余量**。
- **装权重速度**：MEM 可在**不到 40 个周期**内读取 409,600 个权重并装入 4 个 320×320 阵列（含 SRAM 与片上网络传输延迟）。这要求 MEM slice 为 320 个并行 lane 每 lane 提供 32 个 1 字节流操作数，即 **10 TiB/s 的操作数流带宽进入 MXM**。

### 编程模型与调度（第 3–4 页）

- **Producer-consumer 流模型**：向量从 SRAM 读出后进入流寄存器（stream register），即成为"流"，沿指定方向（东向/西向）流动。流寄存器按空间位置编号；在时刻 $t_i$，位于 $x_1$ 的 slice 读到的流 $s_1$ 与 $x_0$、$x_2$ 处不同；下一周期 $t_{i+1}$，该值要么传播到 $x_2$，要么被 $x_1$ 产生的结果覆盖。**流不断流过芯片，是各切片的通信手段**。
- **切片可以逻辑"掉头"**：每个功能切片可以选择其结果流的方向，因此可以把东→西翻转为西→东，从而把多个切片串成更复杂的操作，无需把中间结果写回内存：
  $$F(x,y,z) = \text{MEM}(x) \to \text{SXM}(y) \to \text{MXM}(z)$$
  这就是作者说的"dataflow locality"。
- **指令执行时间是可解析的**（式 4）：
  $$T = N + d_{func} + \delta(j,i)$$
  其中 $N$ 是切片中的 tile 数（20），$d_{func}$ 是功能延迟，$\delta(j,i)$ 是流寄存器 $i$ 到 $j$ 的传输距离（周期）。**ISA 把这条时间信息暴露给编译器**，使编译器后端能精确追踪任意流在片上的位置与使用时刻——这就是"software-defined hardware"。
- **编译器解的是二维调度**：指令与数据在**时间与空间**两个维度上同时调度。
- **同步**：`Sync` 把切片停在指令派发队列头部等待屏障通知，`Notify` 释放挂起的屏障操作。**整个程序只在开始时同步一次**，之后靠静态调度。

### 三大设计决策

1. **显式管理内存**（第 5 页）：为最大化流并发，编译器把一个张量的并发流操作数分配到**不同的 MEM slice**。这要求把内存并发度在 ISA 里暴露出来，让编译器显式调度每个 MEM slice 内的 bank。同一 slice 的两个 bank 可同时读操作数、写结果（利用伪双端口 SRAM）。
2. **没有 cache，没有仲裁器**：传统 CPU 靠层次结构隐式搬数据，TSP 改为提供"薄薄一层内存管理"，按操作逐条标识内存并发度。片内网络也不同——**TSP 不跟踪流的源与目的切片**，流只是向东或向西传播，直到掉出芯片边缘或被某个功能切片覆盖（对比常规 NoC 的包路由 + 仲裁 + 出口调度 + 流控）。
3. **ECC 只在生产端计算**（第 3 页）：128-bit 内存字配 **9-bit ECC（共 137 bit）**，SECDED（单比特纠错 + 双比特检错）。因为内存高度 bank 化且被复制，若在每个消费端都做 XOR 树会重复计算；利用 producer-consumer 特性，**校验位只在 producer 端生成**并随数据同行，消费者在操作前检查，于是 ECC 同时覆盖了 SRAM 软错误与流寄存器上的数据通路软错误。SEU 被自动纠正并记入 CSR 供错误处理程序查询——作者把这类"可自动纠正的瞬态错误"当作**识别大规模系统里边缘芯片的早期信号**。

## 实验与结果

### 实验设置

- **硅状态**：**A0 硅 2019 年 7 月回片，距离 ISCA 论文截止只有 5 个月**。作者明确说明 ResNet50 的这次实现是**编译器验证用的 proof-point 与参考实现**，同期从零建起了新架构、编译器、汇编器与调试/可视化工具链。
- **芯片**：14 nm ASIC，**25×29 mm**，**26.8B 晶体管**，标称 900 MHz（正文分析按 **1 GHz** 核心时钟），PCIe CEM 形态。
- **模型**：ResNet50 v2 图像分类；量化策略为**后训练、逐层对称 int8**（卷积与矩阵乘），MXM 收 int8/fp16 并累加到 int32/fp32，VXM 的 fp32 能力按 4 个 MXM 平面的输出速率流式处理。

### 主要结果

**ResNet50 性能**

- **20.4K IPS，batch size = 1**（每个图像样本是一次独立查询），单次推理 <43 µs。
- **单图延迟 49 µs**。论文声称这是相对 Google TPU v3 **大 batch 推理**的 **2.5× 加速**；相对 Intel/Habana Goya（batch 1 推理 240 µs）**端到端延迟降低近 5×**。
- **ResNet101 与 ResNet152 的预测值**：14.3K IPS 与 10.7K IPS。论文强调因为性能是确定性的，这两个数字可以**精确到周期地推算**（ResNet101/152 与 ResNet50 结构相同，只是重复的层组更多）。

**计算密度与"每个晶体管能产出多少操作"**

- 峰值 **820 TeraOps/s**（1 GHz），换算 **30K deep learning Ops/sec/transistor**（26.8B 晶体管）。
- 对照 NVIDIA Volta 100：130 TeraFlops 混合精度、21.1B 晶体管、815 mm²、12 nm → **6.2K Ops/sec/transistor**。
- 摘要口径：芯片的计算密度 **>1 TeraOp/s per square mm**（820/725 ≈ 1.13）。
- 结论段指出相对领先 GPU 有 **5× 的计算密度**。

**优化过程**

- 第一版 ResNet50 把操作铺满全芯片（典型流水是 `Read → Conv2D → Requantize → ReLU → Write`），但流水填充/排空时产生延迟气泡，且初始内存分配让前一条流水排空时无法启动下一条（MEM slice 争用）。
- 通过调整输入输出张量的内存分配模式、把数据分散到多个 slice、并精心编排 slice 内的 bank 交织，做到**在前一条流水写完结果之前就能读走它的输出**。
- 这些优化**减少约 5,500 个周期**，把性能推到当前的 20.4K IPS。

**量化与精度**

- 逐层对称 int8 的后训练量化 + 在矩阵乘与卷积之间保持更高精度（VXM 的 fp32 通路），使**量化损失为 0.5%**（对比逐操作量化）。
- 论文指出流式架构有能力做 axis-based 非对称量化，留给后续版本。

**模型精度与 MXM 容量的错配**

- MXM 容量是 320×320，而 ResNet50 各层通道深度是 2 的幂（256×256 等），**错配导致 MXM 利用率不足**（256×256 的权重需要分多次 pass）。
- 作者反向利用这一点：训练了一个通道深度加大以充分利用 320 元素的替代 ResNet50。标准版 fp32 ResNet50 训练到 **75.6% Top-1 / 92.8% Top-5**，加大通道的版本达到 **77.2% Top-1 / 93.6% Top-5**——**在相同计算成本与延迟下提升精度**。作者把这条总结为"如何利用 maxVL=320 带来的额外模型容量"。

**Roofline 与功耗**

- 图 9 的 roofline 给出两种运行区间：**斜线区是片上内存带宽受限**（把权重装入 MXM 数组时的区间），**roofline 峰值是算术单元饱和区**（算术受限）。论文给出一个具体事实：MEM slice 可读出 409,600 个权重并在不到 40 周期内装入四个 320×320 阵列。
- 图 10 给出 ResNet50 逐层执行时的功耗曲线，**尖峰对应同时执行 4 个 conv2d 且算术吞吐饱和的周期**。

## 局限性与未来方向

- **硅状态与结果的定位需要限缩**：A0 硅回片距论文截止只有 5 个月，ResNet50 实现被作者自己称为 "proof-point and reference model for compiler validation"。20.4K IPS 与 49 µs 延迟是**刚点亮阶段的早期结果**，不是稳定产品化后的性能；论文没有给出多次运行的方差、也没有给出与同代加速器在同一软件成熟度下的对比。
- **对 TPU v3 的 2.5× 比较存在 batching 口径差异**：TSP 侧是 batch=1（每图独立查询），TPU v3 侧是 "large batch inference"。这两者在吞吐-延迟曲线上处在不同位置，2.5× 这个倍数混合了架构差异与批处理策略差异。相对而言，**49 µs vs Goya 240 µs 的 batch-1 延迟对比是同口径的**，是本文更干净的对外比较。
- **计算密度与 Ops/transistor 是峰值口径**：820 TeraOps/s 是峰值，>1 TeraOp/s/mm² 与 30K Ops/transistor 都由峰值除以面积/晶体管数得到。论文的 ResNet50 实测只覆盖一个模型，**没有给出达到峰值百分之多少**（图 9 的 roofline 只能定性看出有两类受限区间）。把它与 V100 的 6.2K Ops/transistor 直接对比时，也要注意分子分母的口径（TSP 用 int8 Ops，V100 用混合精度 Flops）。
- **没有 cache 的代价没有被直接量化**：论文列出了取消 cache/仲裁器的收益（确定性、可精确预测），但没有给出补偿性代价的数据——例如编译器必须把张量铺到 44 个 MEM slice 上这一约束对可编程性的限制有多大，或者当模型形状与 320 的匹配度差时损失多少（ResNet50 的 256 vs 320 错配只给了定性描述与"加大通道"的替代方案，没有给出错配场景的效率数字）。
- **可扩展性的证据在这一篇里是缺的**：C2C 只给出了 3.84 Tb/s 的 pin 带宽与 `Send`/`Receive` 两条指令；多芯片如何组网、如何保持确定性、如何做流控，都不在本文范围内（由 ISCA'22 那篇补齐）。
- **未来方向（论文自身给出）**：axis-based 非对称量化（流式架构已有能力承载，可降低量化精度损失）；以及结论段的一般性主张——随着 ASIC 工艺到约 25B 晶体管，晶体管花在两处（算术 ALU、以及存储与搬运数据），架构的价值用"每晶体管能产出多少深度学习操作"衡量。

## 个人点评

- **这篇论文的核心贡献是把"确定性"做成了一条从头贯穿到尾的设计约束，而不是一个优化目标**。为了让 `T = N + d_func + δ(j,i)` 这条公式成立，硬件必须依次放弃：cache（会引入命中/缺失的不确定性）、仲裁器（会引入队列延迟的不确定性）、常规 NoC 的路由与出口调度（TSP 的流只是"向东或向西传播直到掉出芯片边缘"）、以及流控。代价是编译器要解一个**时间×空间二维调度问题**，而且必须把内存并发度（44 个 slice × 2 bank）在 ISA 层面暴露出来。这套设计的极端程度在商用加速器里很少见——连 ECC 的 XOR 树都因为"内存高度 bank 化且复制"而只在 producer 端生成一次。这些细节串起来说明一件事：作者是真的把确定性当作第一性约束在设计。
- **带宽账写得非常清楚**，这一点值得单独肯定。式 1 与式 2 把流寄存器带宽（20 TiB/s）与 SRAM 带宽（55 TiB/s）、取指带宽（2.25 TiB/s）三者对齐，并明确剩余 25 TiB/s 服务 20 TiB/s 的流——**这是可以看到余量的预算表**，而不是只报一个峰值数字。装权重那一条（400K 权重在 <40 周期装入四个阵列，需要 MEM 每 lane 每周期 32 字节、即 10 TiB/s 进入 MXM）也是可以直接复算的。
- **两处需要按作者自己的诚实度来读**。第一，论文坦白写了 A0 硅回片距截止只有 5 个月，ResNet50 是"编译器验证的 proof-point"——这句话把 20.4K IPS 这个数字的定位讲清楚了，读者不应该把它当成成熟产品性能。第二，20.4K IPS vs TPU v3 的 2.5× 比的是不同 batching 口径；相比之下 49 µs vs Goya 240 µs 是干净的。这两点的区分很重要，因为摘要里的 "4× improvement compared to other modern GPUs and accelerators" 是笼统的。
- **一个意料之外但有意思的结果**：作者发现 256×256 的权重与 320×320 的 MXM 错配会拉低利用率，然后把这条"缺陷"反向变成了模型设计的自由度——加大通道以填满 320，得到 77.2% Top-1（vs 标准版 75.6%），**在相同计算成本与延迟下提高精度**。这是一个少见的"硬件约束反过来改善算法"的例子，与 TPUv4i 那篇里"用 maxVL 换精度"是同一类思路。
- **与 ISCA'22 那篇连起来读价值更大**：本篇给了单芯片的容量与带宽边界（220 MiB SRAM、20 TiB/s 流带宽、3.84 Tb/s 片外），下一篇给了多芯片如何在没有流控的前提下组网。两篇合起来才是完整的 Groq 方案；只看这一篇会误以为它是个孤立的单芯片设计。

## 工程化三问总结

### 1. 它解决了什么瓶颈？

- **应用场景与核心瓶颈**：深度学习推理/训练的算力供给。核心瓶颈被作者定义为**硬件复杂度导致运行时 stall 不可推断**——cache、分支预测、预取器能改善平均性能但无法界定最坏情况。TSP 的判断是：与其管理复杂度，不如消除它。
- **现有方法为何不足**：cache 层次在数据通路里引入"反应式代理"，把顺序一致性的假象建立在其上的同时带来不可预测性；常规 NoC 的包路由 + 仲裁 + 出口调度会引入冲突、非确定性与流控需求。
- **论文证据的分层**：
  - **可复算的结构与预算**：式 1/2 的带宽账（流寄存器 20 TiB/s、SRAM 55 TiB/s、取指 2.25 TiB/s、剩余 25 TiB/s）；220 MiB = 2 半球 × 44 slice × 16 字节字 × …；16 × 4 × 30 Gbps × 2 = 3.84 Tb/s；MXM 4 × 320×320、400K 权重 <40 周期装入；ECC 128 + 9 = 137 bit；5,120 个向量 ALU（320 lanes × 16 ALU）。
  - **A0 硅实测**：ResNet50 20.4K IPS（batch 1）、单次推理 <43 µs、单图延迟 49 µs；优化过程减少约 5,500 周期；量化损失 0.5%；标准 ResNet50 75.6%/92.8% 与加大通道版 77.2%/93.6%；图 10 的逐层功耗曲线。
  - **需要降级的**：820 TeraOps/s、>1 TeraOp/s/mm²、30K Ops/transistor 均为峰值/密度口径；对 TPU v3 的 2.5× 混了 batching 口径；ResNet101/152 的 14.3K/10.7K IPS 是**推算值**而非实测；作者自述这是 5 个月bring-up 期的 proof-point。

### 2. 用了什么结构或训练方法？

- **整体结构**：功能切片式微架构。20 个 tile 组成一个 slice（每个 tile 16-way SIMD），**6 类切片**（ICU、MEM、VXM、MXM、SXM、C2C）各占一列，切片之间用流寄存器连接；指令在 slice 内逐周期错开传播（20 个 tile 需 20 周期）。向量长度从 minVL=16（1 个 superlane）到 maxVL=320（20 个 superlane），未用 superlane 可断电。
- **关键结构与机制**：
  1. **MEM**：东西半球各 44 个并行 SRAM slice，每 slice 对 16 字节字做 13-bit 寻址、每字节映射一个 lane；支持伪双端口（一 bank 读操作数、另一 bank 写结果）。指令含 `Read`/`Write`/`Gather`/`Scatter`。
  2. **MXM**：4 个独立 320×320 MACC 阵列，int8/fp16 输入、int32/fp32 累加，320 元素和只做一次舍入；`LW`/`IW`/`ABC`/`ACC`。
  3. **VXM**：每 lane 4×4 ALU mesh（全芯片 5,120 ALU），含 `ReLU`、`TanH`、`Exp`、`RSqrt` 与类型转换；用于把 MXM 的 int32 输出 requantize 回 int8。
  4. **SXM**：北/南 lane shifter、distributor（重映射 superlane 内 16 lane，可零填充）、bijective permute、`Rotate`（n=3/4）、`Transpose sg16`（16 进 16 出、行列互换）。
  5. **C2C**：`Deskew`（管理准同步链路的 skew）、`Send`/`Receive` 320 B 向量。
- **数据流与调度**：producer-consumer 流模型，切片可逻辑掉头，操作可串成 $F(x,y,z) = \text{MEM}(x) \to \text{SXM}(y) \to \text{MXM}(z)$；编译器解时间×空间二维调度；程序只在开始时同步一次。
- **量化与数值策略**：后训练逐层对称 int8；int8/fp16 输入（MXM）、int32/fp32 累加、VXM 的 fp32 通路做中间精度保持；量化损失 0.5%。

### 3. 对芯片架构和 RTL 有什么启发？

- **芯片架构**：五条。第一，**"消除反应式部件"是一个可执行的架构选项，而不是口号**：没有 cache、没有仲裁器、没有虚通道、没有流控，片内网络退化成"流沿固定方向传播直到掉出边缘或被覆盖"。这样做的前提是编译器能精确追踪每条流的位置与时刻，而这要求 ISA 暴露功能延迟与流传输距离（式 4 的 $T = N + d_{func} + \delta(j,i)$）。任何想走这条路的设计都必须先回答"编译器能否拿到这些时间信息"。第二，**内存并发度必须上到 ISA**：TSP 把 44 个 MEM slice × 2 bank 的并发暴露给编译器显式调度，才使得前一条流水的输出能在写完之前被读走。这是一条与"cache 自动管理"完全相反的取向值得对照的设计。第三，**带宽预算要算完整**：TSP 把流寄存器带宽、SRAM 带宽、取指带宽并列列出并留出余量（25 TiB/s 对 20 TiB/s），这比只报峰值带宽更能说明设计是否有堵塞点。第四，**功能单元的尺寸选择会反向影响模型设计**：320×320 与 ResNet50 的 256×256 错配导致利用率下降，但反过来把通道加大到填满 320 后精度提升（77.2% vs 75.6%）。所以"功能单元尺寸"这个参数在架构权衡表里应当被视为一个可以与模型协商的自由度。第五，**向量长度的可伸缩性可以直接换成能效**：minVL=16 到 maxVL=320 的 16-lane 步进，配合逐 superlane 断电，得到 energy-proportional 的行为。
- **RTL**：可落到实现层的模块与约束包括：**20 级错开指令传播**（指令在 slice 内逐周期北移，需要与流寄存器时序对齐的流水控制）；**流寄存器阵列的编号与相邻转发逻辑**（数据每周期跳一个寄存器，切片可选择结果流方向，即"逻辑掉头"）；**SXM 的 lane shifter 与 distributor**（北/南移位常成对分配并做三路选择：北移、南移、不移位；distributor 在满带宽下重映射 16 lane 并可零填充）；**`Transpose sg16` 的 16 进 16 出转置网络**；**`Rotate`（n=3/4）产生 $n^2$ 个输出流**；**MEM 的 13-bit 地址解码与伪双端口 bank 控制**；**只在 producer 端生成的 SECDED ECC（128+9 = 137 bit）** 以及消费端的检查逻辑，纠错事件记入 CSR；**`NOP(N)` 这类可重复指令的周期精确延迟控制**与 `Repeat n, d`；**`Sync`/`Notify` 的片级屏障**；**`Deskew` 对准同步链路**。上述模块论文都只给了功能描述，**没有任何面积、功耗、时序或工作频率的实现数据**。
- **推断边界**：第 1 问的结构与预算来自第 2–5 页（可复算），性能来自 A0 硅的早期实测但按作者自述的"proof-point"定位与 batching 口径差异降级，峰值密度指标按峰值口径处理。第 2 问的结构与指令集来自第 2–4 页与 Table I。第 3 问的芯片架构与 RTL 内容为基于论文结构的工程推断；论文未提供的量值——各切片的面积/功耗分解、编译器调度失败时的回退机制、44 个 MEM slice 的 bank 冲突率、ECC 检查对关键路径的影响、以及模型形状与 320 不匹配时的效率损失——均 `TBD`。
