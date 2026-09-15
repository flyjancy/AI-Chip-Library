# 深度分析：《Ten Lessons From Three Generations Shaped Google's TPUv4i Industrial Product》

## 基本信息

- **标题**：Ten Lessons From Three Generations Shaped Google's TPUv4i Industrial Product
- **文档类型**：论文（ISCA 2021 Industry Track，14 页）
- **作者**：Norman P. Jouppi、Doe Hyun Yoon、Matthew Ashcraft、Mark Gottscho、Thomas B. Jablin、George Kurian、James Laudon、Sheng Li、Peter Ma、Xiaoyu Ma、Thomas Norrie、Nishant Patil、Sushma Prasad、Cliff Young、Zongwei Zhou、David Patterson
- **机构**：Google LLC
- **发表 venue**：2021 ACM/IEEE 48th Annual International Symposium on Computer Architecture (ISCA)
- **年份**：2021
- **链接**：DOI 10.1109/ISCA52012.2021.00010

## 一句话总结

> Google 用前三代 TPU 的生产经验提炼出十条教训，并用它们塑造了纯推理芯片 TPUv4i：因为**逻辑比导线与 SRAM 进步快得多**（45 nm→7 nm，逻辑能量改善 2.4–4.3×、SRAM 只有 1.3–2.4×、导线 <2×），所以推理芯片该把面积花在逻辑与片上内存上——结果是 4 个 MXU + 128 MB CMEM（占 die 面积 28%）、1.05 GHz、175 W 风冷、bf16+int8+fp32 全支持，perf/TDP 达到 TPUv3 的 **2.3×**，尽管晶体管只多 1.6×。

## 研究动机与问题定义

- **要解决的核心问题**：TPUv1 之后 Google 用一颗芯片同时兼顾训练与推理（TPUv2/v3），但设计资源有限，很难并行推进"训练优化芯片 + 推理优化芯片"。同时摩尔定律放缓、Dennard scaling 结束，使架构决策必须从"峰值性能"转向"性能/总拥有成本（TCO）"。
- **现有方法的不足**：
  - 以 benchmark 为目标的 perf/CapEx 优化会误导设计。论文举了 Alibaba HanGuang 800 的例子：时钟从 0.7 GHz 提升 1.5×，功耗涨 2.6×（第 4 页）；关闭 ECC 提速、删掉性能计数器等做法都能让 perf/mm² 好看，但损害 perf/TCO。
  - 近期的一些商用 ML 加速器忽略了文中若干教训（第 6 页 Related Work 逐条指出）。
- **切入角度**：把教训分成三组——前三条适用于任何 DSA（半导体技术不均衡进步、复用既有编译器优化、为 TCO 而非 CapEx 设计），中间三条关于 DNN DSA 本身（向后 ML 兼容、推理需要风冷、部分推理任务需要浮点），最后四条关于 DNN 应用（多租户、DNN 每年增长、工作负载随突破演化、SLO 是 P99 延迟而非 batch size）。

## 核心方法

### 十条教训（第 3–6 页）

**① 逻辑、导线、SRAM、DRAM 的进步速度不均衡**

论文把 Horowitz 的 45 nm 操作能耗表更新到 7 nm（Table 2，单位 pJ/operation；SRAM/DRAM 为每 64-bit 访问）：

| 操作 | 45 nm | 7 nm | 改善倍数 |
| --- | ---: | ---: | ---: |
| Int 8 加法 | 0.03 | 0.007 | 4.3 |
| Int 32 加法 | 0.1 | 0.03 | 3.3 |
| BFloat16 加法 | — | 0.11 | — |
| IEEE FP16 加法 | 0.4 | 0.16 | 2.5 |
| IEEE FP32 加法 | 0.9 | 0.38 | 2.4 |
| Int 8 乘法 | 0.2 | 0.07 | 2.9 |
| Int 32 乘法 | 3.1 | 1.48 | 2.1 |
| BFloat16 乘法 | — | 0.21 | — |
| IEEE FP16 乘法 | 1.1 | 0.34 | 3.2 |
| IEEE FP32 乘法 | 3.7 | 1.31 | 2.8 |
| SRAM 8 KB | 10 | 7.5 | 1.3 |
| SRAM 32 KB | 20 | 8.5 | 2.4 |
| SRAM 1 MB | 100 | 14 | 7.1 |
| **几何平均** | | | **2.6** |
| DRAM DDR3/4 | 1300 | 1300 | 1.0 |
| DRAM HBM2 | — | 250–450 | — |
| DRAM GDDR6 | — | 350–480 | — |

结论：**逻辑进步最快，所以逻辑相对"免费"**；SRAM 改善 1.3–2.4×（65 nm→7 nm 的 SRAM 容量密度比理想缩放差约 5×）；**DRAM 因封装创新（HBM）改善 6.3×**；**单位长度导线的能量改善不到 2×**——正是后者迫使 TPUv2/v3 从 TPUv1 的 1 个大核改成 2 个小核。HBM 也比 GDDR6/DDR 更节能，且每 GB/s 带宽的成本最低。

**② 复用既有的编译器优化**

TPU 依赖 2016 年启动的 XLA 编译器，GPU 从 2007 年起用 CUDA。MLPerf Training 0.5 → 0.7 的 20 个月里，**CUDA 让 GPU 提升 1.8×，XLA 让 TPU 提升 2.2×**（图 2），而 C 编译器每年只改善通用代码 1%–2%。教训是：在模拟器上做编译器远不如有真硬件后测量有效。

**③ 为 perf/TCO 而非 perf/CapEx 设计**

$TCO = CapEx + 3 \times OpEx$（CapEx 按 3–5 年摊销），而 OpEx 中**电力供应与冷却的成本是电费本身的两倍**。图 3 给出 TCO 与 system TDP 的关系：5 款 DNN DSA 的相关系数 **R = 0.99**，扩展到 15 款 CPU/GPU/TPU 仍有 **R = 0.88**（$R^2 = 0.78$）。作者据此建议：**拿不到 TCO 就用 TDP 做代理**。

**④ 支持向后 ML 兼容（backwards ML compatibility）**

新推理 TPU 至少要提供相同的数值行为（bf16 与 IEEE fp32）与相同的异常行为，否则编译器无法保证兼容。由于浮点加法不满足结合律，无法在允许性能优化的同时固定运算顺序，因此**同一款编译器必须以相似方式为所有目标生成代码**——这意味着新 TPU 从编译器视角看必须"像"前代 TPU。

**⑤ 推理 DSA 需要风冷以支持全球部署**

75 W 的 TPUv1 与 280 W 的 TPUv2 都是风冷，450 W 的 TPUv3 需要液冷。液冷要求把 TPU 放在若干相邻机架以摊薄冷却基础设施——这对训练超算不是问题（本来就是相邻机架），但对面向用户的推理是问题，因为低延迟要求全球部署，而有些战略数据中心已经很挤。**结论：推理 DSA 应当风冷。**

**⑥ 部分推理应用需要浮点**

TPUv1 只支持整数，强制量化。开发早期应用团队说 1% 的质量下降可接受，硬件到位后他们改变了主意——因为 DNN 整体质量提升后，1% 加在 40% 的错误上影响小，加在 12% 的错误上影响大（第 5 页）。恢复整数下的质量分数可能要多花数月开发。教训：DSA 可以提供量化，但**不应要求量化**。

**⑦ 生产推理通常需要多租户**

论文列出的三个理由：多语言对/多方言的翻译与语音模型需要多模型共存；多租户可以混搭不同 batch size 以平衡吞吐与延迟；工程实践上要灰度发布。Table 3 显示 Google 的生产推理负载中 **>80% 需要多租户**。多租户还要求**快速的模型切换**——开发者要求 <100 µs，而通过 PCIe 从主机加载权重需要 >10 ms，因此权重必须在本地内存。若全放片上 SRAM，需要以 >900 GB/s 的外部内存带宽在 100 µs 内载入 >90 MB（如 MLP1），已超过当时推理芯片的能力。这直接推出：**多租户需要快速 DRAM**。

**⑧ DNN 在内存与算力上每年增长约 1.5×**

Table 4 给出四个最早的生产推理应用的年度增长：CNN1 内存 0.97× / 算力 1.46×；MLP1 1.26× / 1.26×；CNN0 1.63× / 1.63×；MLP0 2.16× / 2.16×。四个数字的几何平均约 **1.44×**，即生产 DNN 的增长速度与摩尔定律相当。作者据此主张架构师必须留出余量，使 DSA 在整个生命周期内都保持有用。

**⑨ DNN 工作负载随 DNN 突破而演化**

Table 3 显示 2016 到 2020 年的负载迁移：MLP 从 65% 降到 25%（部分应用转向 BERT），**BERT 2018 年才出现，2020 年已占 28%**；为提升质量，transformer encoder + LSTM decoder（RNN0）与 Wave RNN（RNN1）取代了 LSTM（占 29%）。教训是可编程性与灵活性对推理 DSA 至关重要。

**⑩ 推理的 SLO 限制是 P99 延迟，不是 batch size**

论文批评近期 DSA 论文把延迟限制重新定义为 batch size（常取 1）。Table 5 给出生产应用的 P99 时间 SLO 与满足 SLO 的最大 batch size：

| 生产应用 | ms | batch | MLPerf 0.7 | ms | batch |
| --- | ---: | ---: | --- | ---: | ---: |
| MLP0 | 7 | 200 | ResNet50 | 15 | 16 |
| MLP1 | 20 | 168 | SSD | 100 | 4 |
| CNN0 | 10 | 8 | GNMT | 250 | 16 |
| CNN1 | 32 | 32 | | | |
| RNN0 | 60 | 8 | | | |
| RNN1 | 10 | 32 | | | |
| BERT0 | 5 | 128 | | | |
| BERT1 | 10 | 64 | | | |

**Google 的生产负载比 MLPerf 平均使用约 9× 更大的 batch size，同时受约 7× 更严格的延迟约束。** 这是因为向后 ML 兼容（④）让训练与推理贯通，Google 内部模型可以针对 TPUv4i 预先调优，而 MLPerf 的推理模型在 GPU 上训练，对 TPU 的调优较差。

### TPUv4i 的设计决策（第 6–8 页）

- **单核推理 + 双核训练的"一石二鸟"**：改用同一套 core、同一套 uncore 的缩放版本、同一团队在同一代码库中开发，得到单核的 TPUv4i（推理）与双核的 TPUv4（训练，可扩到 4096 芯片）。TPUv4 在 MLPerf Training 0.7 中比 TPUv3 快 2.7×，并匹配 Ampere GPU。
- **编译器兼容而非二进制兼容**：TPUv2/v3 共享 322-bit VLIW 指令包长度，但没有保持二进制兼容，理由有四——VLIW 的初衷是通过重编译启用新的指令级并行；Itanium 的教训（包括 XLA 团队中的工程师参与过）；XLA 同时接受 JAX/PyTorch/TensorFlow，靠一个编译器而非多编译器接口；TPU 软件以源码而非二进制分发。XLA 把编译分成机器无关的 HLO 与机器相关的 LLO，**只要新 TPU 把改动限制在 LLO 层就保持了编译器兼容**。
- **CMEM（Common Memory）128 MB**：SRAM 比 DRAM 节能 20×，但有大数据结构装不进 Vector Memory。128 MB 被选为"性能与芯片尺寸之间的拐点"，占 **die 面积 28%**。芯片面积 <400 mm²，更接近 TPUv1 而非 TPUv3。
- **四维张量 DMA**：XLA 团队对 TPUv2/v3 的二维（单 stride）DMA 的反馈促成了三维 stride 的 4D DMA。支持任意 steps-per-stride 与正/负 stride，**内层向量是 512 B**（匹配 128-lane 32-bit 向量单元）。源端与目的端的 striding 参数独立可编程，因而可在芯片内任意两个架构内存之间做 512 B 粒度的 4D 拷贝、reshape、scatter、gather、memset。DMA 带宽**设计为与 striding 参数无关**，以保证性能可预测。DMA 架构在 local（片上）、remote（片间）、host（主机）三类传输间统一。超过四维可用多个 DMA 模拟。
- **共享片上互连 OCI**：TPUv3 的点对点连接在带宽与组件数增长后过于昂贵，且要求预先决定通信模式（例如 TPUv3 的 TensorCore 只能把一半 HBM 当本地内存，另一半必须过 ICI）。TPUv4i 引入共享 OCI，拓扑可随组件缩放。
- **四路分组的 NUMA 式内存**：512 B 是原生访问尺寸（不是 64 B cache line），HBM 每核带宽比 TPUv3 增加 1.3×。借鉴 NUMA 的"用空间局部性降低延迟与对分带宽"，但把 NUMA 边界放在**同一个 core 内部**：HBM、CMEM、VMEM 各被物理分成四个 128 B 宽的组，与 OCI 的对应段连接、彼此重叠最小，形成**四个不重叠的网络，每个服务 153 GB/s 的 HBM 带宽，而不是一个网络服务全部 614 GB/s**。论文明确说这个四路拆分是必需的，因为布线资源不如逻辑那样缩放（对应教训 ①）。
- **4 个 MXU + 定制四输入浮点加法器**：XLA 团队反馈他们能处理 TPUv3 两倍的 MXU。为降低 systolic array 延迟，TPUv4i **先把四个乘法结果相加，再加到部分和上，用 32 个二输入加法器**，而不是串行地用 128 个二输入加法器累加——临界路径降到基线的 **1/4**。进一步做成定制四输入浮点加法器，**去掉了中间结果的四舍五入与规格化逻辑**；虽然数值上不等价，但去掉舍入反而提高了精度（与二输入加法器的差异小到不影响 ML 结果）。收益：相对 128 个二输入加法器**省 40% 面积、25% 功耗**，并**降低 MXU 峰值功耗 12%**（MXU 是芯片上功率密度最高的部件，直接影响 TDP 与冷却设计）。
- **时钟与 TDP**：1.05 GHz、芯片 TDP **175 W**，更接近 TPUv1（75 W）而非 TPUv3（450 W）。
- **ICI 配置**：为给未来的 DNN 增长留余量，TPUv4i 保留 **2 条 ICI 链路**（TPUv3 是 4 条），使每板 4 颗芯片可以通过模型切分快速访问近邻芯片内存。

## 实验与结果

### 实验设置

- **生产应用**：Google 内部八个推理应用（MLP0/MLP1/CNN0/CNN1/RNN0/RNN1/BERT0/BERT1），2020 年占 Google 推理负载约 100%（按 TPUv1/v2/v3 系统的 TCO 加权）。Table 3 给出它们的规模与多租户情况：MLP0 平均 580 MB / 最大 2500 MB / 27 个程序（±17，1–93）/ 2016 年占比 61% → 2020 年 25%；RNN0 1300/1300 MB / 13 个程序 / 0% → 29%；BERT0 **3000 MB** / 9 个程序 / 0% → **28%**。
- **基准**：MLPerf Inference 0.5–0.7 的 Server 与 Offline 场景（ResNet50、SSD、NMT；MLPerf 0.7 的 Server 含 bf16 与 int8）。论文明确标注其中若干结果是 "unofficial / unverified by MLPerf"。
- **对照**：NVIDIA T4（论文选择 T4 而非 A100 的理由是：A100 是 826 mm²、54B 晶体管、400 W，TDP 是 T4 的 5.7×，对大功耗芯片做推理是 perf/TCO 上的错配）；另在 Related Work 中列出 Goya、Nervana NNP-I、Zebra、HanGuang 800。

### 主要结果

**相对 TPUv2 的生产应用表现（图 8，第 9 页）**

TPUv3 与 TPUv4i 都是约 **1.9×** 于 TPUv2，TPUv1 为 0.7×。TPUv4i 在 perf/TDP 上是 TPUv3 的 **2.3×**。这个 2.3× 的分解是：**1.1× FLOPS**（123T → 138T）、**4.5× SRAM 容量**（32 → 144 MB）、**0.7× DRAM 带宽**（900 → 614 GB/s）、**0.4× TDP**（175 vs 450 W），加上把每核 MXU 数翻倍等微架构改动带来的利用率提升。其中 **CMEM 贡献约 1.5×，7 nm 工艺贡献约 1.3×，其余约 1.2×**。

论文特别指出一个与既有预测冲突的地方：accelerator wall 论文 [9] 用晶体管数增长的对数来预测代际 perf/TDP，而 TPUv4i 用 **1.6× 的晶体管**做到了 TPUv3 的 **2.3× perf/TDP**。

**相对 T4 的 MLPerf Inference 表现（图 9，第 9 页）**

TPU 全部用 bf16 跑（为维持与前代的向后 ML 兼容），而 T4 在 ResNet 与 SSD 上用 int8、在 NMT 上用 fp16（NVIDIA 无法让 int8 在 NMT 上工作）。结果：**TPUv4i 比 T4 快 1.3–1.6×，但 perf/TDP 只有 0.9–1.0×**；NMT 的 perf/TDP 是 1.3×（两者都用浮点）。按平均功耗而非 TDP 计算，TPUv4i 相对 T4 是 NMT 1.6–2.0×、ResNet50 1.0×、SSD 0.5–0.6×，几何平均 1.0×。SSD 差的根因是 NonMax Suppression 含大量 gather 与高内存强度操作，GPU 的合并访存比 TPU 的 HBM 更快。

论文对这个结果的评价是诚实的：**"向后 ML 兼容使 TPUv4i 对 Google 有价值，即使它的 int8 perf/TDP 并不比 T4 高多少。"**

**CMEM 的贡献（第 9–10 页）**

- **MLPerf Inference 打开 CMEM 的增益（图 10）**：Server 场景平均 **1.3×**，Offline 场景平均只有 **1.1×**；逐项为 ResNet50 Server 1.50、ResNet50 Offline 1.11、SSD Server 1.12、SSD Offline 1.07、NMT Server 1.41、NMT Offline 1.14、BERT Server 1.33、BERT Offline 1.11。
- **CMEM vs 把 HBM 带宽翻倍（图 11）**：禁用 CMEM 并让 TPUv4 的一个核关掉，即可让单个核的 HBM 带宽翻倍。几何平均上，2× HBM 带宽给 **1.3×**，而 CMEM 给 **1.5×**。结论：**加 CMEM 比翻倍 HBM 带宽更便宜、更低功耗、更容易**。三个应用在 CMEM 下增益达 1.7–2.2×：
  - **BERT1 1.71×**：HBM footprint 93 MB 本可装进 CMEM，但 XLA 为降低上下文切换时间把所有参数放在 HBM 再预取进 CMEM，因此只省下 58% 的 HBM 流量。真正的大头是 `Gather` 算子——它对约 63 MiB 的大缓冲做索引访存，随机访问模式使 HBM 表现很差；放进 CMEM 后 **Gather 快约 15×**。
  - **BERT0 1.85×**：HBM footprint 189 MB 大于 CMEM，预取省 50% HBM 流量；embedding 表约 98 MiB，无 CMEM 时占步进时间 >30%，进 CMEM 后 **Gather 快约 13×**。
  - **RNN1 2.22×**：HBM footprint 254 MB 大于 CMEM，但 CMEM 过滤掉 **98% 的 HBM 流量**；该模型跑很多轮 GRU，中间张量放进 CMEM 后避免昂贵的 HBM 流量，模型随后变成 **CMEM-bound**。
  - 其余 5 个应用的增益在 1.1–1.4×，论文的解释是**主要收益来自 CMEM 的高带宽**而非容量。
- **Roofline 分析（图 12）**：CMEM 读带宽 **2000 GB/s**、写带宽 **1000 GB/s**，且不像 HBM 那样可以同时读写。HBM 的 ridgepoint 是 **301**（FLOPS/byte）。BERT1 的 operational intensity 从无 CMEM 的 106 升到有 CMEM 的 124，**仍低于 HBM ridgepoint**。
- **CMEM 容量的敏感性（图 13）**：把 CMEM 容量从 128 MB 往下调，生产应用平均性能为 0 MB 时 69%、16 MB 时 82%、72 MB 时 90%、112 MB 时 98%；MLPerf 为 0 MB 时 76%、16 MB 时 87%、72 MB 时 97%、112 MB 时 100%。

**利用率（Table 7，第 12 页）**

| | TPUv1 | TPUv2 | TPUv3 | TPUv4i |
| --- | ---: | ---: | ---: | ---: |
| MXU/Chip | 1 | 2 | 4 | 4 |
| MXU 尺寸 | 256×256 | 128×128 | 128×128 | 128×128 |
| MXU 占 die 面积 | 24% | 8% | 11% | 11% |
| FLOPS/s 利用率 | 20% | 51% | 38% | 33% |
| HBM Roofline 利用率 | 20% | 66% | 63% | **99%** |

**没有 CMEM 时 TPUv4i 的 FLOPS/s 掉到 22%、roofline 掉到 67%。** 论文坦承峰值 FLOPS 利用率随时间**下降**（从 TPUv2 的 51% 降到 TPUv4i 的 33%），因为 MXU 数增加而算子并非都吃满，但 roofline 利用率（受内存带宽或 FLOPS 限制中更大的那个约束）反而升到 99%——这才是更有用的指标。

**T4 在 Google 数据中心的表现（第 11 页）**

论文做了两项对 T4 不利但真实的披露：

- **Turbo 模式与温度**：T4 最低时钟空载 35 °C；跑 1.6 GHz 的 MLPerf 后 30 秒内升到 75 °C，此后时钟在 **0.9–1.3 GHz** 之间波动。MLPerf 要求单次运行至少 1 分钟，刚好接近芯片与散热片热平衡的时间。
- **ECC**：MLPerf Inference 0.5/0.7 的 ECC 是可选的。在 Google 数据中心跑 10 分钟：**开启 ECC 后 T4 性能比其 MLPerf 报分下降 19%–26%**（T4 的 inline ECC 消耗内存带宽）。相比之下 TPUv4i 从约 37 °C 起、10 分钟只升约 5 °C，**ECC 开或关、运行多久都不影响速度**，因为 Google 特意在数据中心按恒定延迟供足了电力与冷却，并多配内存让 ECC 可以常开。

**A100 的对照（第 12 页）**

MLPerf 0.7 推理显示 A100 的 ResNet50 server 比 T4 快 4.7–5.7×，SSD server 快 6.0–7.0×，但 **A100 的 perf/TDP 在 T4 的 ±20% 以内**。

**TCO 与 TDP 的相关性**

5 款 DNN DSA：**R = 0.99**；15 款 CPU/GPU/TPU 跨多代：**R = 0.88（$R^2 = 0.78$）**。论文据此呼吁未来的 DSA 论文应报告"整个生命周期内的 perf/system TDP"。

### 消融与敏感性实验要点

- **CMEM 开关**（图 10、11）：Server 1.3× / Offline 1.1×；相对"2× HBM 带宽"是 1.5× vs 1.3×。
- **CMEM 容量**（图 13）：从 0 到 112 MB 的性能曲线，作者自己承认"少 10%–20% 可能对当前 DNN 影响不大，但要过几年才知道 128 MB 是否偏大"。
- **MXU 数与加法器结构**：4 输入加法器省 40% 面积 / 25% 功耗、降 MXU 峰值功耗 12%、把 systolic array 临界路径降到 1/4。
- **量化**：八个生产 DNN 中只有 RNN0 被量化，且主要收益是把内存占用减半；MLPerf 中 3D-Unet 与 DLRM 的后训练 8-bit 量化精度损失 <0.1%（相对 FP32），BERT <1%。

## 局限性与未来方向

- **多项关键结果是未经 MLPerf 验证的自测**：图 9、10、13 的 TPUv4i MLPerf 分数被明确标注为 "unofficial / unverified by MLPerf"，其中 NMT 用的是 MLPerf Inference 0.5 的代码而 ResNet/SSD 用 0.7。文中的横比因此不是官方口径，作者对此标注是清楚的，但引用时应保留这层限定。
- **perf/TDP 上的表现并不优于 T4**：TPUv4i 相对 T4 是 1.3–1.6× 性能但 0.9–1.0× perf/TDP；按平均功耗算是几何平均 1.0×，SSD 甚至只有 0.5–0.6×。论文选择 T4 而非 A100 作对照，理由是 A100 与推理场景错配（400 W、5.7× T4 的 TDP），这个理由成立，但也意味着**论文没有与当时最强推理芯片的正面能量效率对比**。
- **TCO 数据不公开**：作者明确指出 "TCO is confidential"，全文用 system TDP 作代理。R = 0.99 这个数字建立在 5 款自家 DSA 上，扩展到 15 款外部产品后降到 0.88——相关性成立，但样本量小且全来自单一数据中心的运营模式。
- **CMEM 容量选择的依据是事后验证**：128 MB 是"拐点"，图 13 表明 112 MB 已达 98%/100%，但作者自己承认需要数年才能确认 128 MB 是否偏大。换句话说，这个决定在当时并没有可验证的最优性论证。
- **多个教训的量化基础来自很小的样本**：教训 ⑧ 的"1.5×/年"由四个应用的年度增长率几何平均得到，其中 CNN1 的内存增长是 **0.97×**（负增长）；四个数字的离散度（0.97 到 2.16）比它们的均值更有信息量，论文没有给出置信区间。
- **教训 ⑦ 的多租户需求依赖 Google 的工作负载形态**：">80% 的生产推理负载需要多租户"是 Google 内部口径（按 TCO 加权），对外部场景的可迁移性未讨论。
- **未来方向**：论文没有独立的 future work 章节，但结论段给出了方向——在摩尔定律减弱、Dennard scaling 终止的背景下，**硬件/软件/DNN 协同设计**是 DNN DSA 继续跨越 accelerator wall 的最佳机会。同时给出一个值得注意的反向论断：由于 SRAM 缩放极弱（5 nm SRAM 只比 7 nm 密约 13%）而逻辑仍在变密，**内存访问的能量成本将更主导 perf/TDP**；因此与 ML 社区"尽量少做 FLOPS"的习惯相反，在数据中心语境下**降低精度的 FLOPS 相对"免费"，而内存引用才是成本**。

## 个人点评

- **这份论文最有价值的部分是 Table 2 与它推出的设计取向**。把 Horowitz 的 45 nm 能耗表更新到 7 nm，得到一张可以把"该往哪放预算"算清楚的表：逻辑改善 2.4–4.3×、SRAM 只有 1.3–2.4×、单位长度导线不到 2×，而 DRAM 靠 HBM 封装改善 6.3×。由此推出的两个决策都很硬——推理芯片该用大 SRAM（CMEM 占 28% die 面积）而不是靠 DRAM；TPUv2/v3 从 1 个大核改成 2 个小核是因为导线不缩放。这种"技术趋势 → 微架构决策"的推导链，比任何 benchmark 对比都有说服力。
- **CMEM 那一段的分析质量很高**。作者不是简单说"加了 128 MB 更快"，而是区分了两种收益来源：5 个应用的增益（1.1–1.4×）来自带宽，3 个应用（1.7–2.2×）来自容量与随机访问能力。BERT 的 `Gather` 算子对 63/98 MiB 的 embedding 表做索引访存，在 HBM 上表现很差，进 CMEM 后快 13–15×——这是"随机访问带宽"而非"顺序带宽"决定性能的干净案例。尤其可贵的是论文也交代了一个反向细节：XLA 为了降低多租户上下文切换时间把参数留在 HBM 再预取，因此 BERT1 只省了 58% 的 HBM 流量，而不是全部。Roofline 那一页还老老实实写出 BERT1 的 OI 从 106 升到 124 **仍低于 HBM ridgepoint 301**——承认了 CMEM 没把它推到计算受限区。
- **诚实度是这篇论文的突出特征**。Table 7 直接写"TPUv4i 的 FLOPS 利用率是 33%，比 TPUv2 的 51% 低"；§4 明确写"TPUv4i 相对 T4 的 perf/TDP 只有 0.9–1.0×"；NMT 用 fp16 而 T4 也用 fp16 才赢下 1.3×；SSD 在平均功耗口径下只有 0.5–0.6×，作者归因于 NonMax Suppression 的 gather 比 GPU 的合并访存慢。这种把不利数字留在正文里的做法，让其余结论（如 2.3× vs TPUv3 perf/TDP）更可信。
- **两处需要打折**。第一是 MLPerf 数据的口径：图 9/10/13 都是 Google 数据中心自测、未经验证，且 NMT 混用 0.5 版本代码、ResNet/SSD 用 0.7。第二是教训 ⑧ 的"1.5×/年"：四个样本里有一个是 0.97×（下降），几何平均掩盖了 0.97–2.16 的离散度，这类外推结论在引用时应给出原始四个数字而不是均值。另外 Table 6 的 HanGuang 800 在论文中被当作反例（perf/TDP 2.17× 于 T4 但完全没有 DRAM），论文承认它性能指标好却在多租户与 DNN 增长上无解——这其实是一段相当公允的对手分析，值得一读。
- **一个容易被忽略的技术点**：那个定制四输入浮点加法器。它去掉中间结果的舍入与规格化，论文指出结果**不是数值等价的**，但去掉舍入反而提高了精度，且与二输入加法器的差异小到不影响 ML 结果。这条在"向后 ML 兼容"（教训 ④）的框架下是有点冒险的——论文一方面强调浮点结合律不成立导致必须保持运算顺序一致，另一方面又换掉加法器结构改变数值。两者并不直接矛盾（④ 说的是跨代一致性，这里说的是单个芯片内部的加法器），但读者若关心位级可复现性，应该注意这个改动。

## 工程化三问总结

### 1. 它解决了什么瓶颈？

- **应用场景与核心瓶颈**：数据中心的 DNN 推理。真实瓶颈不是峰值算力，而是 **perf/TCO**：$TCO = CapEx + 3\times OpEx$，而 OpEx 中电力供应与冷却的成本是电费的两倍。论文用 system TDP 作 TCO 的代理并给出相关性（5 款 DSA：R = 0.99；15 款处理器：R = 0.88）。
- **现有方法为何不足**：以 benchmark 为目标的 perf/CapEx 或 perf/mm² 优化会误导（HanGuang 800 提速 1.5× 功耗涨 2.6×；关 ECC 提速、删性能计数器都能让指标好看）；TPUv3 的 450 W 需要液冷，而液冷要求相邻机架布局，与推理所需的全球部署冲突；TPUv1 只支持整数，强制量化使部分应用多花数月开发才能恢复质量。
- **论文证据的分层**：
  - **可核算的定量基础**：Table 2（45 nm vs 7 nm 操作能耗，几何平均 2.6×）；Table 4（四个应用的年度增长，几何平均约 1.44×）；Table 7（四代 TPU 的 FLOPS/s 与 roofline 利用率）；图 11（CMEM 1.5× vs 2× HBM 带宽 1.3×）；图 13（CMEM 容量敏感性）；图 3（TCO vs TDP，R = 0.99/0.88）；perf/TDP 2.3× 的因子分解（1.1× FLOPS、4.5× SRAM、0.7× DRAM BW、0.4× TDP）。
  - **明确标注为未验证的**：图 9、10、13 的 TPUv4i MLPerf 分数（"unofficial, not verified by MLPerf"）；图 2 中 TPUv3 的 MLPerf 0.5 分数（Mask R-CNN、Transformer）引自他处且标注 unverified。
  - **不利但被披露的结果**：相对 T4 的 perf/TDP 只有 0.9–1.0×；按平均功耗的几何平均 1.0×，SSD 0.5–0.6×；TPUv4i 的 FLOPS 利用率 33% 低于 TPUv2 的 51%；T4 开 ECC 后性能下降 19%–26%（这条对 T4 不利但影响论文的对比基线）。

### 2. 用了什么结构或训练方法？

- **整体结构**：单核推理芯片 TPUv4i（与双核 TPUv4 共享 core、uncore 缩放版本与代码库）。架构内存为 HBM、CMEM、VMEM、SMEM、IMEM；数据通路为 MXU、VPU、XLU、TCS；uncore 含 OCI、ICI Router、ICI Link Stack、HBM Controller、Unified Host Interface、Chip Manager（图 5）。
- **关键结构**：
  1. **128 MB CMEM（占 die 面积 28%）**，选为性能-面积的拐点；CMEM 读 2000 GB/s、写 1000 GB/s，可同时读写（HBM 不可）。
  2. **4D 张量 DMA**（三维 stride），内层向量 512 B，源/目的端 striding 独立可编程，带宽与 striding 参数无关（保证性能可预测）；统一 local/remote/host 三类传输；保留 TPUv2/v3 的 relaxed DMA ordering（展开后的读写完全无序，HBM 只能由 DMA 访问）。
  3. **共享 OCI**，替代 TPUv3 的点到点连接，拓扑可随组件缩放。
  4. **core 内的四路 NUMA 分组**：HBM/CMEM/VMEM 各分四个 128 B 宽组，形成四张互不重叠的网络，每张服务 153 GB/s，而不是一张网络服务 614 GB/s。
  5. **4 个 MXU + 四输入浮点加法器**：先合并四个乘积再加部分和（32 个二输入加法器），去掉中间舍入与规格化，临界路径降至 1/4，省 40% 面积与 25% 功耗，降 MXU 峰值功耗 12%。
  6. **2 条 ICI 链路**（TPUv3 为 4 条），供 4 芯片板内做模型切分。
  7. **VLIW 指令比 TPUv3 宽 25%**，以容纳 4 个 MXU 与 CMEM scratchpad 的额外字段。
- **数值与量化策略**：支持 bf16、int8 与 IEEE fp32，量化是可选的（教训 ⑥）。八个生产 DNN 中只有 RNN0 被量化（主要收益是把内存占用减半）。

### 3. 对芯片架构和 RTL 有什么启发？

- **芯片架构**：五条可直接套用的判断。第一，**在逻辑比导线与 SRAM 进步快的工艺条件下，把面积投给逻辑与片上 SRAM 是对的**：TPUv4i 用 1.6× 晶体管拿到 2.3× perf/TDP，其中 CMEM 贡献约 1.5×、7 nm 贡献约 1.3×。第二，**大容量片上 SRAM 的价值不只在带宽，更在随机访问**：BERT 的 Gather 对 63–98 MiB 表做索引访存，进 CMEM 后快 13–15×，而这类访问模式在 HBM 上无法通过提高顺序带宽来改善。第三，**"NUMA 边界放在 core 内部"是一个可复用的手法**：把大内存切成四个 128 B 宽组、各连 OCI 的不同段，用空间局部性换更少的布线资源、更低延迟与更简单的仲裁，这是应对导线不缩放的直接手段。第四，**TCO 与 TDP 的高度相关（R = 0.99/0.88）意味着功率密度就是架构约束**：MXU 是最耗电的部件，甚至值得为它专门优化加法器结构（四输入加法器降 MXU 峰值功耗 12%），因为峰值功耗直接决定 TDP 与冷却方案，进而决定能否风冷、能否全球部署。第五，**多租户需求会反向决定内存层级**：需求要求 <100 µs 的模型切换，通过 PCIe 加载权重需 >10 ms，因此权重必须在本地内存；若全放 SRAM 需要 >900 GB/s 的载入带宽（已超过当时芯片能力），所以推理芯片要有快速 DRAM。这条推理链对任何要做多租户推理的架构都适用。
- **RTL**：可落到实现层的模块包括：4D（三维 stride）DMA 引擎——需要支持任意 steps-per-stride、正负 stride、源/目的独立参数、且带宽与 stride 无关（意味着地址生成与数据通路必须解耦）；512 B 原生访问尺寸下的四组 128 B 通道仲裁；OCI 的仲裁与拓扑可配逻辑；CMEM 与 HBM 之间的预取通路（XLA 把参数留在 HBM 再预取进 CMEM，硬件侧需要有对应的预取与 partial-completion 同步机制，论文提到 TPUv4i 支持把 DMA 的部分完成进度与 TensorCore 同步以隐藏 ramp-up/ramp-down）；四输入浮点加法器（去掉中间舍入与规格化，需重新设计对齐与规格化电路）；core-DMA 显式同步逻辑（重叠地址模式需要显式同步避免内存级冒险）；以及 uncore 中大量用于 tracing 与性能计数器的硬件（论文明确说这些增加设计时间与面积，但为了 perf/TCO 是值得的）。需要强调：论文提到的 **relaxed DMA ordering**（展开后的读写完全无序）意味着 RTL 不需要为 DMA 维持顺序语义，但需要为 load/store 与 DMA 的重叠提供显式的同步原语。
- **推断边界**：第 1 问的定量基础（Table 2、4、7，图 11、13，TCO-TDP 相关性）符合要求并可核算；MLPerf 相关结果按 "unverified" 降级。第 2 问的结构描述全部来自论文正文与图 5/6。第 3 问的芯片架构与 RTL 内容为工程推断；论文中未提供的量值——CMEM 各 128 B 通道的仲裁策略细节、DMA 引擎的面积与功耗、四输入加法器的具体电路实现、OCI 拓扑参数、tracing 硬件的面积占比、以及 4D DMA 的地址生成开销——均 `TBD`。
