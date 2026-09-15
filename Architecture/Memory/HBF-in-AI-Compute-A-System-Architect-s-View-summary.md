# 深度分析：《HBF in AI Compute: A System Architect's View》

## 基本信息

- **标题**：HBF in AI Compute — A System Architect's View
- **文档类型**：教程幻灯片（Hot Chips 2026 Tutorial，22 页；文件属性标题为 "dry run v7"）
- **作者**：Anurag Agrawal（System Architecture, OXMIQ Labs）、Radhakrishna Giduthuri（Software Architecture, PRAXMATI）
- **机构**：OXMIQ Labs、PRAXMATI
- **年份**：2026（PDF CreationDate 2026-08-23）
- **链接**：Hot Chips 2026 Tutorial（未提供独立论文链接）。核心规格引用 **OCP HBF Architecture Specification v0.7.0 (2026)**；分析方法引用 `Challenges & Research Directions for LLM Inference HW (arXiv 2601.05047)`、`FineMoE (arXiv 2502.05370)`、`MoE-Beyond (arXiv 2508.17137)`、`MoE-Infinity (arXiv 2401.14361)`、`Tutti (2605.03375)`

> **HBF = High-Bandwidth Flash**。本材料是厂商（OXMIQ Labs）在会议上做的教程宣讲，末尾含招聘页。其中的规格取自 OCP 草案 v0.7.0，性能结论来自其自研模拟器 OxSOL。下文按此分层标注证据强度。

## 一句话总结

> 这份教程用一条成本公式 $mem = β·max(C, I·b/α)$ 把"该买哪种内存"变成一个二维优化问题：HBF 是低 α（带宽/GB）、低 β（$/GB）的**容量点**而非廉价 HBM，在 72-GPU 机架上用同等成本换来 14× 容量但只有 0.6× 带宽，因此它只在 MoE 的小 batch / 低交互度区间与长上下文稀疏 KV 上划算，密集模型与高 batch 仍应留在 HBM。

## 研究动机与问题定义

- **要解决的核心问题**：HBF 的 $/GB 比 HBM 低得多，容易被当成"廉价 HBM"，但作者开篇就给出反驳标题——**"The cheapest $/GB can be the most expensive $/token."**（第 3 页）。问题被重新表述为：在什么工作负载区间内，用容量换带宽是划算的？
- **现有方法的不足**：
  - 单纯比较 $/GB 会得出错误结论，因为内存的成本不是"买 GB"，而是买一个方程的结果（第 6 页）。
  - 现有推理引擎的内存路径是围绕 HBM 与 host CPU pinned memory 设计的：vLLM 的 MoE Expert Pool 路径**目前并不存在**（第 18 页作者自己标注 "MoE Experts Pool is not yet available in vLLM"）。
- **切入角度**：先用公开研究（arXiv 2601.05047）给出的 (β, α) 二维坐标系定位 HBF 的位置，再用两个解析模型（MoE 的专家覆盖概率模型、per-token 字节读取模型）推出工作负载特征，最后用机架级模拟给出 $/token 结论。

## 核心结构（规格与机制）

### 整体结构

材料的论证骨架是四层：**规格（HBF 到底能做什么）→ 坐标系（(β, α) 平面中的位置）→ 工作负载（MoE 与 Dense 的差异、per-token 字节读取）→ 系统模拟（机架级 $/token）**，最后落到软件约束与生态接入。

### 关键规格与机制

**HBF 规格表（第 2 页，引自 OCP HBF Architecture Specification v0.7.0）**

| 指标 | Grade 1 | Grade 2 | Grade 3 |
| --- | ---: | ---: | ---: |
| 最大用户带宽 | 0.384 TB/s | 1.536 TB/s | 3.072 TB/s |
| UCIe 速率 | 8 GT/s | 16 GT/s | 32 GT/s |
| 容量 | 8-high · 256 GiB | 16-high · 512 GiB | 16-high · 512 GiB |
| 访问粒度 | 64 B–4 KiB 读 · 4 KiB 写 · 4 KiB page | | |
| 写寿命 | **Left open-ended**（规格未定义） | | |

幻灯片结论："**8–16× the capacity of HBM — at the same cost**"。

**成本公式（第 6 页）**

$$mem = \beta \cdot \max\left(C,\ \frac{\text{BW demand}}{\alpha}\right), \qquad \text{BW demand} = I \cdot b$$

其中 $\beta$ 是 $/GB，$\alpha$ 是带宽/GB，$C$ 是需要持有的容量（GB to HOLD），$I·b$ 是需要喂给的带宽（GB to FEED）。这个 $\max$ 是全文的分析核心：**容量与带宽谁成为成本主项，取决于工作负载落在哪一侧**。最终结论把它压缩成一个判据：$mem = \beta \cdot \max(C,\ I b/\alpha)$，品质因数是 $(\beta/\alpha)\cdot b$。

**(β, α) 内存景观（第 4 页）**

横轴 β = $/GB（相对 HBF = 1，对数），纵轴 α = 带宽/GB（1/s，对数）。标注的器件位置：

| 器件 | α（相对带宽/GB） | β（相对 $/GB） |
| --- | ---: | ---: |
| NAND SSD | ~0.001 | ~0.1× |
| HBF-G1 / G2 / G3 | ~1 / ~5 / ~10 | 1× |
| LPDDR5X | ~5 | ~5× |
| HBM3E | ~100 | ~12× |
| HBM4 | ~200 | ~15× |
| HBM4E | ~300 | ~20× |
| SRAM-only（Cerebras、Groq） | ~10⁶ | ~10⁵–10⁶× |

图中明确标注 **HBM4 的 α 是 HBF-G2 的 25×（对 G3 是约 13×）**。幻灯片的一句注记很关键：**IMC/PIM/d-Matrix 式存内计算可以"离开这个平面"——它们搬结果而不搬字节**（第 4 页右下角）。

**Per-token 字节读取模型（第 5 页）**

MoE 在 decode 阶段每步读取的字节数：

$$b = \left[1 - \left(1 - \frac{k}{N}\right)^B\right] W + W_0 + B\cdot K$$

其中 $N$ 为总专家数，$k$ 为每 token 激活的专家数，$B$ 为 batch（等于并发用户数），$W$ 为路由专家权重，$W_0$ 为固定权重（共享专家 + 密集部分），$K$ 为每 token 的 KV 字节数。由此得到两个派生量：

- $I = \text{BW}/b$ —— 交互度（tok/s/user）
- $T = B\cdot I$ —— 吞吐（tok/s）

MoE 的 throughput 曲线分三段（第 5 页，基于 OxSOL 对 Kimi-K2 与 Llama-3.1-70B 在 B200 NVL72 上的模拟）：① Experts-Sparse（$T$ 正比于 $B$，$T$ 在图中平缓）；② Experts-All（权重读取被摊薄，$T$ 随 $B$ 增长）；③ Saturated（利用率→100%，$T$ 变平、$I \to 1/B$）。密集模型的转折点被标在 comp-$B \approx 300$。

**专家覆盖概率（第 10 页）**

$$p_{\text{eff}}(B) = 1 - \left(1 - \frac{k}{N}\right)^{B}$$

用六个模型的 $k/N$ 绘图：Mixtral-8x7B（25.0%）、Qwen3-235B-A22B（6.2%）、DeepSeek-V3（3.1%）、Kimi-K2（2.1%）、Kimi-K3（1.8%）、Llama-4-Maverick（0.8%）。结论是：**混合查询下专家热度会摊平，缓存只在低 $B$ 有收益，或需要把"相似"查询批在一起**。这直接削弱"HBM 做热专家缓存"这条思路的普适性。

**软件约束（第 15 页，引自 OCP 规格 v0.7.0）**

- 达到最大带宽所需的访问块：**读 64 KB、写 1 MB（64 KB 对齐）**。
- HBF 的读写**走 DMA**，不是为 GPU cache hierarchy 设计的。
- HBF 与 HBM 混用时**需要独立的内存管理**。
- **上电数据保持能力约 24 小时 @ 85 °C**——因此必须有 host 管理的生命周期（lifecycle）机制。
- 需要仔细的 host 管理才能达到 **~10 年寿命**或 100% 寿命消耗。
- **Scratchpad SRAM 不能直接读写 NAND block**。

幻灯片把这一页总结为："Read-optimized, write-constrained, host-managed — placement is a software problem"。

### 核心主张

1. HBF 是**容量点**（低 α、低 β），不是廉价 HBM；判据是 $(\beta/\alpha)\cdot b$ 对工作负载。
2. HBF 只在**低带宽需求（$I·b$ 小）**的区间取胜：小 batch / 低交互度的 MoE，以及长上下文稀疏 KV。
3. 让 HBF 更有价值的前提是：读带宽上升（α↑）、$/GB 差距保持（β↓）、写寿命与写带宽/延迟被解决。
4. 软件前提：需要 HBF allocator/backend、placement policy、异步预取与寿命遥测——今天的 vLLM 路径指向 CPU/LMCache，不是 HBF。

## 证据、案例与论证

### 机架级模拟设置（第 11 页）

| | HBM-only | HBF-only | HBF + HBM |
| --- | ---: | ---: | ---: |
| 72-GPU 机架的并行策略 | TP8 · DP9 | TP1 · DP72 | TP2 · DP36 |
| 每 DP 内存容量 | 2.3 TB | 4.1 TB | 2.5 TB |
| 每机架内存容量 | 20.7 TB (1×) | 294.9 TB (14×) | 89.3 TB (4.3×) |
| 聚合带宽 | 1,584 TB/s | 922 TB/s (0.6×) | 1,418 → 279 TB/s |
| 每 GPU 带宽 | 22.0 TB/s | 12.8 TB/s | 19.7 → 3.9 TB/s |
| 成本 · 功耗 | 1× · parity | 1× · parity | 1× · parity |

- **模型与工况**：Kimi-K2 1T @ FP4，上下文 1M in / 1K out（另测 32k/8k 与 1k/8k），每 DP 的 batch 从 1 到 512，TP 用于装下容量、DP 填满 72-GPU 机架。成本是 **3 年摊销的 capex+opex，以比值呈现**。工作负载是 decode-centric。
- **一句话结论**："Same rack, same cost → HBF buys ~14× capacity for ~0.6× bandwidth"。

### 案例一：All-HBF 小上下文（第 8 页，256/256）

- Kimi-K2，B = 8（538 GB）→ 32（554 GB）→ 128（615 GB）。
- 图中的基点是 HBF：1 GPU · 4 TB · 12.8 TB/s · 1× 成本。对照 HBM：2 GPU · 0.58 TB · 44 TB/s · 2×。
- Grade 3 的带宽墙被标为 **24 TB/s**。
- 结论句："HBF wins on cost at low B but with **85% dead capacity** — HBM pays well past the wall"。也就是说在小上下文下 HBF 成本低但在低 $B$ 时八成以上容量是闲置的。

### 案例二：All-HBF 长上下文（第 9 页，1M/1K）

- Kimi-K2，B = 8（664 GB）→ 32（1058 GB）→ 128（2631 GB）。
- HBM 对照：4 GPU · 1.1 TB · 88 TB/s · 4×（注记："same cost as 16TB 4x HBF — buys BW, not idle capacity"）；4× HBF 对照：51 TB/s。
- 结论句："Capacity wins only while $I·b$ stays low — past that, HBM is the better buy"。长上下文把容量需求推高，HBF 的成本优势区间比小上下文更宽，但带宽需求一旦越过某个 $I·b$ 就反转。

### 案例三：混合部署 · HBM 做热专家缓存（第 10 页，Enquiry 2）

用 $p_{\text{eff}}(B) = 1-(1-k/N)^B$ 论证：$k/N$ 越小的模型（Kimi-K2 2.1%、Kimi-K3 1.8%、Llama-4-Maverick 0.8%）需要越大的 batch 才能覆盖 90% 的专家。因此"把热门专家放在 HBM 里缓存"只在低 $B$ 有效；高 $B$ 时几乎所有专家都会被读到，缓存失去意义。

### 案例四：机架 $/token 与单机容量的两难（第 12–14 页）

- **左图（CAPEX parity per rack，$/M-token 对交互度 $I$）**：HBM（黑色，2.3 TB，TP8·DP9）在 $I \ge 70$ 之后给出最低的 $/M-token；HBF+HBM（金色，TP2·DP36）最优约在 288 用户附近；HBF-only（红色，TP1·DP72）约在 576 用户附近出现峰值成本点。注记是"same rack · same $ → HBM makes far more tok/s → lowest $/M-token"。
- **右图（cost to stand up one instance，对交互度）**：HBM 8× 机箱（8 GPU · 2.3 TB · 176 TB/s · 8× TDP）最多 112 用户且标注 "$I < 74$: HBM can't serve (OOMs at 112 users)"；HBF+HBM 2× 机箱（2 GPU · 2.5 TB · ~40 TB/s pk · 2× TDP）最多 128 用户；HBF-only 1× 机箱（1 GPU · 4.1 TB · 12.8 TB/s · 1× TDP）最多 232 用户。成本与 TDP 正比于 GPU 数（每 GPU 价格持平）。
- 幻灯片把两图压成一句："Cheaper $/GB ≠ cheaper $/token — **HBM for the rack, HBF for the box**: pick your scale"。这是全文最有操作性的结论：采购单位不同，最优内存不同。

### 案例五：Kimi-K3 的字节分布（第 17 页）

Kimi K3 2.8T 总权重 1.56 TB，其中 **MoE 专家权重 1.45 TB（占字节数的 93%）**，attention 权重 72.2 GB、共享专家 30.2 GB、其他 4.7 GB。1M token 序列的 KV cache 为 **30 GB**。推论：MoE 专家池是 write-once、read-cold，正好落在 HBF 的容量区；剩下 7% 权重（110 GB）留在 HBM。

### 案例六：稀疏注意力（第 19 页）

$$\varphi = \frac{\text{top-}k}{\text{context}} \approx 1\%\text{–}2\% \ll \varphi^* = \frac{\alpha}{I}$$

只要 $\varphi < \varphi^*$，就应当"把整份 KV 便宜地放在 HBF 里，每步只读 top-k"。稀疏注意力（DSA、CSA、Kimi-Linear）每步读约 1–2k 行，而完整 KV 只在长期/多轮场景被保持为冷数据。这是 HBF 最合适的一类负载。

### 软件侧提案（第 16、18 页）

- 生态现状：生产默认是 vLLM（连续批处理、Paged KV、前缀缓存、跨加速器可移植），另有 SGLang、LMDeploy、Modular MAX，以及厂商优化栈（TensorRT-LLM、OpenVINO、AWS Neuron、Google JetStream）。
- vLLM 的内存子系统现状：模型权重（含 router MoE）驻留 HBM，Paged KV cache 与激活在 HBM，KV 可溢写到 SSD 或通过 RDMA 走 remote KV pool。
- **作者提案**：为 vLLM 加 HBF 插件，用 HBF 替代 host-DRAM pinned memory 承担 KV offload pool、prefix cache 与 MoE 专家池（1.45 TB）。并指出 MoE Expert Pool 这条路径 vLLM 尚未提供。
- 硬件形态示例：**GPU 配 4× HBM + 4× HBF stack → 2.2 TB，峰值约 17.4 TB/s**（第 18 页）。

## 局限性与未来方向

- **规格本身是草案**：全部硬件参数引自 **OCP HBF Architecture Specification v0.7.0 (2026)**，是 v0.7 草案而非定版；HBF Grade 1–3 的划分与"同样成本下 8–16× HBM 容量"的说法是规格方立场，本材料未给独立验证。
- **写寿命存在内部张力**：规格表把 Write endurance 写成 "Left open-ended"（未定义），而软件页写"仔细的 host 管理可达到 ~10 年寿命或 100% 寿命消耗"。同一份材料里一处不承诺、一处给出目标，说明寿命仍是未量化风险。这一点是 HBF 落地的最大不确定项，材料只给了"需要 host 管理"的定性结论。
- **断电保持窗口极短**：上电数据保持约 24 小时 @ 85 °C。这意味着 HBF 是"带寿命的缓存/池"而非持久存储，任何把它当存储层的部署都需要额外的生命周期管理逻辑，材料未给该逻辑的开销。
- **NAND 访问粒度与 GPU 访存模式的错配**：最大带宽需要 64 KB 读 / 1 MB 写，而 HBF 走 DMA 且"不是为 GPU cache hierarchy 设计"，scratchpad SRAM 还不能直接读写 NAND block。这三点合起来意味着需要显式的 DMA 引擎与专门的 copy 路径，材料未给这部分的数据搬运开销（例如 KV 从 HBF 搬到 HBM/SRAM 的额外带宽消耗）。
- **所有系统结论来自自研模拟器**：成本与性能结论全部出自 "OxSOL simulations (OXMIQ)"（第 5 页），**没有任何实测**。材料给出了模型公式（$b$、$I$、$T$、$p_{\text{eff}}$），但未给出模拟器的输入参数表、校准方式或与真实部署的偏差。
- **成本假设偏强**：三种配置的 "Cost · power = 1× · parity"，即假设同机架成本与功耗相同。这隐含接受"HBF 与 HBM 每机架总价一致"的设定，而第 4 页又明确 HBF 的 β 只有 HBM 的 1/12–1/20。两种口径并存，读者需要自行判断哪一种是采购假设。
- **结论高度依赖具体模型与工况**：全篇用 Kimi-K2 1T（部分用 Kimi-K3 2.8T）与 1M/1K 上下文。$k/N$ 极小的 MoE 是 HBF 最有利的场景；材料在第 10 页用六个模型的 $k/N$ 分布做了一定泛化，但**密集模型与专家数较少的 MoE 未被纳入机架级模拟**。
- **未来方向（材料自身给出）**：读带宽上升（α↑）、$/GB 差距保持（β↓）、解决写寿命与写带宽/延迟；软件侧需要 HBF allocator/backend、placement policy、异步预取与寿命遥测。OXMIQ 的下一步方向列为 "multi-agent locality · prefetch-BW limits · memory config to workload"（第 22 页）。

## 个人点评

- **这份材料的分析框架是它的主要价值**。$mem = \beta\cdot\max(C, Ib/\alpha)$ 这个式子把"该用哪种内存"从直觉判断变成可计算的判据，并且立刻给出反直觉结论：$/GB 最便宜的内存可能给出最贵的 $/token。第 4 页的 (β, α) 平面把 SSD、HBF、LPDDR、HBM、SRAM-only 放在同一坐标系里，是一张很好的通用参考图；HBM4 的 α 是 HBF-G2 的 25× 这个数字，比任何笼统的"HBM 快得多"都有用。
- **"rack vs box"这个区分是全文最实用的洞察**。第 12–14 页的左图说明在机架采购口径下 HBM 在 $I \ge 70$ 之后给出最低 $/M-token；右图说明在单机箱采购口径下 HBF 能把用户数从 112 提到 232，但因为 OOM 无法服务高交互场景。同一份模拟在两种采购单位下给出不同答案，这解释了为什么"HBF 是否划算"的讨论经常各说各话。
- **两个模型的选择很讲究**。$b = [1-(1-k/N)^B]W + W_0 + BK$ 把 MoE 的权重读取从"每步全读"平滑过渡到"随 batch 摊薄"，$p_{\text{eff}}(B) = 1-(1-k/N)^B$ 则给出专家覆盖随 batch 的饱和行为。两者共同说明同一件事：batch 一大，HBF 的容量优势就失去意义，因为带宽需求涨上去了。这是对"HBF 取代 HBM"式叙事的有效反驳，而且用的是 HBF 支持者自己的模型。
- **薄弱处在验证与寿命**。整份材料的性能与成本结论都出自自研模拟器 OxSOL，没有一次实测；给出的公式可以让读者复算 $b$ 与 $I$，但模拟器的校准与偏差未披露。写寿命在规格表里写 "Left open-ended"，在软件页又变成"~10 年"，这条矛盾的严重性被低估了——它决定的是 HBF 能不能作为可写的池使用，还是只能当一次写入的权重仓库。24 小时上电保持窗口也没有展开讨论其对调度系统的影响。
- **值得记录的定位**：作者自己在标题里写 "bargain or trap?"，结论写 "A precision instrument, not a hammer"。这种把自家技术的适用边界写清楚的做法，比大多数 vendor deck 诚实。第 18 页也直接承认 vLLM 目前没有 MoE Expert Pool 路径。这些自我设限反而提高了材料中其他部分的可信度。

## 工程化三问总结

### 1. 它解决了什么瓶颈？

- **应用场景与核心瓶颈**：LLM 推理的内存采购决策——具体是"该用 HBM 还是 HBF，以及两者怎么混"。核心瓶颈是算力增长与 HBM 带宽/容量的剪刀差，导致大模型（尤其 MoE）的内存成本成为 $/token 的主导项。材料给出的量化形式是：同机架同成本下，全 HBF 配置能拿 14× 容量（20.7 TB → 294.9 TB）但只有 0.6× 聚合带宽（1,584 → 922 TB/s），每 GPU 带宽从 22.0 降到 12.8 TB/s（第 11 页）。
- **现有方法为何不足**：以 $/GB 选择内存会误判——HBF 的 $/GB 只有 HBM 的 1/12 到 1/20（第 4 页），但在 $I \ge 70$ 的交互度下 HBM 给出更低的 $/M-token（第 12 页）。同时现有推理栈（vLLM）的溢出路径指向 host CPU pinned memory 与 SSD，而非 HBF，且缺少 MoE Expert Pool 路径（第 18 页）。
- **论文证据（本材料为模拟，按层级标注）**：
  - **规格级（可引用，但为草案）**：第 2 页 HBF Grade 1–3 带宽/容量/UCIe 速率/访问粒度；第 15 页的软件约束（64 KB 读、1 MB 写、24 h @ 85 °C 保持、DMA 访问、独立内存管理）。
  - **模型级（可复算）**：第 5 页 per-token 字节读取模型与 $I = \text{BW}/b$、$T = B·I$；第 10 页 $p_{\text{eff}}(B) = 1-(1-k/N)^B$ 与六个模型的 $k/N$。
  - **模拟级（无实测支撑）**：第 8–14 页的全部 $/token、用户数、容量与带宽数字，均出自 OxSOL 模拟器，输入参数与校准未披露。
  - **厂商宣称（需降级）**：第 2 页 "8–16× the capacity of HBM — at the same cost"。

### 2. 用了什么结构或方法？

- **整体结构与数据流**：分析结构是"规格 → (β, α) 坐标 → 工作负载建模 → 机架级模拟 → 软件接入"。数据流层面的结构是：GPU 侧保留 4× HBM + 4× HBF stack 形成 2.2 TB / 约 17.4 TB/s 的混合体（第 18 页）；HBF 承担 KV offload pool、prefix cache 与 MoE 专家池（1.45 TB），HBM 承担驻留权重、Paged KV 与激活。HBF 与 HBM 之间需要独立的内存管理，数据搬运走 DMA。
- **关键机制**：成本判据 $mem = \beta\cdot\max(C, Ib/\alpha)$（第 6 页）；per-token 字节模型 $b = [1-(1-k/N)^B]W + W_0 + BK$（第 5 页）；专家覆盖 $p_{\text{eff}}(B)$（第 10 页）；稀疏注意力判据 $\varphi \approx 1\text{–}2\% \ll \varphi^* = \alpha/I$（第 19 页）；EP × HBF 的通信节省——容量够大时可把专家留在本地 HBF，用 2 个节点替代 8 个 GPU 的 all-to-all（第 20 页）。
- **训练/量化策略**：本材料不涉及训练。模型侧设定是 Kimi-K2 1T @ **FP4**，decode-centric 工况。

### 3. 对芯片架构和 RTL 有什么启发？

- **芯片架构**：五条可操作的结论。第一，HBF 走 DMA 且不是为 GPU cache hierarchy 设计的（第 15 页），所以芯片侧要么提供一个显式的 HBF 数据搬运引擎（类似 copy engine / TMA 的角色），要么接受 HBF 数据必须先落到 HBM/SRAM 才能被 SM 使用——两种选择的数据搬运开销材料都没给。第二，最大带宽要求 64 KB 读 / 1 MB 写且 64 KB 对齐，这意味着 HBF 控制器与 DMA 引擎的突发长度、描述符粒度与地址对齐检查都需要按这个量级设计，小粒度随机访问会直接损失带宽（推测）。第三，"scratchpad SRAM 不能直接读写 NAND block"（第 15 页）是一条硬约束：FLA 到 SRAM 的通路必须经过一个中间缓冲阶段，架构上需要预留这块 buffer 及其容量（论文未给容量，`TBD`）。第四，同一机架下 HBF-only 配置的 TP 策略从 TP8·DP9 变为 TP1·DP72（第 11 页），意味着互连拓扑与集合通信库要适配更大的 DP 域、更小的 TP 域——这是对 scale-up/scale-out 互连比例的直接架构要求。第五，把 HBF 当 EP 的容量来源可以大幅削减 all-to-all（第 20 页），等价于用内存容量换互连带宽，这是机架级架构权衡里很少被摆到桌面上的一个选项。
- **RTL**：可落到实现层的项目包括：HBF 侧的 DMA 引擎与描述符处理、64 KB/1 MB 访问块的对齐与拆分逻辑、上电保持窗口的生命周期管理（材料只说要 host 管理，具体是计数/擦除调度还是别的机制，`TBD`）、HBF 与 HBM 双内存空间的地址映射与独立管理逻辑、以及寿命遥测的计数器通路。ECC 层面：NAND 的 ECC 强度与端到端数据完整性方案在本材料中完全没有涉及（`TBD`）。上述均为基于规格的工程推断，本材料**没有**给出任何 HBF 控制器 RTL、时序、面积或功耗数据。
- **推断边界**：第 1 问中 HBF 规格与软件约束来自 OCP 草案 v0.7.0（第 2、15 页），字节模型与覆盖模型属材料直接给出的公式（可复算），而所有 $/token 与用户数结论均为 OxSOL 自研模拟结果，无实测。第 2 问的结构描述来自第 6、10、18、19、20 页。第 3 问的芯片架构与 RTL 内容为工程推断；材料中不存在的具体量值——HBF 到 SRAM 的缓冲容量、DMA 引擎的开销、写寿命的具体圈数、纠错方案、controller 面积与功耗——均标 `TBD`。写寿命这一项在材料内部即自相矛盾（规格表 "Left open-ended" 对软件页 "~10-yr life"），是最大的未决风险。
