# 深度分析：《You Only Cache Once: Decoder-Decoder Architectures for Language Models》

## 基本信息

- **标题**：You Only Cache Once: Decoder-Decoder Architectures for Language Models
- **文档类型**：论文（arXiv preprint，cs.CL）
- **作者**：Yutao Sun\*†‡、Li Dong\*†、Yi Zhu†、Shaohan Huang†、Wenhui Wang†、Shuming Ma†、Quanlu Zhang†、Jianyong Wang‡、Furu Wei†⋄（\*同等贡献，⋄通讯作者）
- **机构**：Microsoft Research（†）、清华大学（‡）
- **发表 venue**：arXiv（未标注正式会议）
- **年份**：2024
- **原始来源**：[arXiv:2405.05254v2](https://arxiv.org/abs/2405.05254)，2024-05-09；代码 https://aka.ms/YOCO
- **本地原文**：`You-Only-Cache-Once-Decoder-Decoder-Architectures-for-Language-Models.pdf`（同目录，20 页）

## 一句话总结

> 论文提出 decoder-decoder 架构 YOCO，把全局 KV cache 的产生与消费解耦：前一半层（self-decoder）用高效自注意力生成唯一一份全局 KV，后一半层（cross-decoder）通过 cross-attention 复用同一份 cache，从而在保持 decoder-only 行为与竞争性精度的前提下把 KV 显存降约 L 倍、prefill 时间从 O(N²) 降为 O(N)。

## 研究动机与问题定义

- **要解决的核心问题**：长上下文 LLM 服务的内存瓶颈与 prefill 延迟瓶颈。decoder-only 的 KV cache 随 `序列长度 × 层数` 线性增长（`O(LND)`），使推理变成 memory-bounded；prefill 的注意力复杂度为 `O(LN²D)`，长输入下延迟不可接受。
- **现有方法的不足**（第 1 页）：
  - **encoder-only（BERT）与 encoder-decoder（T5）**：因双向编码，自回归生成时必须重新编码输入与已生成 token；且生成阶段无法充分利用 encoder 参数，在多轮对话场景尤其低效。
  - **decoder-only（GPT）**：靠 KV cache 避免重编码，但每层都要各自保存 N 个 K/V，显存占用随层数放大。
  - **论文给出的量化锚点**（第 2 页）：65B 模型（已用 GQA + 8-bit KV 量化）在 512K token 时 KV 占约 **86GB**，超过单张 H100-80GB；7B 模型在 4×H100 上（已用 Flash-Decoding + kernel fusion）prefill 450K 需约 **110s**，1M 需约 **380s**。
- **本文的切入角度**：不做"少算 attention"，而是做**结构性的 cache 去重**——把全局 KV 的"生产"与"消费"分配给不同层组，使全局 KV 只生成一次（论文的原话是 "only caches key-value pairs once"，脚注 2 明确说明"once"特指全局 cache，self-decoder 仍有常数级 cache）。

## 核心方法

### 方法概述

YOCO 共 `L` 层，前 `L/2` 层是 **self-decoder**，其余是 **cross-decoder**（Figure 2，第 3 页）。给定输入 `X⁰ = [x₁, …, x_|x|]`：

1. self-decoder 用**高效自注意力**（ESA）逐层得到 `X^l = Self-Decoder(X^{l-1}), l ∈ [1, L/2]`；
2. 自解码器输出 `X^{L/2}` 经线性投影生成**唯一的全局 KV cache** `K̂ = LN(X^{L/2})W_K`、`V̂ = LN(X^{L/2})W_V`；
3. cross-decoder 各层各自产生 Q，全部 cross-attend 到同一份 `K̂, V̂`：`Y^l = Attention(LN(X^l)W_Q^l, K̂, V̂) + X^l`；
4. 最后 `X^L` 经 softmax 分类器做 next-token 预测。

两层都使用**因果 mask**，因此整体对外行为等价于标准 decoder-only 自回归模型，不需要 encoder-decoder 式的双向编码，可直接复用现有 decoder-only 的预训练与推理范式。自解码器的 cache 是常数（滑动窗口的 `O(C)` 或 gated retention 的固定 size 状态 `S ∈ R^{d×d}`），cross-decoder 层共享同一份全局 cache，因此总 cache 为 `O(N + CL)`，长序列下约等于 `O(N)`——即相比 Transformer 省约 L 倍。

Figure 2 的视觉核对确认了数据流方向：KV Cache 是图中**左侧一个独立框**，由下半部分 Efficient Self-Attention 的 K/V 写入，由上半部分 Cross-Attention 的 K/V 读出；Q 在两侧各自本地生成。

### 关键技术细节

- **Self-Decoder 块结构**（式 1）：`Y^l = ESA(LN(X^l)) + X^l`，`X^{l+1} = SwiGLU(LN(Y^l)) + Y^l`，`LN` 为 RMSNorm，ESA 使用因果 mask。核心性质是 **O(1) 推理显存**（常数个 KV cache）。
- **Gated Retention（gRet，默认 ESA 选项，第 3.1 节）**：在 retention 上加入**数据相关的门控衰减** `γ = sigmoid(XW_γ)^{1/τ}`（温度项 `τ` 使 `γ` 趋向 1 以增强记忆）。三种**等价**表示：
  - **并行表示**（训练用）：`gRet(X) = (QK^⊤ ⊙ D)V`，`Q = (XW_Q) ⊙ Θ`，`K = (XW_K) ⊙ Θ`，`V = XW_V`，`Θ_n = e^{inθ}`，`D_nm = ∏_{i=m+1}^{n} γ_i`（n ≥ m，否则 0）。
  - **循环表示**（推理用，常数显存）：`S_n = γ_n S_{n-1} + K_n^⊤ V_n`，`out = Q_n S_n`。
  - **分块循环表示**（prefill / 长序列训练用）：按 chunk size `B` 切块，输出 = inner-chunk（块内并行，`(Q_[i]K_[i]^⊤ ⊙ D_[i])V_[i]`）+ cross-chunk（跨块递推，`(Q_[i]R_{i-1}) ⊙ β_[i]`），中间状态递推为 `R_i = K_[i]^⊤(V_[i] ⊙ β_[i]) + β_{iB} R_{i-1}`。Appendix B（第 16–17 页，式 9–11）给出了三种表示的等价性证明。
  - 衰减被设计为 **head-wise 而非 element-wise**，以便充分利用 NVIDIA tensor core（第 5 页明确说明）。
- **Multi-Head Gated Retention**（式 7）：逐头计算 gRet，`GroupNorm_h` 逐头归一化后拼接，再经 swish gate 与 `W_O` 输出。
- **Cross-Decoder**（式 3）：`Q̂^l = LN(X^l)W_Q^l`，`Y^l = Attention(Q̂^l, K̂, V̂) + X^l`。标准多头注意力，与 GQA 兼容（可进一步压缩 cache）。
- **Sliding-Window Attention（备选 ESA，第 3.2 节）**：窗口因果 mask `B_ij = 0 (i−C < j ≤ i)` 否则 `−∞`，窗口 `C = 1024`（第 4.2 节），cache 复杂度 `O(C)`。
- **Chunk Parallelism（Appendix A，第 16 页，Figure 11）**：长序列训练时按 chunk 切到不同 GPU。数据流经视觉核对为 `X --Split--> X₁/X₂ --> M₁/M₂ --Project--> Q,K,V --> All-Gather Once --> KV --> O₁/O₂`。self-decoder 只有相邻设备依赖（gRet 的递推 state 或滑动窗口），通信量小；cross-decoder 的 KV **只 all-gather 一次**而非每层通信，显著降低通信频率与显存碎片。
- **推理的两阶段模式**（Figure 3，第 4 页）：prefill 阶段 `Cross-Decoder` 以灰色 **(Skipped)** 状态整体跳过，只跑 self-decoder 得到 KV Cache；generation 阶段才启用蓝色 Cross-Decoder 逐 token 生成。

### 核心创新点

1. **全局 KV cache 只生成一次**：把"每层各自 cache"改为"半层生成、半层共享"，是显存复杂度的结构性下降，而非量化/驱逐类的近似压缩。Table 1/2（第 4 页）给出复杂度对比：KV cache 显存 `O(LND) → O((N+L)D)`，prefill 时间 `O(LN²D) → O(LND)`。
2. **Prefill 可提前退出且不改变最终输出**：cross-decoder 只依赖 self-decoder 输出，故 prefill 可"early exit"，至少省一半层的前向计算；由于 cross-decoder 在生成阶段会补算，最终输出与全量 prefill 一致。作者称这是"computation dependency"带来的固有红利。
3. **gated retention 的数据相关门控**：为 self-decoder 提供兼具训练并行性、常数推理显存与长程记忆能力的算子，并使 chunk-parallel 训练友好。
4. **对分布式长序列训练的结构性优势**：KV 只通信一次，降低了长序列训练的通信瓶颈。

### 与现有方法的关键区别

- **vs encoder-decoder（T5）**：形式上像"前半 encode + 后半 decode"，但 YOCO 全部层因果，可直接做 decoder-only 自回归；且不存在 encoder 参数在生成阶段被闲置的问题。
- **vs 纯高效注意力（RetNet / Mamba / Sliding-Window / Sparse Transformer）**：这些方法以"降低注意力复杂度"换效率，但弱化了全局检索能力（Table 8 中 AR-Hit 明显劣于 Transformer）；YOCO 用**廉价的 self-decoder 做局部/递推 + 昂贵的全局 cross-attention 只做一次**，保住了全局注意力（Table 4 multi-needle 的优势主要来源于此）。
- **vs GQA / KV 量化 / KV 驱逐**：正交且可叠加——论文的 Transformer 基线本身已开启 GQA + Flash-Decoding + kernel fusion。

## 实验与结果

### 实验设置

- **数据集与任务**：
  - LM Eval Harness 零样本任务：ARC-C、ARC-E、BoolQ、HellaSwag、OBQA、PIQA、Winogrande、SciQ；
  - Needle-in-a-Haystack（单针，10 次重复取平均）与 Multi-needle Retrieval；
  - book 与 repository-level code 的长序列累计平均 NLL；
  - ZeroSCROLLS 四任务（Qasper / GovReport / QMSum / NarrativeQA，附录 G.2）。
- **基线方法**：OpenLLaMA-v2-3B、StableLM-base-alpha-3B-v2、StableLM-3B-4E1T；Llama 优化版 Transformer；长上下文对比 LWM-1M-text、MiniCPM-128K、ChatGLM3-128K、YaRN-Mistral-128K；架构对比 Mamba、RetNet、Hybrid H3、gRetNet。
- **评估指标**：零样本准确率、验证 loss、检索准确率、累计平均 NLL、GPU 显存 / prefill 延迟 / 吞吐（tokens/s）。
- **YOCO-3B 训练配置**（第 4.1 节 + Table 5）：26 层、hidden 3072、FFN 8192、vocab 100,288、24 个 Q head / 8 个 KV head（GQA）、非嵌入参数 **2.83B**；序列长度 4096、batch 4M token、AdamW β=(0.9, 0.95)、峰值 lr 3.2e-4、warmup 1000 步、5T-token 的线性衰减 schedule（实际训到 1.6T）、weight decay 0.1、dropout 0.0。
- **Scaling 曲线配置**（Table 6）：160M–13B 共 7 个规模，head dim of gRet 固定 256，为对齐参数量 Transformer 的 FFN 为 `8/3·d`、YOCO 为 `3d`（原文如此表述，疑为 `(8/3)d` 与 `3d` 的对照），序列 2048、batch 0.25M token、10B token 训练量。

### 主要结果

**Table 3｜LM Eval Harness 零样本平均（第 7 页）**

| 方法 | 训练 token | ARC-C | ARC-E | OBQA | SciQ | Avg |
|---|---:|---:|---:|---:|---:|---:|
| OpenLLaMA-3B-v2 | 1T | 0.339 | 0.676 | 0.260 | 0.924 | 0.619 |
| StableLM-base-alpha-3B-v2 | 1T | 0.324 | 0.673 | 0.264 | 0.921 | 0.612 |
| **YOCO-3B** | 1T | **0.379** | **0.731** | **0.298** | 0.924 | **0.634** |
| StableLM-3B-4E1T | 1.6T | — | 0.688 | — | 0.913 | — |
| **YOCO-3B** | 1.6T | **0.396** | **0.733** | **0.300** | **0.921** | **0.636** |
| **YOCO-3B-1M**（扩到 1M 上下文） | 1.6T+ | **0.413** | **0.747** | **0.300** | **0.932** | **0.645** |

注：StableLM-3B-4E1T 的 1.6T 行为其技术报告中的中间数字，"—"表示原表未给出该任务的数值。

**Table 4｜Multi-needle 检索准确率（128K 长度，第 8 页）**

| 模型 | 规模 | N=1 | N=2 | N=4 | N=8 |
|---|---:|---:|---:|---:|---:|
| YaRN-Mistral-128K | 7B | 0.02 | 0.12 | 0.08 | 0.20 |
| LWM-1M-text | 7B | 1.00 | 0.90 | 0.76 | 0.62 |
| MiniCPM-128K | 2.4B | 1.00 | 1.00 | 0.54 | 0.56 |
| ChatGLM3-128K | 6B | 0.94 | 0.72 | 0.52 | 0.44 |
| **YOCO-3B-1M** | **3B** | 0.98 | 0.98 | **0.84** | 0.56 |

YOCO 以一半参数量逼近 7B 的 LWM-1M-text，并优于同/更大规模的其他长上下文模型。YaRN-Mistral-128K 因仅做位置插值而显著落后。

**Table 8｜160M 规模精细困惑度（Zoology 诊断集，第 19 页）**

| 方法 | Valid. | AR-Hit | First-Occur |
|---|---:|---:|---:|
| Mamba | 3.645 | 1.555 | 4.126 |
| RetNet | 3.633 | 1.466 | 4.131 |
| Hybrid H3 | 3.591 | 1.251 | 4.130 |
| gRetNet | 3.600 | 1.354 | 4.116 |
| Transformer | 3.564 | 1.219 | 4.104 |
| YOCO_SWA | 3.553 | 1.202 | 4.094 |
| **YOCO_gRet** | **3.530** | **1.199** | **4.067** |

AR-Hit 衡量联想召回能力，First-Occur 反映常规语言建模性能。YOCO_gRet 在两项上均优于所有对比架构。

**推理侧 profiling（Figure 7–10，第 10–11 页）**

硬件为 H100-80GB，模型为 3B；基准 Transformer 已开启 GQA + Flash-Decoding + kernel fusion；gated retention 在 prefill 用 chunk-recurrent 表示（chunk size 256）、生成用 recurrent 表示，并实现了 Triton kernel。评测序列长度 32K–1M，最后 1024 token 为待生成部分。

| 维度 | 结果（已逐图视觉核对） |
|---|---|
| **显存**（Fig 7a，纵轴 GPU Memory 0–120GB） | 加速比 1.95×(32K) / 2.32×(64K) / 3.01×(128K) / 4.16×(256K) / 6.39×(512K) / **9.38×(1M)**；1M 时 YOCO 总推理显存仅 **12.4GB** |
| **显存构成**（Fig 7b） | 图例 KV Cache / Weight / Other；1M 长度时两根柱均标 **9.38×** |
| **KV cache 每 token**（Fig 8，纵轴 KB/token 0–600） | 1.2B **24×**、6.4B **32×**、13B **40×**、30B **64×**、65B **80×**；"YOCO 用 1GB 可服务 128K token，而带 GQA 的 Transformer 65B 只能支持约 1.6K token" |
| **Prefill 延迟**（Fig 9，纵轴 0–300s） | 2.87×(32K) / 5.05×(64K) / 8.36×(128K) / 15.55×(256K) / **30.3×(512K，180s → <6s)** / **71.82×(1M)** |
| **吞吐**（Fig 10，纵轴 0–600 tokens/s） | 2.72×(32K) / 2.57×(64K) / 2.77×(128K) / 4.37×(256K) / **9.56×(512K，4.5 → 43.1 token/s)** |

Figure 1（第 1 页）给出了 512K 长度的汇总：GPU Memory ↓(GB) **6.4X**、Throughput ↑(wps) **9.6X**、Prefilling Latency ↓(s) **30.3X**。

**长上下文验证（第 4.3 节）**

- **Needle-in-a-Haystack**（Fig 5）：横轴 Context Length 128K–1M，纵轴 Depth 0–100%，色条 Score 0.0–1.0。视觉核对显示整张热力图近全绿，格内无具体数值，即 **1M 长度下近完美检索**。
- **长序列 NLL**（Fig 6）：横轴 Sequence Position 对数刻度 10–1M，纵轴 NLL（无数值刻度，越低越好）。book 与 repository-level code 两条曲线均随序列变长持续下降，说明模型确实利用了长距离依赖；作者称曲线近似幂律，并注明差距受验证样本噪声影响。
- **ZeroSCROLLS**（Fig 12，附录 G.2）：四子图 Qasper / GovReport / QMSum / NarrativeQA，横轴 Length 4096–16384，纵轴 PPL（各子图范围不同，如 Qasper 2.5–4.0、NarrativeQA 4.5–6.0）。YOCO_gRet 与 Transformer 在所有任务与长度上一致优于 Mamba、Sparse Transformer、Hybrid H3。

### 消融实验要点

- **self-decoder 算子选择（关键消融）**：YOCO_gRet 在 160M–13B 全区间一致优于 YOCO_SWA 与 Llama 优化 Transformer（Fig 4，横轴 #Parameters (B) 对数刻度 10⁰–10¹，三条曲线 Transformer / YOCO_SWA / YOCO_gRet）。作者归因于 attention 与 retention 的归纳偏置互补，并补充说明自己用 1:3 交织 attention/retention 也能获得类似增益，与 Jamba 等混合架构结论一致。在 Table 8 的 AR-Hit / First-Occur 细分上同样是 YOCO_gRet 最优。
- **模型规模可扩展性**：Fig 4 显示 loss 随参数量（160M→13B）下降且趋势与 Transformer 可比，说明收益不是小模型特例。
- **训练 token 可扩展性**：Table 3 中 1T 与 1.6T 两个 checkpoint 趋势一致（0.634 → 0.636），说明继续加 token 不会退化。
- **上下文可扩展性**：Fig 5 的 1M 检索近满分、Fig 6 的 NLL 单调下降。
- **复杂度对比（Table 1/2）**：KV cache 显存 `O(LND) → O((N+L)D)`，prefill 时间 `O(LN²D) → O(LND)`（N、L、D 分别为序列长度、层数、hidden 维）。
- **正交性**：论文的 Transformer 基线本身已包含 GQA、Flash-Decoding、kernel fusion，说明 YOCO 的收益与这些优化可叠加。

## 局限性与未来方向

- **作者明确提到的局限**：
  - 脚注 2 承认"only cache once"是就全局 cache 而言，**self-decoder 仍需存储常数级 cache**（记为 `O(CL)`），只是在长序列下可忽略。原文未给出该常数项开始可忽略的 crossover 点。
  - 长上下文扩展依赖**渐进式长度训练**（64K → 256K → 1M）加 RoPE θ 调整（Table 7：训练长度 65,536 / 262,144 / 1,048,576 分别对应 lr 8e-5 / 4e-5 / 2e-5、RoPE θ 640K / 5M / 80M、token 量 6B / 4B / 1.5B），与一般的长上下文扩展手段难以完全剥离。
- **作者提出的未来方向（结论第 5 节）**：
  - **YOCO + BitNet + Groq**：Groq 把权重全放 SRAM，但容量瓶颈限制模型规模与输入长度；YOCO 省 KV、BitNet 省权重，作者预期组合后部署成本可再降数量级。这属于**展望，非本文验证结果**。
  - **多模态融合**：cross-attention 天然适合多模态融合，self-decoder 的因果性契合流式视频，异步 MLLM 可避免不同数据流互相阻塞（对机器人等实时应用关键）。同样未被本文实验覆盖。
  - **KV cache 原生机制**：由于 cache 集中且复用，可以做单份压缩、单份检索索引、以及 pre-caching 支持 native RAG / LLM 原生搜索。
- **本文未覆盖 / 证据薄弱处**：
  - **无任何芯片实现数据**：全部效率结论都是 GPU（H100）侧 profiling，属于架构级代理指标；没有 post-synthesis、post-layout 或硅后功耗/面积数据。
  - **总 FLOPs 未减少**：cross-decoder 每层仍要对同一份长 KV 做全局 cross-attention，算力开销并未下降。论文只论证了显存、延迟与吞吐，没有给出总 FLOPs 或能耗口径的完整对比——对算力受限（而非显存受限）的场景，YOCO 未必是净收益。
  - **精度结论集中在 3B 与 ≤1.6T token**：13B 只出现在 10B token 的 scaling 曲线上，没有大规模下游任务评测。
  - **缺少与 KV 量化/驱逐/稀疏化的组合实验**：Table 3 基线本身带 8-bit KV 量化的说法出现在动机部分，但正文未系统评测 YOCO 与这些正交压缩技术叠加后的效果。
  - **缺多轮对话 / prefix caching 的端到端评测**：prefill early-exit 与 cross-decoder 重算对 flush 和调度的影响未量化。
  - **消融不完整**：`L/2` 这个切分比例是固定选择，论文未报告 self-decoder 与 cross-decoder 层数比例（如 1:3、3:1）的扫描结果。
  - **图 12 图例命名**：附录 Figure 12 图例包含 `YOCO_gRet`，与 Table 8 命名一致。

## 个人点评

- **亮点**：
  1. 思路极其干净——不是"减 attention"，而是"消 cache 冗余"。把一条被当作常识的成本（每层都要存 KV）直接砍掉，且**保持输出等价**，因此能与现有 kernel、GQA、量化正交叠加，工程迁移成本低。
  2. "prefill 可 early exit 且不改变输出"是个被低估的红利：它同时改善延迟与调度（可先算完廉价半层就释放计算资源），而且这个性质来自计算依赖图本身，不需要额外训练技巧。
  3. 论证链条完整：小规模 scaling law（160M–13B）→ 3B 真实训练（1.6T token）→ 1M 长度扩展 → 多维度 profiling，覆盖了从"能不能训"到"值不值得部署"的完整问题。
  4. 用 Zoology 的 AR-Hit / First-Occur 拆解对比 Mamba / RetNet / H3，比只报一个平均 PPL 有说服力得多，也直接回应了"高效注意力丢不丢检索能力"这个核心质疑。
  5. Appendix A 的 chunk parallelism 说明作者确实考虑了长序列训练的通信瓶颈，而不是只做单卡 profiling。
- **不足**：
  1. **"only cache once" 有营销成分**：self-decoder 的 state 仍是 `O(CL)`，短序列与浅层模型下收益会明显缩水；论文未给 crossover 点分析，读者容易误以为短上下文也自动受益（Fig 9 显示 32K 只有 2.87×，其实主要来自 early-exit 的 2× 而非 cache 节省）。
  2. **FLOPs 不降反升的风险未讨论**：cross-decoder 半层对同一份 1M 长 KV 反复做全局 attention，计算量并没有减少；论文没有给出 FLOPs/能耗口径的对比，也没有给出"显存受限 vs 算力受限"的适用边界。
  3. **精度优势的可解释性偏弱**：YOCO_gRet 优于 Transformer，作者归因于"attention 与 retention 归纳偏置互补"，但同时也承认 1:3 交织 attention/retention 有类似效果——这意味着部分增益可能来自混合架构本身，而非 decoder-decoder 结构。缺少"纯 SWA 版 YOCO vs 纯 SWA Transformer"的干净对照来分离两个因素（Table 8 中 YOCO_SWA 3.553 vs Transformer 3.564 的差距确实很小，这一点部分支持了我的怀疑）。
  4. **缺少与 KV 压缩方法的组合实验**，而这类方法在工程落地中往往先行。
  5. **3B / 1.6T token 的规模**与当前主流（数十 B、数十 T token）有差距，61.9%–64.5% 的零样本平均分处在同一档位内，精度结论的说服力有限——**本文真正的贡献在效率侧而非精度侧**。
- **启发**：对硬件设计者而言，本文最有价值的不是"又一个高效注意力变体"，而是**把 KV cache 从"每层分散"变成"单点集中"**——这直接改变了 memory hierarchy 的设计约束。集中式 cache 意味着可以做单份压缩硬件、单份 K-V 检索索引、单份 KM（key-match）加速器，而不需要每层复制一套控制逻辑。这是一条很适合往芯片方向推进的线索。

## 工程化三问总结

### 1. 它解决了什么瓶颈？

- **应用场景与核心瓶颈**：长上下文 LLM serving（长文档问答、仓库级代码、1M 检索）。瓶颈是 KV cache 的显存占用（decoder-only 为 `O(LND)`）与 prefill 的 `O(LN²D)` 时间。论文给出的具体锚点：65B 模型 512K token 需约 86GB KV（超过 H100-80GB 容量），7B 模型在 4×H100 上 prefill 1M 需约 380s。
- **现有方法为何不足**：encoder-decoder 需要重编码且生成阶段闲置 encoder 参数；纯高效注意力（Mamba / RetNet / SWA / Sparse Transformer）牺牲全局检索能力（Table 8 的 AR-Hit 明显更差，Mamba 1.555 / RetNet 1.466 vs Transformer 1.219）；KV 量化与驱逐是近似压缩，且仍需逐层维护 cache 结构。
- **论文用什么证据证明问题得到缓解**：显存 9.38×（1M，3B 模型，仅占 12.4GB）、KV 每 token 最高 **80×**（65B，Fig 8）、prefill 71.82×（1M）、吞吐 9.56×（512K）；同时精度在 1T / 1.6T token 下与 StableLM / OpenLLaMA 持平或更好（Avg 0.634 / 0.636 vs 0.619 / 0.612），1M 长度 needle 检索近满分（Fig 5），multi-needle 用 3B 逼近 7B 的 LWM（Table 4）。以上均为**论文证据**；跨平台/跨芯片的收益属**推断**。
- **限定条件**：32K 时显存只省约 2×、prefill 加速 2.87×（其中约 2× 来自 early-exit 的结构性收益，而非 cache 节省），说明短上下文下收益明显缩小，这是论文证据本身给出的边界。

### 2. 用了什么结构或训练方法？

- **整体结构与数据流**：`X⁰ →(L/2 层 self-decoder，高效自注意力)→ X^{L/2} →(线性投影一次)→ K̂,V̂ →(L/2 层 cross-decoder，共享 K̂,V̂)→ X^L → softmax`。推理时 prefill 阶段跳过全部 cross-decoder（Figure 3 中的 "Cross-Decoder (Skipped)"），generation 阶段逐 token 补算；训练时用 chunk parallelism，KV 只 all-gather 一次（Figure 11）。
- **关键模块/结构**：gated retention（三种等价表示：parallel / recurrent / chunkwise recurrent，Appendix B 有完整等价性证明；门控为 head-wise 以便走 tensor core）、滑动窗口 attention（C=1024）、GroupNorm 逐头归一化的多头融合 + swish gate、pre-RMSNorm + SwiGLU、与 GQA 兼容的 cross-attention、Triton 实现的 gRet kernel（基于 FLA）。
- **训练目标、损失函数或优化方法**：标准 **next-token prediction 的 softmax 交叉熵**——架构改动不引入任何新损失项，这是该工作工程可迁移性强的重要原因。优化器 AdamW β=(0.9, 0.95)（3B）/ (0.9, 0.98)（scaling），lr 3.2e-4，warmup 1000（3B）/ 375（scaling）步，线性衰减，weight decay 0.1 / 0.05。
- **数据与训练策略**：与 StableLM-3B-4E1T 同源的 curated corpus，tokenizer 为 tiktoken-cl100k_base；3B 用 batch 4M token、序列 4096、5T-token schedule 实跑 1.6T；scaling 曲线用 batch 0.25M token、序列 2048、10B token；长上下文扩展按 64K → 256K → 1M 递进，逐级降 lr（8e-5 / 4e-5 / 2e-5）并调大 RoPE θ（640K / 5M / 80M），训练数据**按序列长度上采样**，且为公平对比**不使用长指令微调数据**。

### 3. 对芯片架构、RTL、验证有什么启发？

- **芯片架构**：
  - **KV cache 集中化改变 memory hierarchy 假设**：全局 KV 只有一份，且物理位置固定在 self-decoder 与 cross-decoder 的交界。这适合做成**单一集中式 on-chip SRAM 或近存 cache**，而不是每层私有的 KV 切片。对片上存储规划的影响是实质性的：总容量需求从 `O(L·N)` 降到 `O(N)`，地址生成与控制逻辑也从 L 套降为 1 套。
  - **访问模式变化**：cross-decoder 的半层层**对同一份 KV 反复读取**，是典型的**广播/多播型带宽受限**负载，可用共享总线或多播互连替代点对点读，降低互连压力。同时 self-decoder 的 K/V 是写入侧、Q 是本地生成，写少读多的不对称性有利于简化仲裁。
  - **算子选型**：gated retention 的递推 `S ← γS + K^⊤V` 是**固定尺寸状态 + 逐元素标量缩放**，与脉动阵列 / 权重驻留数据流的契合度高于 softmax attention（无 softmax、无动态序列长度的归约）。但 `γ` 是数据相关的 head 级标量，需要一条低精度指数/累乘通路（论文用 `logsigmoid + cumsum`）。
  - **算力换取显存**：论文明确承认 FLOPs 未减少，因此**算力受限芯片（而非显存/带宽受限芯片）不会自动受益**。这是选型时最关键的一条边界。
  - **可扩展性**：KV 只 all-gather 一次意味着跨 die / 跨 GPU 的通信模式从"每层一次"变成"一次"，对 chiplet 或 scale-up 互连的带宽预算有直接好处。
  - `TBD / 推测`：论文没有任何面积、功耗、SRAM 容量建议，也没有给出集中式 KV 的命中率或 bank 冲突模型。上述收益取决于 kernel 侧容量与访问模型，属**工程推断**而非论文结论。
- **RTL**：
  - **可直接映射的模块**：`Project`（一次 K/V 投影，可参数化 head 数与非对称 GQA 分组比 24:8）、`RecurrentRetention`（`S` 寄存器阵列 + `γ` 缩放 + 外积累加，是天然的状态机/累加器结构）、`ChunkwiseRetention`（块内并行单元 + 跨块 `R` 递推 + 块间串行）、`GroupNorm`（逐 head 归一化，需要 per-head 统计与倒数）、`SwiGLU` / `RMSNorm` 标准件。
  - **双模式接口**：必须区分 **prefill（chunkwise，块内并行、高吞吐）** 与 **generation（recurrent，batch=1、低延迟）** 两条路径，是典型的多模式 FSM + 模式选择寄存器设计。此外 prefill 需要一条"跳过 cross-decoder"的控制路径（Figure 3 的 Skipped 状态），这是 YOCO 特有的控制复杂度——本质上是把"层数减半"做成可运行时切换的行为。
  - **数值与时序风险**：`γ` 的累乘（`β`、`D`、`R` 的递推）在长 chunk 上极易下溢，定点化需要**对数域递推或分段重缩放**；论文的 `log/exp` 路径对精度敏感，是定点化的重点与风险点。窗口 `C`、chunk 大小 `B`（论文取 256）、head dim（gRet 取 256）都是可参数化设计旋钮。
  - `TBD`：论文未涉及 RTL、流水线级数、时钟频率与面积评估，以上为**基于算法结构的映射推断**。
- **验证**：
  - **三种表示的数值等价性**是最核心的功能性质：`parallel ↔ recurrent ↔ chunkwise` 必须给出相同（或容差内一致）的结果。应建立 golden model（PyTorch parallel 实现作参考）做逐 token 比对。这正是 Appendix B 证明的内容，也是实现中最易出错处。
  - **KV cache 一致性**：cross-decoder 所有层共享同一份 `K̂, V̂`，需验证"全局 cache 只写一次、读 L/2 次"的一致性；并覆盖 **prefill early-exit 路径与完整 prefill 路径输出完全相同**这一核心声明（论文的关键卖点，也是回归测试的第一优先级）。
  - **边界条件**：`N=1`、`N < chunk size`、`N` 非 chunk 整数倍、窗口边界（`i−C < j ≤ i`）、mask 的 `j ≤ k` 与 `j < m` 分支、`γ → 1` 的温度极端值、超长序列（1M）下 cache 地址空间与计数器溢位、GQA 分组非整除情形。
  - **随机约束与参考模型**：以 parallel 实现为参考模型，对 recurrent / chunkwise 硬件通路做 constrained-random 对比；对 GQA 分组数、head 数、GroupNorm 统计做参数化覆盖。
  - **性能指标与软硬件一致性**：除延迟/吞吐外，应单独验证 **KV 显存占用随 N 的斜率是否真的接近 `O(N)` 而非 `O(NL)`**；用 32K 与 1M 两个点即可暴露"是否只在单边长上下文才有效"的回归行为（对应 Fig 7a 的 1.95× → 9.38× 曲线）。
  - `TBD`：论文没有验证方法学章节，以上验证计划为**基于方法结构的工程推断**。
