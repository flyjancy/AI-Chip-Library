# 深度分析：《You Only Cache Once: Decoder-Decoder Architectures for Language Models》

## 基本信息

- **标题**：You Only Cache Once: Decoder-Decoder Architectures for Language Models
- **文档类型**：论文（arXiv 预印本，20 页）
- **作者**：Yutao Sun（共同一作）、Li Dong（共同一作）、Yi Zhu、Shaohan Huang、Wenhui Wang、Shuming Ma、Quanlu Zhang、Jianyong Wang、Furu Wei（通信作者）
- **机构**：Microsoft Research、Tsinghua University
- **发表 venue**：arXiv:2405.05254v2 [cs.CL]（2024-05-09 的 v2）
- **年份**：2024
- **链接**：https://aka.ms/GeneralAI、https://aka.ms/YOCO

## 一句话总结

> YOCO 把 decoder-only 模型拆成「前半 self-decoder（用高效注意力）+ 后半 cross-decoder（对同一份全局 KV 做 cross-attention）」，从而**只缓存一次 KV**：KV 缓存复杂度从 $O(LND)$ 降到 $O((N+L)D)$、prefill 复杂度从 $O(LN^2D)$ 降到 $O(LND)$，在 H100 上 1M 上下文时显存降到 1/9.4、prefill 从约 300 s 降到约 4 s（71.8×）、512K 时吞吐提升 9.6×，代价是 3B 模型上 1.6T token 训练后的下游平均分 0.636 对同期 Transformer 基线的 0.619–0.634。

## 研究动机与问题定义

- **要解决的核心问题**：KV 缓存既是推理容量的瓶颈，也是 prefill 延迟的根源。缓存大小随序列长度与层数同时增长（Transformer 为 $O(LND)$），而 prefill 的自注意力复杂度是 $O(LN^2D)$。论文用一个具体数字锚定成本：**512K 上下文时 Transformer 的 prefill 需要约 180 秒**。
- **现有几条路线的不足**（第 1 页）：
  - **encoder-only**（BERT）：双向编码，自回归生成时每一步都要重新编码整个输入与输出序列。
  - **encoder-decoder**（T5）：双向编码器 + 单向解码器，但解码器生成时**没有充分利用编码器参数**，多轮对话场景尤其明显。
  - **decoder-only**（GPT）：靠缓存 KV 避免重编码历史，是当前标准做法，但缓存量随层数线性增长。
- **切入角度**：既然 KV 缓存的冗余在于**每一层都存一份自己的 KV**，那就让所有层**共享同一份全局 KV**——用 cross-attention 让后半部分层去读前半部分层产出的那一份。同时把前半部分层换成"推理内存为常数"的高效注意力（sliding-window 或 gated retention），使这份全局 KV 可以一次性算完、只存一份。

## 核心方法

### 架构结构（第 3–4 页）

YOCO 共 $L$ 层，**前 $L/2$ 层是 self-decoder，后 $L/2$ 层是 cross-decoder**：

- **Self-decoder**：输入 $X^0$，用高效自注意力（ESA）与 SwiGLU 逐层计算
  $$Y^l = \text{ESA}(\text{LN}(X^l)) + X^l, \qquad X^{l+1} = \text{SwiGLU}(\text{LN}(Y^l)) + Y^l$$
  带 causal mask。**关键性质是高效自注意力的推理内存为 $O(1)$，即 KV 缓存数量是常数**——例如 sliding-window attention 的缓存只取决于窗口大小而与输入长度无关。
- **Cross-decoder**：用 self-decoder 的输出 $X^{L/2}$ 生成**唯一一份全局 KV**：
  $$\hat K = \text{LN}(X^{L/2})W_K, \qquad \hat V = \text{LN}(X^{L/2})W_V$$
  然后后 $L/2$ 层各自用 query 去 attend 这一份 KV：
  $$\hat Q^l = \text{LN}(X^l)W_Q^l, \qquad Y^l = \text{Attention}(\hat Q^l, \hat K, \hat V) + X^l$$
  同样带 causal mask。cross-attention 与 grouped-query attention（GQA）兼容，可进一步压缩 KV 内存。
- 两种 decoder 块的其余布局与 Transformer 一致：pre-RMSNorm、SwiGLU、GQA。

### 两个复杂度结论（Table 1、Table 2）

| | Transformer | YOCO |
| --- | --- | --- |
| KV 缓存内存复杂度 | $O(LND)$ | $O((N+L)D)$ |
| Prefill 注意力时间复杂度 | $O(LN^2D)$ | $O(LND)$ |

（$N$ 为序列长度，$L$ 为层数，$D$ 为隐藏维度。）由于 $CL \ll N$，需要的缓存数约为 $O(N)$——**"you only cache once"**。相对 Transformer，YOCO 大约省 $L$ 倍的缓存显存。

### 推理优势的两条机制（第 4–5 页）

1. **省显存、服务更多 token**：推理容量瓶颈从权重变为 KV 缓存（图 7b 的分解显示随上下文增长 KV 缓存成为主项），减少缓存后可以增大 batch，进而提升吞吐。
2. **Prefill 提前退出**：由于 cross-decoder 复用 self-decoder 的输出，**prefill 阶段可以在进入 cross-decoder 之前提前退出**（图 3 标注 "Cross-Decoder (Skipped)"）。论文据此给出两层收益：前向计算只需一半层数（至少减半 prefill 延迟）；self-decoder 的高效注意力本身也快。

### Self-decoder 的设计选择（第 5–6 页）

只要模块的推理内存为常数即可用。论文实验了两种：

- **Gated retention**（记为 $\text{YOCO}_{\text{gRet}}$）：prefill 阶段用 chunk-recurrent 表示，生成阶段用 recurrent 表示，**chunk size 设为 256**，作者用 Triton 实现了 kernel。
- **Sliding-window attention**（$\text{YOCO}_{\text{SWA}}$）：缩放实验中的窗口大小为 1024。

缩放实验的结论是 **$\text{YOCO}_{\text{gRet}}$ 优于 Transformer 与 $\text{YOCO}_{\text{SWA}}$**，作者归因于"注意力与 retention 的混合架构，两者的归纳偏置互补"，并补充说以 1:3 交错注意力与 retention 模块也能获得类似收益。

## 实验与结果

### 实验设置

- **3B 主实验**：跟随 StableLM-3B-4E1T 的训练配方；隐藏维度 3072、26 层、head dim 128（StableLM 为 80，改为 128 是为了更好的 kernel 支持）、GQA（24 个 query head / 8 个 KV head）、使用 gated retention。**非嵌入参数量 2.8B**（StableLM-3B-4E1T 为 2.7B，OpenLLaMA-v2-3B 为 3.2B）。序列长度 4096，batch 4M token，AdamW（$\beta = 0.9, 0.95$），最大学习率 3.2e-4（1000 步 warmup，线性衰减到 1.28e-5），总调度 5T token，实际训练 400k 步即 **1.6T token**。tokenizer 为 `tiktoken-cl100k_base`。
- **缩放实验**：160M、400M、830M、1.4B、2.7B、6.8B、13B，用相同数据与设置训练，**batch 0.25M token、序列长度 2k、共 40k 步即 10B token**，用验证损失作指标并拟合 scaling law。
- **长上下文**：把 YOCO-3B 逐级扩到 64K → 256K → 1M，batch 保持不变，按序列长度上采样训练数据，**不使用长指令微调数据**。
- **推理测速**：H100-80GB，序列长度 32K 到 1M，假设给定上下文后生成最后 1,024 个 token；对照的 Transformer **使用 GQA + Flash-Decoding + kernel fusion**（论文称这是为了公平比较）。

### 主要结果

**3B 模型的下游任务（Table 3，LM Eval Harness 零样本）**

| 模型 | ARC-C | ARC-E | BoolQ | Hellaswag | OBQA | PIQA | Winogrande | SciQ | **平均** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1T token 训练** | | | | | | | | | |
| OpenLLaMA-3B-v2 | 0.339 | 0.676 | 0.657 | 0.700 | 0.260 | 0.767 | 0.629 | 0.924 | 0.619 |
| StableLM-base-alpha-3B-v2 | 0.324 | 0.673 | 0.646 | 0.686 | 0.264 | 0.760 | 0.621 | 0.921 | 0.612 |
| **YOCO-3B** | 0.379 | 0.731 | 0.645 | 0.689 | 0.298 | 0.763 | 0.639 | 0.924 | **0.634** |
| **1.6T token 训练** | | | | | | | | | |
| StableLM-3B-4E1T | — | 0.688 | — | — | — | 0.762 | 0.627 | 0.913 | — |
| **YOCO-3B** | 0.396 | 0.733 | 0.644 | 0.698 | 0.300 | 0.764 | 0.631 | 0.921 | **0.636** |
| **扩到 1M 上下文** | | | | | | | | | |
| **YOCO-3B-1M** | 0.413 | 0.747 | 0.638 | 0.705 | 0.300 | 0.773 | 0.651 | 0.932 | **0.645** |

论文的表述是 "comparable results with previous well-tuned Transformer language models"。1T token 时 YOCO 平均分 0.634 高于两个对照（0.619/0.612），但对照的 StableLM-3B-4E1T 在该行缺多项数据（原文以 "—" 表示），可比性有限。

**模型规模缩放（图 4）**

160M 到 13B 的验证损失与 Llama 式 Transformer 相当，$\text{YOCO}_{\text{gRet}}$ 优于 Transformer 与 $\text{YOCO}_{\text{SWA}}$。**需要强调训练预算很小：仅 10B token**。

**长上下文（图 5、图 6、Table 4）**

- **Needle-In-A-Haystack（1M 上下文）**：YOCO-3B-1M **接近满分**（图 5 的分数矩阵）。评估设置跟随 Gemini 1.5 与 LWM，同一深度与长度跑 10 次取平均。
- **多针检索（128K，Table 4）**：

| 模型 | 规模 | N=1 | N=2 | N=4 | N=8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| YaRN-Mistral-128K | 7B | 0.02 | 0.12 | 0.08 | 0.20 |
| LWM-1M-text | 7B | 1.00 | 0.90 | 0.76 | 0.62 |
| MiniCPM-128K | 2.4B | 1.00 | 1.00 | 0.54 | 0.56 |
| ChatGLM3-128K | 6B | 0.94 | 0.72 | 0.52 | 0.44 |
| **YOCO-3B-1M** | 3B | 0.98 | 0.98 | **0.84** | 0.56 |

  论文的结论是：YOCO-3B-1M 用**一半的模型规模**达到与 LWM-1M-text（7B，由 Llama-2-7B 继续训练）相当的水平，并超过 MiniCPM-128K 与 ChatGLM3-128K。注意 N=8 时 YOCO 的 0.56 与 MiniCPM 相同、低于 LWM 的 0.62。
- **长序列困惑度**：在 book 与 repository-level code 数据上，累积平均 NLL 随上下文长度持续下降，曲线接近幂律。

**推理剖析（图 7–10，H100-80GB）**

| 指标 | 结果 |
| --- | --- |
| 1M 上下文推理显存 | YOCO **12.4 GB**，Transformer 为 **9.4×**（图 7a 各长度下的放大倍数为 32K 1.95×/2.32×、128K 3.01×、256K 4.16×、512K 6.39×、1M 9.38×） |
| 32K 上下文显存 | YOCO 约省 **2×** |
| 每 token 的 KV 显存 | 65B 模型下，**1 GB 显存 YOCO 可服务 128K token，而带 GQA 的 Transformer 只能服务 1.6K token**（约 **80×**）；模型越大节省越多（1.2B 24×、6.4B 32×、13B 40×、30B 64×、65B 80×） |
| Prefill 延迟 | 512K：约 180 s → **<6 s**；1M：约 300 s → **71.82×** 加速；各长度加速比 32K 2.87×、256K 15.55×、512K 30.3×、1M 71.82× |
| 吞吐 | 512K 时 Transformer **4.5 token/s** vs YOCO **43.1 token/s**，即 **9.6×**；各长度 2.57×（64K）、2.72×（32K）、2.77×（128K）、4.37×（256K） |

论文对吞吐提升的两条解释：prefill 时间减少；显存降低后可以用更大的 batch。

**图 1 摘要给出的 512K 口径**：显存 9.6×、吞吐 6.4×、prefill 延迟 30.3×。注意图 1 的显存 9.6× 与图 7 的 9.38×（1M）不是同一长度，图 1 的标注为 "@512k" 但数值与图 7/9/10 在 512K 下的 6.39×/30.3×/9.56× 不一致，引用时需以正文图为准。

### 消融与设计选择

- **self-decoder 模块选择**：$\text{YOCO}_{\text{gRet}}$ > Transformer > $\text{YOCO}_{\text{SWA}}$（图 4，160M–13B，10B token）。窗口大小 1024。
- **混合注意力与 retention**：以 1:3 交错两种模块也有类似收益，作者引用同期 hybrid 架构工作佐证。
- **chunk size**：gated retention 的 chunk size 设为 256。

## 局限性与未来方向

- **缩放实验的训练预算偏小**：160M–13B 的缩放曲线只用了 **10B token**（40k 步 × 0.25M）。在这个预算下拟合出的 scaling law 与结论，与当代动辄万亿 token 的实践距离较大；论文用"scaling laws can be well-fitted"作为依据，但未给拟合的外推误差。
- **3B 主实验的对照组数据不完整**：Table 3 中 1.6T token 行的 StableLM-3B-4E1T 有 5 项为 "—"（论文取自其技术报告），因此这一行不是完整对照；1T 行 YOCO 的 0.634 对 0.619/0.612 是完整可比的，但优势幅度不大，且三项（BoolQ、Hellaswag、PIQA）实际低于 OpenLLaMA-3B-v2。
- **"prefill 提前退出不改变最终输出"这一说法需要限定**：KV 缓存由 self-decoder 的输出 $X^{L/2}$ 完全决定，所以跳过 cross-decoder 不影响缓存本身；但生成第一个 token 仍需要 cross-decoder 的末层隐藏状态。论文与图 3 的表述容易让人理解为 prefill 完全不需要 cross-decoder。这个机制本身是成立的，但"至少减半 prefill 延迟"的收益中，有一部分会以第一个 token 的首次解码步（单 query 位置、代价小）形式出现。
- **推理测速的对照口径需注意**：对照的 Transformer 已经用了 GQA + Flash-Decoding + kernel fusion（论文称是为公平比较），但 **YOCO 侧用的是为 gated retention 专门写的 Triton kernel**，且 chunk size 固定为 256。两侧的工程优化程度是否完全对等，论文没有给出逐项拆解。
- **摘要图的数字与正文图不一致**：图 1 的 "@512k" 标注下给出显存 9.6×、吞吐 6.4×、prefill 30.3×，而正文图 7 在 512K 是 6.39×（显存）、图 10 在 512K 是 9.56×（吞吐）。两组数字疑似互换或标注错位，引用时应以正文图为准。
- **长上下文评估的口径**：多针检索只在 128K 下做（因为多数对照模型按此长度调优），而 1M 只做了单针与 NLL。1M 的"N 针"能力没有直接证据。
- **未来方向（论文明确给出的三条，其中第一条直接指向硬件）**：
  1. **YOCO + BitNet + Groq**：Groq 把所有东西放进 SRAM 获得极高吞吐，但内存容量瓶颈限制了模型规模与输入 token 数，需要几百颗芯片才能承载一个模型。**YOCO 压缩 KV 缓存、BitNet 压缩权重，两者组合有望把 LLM 部署成本降低数个数量级。**
  2. **YOCO 用于多模态 LLM**：布局天然支持多个 self-decoder，cross-attention 层适合多模态融合；self-decoder 的因果依赖适合流式视频；异步多模态可以避免不同数据流互相阻塞（对机器人等实时应用关键）。
  3. **KV 缓存模块的专门机制**：图 2 把 KV 缓存显式画出来，为原生内存机制留出空间——可以集成缓存压缩机制；可以为键值检索建索引（**因为缓存被复用，只需维护一份索引而不是每层一份**）；解耦建模支持预先缓存上下文，对原生 RAG 与 LLM 原生搜索引擎有用。

## 个人点评

- **这篇工作的核心洞察很简洁：KV 缓存的冗余不在"缓存"这件事本身，而在"每层都缓存一份"**。Transformer 存 $N \times L$ 份 KV，YOCO 只存一份（加上 self-decoder 的常数窗口缓存），因此直接省 $L$ 倍显存。这个视角比"用什么高效注意力压 KV"更彻底——sliding window、retention 之类的方案是在压单层的缓存长度，而 YOCO 是在压层数这个维度。用 cross-attention 让后半个模型读同一份 KV，代价是后半个模型的注意力表达力被限制在同一组 $\hat K, \hat V$ 上，论文用实验证明这个代价在 3B 与 13B 规模下可以接受。
- **Prefill 的复杂度降低是附带但很有价值的结果**。因为 KV 只来自 self-decoder，prefill 的注意力从 $O(LN^2D)$ 变成 $O(LND)$，1M 上下文下 prefill 从约 300 秒降到约 4 秒（71.8×）、512K 从 180 秒降到 6 秒以内。同时吞吐在 512K 从 4.5 提升到 43.1 token/s。这些数字是在 H100 上实测的，且对照的 Transformer 已用了 Flash-Decoding 与 kernel fusion，因此量级可信。
- **最值得记下的是结论里那条硬件建议**。作者把 YOCO 与 BitNet（权重压缩）和 Groq（全 SRAM 架构）放在一起：Groq 的瓶颈是内存容量不足以承载大模型与大输入，需要几百颗芯片供一个模型；**YOCO 压 KV、BitNet 压权重，组合起来有望把部署成本降低数个数量级**。这条把算法侧的两种压缩与我们仓库里另一篇 Groq TSP 论文的容量短板直接对上了——Groq TSP 每芯片只有 220 MiB SRAM，全局内存是物理分布的片上 SRAM，容量正是它的约束。这是一处少见的"算法论文主动指向具体硬件架构"的表述。
- **需要打折扣的地方有三处**。第一，缩放实验只用了 10B token，与动辄万亿 token 的实践差距很大，"scaling law 拟合良好"这个论断的证据强度不高。第二，Table 3 中 1.6T 行的对照数据有 5 项缺失，1T 行的优势（0.634 vs 0.619/0.612）幅度小且在 BoolQ/Hellaswag/PIQA 三项上落后。第三，图 1 与正文图的数字对不上（图 1 标 *@512k* 却给出 9.6× 显存，而图 7 在 512K 是 6.39×），这类标注错误虽不影响主结论，但说明图表没做交叉核对。
- **一个工程上值得注意的细节**：GQA 与 cross-attention 兼容，且因为缓存只有一份，**全模型共用一个检索索引**（未来方向第三条）。这意味着如果要做 KV 的检索、压缩或异位存储（比如本仓库 HBF 那篇讨论的 KV offload 池），YOCO 结构让这些机制只需实现一次而不是每层实现一次——这是架构层面少见的"压缩即简化"效应。

## 工程化三问总结

### 1. 它解决了什么瓶颈？

- **应用场景与核心瓶颈**：长上下文 LLM 的推理部署。两个瓶颈都被量化：**KV 缓存显存**（Transformer 为 $O(LND)$，随长度与层数同时增长，成为推理容量瓶颈；512K 上下文的 prefill 需要约 180 s，1M 约 300 s）与 **prefill 计算**（自注意力 $O(LN^2D)$）。实测锚点：1M 上下文时 Transformer 的推理显存是 YOCO 的 9.4×；65B 模型下 1 GB 显存 YOCO 能服务 128K token，带 GQA 的 Transformer 只能服务 1.6K token（约 80×）。
- **现有方法为何不足**：encoder-only 与 encoder-decoder 在自回归生成上都要重复编码或无法复用编码器参数；decoder-only 靠缓存 KV 解决了重编码问题，但缓存量随层数线性增长。
- **论文证据的分层**：
  - **可核算的复杂度**：Table 1（KV 内存 $O(LND)$ → $O((N+L)D)$）与 Table 2（prefill $O(LN^2D)$ → $O(LND)$）是结构性的，不依赖测量。
  - **H100-80GB 实测**：显存突破（图 7，1M 时 12.4 GB vs 9.4×）、per-token KV 显存（图 8，65B 时 80×）、prefill 延迟（图 9，32K 2.87× 到 1M 71.82×）、吞吐（图 10，512K 时 4.5 → 43.1 token/s，9.6×）。这些是在明确硬件与明确对照配置下测得的。
  - **需要打折的**：图 1 的 "@512k" 数字（9.6×/6.4×/30.3×）与正文图 7/10 在 512K 下的 6.39×/9.56× 不一致；3B 模型的下游对比在 1.6T 行有 5 项对照缺失；160M–13B 的缩放曲线只有 10B token 预算；多针检索只在 128K 上评估。

### 2. 用了什么结构或训练方法？

- **整体结构**：$L$ 层拆成前 $L/2$ 层 self-decoder + 后 $L/2$ 层 cross-decoder。self-decoder 用高效自注意力（gated retention 或 sliding-window attention），其推理内存为 $O(1)$；cross-decoder 用 GQA 兼容的 cross-attention 去读由 $X^{L/2}$ 生成的**唯一一份**全局 $\hat K, \hat V$。块内布局沿用 Transformer（pre-RMSNorm、SwiGLU、GQA）。
- **关键机制**：
  1. **KV 只生成一次**：$\hat K = \text{LN}(X^{L/2})W_K$、$\hat V = \text{LN}(X^{L/2})W_V$，被全部 $L/2$ 个 cross-decoder 层复用。
  2. **Prefill 提前退出**：KV 缓存由 self-decoder 决定，因此 prefill 可跳过 cross-decoder。
  3. **Gated retention 的 chunk-recurrent 表示**（prefill）与 recurrent 表示（生成），chunk size 256，Triton kernel 实现。
- **训练与数据策略**：3B 模型跟随 StableLM-3B-4E1T 配方（26 层、hidden 3072、head dim 128、GQA 24/8、AdamW、lr 3.2e-4 线性衰减到 1.28e-5、400k 步 ≈ 1.6T token、序列长度 4096、batch 4M token）。缩放实验 160M–13B、10B token、2k 序列、0.25M batch。长上下文按 64K → 256K → 1M 逐级扩展，数据按长度上采样，不使用长指令微调数据。论文另提出用于 1M 训练的 chunk parallelism 算法以降低通信开销与显存碎片。

### 3. 对芯片架构和 RTL 有什么启发？

- **芯片架构**：四条。第一，**KV 缓存的压缩是算法层面对显存的直接释放，会改变加速器的容量-带宽权衡点**。65B 模型下每 GB 显存的服务 token 数提升约 80×，这意味着在同一片 HBM 容量下可以支撑的上下文长度或 batch 数大幅增加；反过来，如果加速器的容量本来就紧张（如全 SRAM 架构），这一压缩直接决定模型能否装下。第二，**论文自己给出的组合路线值得作为设计输入**：YOCO（压 KV）+ BitNet（压权重）+ Groq（全 SRAM、极高吞吐但容量受限）。Groq TSP 每芯片只有 220 MiB SRAM、全局内存是物理分布的片上 SRAM，容量是硬约束；YOCO 与权重压缩同时减小"必须放下的字节"，这是把两类压缩算法与具体内存架构（SRAM-only vs HBM vs HBF/NAND 分级）配对的一个具体案例。第三，**prefill 与 decode 的瓶颈在新结构下分离得更清楚**：prefill 从 $O(LN^2D)$ 降到 $O(LND)$ 且可跳过一半层，而 decode 阶段每步只需读一份 KV——这对加速器的双阶段调度（prefill 用大算力、decode 用大带宽）是有利的，但它同时意味着**加速器如果按"每层一份 KV"来规划片上 buffer，会出现严重过配**（推测）。第四，**跨层共享 KV 使"一份数据被多个消费者读"成为常态**：$L/2$ 个 cross-decoder 层读同一份 $\hat K, \hat V$，在片上 SRAM 组织上这是一个天然的多播/共享缓冲场景，而不是每层私有的 buffer（此为工程推断，论文未涉及硬件实现）。
- **RTL**：论文是纯算法/建模工作，**没有任何硬件或 RTL 数据**。可推断的实现含义包括：cross-attention 的 K/V 读取路径需要支持 $L/2$ 个消费者共享同一 buffer（多播或分级共享）；self-decoder 的高效注意力（sliding window 或 gated retention）需要常数大小的环形缓冲与 chunk 级的部分和累加（chunk size 256 是一个可用的参考值）；prefill 阶段"跳过 cross-decoder"需要一组可旁路的数据通路与节拍控制；以及论文提到的 chunk parallelism（用于 1M 训练，涉及跨设备的 chunk 划分与通信）。上述均为工程推断，论文未给出任何循环次数、面积、功耗或时序数据。若要在 RTL 中落地，需要补的量值包括：共享 KV buffer 的带宽需求与端口数、多播网络结构、chunk-recurrent 累加器的位宽与精度、以及旁路路径对时序的影响，均 `TBD`。
- **推断边界**：第 1 问的复杂度结论来自 Table 1/2（结构性论证），性能数字来自 H100 实测（但有图 1 与正文图不一致、对照配置不完全对等两点需注意）；第 2 问的结构与机制为论文直接内容。第 3 问的芯片架构与 RTL 内容为工程推断，其中"YOCO + BitNet + Groq"是论文自身的表述，其余为实现含义的推断。模型在更大规模（>13B）与更大训练预算下的表现、1M 上下文的多针检索能力、以及共享 KV 在真实加速器上的带宽开销，均 `TBD`。
