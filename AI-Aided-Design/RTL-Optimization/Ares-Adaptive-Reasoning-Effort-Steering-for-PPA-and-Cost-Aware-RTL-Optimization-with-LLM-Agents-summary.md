# 深度分析：《Ares: Adaptive Reasoning-Effort Steering for PPA- and Cost-Aware RTL Optimization with LLM Agents》

## 基本信息

- **标题**：Ares: Adaptive Reasoning-Effort Steering for PPA- and Cost-Aware RTL Optimization with LLM Agents
- **文档类型**：论文（预印本）
- **作者**：Stef Cuyckens、Mihaela Jivanescu、Jun Yin、Chao Fang（通讯作者）、Marian Verhelst
- **机构**：KU Leuven、Nokia Bell Labs
- **发表 venue**：arXiv preprint（arXiv:2607.27879v1 [cs.AR]，2026-07-30）
- **年份**：2026
- **链接**：https://arxiv.org/abs/2607.27879
- **资助**：ERC grant No. 101088865、Flanders AI Research Program、Flemish Government Methusalem funding

## 一句话总结

> 把「每次 LLM 调用的美元成本」和 PPA 收益放在同一张坐标图上报，Ares 发现长期记忆怎么构造差别不大，真正的杠杆是推理强度——用 patience counter 只在停滞时升到 high effort，在三个测试设计上把 FoM 降低 23–27%，而最好的固定强度只能做到 16–23%。

## 研究动机与问题定义

- **要解决的核心问题**：LLM agent 优化 RTL 的 PPA 时，质量与推理花费被分开评价。已有 agent 报最终 FoM 不报成本，把质量归因于精心构造的跨设计记忆，并且整轮 run 固定推理强度。
- **现有方法的不足**：论文给出三条。(1) 成本披露不足：最好的情况只报整个研究的单一美元总额，没有能把优化质量与花费联立的指标，effort 级别之间、不同 optimizer 之间无法公平比较（第 2 页）。(2) 记忆归因可疑：Dr. RTL 等把收益归因于 markdown 规则库的构造方式，但这建立在只比较最终 FoM 的对比上，没有控制花费（第 3 页）。(3) 固定推理强度：低强度在难迭代上想得太少，高强度在简单迭代上过度思考，而固定预算的浪费在 test-time compute 文献里已有共识（第 2–3 页）。
- **本文的切入角度**：把每次调用的归一化美元成本作为一等指标，用它重新检验记忆的价值，再把省下来的预算投到推理强度这一「每次调用的变量」上。

## 核心方法

### 方法概述

Ares 的循环很直接（Figure 1，第 1–2 页）：把当前最优 RTL（初始为未优化输入 $v_0$）连同本轮对话历史和长期 markdown 记忆交给 LLM agent，agent 在自适应策略指定的强度上提出一次编辑；候选设计必须先通过功能等价验证，再走商业综合流程得到面积、功耗、延迟，合成 FoM；FoM 更低就替换当前最优，否则丢弃，两种情况都更新 patience counter，计数达到阈值就把下一次调用的强度升到 high。run 在累计成本达到用户设定预算时结束。

两条主线贯穿全文：一是把每次 LLM 调用的成本与它带来的 FoM 变化配对记录，二是用 patience counter 把推理强度在两档之间切换。

### 关键技术细节

- **FoM 定义**（式 1，第 3 页）：$FoM = (area/area_0)\cdot(power/power_0)\cdot(delay/delay_0)$，三项均为后综合 PPA 相对未优化输入 $v_0$ 的归一化值，$FoM(v_0)=1$，越低越好。三项相乘意味着「牺牲一项换另一项」按净效果评判。
- **成本指标**（式 2，第 3 页）：每次调用记录 input、cache read、cache write、output 四类 token 数 $t_i$，用公开发布的 OpenRouter 单价 $p$ 加权求和得到该次调用的美元成本，累计得到 run 的总成本。论文用 **high-calls** 作为成本单位，即美元总额除以该设计上一次 high-effort 调用的平均成本。这样定价变动不影响对比基准。
- **长期记忆的三种构造**（第 3–4 页）：
  - memoryless：没有长期记忆，只在单个设计内部学习。
  - baseline memory：把每个训练 run 蒸馏成「该设计上生效的优化」简单描述列表并拼接。
  - engineered memory：同一批经验，但把每个被接受的编辑记成结构化 (context, action, result) 案例，显式保留无效与有害编辑作为 anti-optimization，跨设计合并重复项并按实测有效性排序，另外把设计专有名称匿名化，让规则匹配问题结构而不是来源。论文称这是现有工作的技术超集。
- **自适应推理强度**（第 4 页，Figure 3）：两状态策略，从 MEDIUM 起步，只有停滞才升到 HIGH。
  - stalled 迭代（综合且验证通过但没降低 FoM）使 $C \rightarrow C+1$；
  - failed 迭代（连有效候选都没产出）使 $C \rightarrow C+w$；
  - 被接受的改进按相对增益放电：$C \rightarrow C\cdot \max(0, 1-\Delta FoM/\kappa)$，其中 $\Delta FoM = (FoM_{best}-FoM_{new})/FoM_{best}$；
  - $C \geq p$ 时下一次调用用 high，然后计数清零。
  - 三个常数 $p=3$、$w=2.8$、$\kappa=0.05$ 在 21 个训练设计上**一次性联合拟合**，不做逐设计调参。拟合准则：在 medium 于后三次迭代内不再产生改进的迭代点上升是「值得的」，grid search 在保证至少 90% 触发的升级是值得的前提下尽可能多命中值得点，最终命中 94%。失败权重 $w=2.8>1$，说明训练数据里一次失败几乎等于三次停滞。
- **验证与综合流程**（第 5 页）：测试平台为固定种子生成的 $10^4$ 个随机输入向量；每个候选在 RTL 和综合后 netlist 两级都与 $v_0$ 输出比对，另用 JasperGold 做 RTL 级顺序等价检查（SEC）。综合用 Synopsys Design Compiler + PrimeTime，开源 Nangate 45 nm 工艺库；先求最短可达时钟周期作为 delay 项，再在该频率下重新综合取面积，功耗由 PrimeTime 基于测试向量切换活动估计。用商业综合器的理由是弱开源综合器会把工具本来就会做的重写算作 agent 的功劳。
- **LLM 配置**：Claude Code 命令行助手 + Claude Opus 4.6，全程开启 extended thinking 与 prompt caching。论文提醒：不指定 effort 的调用会走助手默认值，而默认等同 high，所以 effort 调度必须由 optimizer 显式施加。

### 核心创新点

1. 提出可跨 effort 级别、跨 optimizer 比较的归一化 per-call 成本指标，并把它与 FoM 并列报告。
2. 在成本受控条件下推翻「记忆构造决定优化质量」这一先前归因：engineered memory 相对简单拼接没有稳定收益，但两者都优于无长期记忆。
3. 把推理强度当作 run 内可调变量，用 patience counter 实现两状态自适应调度。

### 与现有方法的关键区别

Table 1（第 3 页）按任务、记忆形式、是否跨设计、是否含功耗、是否有成本轴、是否自适应 effort、是否开源对比了 5 个系统。只有 REvolution 与 Dr. RTL 公开代码，因此只有这两个进入对比实验。与 Dr. RTL 的本质差别是目标函数：Dr. RTL 只优化 timing 与 area，不计功耗，且每轮跑 critical-path 分析 agent、多个并行重写 agent 和评估 agent 的多 agent 流水线；Ares 用单一 agent 串行迭代并按 per-call 成本记账。

## 实验与结果

### 实验设置

- **数据集**：24 个单模块开源 RTL，规模数百到数千行。19 个来自 Dr. RTL 的 20 个设计（其 LSTM 因含推断 memory 被排除），加 RTL-Rewriter 的 FFT butterfly、Huffman decoder，以及 OpenCores 的 CORDIC、pipelined FFT、JPEG DCT。测试集为三个可优化空间跨度较大的设计：tv80（Z80 核的算术逻辑单元）、uart（串行收发器）、controller（AES 核的控制单元）；其余 21 个为训练集，用于收集长期记忆并拟合升级常数。
- **基线方法**：REvolution（有代码的最强演化式 LLM-RTL 框架，适配到同一流程）、Dr. RTL（skill-library 优化器，使用其自带 memory 文件）。另外内部对比三种记忆构造与三种固定 effort。
- **评估指标**：归一化 FoM（越低越好）、累计成本（high-calls）、通过率性质的等价性验证。

### 主要结果

固定 effort 的对比（第 5 页，Figure 4）：

| 观测项 | 结果 |
| --- | --- |
| low effort | 每个设计都差于 medium；controller 上均值 FoM 0.82（medium 可达 0.77）；tv80 上无一次低于 0.84（medium 最好 0.71） |
| high effort 早期 | 首次成功优化要花 4.6–7.1 high-calls，medium 砍掉前 5% FoM 只需 0.9–3.9 |
| high effort 终局 | tv80 上 0.79 对 medium 0.84；uart 与 controller 只差 0.004 与 0.016 |
| adaptive | 平均 FoM 0.76 / 0.77 / 0.73，全部深于固定 high 的 0.79 / 0.84 / 0.79；uart 上以固定 high 的 0.89× 成本先达到其终局 FoM |

自适应策略相对固定强度的收益：三个设计上把输入 FoM 降低 **23–27%**，而最好的固定强度是 **16–23%**（第 5 页第 5.2 节）。

MX MAC 案例（第 6 页，Figure 5）：从规格与测试平台出发，Claude Opus 4.6 先起草 MX_LLM，Ares 把它朝手工优化版 MX_fp32 优化。未归一化 FoM 从 68.8 降到六次 run 的均值 33.6（最深 27.5），MX_fp32 公布值为 18.9，即**最多补上 83% 的差距**。同一个优化器对已经手工优化的 MX_fp32 只能再改进 16%。run 间方差从 std 8.9 降到 5.8（**降 58%**），端点区间从 22.6 缩到 17.6 FoM 点，代价是平均多花 7.3 个 high-calls；固定 medium 与升级分支在同成本下均值从 42.1 降到 33.6。

与 SotA 的对比（第 6 页，Figure 6，三者优化同一个 controller）：

| 方法 | 最终 FoM | 成本 |
| --- | ---: | --- |
| Ares（自适应） | **0.694** | 约 15 high-calls |
| REvolution | 0.943 | 5 候选项 × 5 轮，约等于 10 high-effort 调用 |
| Dr. RTL | 0.923 | Ares 的 8.7×，token 数为 Ares 的 8.3×（Ares 只用其 12%） |

Dr. RTL 的 timing-focused 选择还会丢弃自己在这个 power-aware FoM 下最好的候选（0.909），因为它的目标函数不含功耗。论文把 Dr. RTL 的高成本归因于多 agent 流水线：每个提议设计都要跑一个 critical-path 分析 agent、若干并行重写 agent 和一个评估 agent。

### 消融实验要点

- **记忆构造**（第 4 页，Figure 2）：三种记忆模式在三个测试设计上并排对比，engineered memory 与 baseline concatenation 的曲线同步下降、终点接近，没有一方稳定领先；两者通常都好于 memoryless。结论是同等成本下记忆的书写方式影响很小。
- **升级常数**：$p$、$w$、$\kappa$ 一次性在训练设计上拟合，不逐设计调参，这是为了避免把测试集信息泄漏进调度策略。论文未给出常数敏感性分析。
- **单次 run 内分支对照**（Figure 5 左）：六个 run 各自在自己首个停滞点分叉成「继续 medium」与「按策略升级」两支，六次中五次升级支在相同迭代数下取得更低 FoM，第六次落在固定 medium 的 2% 以内。这比跨 run 平均更有说服力。

## 局限性与未来方向

- **作者未设局限章节**，以下为从实验设置直接可见的边界：
  - 测试集只有 3 个设计，全部是单模块、开放源码、数百到数千行；无法判断结论能否外推到多模块、多时钟域或含 IP 的设计。
  - 全部实验只用 Claude Opus 4.6 一个模型。自适应强度的收益是否与其他模型（尤其开源模型）的 effort–成本曲线一致，未验证。
  - 成本指标依赖 OpenRouter 公开单价，换供应商或换计费方式需重算 high-call 基准。
  - 拟合常数用的是 21 个训练设计上的 medium-effort run，训练集里排除了 Dr. RTL 的 LSTM（理由是其含推断 memory），也就是说方法对「需要推断 memory 的设计」未做验证。
  - 验证依赖 $10^4$ 随机向量加 RTL 级 SEC。随机向量无法覆盖全部结构，论文自己也承认这一点，因此「功能等价」的证据强度受限于测试平台质量。
  - 所有 PPA 数据来自 Nangate 45 nm 商业综合流程的 post-synthesis 结果，不是 post-layout，也不是硅测。面积、功耗、延迟之间的取舍在布局布线后可能变化，论文未讨论。
  - 这是预印本，尚未同行评审；代码称将在同行评审发表后公开。

## 个人点评

- **亮点**：把成本提到与质量同等的地位，并用它反过来审查社区里一个被广泛接受的归因（记忆构造决定质量）。这类「先建立公平度量，再验证既有归因」的工作比再加一层记忆结构更有价值。run 内分叉对照的设计也克制：不靠跨 run 平均去掩盖采样噪声。成本单位选 high-calls 而不是美元或 token，避免了定价变动导致的历史结果不可比，这个细节值得借鉴。
- **不足**：只报 post-synthesis 的 PPA 归一化乘积。FoM 三项相乘会把「延迟显著变差、面积略微变好」这类工程上不可接受的方案判为可接受，论文没有给三项各自的约束或 Pareto 视图。测试设计只有三个，其中 controller 和 uart 的 high-vs-medium 差距只有 0.004 和 0.016，说明「固定高强度浪费」的结论主要由 tv80 一个设计支撑。另外 $w=2.8$ 这类拟合常数缺少敏感性分析，读者无法判断策略对它的依赖程度。
- **启发**：真正可迁移的部分不是记忆结构，而是「停滞信号 → 升级预算 → 改进后放电」这个反馈控制回路。它与领域无关，可以套用到验证用例生成、时序约束收敛、布局参数调优等其它 EDA 迭代任务。对做 agent 系统的人，$w>1$ 的拟合结果给出一个有价值的先验：一次失败比一次停滞携带更多信息。

## 工程化三问总结

### 1. 它解决了什么瓶颈？

- **应用场景与核心瓶颈**：用 LLM agent 反复编辑综合前 RTL 来改善 PPA。瓶颈是双重约束——既要更深地降低 FoM，又要在每次 LLM 调用都计费的前提下控制花费。
- **现有方法为何不足**：已有 agent 只报最终质量不报成本，把质量归因于记忆构造，并全程固定推理强度；固定高强度在早期会用 4.6–7.1 high-calls 才换到第一次成功优化，而 medium 只需 0.9–3.9 就能砍掉前 5% FoM（第 5 页）。
- **论文证据**：同等成本下 engineered memory 相对简单拼接无稳定收益（Figure 2，三个测试设计）；自适应强度把 FoM 降低 23–27%，而最好的固定强度只有 16–23%；controller 上以 0.694 对 REvolution 0.943、Dr. RTL 0.923，且只用 Dr. RTL 12% 的 token（Figure 6）。这些是论文的直接证据。注意 FoM 为 post-synthesis 归一化值，不含布局后与硅测。

### 2. 用了什么结构或训练方法？

- **整体结构与数据流**：单一 LLM agent 的串行迭代环。输入 $v_0$ 与测试平台 → agent 在指定 effort 下提出编辑 → $10^4$ 随机向量测试 + JasperGold SEC 验证功能等价 → Synopsys DC 综合（先求最短时钟周期，再在该频率重综合取面积）→ PrimeTime 估功耗 → 计算三项乘积 FoM → 更好则替换当前最优 → 更新 patience counter 决定下一次 effort。run 到达成本预算即停。
- **关键模块/结构**：短期记忆（当前 run 的对话）、长期记忆（跨设计的 markdown 经验，三种构造变体）、patience counter 两状态 effort 策略、商业综合与验证流程。
- **训练目标与优化方法**：论文不做模型训练。优化目标是 FoM 最小化，调度策略由 $p=3$、$w=2.8$、$\kappa=0.05$ 三个常数定义，在 21 个训练设计上一次性网格搜索拟合。
- **数据与训练策略**：24 个开源 RTL 模块，21 训练 / 3 测试；测试平台由 $10^4$ 个固定种子随机向量构成；长期记忆从训练 runs 的已接受编辑中蒸馏，engineered 版本做 (context, action, result) 结构化、反例保留、跨设计去重与匿名化。

### 3. 对芯片架构和 RTL 有什么启发？

- **芯片架构**：论文不含任何芯片架构或硬件实现内容，这一层全部是推断。从负载构成看，agentic RTL 优化 loop 的 wall-clock 主导项是每轮一次的综合与功耗分析，而不是 LLM 推理本身，因此若要本地化部署，值得考虑把 DC/PrimeTime 这类工具的调用与 LLM 服务放在同一节点以压缩环回延迟。论文使用 prompt caching 且反复回灌同一会话历史，说明长对话的 KV Cache 复用会直接影响单轮成本——这对推理加速器的缓存层次设计是一个可量化的需求来源，但论文没有给出 token 组成或缓存命中率数据，具体收益 `TBD`。
- **RTL**：论文对 RTL 层面的启发在流程接口而不是电路结构。其一，把 $area\cdot power\cdot delay$ 归一化乘积作为可计算目标，配合 SEC 与随机向量验证作为护栏，是可以直接搬进 RTL 优化流程的组合；其二，patience counter 本身是一个极简两状态 FSM（停滞 +1、失败 +$w$、改进按 $\Delta FoM/\kappa$ 放电、越过阈值 $p$ 切换状态），若要把这种调度逻辑硬件化（例如放进 chiplet 的 DVFS 或自适应调优硬件），定点化实现难度低，这是推测，论文没有做任何硬件实现或综合评估。
- **推断边界**：第 1、2 问基于论文正文与图表，属论文证据；第 3 问中芯片架构与 RTL 两部分论文均未涉及，全部为工程推断。论文所有 PPA 数据来自 Nangate 45 nm 的 post-synthesis 商业流程，无 post-layout、无 silicon measurement、无 RTL 实现面积或时序数据。
