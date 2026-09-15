# 深度分析：《Ares: Adaptive Reasoning-Effort Steering for PPA- and Cost-Aware RTL Optimization with LLM Agents》

## 基本信息

- **标题**：Ares: Adaptive Reasoning-Effort Steering for PPA- and Cost-Aware RTL Optimization with LLM Agents
- **文档类型**：论文（arXiv 预印本）
- **作者**：Stef Cuyckens、Mihaela Jivanescu、Jun Yin、Chao Fang、Marian Verhelst
- **机构**：KU Leuven、Nokia Bell Labs
- **发表 venue**：arXiv，cs.AR
- **年份**：2026
- **链接**：[arXiv:2607.27879](https://arxiv.org/abs/2607.27879)
- **阅读范围**：全文 7 页，包括全部图表与参考文献

## 一句话总结

> Ares 将 LLM 的单次调用成本与 RTL 优化得到的功耗、性能、面积结果统一计量，并用基于停滞状态的 patience counter 动态提升推理强度，以更少 token 获得更深且更稳定的 PPA 优化。

## 研究动机与问题定义

- **要解决的核心问题**：LLM Agent 可以反复修改 RTL 并借助 EDA 工具反馈优化 PPA，但不同推理强度的调用成本差异很大；现有方法既缺少统一的“效果-成本”坐标，也通常在整次运行中固定推理强度。
- **现有方法的不足**：已有 RTL Agent 常只优化面积与时序而忽略功耗；部分工作只报告整个实验的总美元成本；调用次数和 token 数又不能准确反映缓存输入、推理输出等不同 token 类型的真实价格。已有方法还把优化质量主要归因于精心设计的长期记忆，但缺少等成本对照。
- **本文的切入角度**：将每次 LLM 调用的真实计价归一化为 `high-call`，在相同成本下比较策略；固定其他条件，只改变长期记忆的组织形式；最后通过停滞计数器在 medium 与 high reasoning effort 之间自适应切换。

## 核心方法

### 方法概述

Ares 的输入是功能正确但未优化的 RTL 设计 `v0` 与测试平台（Testbench，TB）。每轮将当前最优 RTL、当前会话历史形成的短期记忆，以及跨设计积累的 Markdown 长期记忆交给 LLM。LLM 产生一个候选修改后，系统先做随机测试与顺序等价性检查，再综合、进行门级验证并测量功耗。只有功能正确且降低品质因数（Figure of Merit，FoM）的候选才成为新的当前最优设计，其余候选均丢弃（图 1，第 2 页）。

论文同时跟踪两条轴：优化深度和累计成本。FoM 是相对输入设计归一化后的面积、功耗、延迟乘积，输入设计的 FoM 固定为 1，越低越好。成本则按实际输入、缓存读、缓存写和输出 token 的公开单价求和，再除以该设计一次 high-effort 调用的平均美元成本，得到跨推理强度可比较的 `high-call` 单位（公式 1、2，第 3 页）。

自适应策略从 medium effort 开始。能通过综合与验证但没有改善 FoM 的候选记为 stall，计数器 `C` 加 1；未能编译、综合或验证的候选记为 fail，`C` 加权重 `w`；成功候选按相对 FoM 改善量降低 `C`。当 `C >= p` 时，下一次调用提升为 high effort，然后重置计数器。三个常数只在训练集上拟合一次（图 3，第 4 页）。

### 关键技术细节

- **闭环 RTL 优化**：每轮从当前最优版本继续编辑，候选只有在功能验证通过且 FoM 更低时才保留。
- **PPA 评价**：先用 Synopsys Design Compiler 搜索最大频率，再在该频率下综合得到面积；PrimeTime 根据门级网表测试向量的翻转活动估计功耗。标准单元库为 Nangate 45 nm（第 5 页）。
- **功能安全门槛**：使用固定随机种子的 `10^4` 个输入向量，在 RTL 和综合后网表上与 `v0` 对比，并用 JasperGold 顺序等价性检查（Sequential Equivalence Checking，SEC）补足随机测试覆盖（第 5 页）。
- **成本模型**：分别统计 input、cache read、cache write、output token，以 OpenRouter 单价计费；内部推理 token 计入 output token（公式 2，第 3 页）。
- **长期记忆对照**：比较无长期记忆、简单拼接成功优化描述的 baseline memory，以及增加结构化案例、负面案例、去重排序和名称匿名化的 engineered memory（图 2，第 4 页）。
- **自适应参数**：`p = 3`、`w = 2.8`、`kappa = 0.05`。网格搜索目标是在至少 90% 触发正确率下覆盖尽可能多的应升级点，最终覆盖 94%（第 4-5 页）。
- **模型与执行环境**：Claude Code + Claude Opus 4.6，启用 extended thinking 与 prompt caching；模型调用不可固定随机种子，因此各配置重复运行（第 5 页）。

### 核心创新点

1. 将每次调用的真实美元成本换算为设计相关但价格变化不敏感的 `high-call`，使不同 effort 和不同 Agent 能在统一成本轴上比较。
2. 在相同原始经验和相同成本下隔离长期记忆的组织方式，发现复杂的结构化、去重和抽象并未稳定优于简单拼接。
3. 用可解释的 patience counter 根据 stall、fail 和实际 FoM 改善动态分配推理预算，而不是固定使用低、中或高 effort。

### 与现有方法的关键区别

REvolution、POET 和 CktEvo 主要保存当前运行中的候选种群或归档；VeriAgent 与 Dr. RTL 使用跨设计长期记忆，但不逐调用报告成本，也不动态改变 reasoning effort。Ares 同时包含功耗感知 FoM、逐调用成本轴、跨设计记忆和 effort adaptation（表 1，第 3 页）。与更复杂的多 Agent Dr. RTL 相比，Ares 采用单 Agent 闭环，并让商业 EDA 与形式验证直接决定候选去留。

## 实验与结果

### 实验设置

- **数据集**：24 个开源单模块 RTL，规模从数百到数千行；21 个用于记忆积累和参数拟合，3 个 held-out 测试设计为 Z80 ALU `tv80`、UART 收发器 `uart`、AES 控制模块 `controller`。另用 MX 微缩放（Microscaling）MAC 比较 LLM 草稿与人工优化实现。
- **基线方法**：固定 low/medium/high effort；无记忆、简单拼接记忆、工程化记忆；REvolution；Dr. RTL；人工优化的 `MX_fp32`。
- **评估指标**：归一化 area-power-delay FoM、累计 `high-call` 成本、token 数、运行间方差，以及相对人工实现所缩小的 FoM 差距。

### 主要结果

| 对比 | 结果 | 证据位置 |
| --- | --- | --- |
| Adaptive vs. 最佳固定 effort | Adaptive 在 `tv80`、`uart`、`controller` 的平均 FoM 为 0.76、0.77、0.73；fixed high 为 0.79、0.84、0.79 | 图 4，第 5 页 |
| 相对输入设计的优化 | Adaptive 降低 FoM 23%-27%，最佳固定 effort 为 16%-23% | 第 5 页 |
| MX LLM 草稿 | FoM 从 68.8 降至六次运行平均 33.6，最好 27.5；人工版本为 18.9，最多缩小 83% 差距 | 图 5，第 6 页 |
| 运行稳定性 | fixed medium 的标准差从 8.9 降至 adaptive 的 5.8，下降 58%；平均增加 7.3 次 high-call | 第 6 页 |
| `controller` 上的 Agent 对比 | Ares 为 0.694，REvolution 为 0.943，Dr. RTL 返回 0.923；Ares 使用 Dr. RTL 约 12% 的 token | 图 6，第 6 页 |
| 成本对比 | Dr. RTL 在同为 25 个候选时成本为 Ares 的 8.7 倍 | 第 6 页 |

### 消融实验要点

- **记忆结构消融**：在三个 held-out 设计上，engineered memory 与简单拼接 baseline memory 的曲线接近，双方没有一致领先；两者通常都优于 memoryless。证据支持“跨设计经验有用”，但不支持“复杂记忆工程稳定更好”（图 2，第 4 页）。
- **固定 effort 消融**：low 最弱且不稳定；high 早期付出更高成本，优势主要出现在优化后期；medium 早期性价比最好但容易进入平台期（图 4，第 5 页）。
- **逐运行分叉实验**：MX 的六次 fixed-medium 运行在首次停滞点分叉，五次 adaptive 分支更好，第六次与 fixed 分支相差 2% 以内；说明收益集中在确实卡住的运行（图 5，第 6 页）。
- **功耗感知选择**：Dr. RTL 的时序导向选择返回 FoM 0.923，却丢弃了按论文功耗感知 FoM 更好的 0.909 候选，显示目标函数会直接改变 Agent 的选择（图 6，第 6 页）。

## 局限性与未来方向

- **作者明确或材料直接显示的限制**：held-out 测试集只有 3 个单模块 RTL；模型仅使用 Opus 4.6；代码要等同行评审发表后才开放；LLM 无法固定采样种子。
- **实验外推限制**：结果来自 Nangate 45 nm 和 Synopsys 商业流程，未验证先进工艺、FPGA、层次化 SoC 或百万行级仓库；归一化成本能缓解价格变化，但仍依赖提供商的 token 分类与计价日志。
- **验证限制**：形式等价检查强化了功能可信度，但测试对象是单模块且输入设计本身被当作黄金参考；没有讨论未定义行为、X 传播、跨时钟域、模拟接口或系统级软件可见行为。
- **方法限制**：策略只在 medium/high 间进行简单两状态调度，参数在 21 个设计上网格搜索；尚未证明这些常数能跨模型、跨任务和跨工具流泛化。
- **记忆结论边界**：engineered memory 未优于简单拼接不等于所有记忆工程都无价值；结论仅适用于相同经验池、当前三个测试设计和当前 Agent。
- **潜在改进方向**：扩大多模型、多工艺、多层次 RTL 基准；联合估计改进概率与 EDA 成本；加入动态降级或多档 effort；公开每轮编辑、验证失败类型和完整成本日志以支持复现。

## 个人点评

- **亮点**：论文把“优化得多深”和“为此花多少钱”拆成两个明确轴，并用严格的功能验证与商业 PPA flow 关闭了只看代码表面质量的漏洞。最有价值的结论不是某个 Agent 结构，而是推理预算应投在停滞点。
- **不足**：测试规模很小，且 adaptive 参数和记忆结论都可能与 Opus 4.6、选定模块及工具链强相关。与 Dr. RTL 的比较还混合了目标函数和 Agent 拓扑差异，不能单独归因于 effort policy。
- **启发**：对工程团队而言，一个简单、可审计的状态机加严格 EDA 反馈，可能比继续堆叠 manager、skill library 和多 Agent 更值得优先验证。成本、功能正确性和 PPA 应从一开始就是统一实验协议的一部分。

## 工程化三问总结

### 1. 它解决了什么瓶颈？

- **应用场景与核心瓶颈**：在 LLM Agent 反复修改 RTL 的长周期优化中，固定推理强度要么早期浪费昂贵 reasoning，要么后期缺少跳出局部平台的能力；已有结果还缺乏可比的真实成本轴。
- **现有方法为何不足**：调用数忽略不同 effort 的价格，token 数忽略 token 类型单价；仅看面积/时序会遗漏功耗回退；复杂长期记忆的收益也未在等成本条件下隔离验证。
- **论文证据**：Adaptive 在三个 held-out 设计上降低 FoM 23%-27%，优于固定策略的 16%-23%；MX 案例最多缩小 83% 的人工实现差距，并将运行间标准差从 8.9 降至 5.8。这些是论文实验结果，不代表大规模 SoC 上也会保持同等收益。

### 2. 用了什么结构或训练方法？

- **整体结构与数据流**：当前最优 RTL -> LLM 修改 -> RTL 随机测试与 SEC -> 最大频率综合 -> 固定频率面积综合 -> 门级验证与翻转活动 -> PrimeTime 功耗 -> 归一化 FoM -> 接受或丢弃候选。
- **关键模块/结构**：单 LLM Agent、短期会话记忆、跨设计 Markdown 长期记忆、真实 token 成本计量、商业综合与功耗分析、顺序等价检查、patience counter effort controller。
- **训练目标、损失函数或优化方法**：没有训练或微调 LLM。控制参数通过 21 个训练设计上的记录做网格搜索，以“至少 90% 的升级触发是正确的”为约束并最大化应升级点覆盖率。
- **数据与训练策略**：21 个 RTL 用于积累记忆和拟合 `p=3`、`w=2.8`、`kappa=0.05`，3 个 RTL 完全留作测试；各运行从 medium 开始，stall/fail 累积到阈值后调用 high effort。

### 3. 对芯片架构、RTL、验证有什么启发？

- **芯片架构**：论文没有提出新的 NPU/GPU 微架构。可落地的启发是架构探索目标必须同时覆盖功耗、性能和面积，并显式计入探索成本；只优化某一指标可能选择整体更差的设计。这是基于论文 flow 的工程推断。
- **RTL**：Agent 应只修改当前已验证的最佳版本，每次修改都经过编译、综合、等价与 PPA 门槛；patience counter 可实现为编排层的简单状态机，无需把调度逻辑塞进 LLM prompt。对层次化设计、约束文件和多时钟 RTL 的处理为 `TBD`。
- **验证**：以原始 RTL 为黄金参考，至少组合随机仿真、RTL SEC、综合后网表仿真，并对每个候选记录 compile/synthesis/verification failure 类型。还应补充 X/复位、边界输入、时序约束一致性和低功耗语义检查；这些补充项是工程建议，不是论文已验证内容。
- **推断边界**：论文直接证明的是三个单模块在给定 LLM、45 nm 库和商业工具流上的优化效果；对先进节点、复杂 SoC、形式验证容量和真实签核流程的收益均需重新实验。
