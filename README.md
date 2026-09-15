<div align="center">

# AI Chip Library

**面向 AI 芯片架构、算法与 AI 辅助设计的个人技术资料库**<br>
**A personal technical library for AI chip architecture, algorithms, and AI-aided design**

<p>
  <a href="#中文">中文</a> · <a href="#english">English</a>
</p>

</div>

---

<a id="中文"></a>

## 中文

AI Chip Library 持续归档 AI 芯片及相关技术资料，并为每份资料整理中文深度分析。内容覆盖芯片架构、硬件感知算法、AI 辅助芯片设计，以及与技术研发相关的软件工程主题。

### 核心内容

| 内容 | 说明 |
| --- | --- |
| 原始资料 | 保存论文、技术报告、博客和技术文档，便于离线查阅与长期归档 |
| 中文深度分析 | 提炼问题、方法、实验、局限与工程启示，并明确区分论文证据和工程推断 |
| 统一索引 | 按年份、主分类和相关主题组织资料，集中维护本地 Summary 链接 |
| 工程视角 | 关注算法瓶颈、结构或训练方法，以及对芯片架构、RTL 和验证的影响 |

### 目录结构

资料按两级主题组织：一级目录表示技术领域，二级目录表示该领域内的模块、技术或设计任务。二级目录只在有实际资料时创建。

```text
.
├── AI-Aided-Design/
│   ├── RTL-Generation/
│   └── RTL-Optimization/
├── Algorithm/
│   ├── Deep-Learning/
│   └── Quantization/
├── Architecture/
│   ├── GPU/
│   ├── Memory/
│   └── NPU/
└── Software-Engineering/
    └── Project-Management/
```

- `Architecture`：按硬件模块分类，例如 `GPU`、`NPU`、`NoC`、`DMA`。
- `Algorithm`：按算法技术分类，例如 `Quantization`、`Sparsity`、`Training`。
- `AI-Aided-Design`：按设计任务分类，例如 `RTL-Generation`、`RTL-Optimization`、`Verification`、`Physical-Design`。
- `Software-Engineering`：按软件工程主题分类，例如 `Project-Management`、`Architecture`、`Testing`。

### 资料索引

| 标题 | 年份 | 分类 | Summary | 相关主题 |
| --- | ---: | --- | --- | --- |
| ACE-RTL: When Agentic Context Evolution Meets RTL-Specialized LLMs | 2026 | AI-Aided-Design / RTL-Generation | [深度分析](AI-Aided-Design/RTL-Generation/ACE-RTL-When-Agentic-Context-Evolution-Meets-RTL-Specialized-LLMs-summary.md) | RTL Generation、LLM Agent、Simulation Feedback、Context Evolution |
| Memory: Feeding AI’s Voracious Hunger for Data | 2026 | Architecture / Memory | [深度分析](Architecture/Memory/Memory-Feeding-AI-s-Voracious-Hunger-for-Data-summary.md) | AI Memory、HBM、PIM、Memory Wall、DRAM Pricing |
| Evolving Memory Architectures for AI | 2026 | Architecture / Memory | [深度分析](Architecture/Memory/Evolving-Memory-Architectures-for-AI-summary.md) | HBM、HBM4、DDR5、封装、RAS、热设计 |
| HBF in AI Compute: A System Architect’s View | 2026 | Architecture / Memory | [深度分析](Architecture/Memory/HBF-in-AI-Compute-A-System-Architect-s-View-summary.md) | HBF、Flash、MoE、KV Cache、Memory Tiering、Endurance |
| Raptor: The First 3D-DRAM Accelerator for Generative Inference | 2026 | Architecture / Memory | [深度分析](Architecture/Memory/Raptor-The-First-3D-DRAM-Accelerator-for-Generative-Inference-summary.md) | 3D DRAM、HBM、ECC、DBI、Bank Mapping、Thermal |
| AMD Instinct MI400 Series GPU Architecture | 2026 | Architecture / GPU | [深度分析](Architecture/GPU/AMD-Instinct-MI400-Series-GPU-Architecture-summary.md) | AMD、MI400、MI455X、CDNA、HBM4、MXFP4、CoWoS-L |
| System Architecture of the AMD MI400 Series GPU | 2026 | Architecture / GPU | [深度分析](Architecture/GPU/System-Architecture-of-the-AMD-MI400-Series-GPU-summary.md) | AMD、MI400、UALoE、AI-NIC、Infinity Fabric、VPod |
| Crescent Island: GPU Designed for Agentic AI Inference | 2026 | Architecture / GPU | [深度分析](Architecture/GPU/Crescent-Island-GPU-Designed-for-Agentic-AI-Inference-summary.md) | Intel、GPU、Agentic AI、LPDDR5x、Speculative Decoding |
| NVIDIA Rubin GPU: Driving the Era of Agentic AI | 2026 | Architecture / GPU | [深度分析](Architecture/GPU/NVIDIA-Rubin-GPU-Driving-the-Era-of-Agentic-AI-summary.md) | NVIDIA、Rubin、NVFP4、Sparsity、NVLink、NVL72 |
| Rack-Scale Architecture for Wafer Scale Engine | 2026 | Architecture / NPU | [深度分析](Architecture/NPU/Rack-Scale-Architecture-for-Wafer-Scale-Engine-summary.md) | Cerebras、WSE、Wafer-scale、Rack-scale、Networking |
| Meta’s Custom AI Silicon: From Recommendation to Dual-Mandate with GenAI | 2026 | Architecture / NPU | [深度分析](Architecture/NPU/Meta-s-Custom-AI-Silicon-From-Recommendation-to-Dual-Mandate-with-GenAI-summary.md) | Meta、MTIA、Recommendation、GenAI、HBM |
| Maia 200: A Data Center Scale AI Accelerator for Large Scale Inference using Software Defined Dataflow | 2026 | Architecture / NPU | [深度分析](Architecture/NPU/Maia-200-A-Data-Center-Scale-AI-Accelerator-for-Large-Scale-Inference-using-Software-Defined-Dataflow-summary.md) | Microsoft、Maia 200、Dataflow、Inference、HBM |
| The Eighth Generation TPU Family: Two Chips Optimized for the Agentic Era | 2026 | Architecture / NPU | [深度分析](Architecture/NPU/The-Eighth-Generation-TPU-Family-Two-Chips-Optimized-for-the-Agentic-Era-summary.md) | Google、TPU8t、TPU8i、Agentic AI、Pod |
| Dataflow at Scale: the SN50 RDU | 2026 | Architecture / NPU | [深度分析](Architecture/NPU/Dataflow-at-Scale-the-SN50-RDU-summary.md) | SambaNova、RDU、Dataflow、MBU、MoE |
| Think Fast: LPU Accelerator for Heterogeneous Compute | 2026 | Architecture / NPU | [深度分析](Architecture/NPU/Think-Fast-LPU-Accelerator-for-Heterogeneous-Compute-summary.md) | NVIDIA、Groq、LPU、VLIW、Heterogeneous Inference |
| Jalapeño ASIC + System | 2026 | Architecture / NPU | [深度分析](Architecture/NPU/Jalapeno-ASIC-System-summary.md) | OpenAI、ASIC、HBM4、Spatial Architecture、PPA |
| Patterns behind Chaos: Forecasting Data Movement for Efficient Large-Scale MoE LLM Inference | 2026 | Architecture / NPU | [深度分析](Architecture/NPU/Patterns-behind-Chaos-Forecasting-Data-Movement-for-Efficient-Large-Scale-MoE-LLM-Inference-summary.md) | MoE、LLM Inference、GPU、Chiplet、Wafer-scale、Data Movement |
| DSPE: An Energy-Efficient Edge Processor for DeepSeek Inference with MerkleTree-based Incremental Pruning, Multi-Stage Boothing Lookup and Dynamic Adaptive Posit Processing | 2026 | Architecture / NPU | [深度分析](Architecture/NPU/DSPE-An-Energy-Efficient-Edge-Processor-for-DeepSeek-Inference-with-MerkleTree-based-Incremental-Pruning-Multi-Stage-Boothing-Lookup-and-Dynamic-Adaptive-Posit-Processing-summary.md) | Edge Inference、Pruning、Approximate Computing、Merkle Tree、Booth、Posit |
| Designing AI Chip Hardware and Software | 2026 | Architecture / NPU | [深度分析](Architecture/NPU/Designing-AI-Chip-Hardware-and-Software-summary.md) | AI CPU、Systolic Array、LLM Inference、Co-design、Compiler |
| Ares: Adaptive Reasoning-Effort Steering for PPA- and Cost-Aware RTL Optimization with LLM Agents | 2026 | AI-Aided-Design / RTL-Optimization | [深度分析](AI-Aided-Design/RTL-Optimization/Ares-Adaptive-Reasoning-Effort-Steering-for-PPA-and-Cost-Aware-RTL-Optimization-with-LLM-Agents-summary.md) | LLM Agent、EDA、PPA、RTL |
| SigmaQuant: Hardware-Aware Heterogeneous Quantization Method for Edge DNN Inference | 2026 | Algorithm / Quantization | [深度分析](Algorithm/Quantization/SigmaQuant-Hardware-Aware-Heterogeneous-Quantization-Method-for-Edge-DNN-Inference-summary.md) | Edge Inference、Mixed Precision、KL Divergence、Shift-Add MAC |
| DeepSeek-V3 Technical Report | 2025 | Architecture / NPU | [深度分析](Architecture/NPU/DeepSeek-V3-Technical-Report-summary.md) | DeepSeek、MoE、MLA、FP8、LLM Training |
| Hardware-Centric Analysis of DeepSeek’s Multi-Head Latent Attention | 2025 | Architecture / NPU | [深度分析](Architecture/NPU/Hardware-Centric-Analysis-of-DeepSeek-s-Multi-Head-Latent-Attention-summary.md) | MLA、KV Cache、Dataflow Accelerator、Roofline、Operational Intensity |
| Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures | 2025 | Architecture / NPU | [深度分析](Architecture/NPU/Insights-into-DeepSeek-V3-Scaling-Challenges-and-Reflections-on-Hardware-for-AI-Architectures-summary.md) | DeepSeek、MoE、FP8、Multi-Plane Network、Co-design |
| An Efficient Layer Normalization Training Module With Dynamic Quantization for Transformers | 2025 | Architecture / NPU | [深度分析](Architecture/NPU/An-Efficient-Layer-Normalization-Training-Module-With-Dynamic-Quantization-for-Transformers-summary.md) | Transformer Training、Dynamic Quantization、FPGA、ASIC |
| You Only Cache Once: Decoder-Decoder Architectures for Language Models | 2024 | Architecture / NPU | [深度分析](Architecture/NPU/You-Only-Cache-Once-Decoder-Decoder-Architectures-for-Language-Models-summary.md) | LLM Inference、KV Cache、Gated Retention、Long Context、Prefill |
| Understanding Deep Learning | 2023 | Algorithm / Deep-Learning | [全书总结](Algorithm/Deep-Learning/Understanding-Deep-Learning-summary.md) | Neural Networks、Optimization、CNN、Transformer、Generative Models、Reinforcement Learning |
| A Research Retrospective on the AMD Exascale Computing Journey | 2023 | Architecture / GPU | [深度分析](Architecture/GPU/A-Research-Retrospective-on-the-AMD-Exascale-Computing-Journey-summary.md) | AMD、Frontier、EHP、Chiplet、封装、Co-design |
| TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings | 2023 | Architecture / NPU | [深度分析](Architecture/NPU/TPU-v4-An-Optically-Reconfigurable-Supercomputer-for-Machine-Learning-with-Hardware-Support-for-Embeddings-summary.md) | TPU、OCS、SparseCore、Embedding、3D Torus |
| A Software-defined Tensor Streaming Multiprocessor for Large-scale Machine Learning | 2022 | Architecture / NPU | [深度分析](Architecture/NPU/A-Software-defined-Tensor-Streaming-Multiprocessor-for-Large-scale-Machine-Learning-summary.md) | Groq、TSP、Dragonfly、Deterministic Network |
| Ten Lessons From Three Generations Shaped Google’s TPUv4i Industrial Product | 2021 | Architecture / NPU | [深度分析](Architecture/NPU/Ten-Lessons-From-Three-Generations-Shaped-Google-s-TPUv4i-Industrial-Product-summary.md) | TPU、DSA、CMEM、Compiler、Perf/TCO |
| Think Fast: A Tensor Streaming Processor (TSP) for Accelerating Deep Learning Workloads | 2020 | Architecture / NPU | [深度分析](Architecture/NPU/Think-Fast-A-Tensor-Streaming-Processor-TSP-for-Accelerating-Deep-Learning-Workloads-summary.md) | Groq、TSP、Dataflow、Batch-1 Inference |
| The Mythical Man-Month: Essays on Software Engineering | 1995 | Software-Engineering / Project-Management | [全书总结](Software-Engineering/Project-Management/The-Mythical-Man-Month-Essays-on-Software-Engineering-summary.md) | Software Project Management、Brooks's Law、Conceptual Integrity、No Silver Bullet |

### 收录规则

1. 每份资料只保存一份，并归入一个主目录；跨领域信息记录在索引的“相关主题”中。
2. 文件名使用正式完整标题，以连字符分隔并保留原始大小写和数字；完整标题写入索引。Summary 与原文同目录，命名为 `<正式标题>-summary.md`。
3. 新资料先检查标题、来源、重复情况和候选分类。分类明确时提出归档建议；跨一级领域时列出候选位置及理由，由资料库维护者决定。
4. 分类确认后再移动或重命名文件，并在同一次整理中更新本索引。

---

<a id="english"></a>

## English

AI Chip Library is a personal archive of source materials and Chinese technical reviews focused on AI chips. It covers chip architecture, hardware-aware algorithms, AI-aided chip design, and selected software engineering topics relevant to technical development.

### What It Contains

| Content | Purpose |
| --- | --- |
| Source materials | Papers, technical reports, blog posts, and technical documents preserved for offline reference |
| Chinese reviews | Structured analysis of the problem, method, experiments, limitations, and engineering implications |
| Topic-based archive | One primary location per item, organized by technical domain and subject |
| Engineering perspective | Connections from algorithmic bottlenecks and methods to chip architecture, RTL, and verification |

The Chinese section above is the canonical index for all archived materials and reviews.
