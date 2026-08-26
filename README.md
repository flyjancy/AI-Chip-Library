# AI Chip Library

这是一个以 AI Chip 为核心、持续收录其他技术主题资料的资料库，内容包括论文、博客和技术文档。

## 目录结构

资料按两级主题组织：一级目录表示技术领域，二级目录表示该领域内的模块、技术或设计任务。二级目录只在有实际资料时创建。

```text
.
├── AI-Aided-Design/
│   └── RTL-Optimization/
├── Algorithm/
│   └── Quantization/
└── Architecture/
    ├── GPU/
    ├── Memory/
    └── NPU/
```

- `Architecture`：按硬件模块分类，例如 `GPU`、`NPU`、`NoC`、`DMA`。
- `Algorithm`：按算法技术分类，例如 `Quantization`、`Sparsity`、`Training`。
- `AI-Aided-Design`：按设计任务分类，例如 `RTL-Generation`、`RTL-Optimization`、`Verification`、`Physical-Design`。

## 资料索引

| 标题 | 年份 | 分类 | Summary | 相关主题 |
| --- | ---: | --- | --- | --- |
| Patterns behind Chaos: Forecasting Data Movement for Efficient Large-Scale MoE LLM Inference | 2026 | Architecture / NPU | [深度分析](Architecture/NPU/summary-Patterns-behind-Chaos-Forecasting-Data-Movement-for-Efficient-Large-Scale-MoE-LLM-Inference.md) | MoE、LLM Inference、GPU、Chiplet、Wafer-scale、Data Movement |
| Ares: Adaptive Reasoning-Effort Steering for PPA- and Cost-Aware RTL Optimization with LLM Agents | 2026 | AI-Aided-Design / RTL-Optimization | [深度分析](AI-Aided-Design/RTL-Optimization/summary-Ares-Adaptive-Reasoning-Effort-Steering-for-PPA-and-Cost-Aware-RTL-Optimization-with-LLM-Agents.md) | LLM Agent、EDA、PPA、RTL |
| An Efficient Layer Normalization Training Module With Dynamic Quantization for Transformers | 2025 | Architecture / NPU | [深度分析](Architecture/NPU/summary-An-Efficient-Layer-Normalization-Training-Module-With-Dynamic-Quantization-for-Transformers.md) | Transformer Training、Dynamic Quantization、FPGA、ASIC |
| SigmaQuant: Hardware-Aware Heterogeneous Quantization Method for Edge DNN Inference | 2026 | Algorithm / Quantization | [深度分析](Algorithm/Quantization/summary-SigmaQuant-Hardware-Aware-Heterogeneous-Quantization-Method-for-Edge-DNN-Inference.md) | Edge Inference、Mixed Precision、Accelerator |
| Designing AI Chip Hardware and Software | 2026 | Architecture / NPU | [深度分析](Architecture/NPU/summary-Designing-AI-Chip-Hardware-and-Software.md) | AI CPU、Systolic Array、LLM Inference、Co-design、Compiler |
| DeepSeek-V3 Technical Report | 2025 | Architecture / NPU | [深度分析](Architecture/NPU/summary-DeepSeek-V3-Technical-Report.md) | DeepSeek、MoE、MLA、FP8、LLM Training |
| Hardware-Centric Analysis of DeepSeek’s Multi-Head Latent Attention | 2025 | Architecture / NPU | [深度分析](Architecture/NPU/summary-Hardware-Centric-Analysis-of-DeepSeek-s-Multi-Head-Latent-Attention.md) | MLA、KV Cache、Dataflow Accelerator、Roofline |
| DSPE: An Energy-Efficient Edge Processor for DeepSeek Inference with MerkleTree-based Incremental Pruning, Multi-Stage Boothing Lookup and Dynamic Adaptive Posit Processing | 2026 | Architecture / NPU | [深度分析](Architecture/NPU/summary-DSPE-An-Energy-Efficient-Edge-Processor-for-DeepSeek-Inference-with-MerkleTree-based-Incremental-Pruning-Multi-Stage-Boothing-Lookup-and-Dynamic-Adaptive-Posit-Processing.md) | Edge Inference、Pruning、Approximate Computing、Posit |
| A Software-defined Tensor Streaming Multiprocessor for Large-scale Machine Learning | 2022 | Architecture / NPU | [深度分析](Architecture/NPU/summary-A-Software-defined-Tensor-Streaming-Multiprocessor-for-Large-scale-Machine-Learning.md) | Groq、TSP、Dragonfly、Deterministic Network |
| A Research Retrospective on the AMD Exascale Computing Journey | 2023 | Architecture / GPU | [深度分析](Architecture/GPU/summary-A-Research-Retrospective-on-the-AMD-Exascale-Computing-Journey.md) | AMD、Frontier、CDNA、Chiplet、HBM、Co-design |
| TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings | 2023 | Architecture / NPU | [深度分析](Architecture/NPU/summary-TPU-v4-An-Optically-Reconfigurable-Supercomputer-for-Machine-Learning-with-Hardware-Support-for-Embeddings.md) | TPU、OCS、SparseCore、Embedding、3D Torus |
| Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures | 2025 | Architecture / NPU | [深度分析](Architecture/NPU/summary-Insights-into-DeepSeek-V3-Scaling-Challenges-and-Reflections-on-Hardware-for-AI-Architectures.md) | DeepSeek、MoE、FP8、Multi-Plane Network、Co-design |
| Think Fast: A Tensor Streaming Processor (TSP) for Accelerating Deep Learning Workloads | 2020 | Architecture / NPU | [深度分析](Architecture/NPU/summary-Think-Fast-A-Tensor-Streaming-Processor-TSP-for-Accelerating-Deep-Learning-Workloads.md) | Groq、TSP、Dataflow、Batch-1 Inference |
| Ten Lessons From Three Generations Shaped Google’s TPUv4i Industrial Product | 2021 | Architecture / NPU | [深度分析](Architecture/NPU/summary-Ten-Lessons-From-Three-Generations-Shaped-Google-s-TPUv4i-Industrial-Product.md) | TPU、DSA、CMEM、Compiler、Perf/TCO |
| Memory: Feeding AI’s Voracious Hunger for Data | 2026 | Architecture / Memory | [深度分析](Architecture/Memory/summary-Memory-Feeding-AI-s-Voracious-Hunger-for-Data.md) | AI Memory、HBM、PIM、Memory Wall |
| Evolving Memory Architectures for AI | 2026 | Architecture / Memory | [深度分析](Architecture/Memory/summary-Evolving-Memory-Architectures-for-AI.md) | HBM、DDR5、封装、RAS、热设计 |
| HBF in AI Compute: A System Architect’s View | 2026 | Architecture / Memory | [深度分析](Architecture/Memory/summary-HBF-in-AI-Compute-A-System-Architect-s-View.md) | HBF、Flash、MoE、KV Cache、Memory Tiering |
| Raptor: The First 3D-DRAM Accelerator for Generative Inference | 2026 | Architecture / Memory | [深度分析](Architecture/Memory/summary-Raptor-The-First-3D-DRAM-Accelerator-for-Generative-Inference.md) | 3D DRAM、HBM、ECC、DBI、Memory Bandwidth |
| AMD Instinct MI400 Series GPU Architecture | 2026 | Architecture / GPU | [深度分析](Architecture/GPU/summary-AMD-Instinct-MI400-Series-GPU-Architecture.md) | AMD、MI400、CDNA、HBM4、Rack-scale |
| System Architecture of the AMD MI400 Series GPU | 2026 | Architecture / GPU | [深度分析](Architecture/GPU/summary-System-Architecture-of-the-AMD-MI400-Series-GPU.md) | AMD、MI400、AI-NIC、Infinity Fabric、Co-design |
| Crescent Island: GPU Designed for Agentic AI Inference | 2026 | Architecture / GPU | [深度分析](Architecture/GPU/summary-Crescent-Island-GPU-Designed-for-Agentic-AI-Inference.md) | Intel、GPU、Agentic AI、Inference、Memory |
| NVIDIA Rubin GPU: Driving the Era of Agentic AI | 2026 | Architecture / GPU | [深度分析](Architecture/GPU/summary-NVIDIA-Rubin-GPU-Driving-the-Era-of-Agentic-AI.md) | NVIDIA、Rubin、Agentic AI、NVL72、Co-design |
| Rack-Scale Architecture for Wafer Scale Engine | 2026 | Architecture / NPU | [深度分析](Architecture/NPU/summary-Rack-Scale-Architecture-for-Wafer-Scale-Engine.md) | Cerebras、WSE、Wafer-scale、Rack-scale、Networking |
| Meta’s Custom AI Silicon: From Recommendation to Dual-Mandate with GenAI | 2026 | Architecture / NPU | [深度分析](Architecture/NPU/summary-Meta-s-Custom-AI-Silicon-From-Recommendation-to-Dual-Mandate-with-GenAI.md) | Meta、MTIA、Recommendation、GenAI、HBM |
| Maia 200: A Data Center Scale AI Accelerator for Large Scale Inference using Software Defined Dataflow | 2026 | Architecture / NPU | [深度分析](Architecture/NPU/summary-Maia-200-A-Data-Center-Scale-AI-Accelerator-for-Large-Scale-Inference-using-Software-Defined-Dataflow.md) | Microsoft、Maia 200、Dataflow、Inference、HBM |
| The Eighth Generation TPU Family: Two Chips Optimized for the Agentic Era | 2026 | Architecture / NPU | [深度分析](Architecture/NPU/summary-The-Eighth-Generation-TPU-Family-Two-Chips-Optimized-for-the-Agentic-Era.md) | Google、TPU8t、TPU8i、Agentic AI、Pod |
| Dataflow at Scale: the SN50 RDU | 2026 | Architecture / NPU | [深度分析](Architecture/NPU/summary-Dataflow-at-Scale-the-SN50-RDU.md) | SambaNova、RDU、Dataflow、MBU、MoE |
| Think Fast: LPU Accelerator for Heterogeneous Compute | 2026 | Architecture / NPU | [深度分析](Architecture/NPU/summary-Think-Fast-LPU-Accelerator-for-Heterogeneous-Compute.md) | NVIDIA、Groq、LPU、VLIW、Heterogeneous Inference |
| Jalapeño ASIC + System | 2026 | Architecture / NPU | [深度分析](Architecture/NPU/summary-Jalapeno-ASIC-System.md) | OpenAI、ASIC、HBM4、Spatial Architecture、PPA |

## 收录规则

1. 每份资料只保存一份，并归入一个主目录；跨领域信息记录在索引的“相关主题”中。
2. 文件名使用正式完整标题，以连字符分隔并保留原始大小写和数字；完整标题写入索引。
3. 新资料先检查标题、来源、重复情况和候选分类。分类明确时提出归档建议；跨一级领域时列出候选位置及理由，由资料库维护者决定。
4. 分类确认后再移动或重命名文件，并在同一次整理中更新本索引。
