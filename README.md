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
    └── NPU/
```

- `Architecture`：按硬件模块分类，例如 `GPU`、`NPU`、`NoC`、`DMA`。
- `Algorithm`：按算法技术分类，例如 `Quantization`、`Sparsity`、`Training`。
- `AI-Aided-Design`：按设计任务分类，例如 `RTL-Generation`、`RTL-Optimization`、`Verification`、`Physical-Design`。

## 资料索引

| 标题 | 年份 | 类型 | 分类 | 本地文件 | Summary | 原始来源 | 相关主题 |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| Ares: Adaptive Reasoning-Effort Steering for PPA- and Cost-Aware RTL Optimization with LLM Agents | 2026 | 论文 | AI-Aided-Design / RTL-Optimization | [PDF](AI-Aided-Design/RTL-Optimization/Ares-Adaptive-Reasoning-Effort-Steering-for-PPA-and-Cost-Aware-RTL-Optimization-with-LLM-Agents.pdf) | [深度分析](AI-Aided-Design/RTL-Optimization/summary-Ares-Adaptive-Reasoning-Effort-Steering-for-PPA-and-Cost-Aware-RTL-Optimization-with-LLM-Agents.md) | [arXiv:2607.27879](https://arxiv.org/abs/2607.27879) | LLM Agent、EDA、PPA、RTL |
| An Efficient Layer Normalization Training Module With Dynamic Quantization for Transformers | 2025 | 论文 | Architecture / NPU | [PDF](Architecture/NPU/An-Efficient-Layer-Normalization-Training-Module-With-Dynamic-Quantization-for-Transformers.pdf) | [深度分析](Architecture/NPU/summary-An-Efficient-Layer-Normalization-Training-Module-With-Dynamic-Quantization-for-Transformers.md) | [DOI:10.1109/TCSII.2025.3591633](https://doi.org/10.1109/TCSII.2025.3591633) | Transformer Training、Dynamic Quantization、FPGA、ASIC |
| SigmaQuant: Hardware-Aware Heterogeneous Quantization Method for Edge DNN Inference | 2026 | 论文 | Algorithm / Quantization | [PDF](Algorithm/Quantization/SigmaQuant-Hardware-Aware-Heterogeneous-Quantization-Method-for-Edge-DNN-Inference.pdf) | [深度分析](Algorithm/Quantization/summary-SigmaQuant-Hardware-Aware-Heterogeneous-Quantization-Method-for-Edge-DNN-Inference.md) | [DOI:10.1109/TCASAI.2026.3666506](https://doi.org/10.1109/TCASAI.2026.3666506) | Edge Inference、Mixed Precision、Accelerator |
| Designing AI Chip Hardware and Software | 2026 | 技术文章 | Architecture / NPU | [DOCX](Architecture/NPU/Designing-AI-Chip-Hardware-and-Software.docx) | [深度分析](Architecture/NPU/summary-Designing-AI-Chip-Hardware-and-Software.md) | [LinkedIn 原始发布](https://www.linkedin.com/posts/bjarkeroune_designing-ai-chip-software-and-hardware-activity-7437289720855945216-3OAT) | AI CPU、Systolic Array、LLM Inference、Co-design、Compiler |
| DeepSeek-V3 Technical Report | 2025 | 论文 | Architecture / NPU | [PDF](Architecture/NPU/DeepSeek-V3-Technical-Report.pdf) | [深度分析](Architecture/NPU/summary-DeepSeek-V3-Technical-Report.md) | [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) | DeepSeek、MoE、MLA、FP8、LLM Training |
| Hardware-Centric Analysis of DeepSeek’s Multi-Head Latent Attention | 2025 | 论文 | Architecture / NPU | [PDF](Architecture/NPU/Hardware-Centric-Analysis-of-DeepSeek-s-Multi-Head-Latent-Attention.pdf) | [深度分析](Architecture/NPU/summary-Hardware-Centric-Analysis-of-DeepSeek-s-Multi-Head-Latent-Attention.md) | [arXiv:2506.02523](https://arxiv.org/abs/2506.02523) | MLA、KV Cache、Dataflow Accelerator、Roofline |
| DSPE: An Energy-Efficient Edge Processor for DeepSeek Inference with MerkleTree-based Incremental Pruning, Multi-Stage Boothing Lookup and Dynamic Adaptive Posit Processing | 2026 | 论文 | Architecture / NPU | [PDF](Architecture/NPU/DSPE-An-Energy-Efficient-Edge-Processor-for-DeepSeek-Inference-with-MerkleTree-based-Incremental-Pruning-Multi-Stage-Boothing-Lookup-and-Dynamic-Adaptive-Posit-Processing.pdf) | [深度分析](Architecture/NPU/summary-DSPE-An-Energy-Efficient-Edge-Processor-for-DeepSeek-Inference-with-MerkleTree-based-Incremental-Pruning-Multi-Stage-Boothing-Lookup-and-Dynamic-Adaptive-Posit-Processing.md) | [DOI:10.1145/3770743.3805813](https://doi.org/10.1145/3770743.3805813) | Edge Inference、Pruning、Approximate Computing、Posit |
| A Software-defined Tensor Streaming Multiprocessor for Large-scale Machine Learning | 2022 | 论文 | Architecture / NPU | [PDF](Architecture/NPU/A-Software-defined-Tensor-Streaming-Multiprocessor-for-Large-scale-Machine-Learning.pdf) | [深度分析](Architecture/NPU/summary-A-Software-defined-Tensor-Streaming-Multiprocessor-for-Large-scale-Machine-Learning.md) | [DOI:10.1145/3470496.3527405](https://doi.org/10.1145/3470496.3527405) | Groq、TSP、Dragonfly、Deterministic Network |
| A Research Retrospective on the AMD Exascale Computing Journey | 2023 | 论文 | Architecture / GPU | [PDF](Architecture/GPU/A-Research-Retrospective-on-the-AMD-Exascale-Computing-Journey.pdf) | [深度分析](Architecture/GPU/summary-A-Research-Retrospective-on-the-AMD-Exascale-Computing-Journey.md) | [DOI:10.1145/3579371.3589349](https://doi.org/10.1145/3579371.3589349) | AMD、Frontier、CDNA、Chiplet、HBM、Co-design |
| TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings | 2023 | 论文 | Architecture / NPU | [PDF](Architecture/NPU/TPU-v4-An-Optically-Reconfigurable-Supercomputer-for-Machine-Learning-with-Hardware-Support-for-Embeddings.pdf) | [深度分析](Architecture/NPU/summary-TPU-v4-An-Optically-Reconfigurable-Supercomputer-for-Machine-Learning-with-Hardware-Support-for-Embeddings.md) | [DOI:10.1145/3579371.3589350](https://doi.org/10.1145/3579371.3589350) | TPU、OCS、SparseCore、Embedding、3D Torus |
| Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures | 2025 | 论文 | Architecture / NPU | [PDF](Architecture/NPU/Insights-into-DeepSeek-V3-Scaling-Challenges-and-Reflections-on-Hardware-for-AI-Architectures.pdf) | [深度分析](Architecture/NPU/summary-Insights-into-DeepSeek-V3-Scaling-Challenges-and-Reflections-on-Hardware-for-AI-Architectures.md) | [DOI:10.1145/3695053.3731412](https://doi.org/10.1145/3695053.3731412) | DeepSeek、MoE、FP8、Multi-Plane Network、Co-design |
| Think Fast: A Tensor Streaming Processor (TSP) for Accelerating Deep Learning Workloads | 2020 | 论文 | Architecture / NPU | [PDF](Architecture/NPU/Think-Fast-A-Tensor-Streaming-Processor-TSP-for-Accelerating-Deep-Learning-Workloads.pdf) | [深度分析](Architecture/NPU/summary-Think-Fast-A-Tensor-Streaming-Processor-TSP-for-Accelerating-Deep-Learning-Workloads.md) | [DOI:10.1109/ISCA45697.2020.00023](https://doi.org/10.1109/ISCA45697.2020.00023) | Groq、TSP、Dataflow、Batch-1 Inference |
| Ten Lessons From Three Generations Shaped Google’s TPUv4i Industrial Product | 2021 | 论文 | Architecture / NPU | [PDF](Architecture/NPU/Ten-Lessons-From-Three-Generations-Shaped-Google-s-TPUv4i-Industrial-Product.pdf) | [深度分析](Architecture/NPU/summary-Ten-Lessons-From-Three-Generations-Shaped-Google-s-TPUv4i-Industrial-Product.md) | [DOI:10.1109/ISCA52012.2021.00010](https://doi.org/10.1109/ISCA52012.2021.00010) | TPU、DSA、CMEM、Compiler、Perf/TCO |

## 收录规则

1. 每份资料只保存一份，并归入一个主目录；跨领域信息记录在索引的“相关主题”中。
2. 文件名使用正式完整标题，以连字符分隔并保留原始大小写和数字；完整标题写入索引。
3. 新资料先检查标题、来源、重复情况和候选分类。分类明确时提出归档建议；跨一级领域时列出候选位置及理由，由资料库维护者决定。
4. 分类确认后再移动或重命名文件，并在同一次整理中更新本索引。
