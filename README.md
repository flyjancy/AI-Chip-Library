# Technical Library

这是一个以 AI Chip 为核心、持续收录其他技术主题资料的个人阅读资料库，内容包括论文、博客和技术文档。

## 目录结构

资料按两级主题组织：一级目录表示技术领域，二级目录表示该领域内的模块、技术或设计任务。二级目录只在有实际资料时创建。

```text
.
|-- AI-Aided-Design/
|   `-- RTL-Optimization/
|-- Algorithm/
|   `-- Quantization/
`-- Architecture/
    `-- NPU/
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

## 收录规则

1. 每份资料只保存一份，并归入一个主目录；跨领域信息记录在索引的“相关主题”中。
2. 文件名使用正式完整标题，以连字符分隔并保留原始大小写和数字；完整标题写入索引。
3. 新资料先检查标题、来源、重复情况和候选分类。分类明确时提出归档建议；跨一级领域时列出候选位置及理由，由资料库维护者决定。
4. 分类确认后再移动或重命名文件，并在同一次整理中更新本索引。
