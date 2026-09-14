# EVA Flame-Retardant Formulation Optimization

> 基于 **Machine Learning + Multi-objective Optimization + Bayesian Optimization** 的 EVA 阻燃配方优化项目。

## Project Overview

本项目使用 **100 组 EVA 配方记录**建立 LOI、UL-94 和透光率预测模型，并进一步用于多目标配方优化和下一轮实验推荐。

这些记录采用混合来源：**部分来自实际研发实验数据，部分是在材料配方约束、合理性能范围和已有实验规律基础上构建的数据**。构建数据仅用于开发和验证完整的材料机器学习 workflow，不被描述为真实测量结果。

项目目标不是直接用 AI 给出“最终配方”，而是利用已有配方数据缩小搜索空间，提高下一轮实验设计的效率。

核心流程：

```text
EVA Formulation Dataset
        ↓
LOI / UL-94 / Transmittance Models
        ↓
Pareto Optimization
        ↓
Bayesian Optimization
        ↓
Uncertainty + Applicability Domain
        ↓
Next-Experiment Selection
        ↓
Real Experimental Feedback
```

---

## Data Provenance

当前 100 组配方记录由两部分组成：

```text
Real experimental records
        +
Domain-informed constructed records
```

其中：

- 真实实验数据来自实际材料研发过程，并进行了适合公开项目的抽象化处理；
- 构建数据依据材料配方约束、合理变量范围和已有实验规律生成，用于补充小样本条件下的 workflow 开发；
- 构建数据不会被描述为真实实验测量值；
- 当前交叉验证指标反映的是该**混合来源开发数据集**上的模型表现，不能直接等同于真实工业场景中的最终泛化能力；
- 最终推荐配方仍需要真实实验验证，只有真实测得的性能数据才允许进入后续闭环训练。

因此，本项目重点展示的是：

> **如何把材料研发问题转化为“数据建模 → 多目标优化 → 可靠性筛选 → 下一轮实验设计”的完整流程。**

---

## Key Methods

| Task | Method |
|---|---|
| LOI prediction | Regression |
| UL-94 prediction | Classification |
| Transmittance prediction | Regression |
| Multi-property trade-off | Pareto Optimization |
| Formulation search | Bayesian Optimization |
| Reliability control | Uncertainty + Applicability Domain |
| Experiment planning | Performance + Uncertainty + Diversity |

模型选择并不默认“复杂模型一定更好”，而是结合交叉验证、数据规模和可解释性进行比较。

---

## Screening Result

当前 workflow：

```text
100 mixed-source formulation records
        ↓
20 safety-aware BO candidates
        ↓
10 candidates within experimental ranges
        ↓
7 Pareto candidates
        ↓
5 recommended next experiments
```

最终候选并不是简单选择预测分数最高的配方，而是综合考虑：

```text
Predicted Performance
        +
Model Uncertainty
        +
Applicability Domain
        +
Formulation Diversity
```

从而降低模型外推风险，并避免下一轮实验集中在过于相似的配方区域。

---

## Prediction Targets

项目同时考虑三个主要材料性能指标：

```text
LOI ↑
UL-94 ↑
Transmittance ↑
```

其中：

- **LOI**：回归任务
- **UL-94**：分类任务
- **Transmittance**：回归任务

由于阻燃性能和透明性能之间可能存在 trade-off，因此进一步使用 Pareto Optimization 进行多目标筛选。

---

## Reliability-Aware Optimization

Bayesian Optimization 用于进一步搜索潜在候选配方，同时考虑：

```text
Predicted utility μ(x)
+
Predictive uncertainty σ(x)
```

为了避免优化器把配方推入当前数据覆盖不足的区域，项目进一步加入：

```text
Formulation constraints
Experimental property ranges
Gaussian-process uncertainty
Applicability-domain distance
Distance from existing formulations
Pareto filtering
```

因此项目中的候选筛选不仅关注“预测性能”，也关注模型是否处在相对可信的适用范围内。

---

## Important Limitations

当前 UL-94 数据分布：

```text
NR   37
V-2  43
V-1  20
V-0   0
```

因此：

> 当前模型 **不能被认为已经具备可靠的 V-0 预测能力**。

此外，当前数据集包含构建数据，因此模型性能指标主要用于验证建模和优化流程，**不能直接作为真实工业数据上的最终泛化性能结论**。

同时，最终推荐的 5 个配方目前属于：

```text
Model-based next-experiment candidates
```

并不是已经完成真实实验验证的最终配方。

项目不会把模型预测结果直接作为新的实验标签加入训练集。

只有真实实验测得的：

```text
LOI
UL-94
Transmittance
Haze
```

才可以进入下一轮模型训练。

---

## Experimental Closed Loop

项目已经预留真实实验反馈接口：

```text
Current Formulation Dataset
        ↓
Model Training
        ↓
Optimization
        ↓
Candidate Recommendation
        ↓
Laboratory Experiment
        ↓
Real Measurement
        ↓
Dataset Update
        ↓
Model Retraining
```

当前 closed-loop workflow 已实现，但尚未声称已经完成真实实验反馈循环。

---

## Full Project

完整项目包含：

```text
EVA project/
│
├── data/
├── models/
├── src/
├── results/
├── docs/
├── requirements.txt
└── README.md
```

详细的数据处理、模型评估、Pareto Optimization、Bayesian Optimization、不确定性分析、候选配方以及实验反馈流程：

### [View Full EVA Project](EVA%20project/README.md)

---

## Project Highlights

```text
100 mixed-source formulation records
        ↓
Small-data machine learning
        ↓
Multi-objective optimization
        ↓
Uncertainty-aware screening
        ↓
Applicability-domain control
        ↓
Batch experiment planning
        ↓
Real-data feedback interface
```

项目重点不是算法堆叠，而是将：

```text
Materials R&D
+
Machine Learning
+
Optimization
+
Experimental Decision-Making
```

整合为一个完整的材料配方研发 workflow。

---

## Core Idea

> **The goal is not to replace experiments, but to make the next experiment more informative.**

```text
Experiment
   ↓
Learn
   ↓
Screen
   ↓
Optimize
   ↓
Recommend
   ↓
Experiment again
```