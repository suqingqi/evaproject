# EVA Flame-Retardant Formulation Optimization

> 基于 **Machine Learning + Multi-objective Optimization + Bayesian Optimization** 的 EVA 阻燃配方优化项目。

## Project Overview

本项目基于 **100 组 EVA 历史配方实验数据**，建立 LOI、UL-94 和透光率预测模型，并进一步用于多目标配方优化和下一轮实验推荐。

核心流程：

```text
Historical EVA Experiments
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