# EVA Flame-Retardant Formulation Optimization

> 基于 **Machine Learning + Multi-objective Optimization + Bayesian Optimization** 的 EVA 阻燃配方研发项目。  
> 核心目标：利用有限实验数据，在 **阻燃性能、透明度与模型可靠性** 之间寻找更值得验证的下一批配方。

---

## 1. Project Overview

本项目来源于 EVA 阻燃材料研发场景，关注一个实际的配方设计问题：

> **如何利用已有实验数据，在 LOI、UL-94 和透光率存在性能权衡的情况下，减少盲目试配，并推荐更值得进行下一轮实验的配方？**

项目使用 **100 组 EVA 历史配方实验数据**，分别建立：

- LOI 回归模型
- UL-94 分类模型
- Transmittance 回归模型

随后结合：

```text
Machine Learning
        ↓
Multi-objective Prediction
        ↓
Pareto Optimization
        ↓
Bayesian Optimization
        ↓
Uncertainty + Applicability Domain
        ↓
Batch Experiment Selection
        ↓
Recommended Next Experiments
```

最终目标不是寻找“预测值最高”的配方，而是寻找：

> **性能较好、模型相对可信、同时具有实验价值的候选配方。**

---

## 2. Dataset

当前数据集包含：

```text
100 EVA formulation experiments
```

主要输入变量：

| Type | Variables |
|---|---|
| Polymer | EVA_content, Polymer_A, Polymer_B |
| Flame retardants | FR_A, FR_B, FR_C, FR_D |
| Additives | Additive_1, Additive_2 |

预测目标：

| Target | Task | Objective |
|---|---|---|
| LOI | Regression | Higher is better |
| UL-94 | Classification | Improve flame-retardant level |
| Transmittance | Regression | Higher is better |
| Haze | Experimental reference | Feedback / validation |

所有候选配方需要满足：

```text
Total formulation ≈ 100 wt%
```

同时将连续优化结果转换为实际实验可执行的配方精度，避免出现没有实际意义的过多小数位。

---

## 3. Predictive Models

### LOI

比较：

```text
Linear Regression
Random Forest
```

5-fold CV：

| Metric | Result |
|---|---:|
| MAE | 0.538 ± 0.078 |
| RMSE | 0.681 ± 0.095 |
| R² | 0.926 ± 0.015 |

最终选择：

```text
Linear Regression
```

模型选择完成后，使用全部 100 组已有实验数据重新训练最终模型，用于候选配方预测。

---

### UL-94

比较：

```text
Logistic Regression
Random Forest
```

最终选择：

```text
Logistic Regression
```

结果：

| Metric | Score |
|---|---:|
| Accuracy | 0.900 |
| Macro Precision | 0.926 |
| Macro Recall | 0.926 |
| Macro F1 | 0.917 |

当前类别分布：

| UL-94 | Samples |
|---|---:|
| NR | 37 |
| V-2 | 43 |
| V-1 | 20 |
| V-0 | **0** |

> **重要限制：当前训练数据没有 V-0 样本，因此模型不能被描述为已经具备可靠的 V-0 预测能力。**

---

### Transmittance

比较：

```text
Linear Regression
Random Forest
```

最终选择：

```text
Linear Regression
```

5-fold CV：

| Metric | Result |
|---|---:|
| MAE | 0.920 ± 0.109 |
| RMSE | 1.124 ± 0.180 |
| R² | 0.958 ± 0.018 |

---

## 4. Multi-Objective Optimization

真实配方研发中：

```text
LOI ↑
UL-94 ↑
Transmittance ↑
```

并不一定能够同时达到最优。

例如，提高阻燃剂用量可能改善阻燃性能，但同时影响：

```text
transparency
processability
formulation balance
```

因此项目没有简单构造一个“最高预测分数”，而是使用：

```text
Pareto Optimization
```

寻找无法在所有目标上被其他候选同时超越的配方。

![Pareto Front](docs/images/pareto_loi_transmittance.png)

---

## 5. Bayesian Optimization & Reliability Control

在 Pareto 分析基础上进一步使用 Gaussian Process 进行 Bayesian Optimization。

模型同时考虑：

```text
Predicted utility μ(x)
+
Predictive uncertainty σ(x)
```

从而在：

```text
Exploitation
vs
Exploration
```

之间进行平衡。

但材料优化存在一个重要问题：

> **优化器很容易把配方推向历史数据覆盖不足的区域。**

因此项目进一步加入 reliability screening：

```text
Formulation constraints
        ↓
Experimental property ranges
        ↓
Gaussian-process uncertainty
        ↓
Applicability-domain distance
        ↓
Distance from historical formulations
        ↓
Pareto filtering
```

这里使用的是：

> **Safety-aware Bayesian Optimization**

它是一套面向工程应用的约束与可靠性筛选流程，并不声称实现理论上的 SafeOpt 安全保证。

当前 pipeline：

```text
100   historical experiments
 ↓
20    safety-aware BO candidates
 ↓
10    within experimental performance ranges
 ↓
7     Pareto candidates
 ↓
5     final model-based recommendations
```

这 5 个候选目前仍然是：

```text
Model-based recommendations
```

而不是已经完成实验验证的最终配方。

---

## 6. Next-Experiment Selection

最终没有直接选择预测分数最高的 5 个配方。

下一轮实验设计综合考虑：

```text
60%  Predicted performance / safety
25%  Model uncertainty
15%  Formulation diversity
```

这样既包含 exploitation，也保留 exploration，并避免一次实验全部集中在非常相似的配方区域。

当前推荐：

| Rank | EVA | Polymer A | Polymer B | Pred. LOI | UL-94 | Trans. | Utility STD | Reason |
|---:|---:|---:|---:|---:|---|---:|---:|---|
| 1 | 46.0 | 10 | 15 | 30.3 | V-1 | 70.9 | 0.070 | High performance |
| 2 | 56.6 | 0 | 15 | 29.6 | V-1 | 72.6 | 0.056 | High performance |
| 3 | 53.8 | 15 | 5 | 29.4 | V-1 | 72.1 | 0.069 | Diversity |
| 4 | 58.4 | 5 | 10 | 29.1 | V-1 | 73.0 | 0.071 | Diversity |
| 5 | 52.4 | 5 | 10 | 30.5 | V-1 | 69.4 | 0.081 | Exploration |

完整结果：

```text
results/next_experiment_selection/next_experiments.csv
```

![Final Optimization Result](docs/images/final_optimization_result.png)

---

## 7. Experimental Closed Loop

项目预留了真实实验结果回填接口：

```text
Model recommendation
        ↓
Laboratory experiment
        ↓
Real LOI / UL-94 / Transmittance
        ↓
Data validation
        ↓
Dataset update
        ↓
Model retraining
        ↓
Next optimization cycle
```

创建实验记录模板：

```bash
python src/update_experimental_data.py --create-template
```

真实实验完成后，将测量结果填写到：

```text
new_experimental_results.csv
```

首先进行 dry-run validation：

```bash
python src/update_experimental_data.py
```

检查：

```text
Missing values
Numeric fields
UL-94 labels
Property ranges
Formulation balance
```

验证通过后：

```bash
python src/update_experimental_data.py --commit
```

再运行：

```bash
python src/run_closed_loop.py
```

重新完成：

```text
Model retraining
→ Multi-objective prediction
→ Pareto
→ Bayesian Optimization
→ Reliability screening
→ Next-experiment recommendation
```

> **模型预测值不会作为新的实验标签回填。只有真实测量数据才允许进入下一轮训练。**

当前 closed-loop interface 已实现，但尚未声称已经完成真实实验反馈循环。

---

## 8. Model Diagnostics

项目保留了必要的模型诊断与优化结果，包括：

```text
Actual vs Predicted
Residual analysis
Model comparison
Regression coefficients
UL-94 confusion matrix
Pareto front
Bayesian optimization outputs
Next-experiment recommendations
```

![Actual vs Predicted](docs/images/actual_vs_predicted.png)

模型选择原则不是：

```text
more complex model = better model
```

而是优先考虑：

```text
Cross-validation performance
+
Small-data robustness
+
Interpretability
+
Materials-domain consistency
```

---

## 9. Project Structure

```text
EVA project/
│
├── data/
│   ├── EVA data 100.xlsx
│   ├── polymer_dataset_clean.csv
│   └── new_experimental_results_template.csv
│
├── docs/images/
│
├── models/
│   ├── loi_model.pkl
│   ├── ul94_model.pkl
│   └── transmittance_model.pkl
│
├── src/
│   ├── data_cleaning.py
│   ├── loi_model.py
│   ├── ul94_model.py
│   ├── transmittance_model.py
│   ├── multi_objective_model.py
│   ├── pareto_analysis.py
│   ├── bayesian_optimization.py
│   ├── safe_bayesian_optimization.py
│   ├── final_recommendation.py
│   ├── next_experiment_recommendation.py
│   ├── update_experimental_data.py
│   └── run_closed_loop.py
│
├── results/
├── requirements.txt
└── README.md
```

安装依赖：

```bash
pip install -r requirements.txt
```

---

## 10. Limitations

当前项目边界明确：

- 数据量只有 **100 组历史实验**
- UL-94 数据没有 **V-0** 样本
- 推荐的 5 个候选配方尚未进行真实实验验证
- 当前 uncertainty / applicability-domain 方法属于工程可靠性控制，而不是严格概率安全保证
- 模型能力依赖于现有实验设计空间的覆盖范围

因此项目的目标不是：

```text
AI directly gives the final formulation
```

而是：

> **利用历史实验数据缩小配方搜索空间，并帮助研发人员决定下一批最值得做的实验。**

---

## Why This Project Matters

这个项目更关注材料研发 workflow，而不是单纯比较算法 leaderboard。

它展示了：

```text
Experimental data
+
Machine Learning
+
Multi-objective Optimization
+
Uncertainty
+
Materials Constraints
+
Experiment Planning
```

如何组合成一个实际可执行的配方研发流程。

### Interview Summary

> **我基于 100 组 EVA 配方实验数据分别建立 LOI、UL-94 和透光率模型，再通过 Pareto 与 Bayesian Optimization 搜索多目标候选。考虑到小样本材料模型容易在数据覆盖不足区域产生不可靠外推，我进一步加入模型不确定性、适用域和配方距离进行筛选。最终不是直接选择预测值最高的配方，而是结合性能、不确定性和配方多样性推荐下一批 5 组实验，并建立真实实验结果回填与模型重训练接口。**

---

## Core Takeaway

> **The goal is not to replace experiments, but to make the next experiment more informative.**

即：

```text
Historical experiments
        ↓
Learn
        ↓
Screen
        ↓
Recommend
        ↓
Experiment
        ↓
Learn again
```