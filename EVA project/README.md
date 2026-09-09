# EVA Polymer Multi-Objective Optimization

> Data-driven EVA formulation optimization with machine learning, multi-objective optimization, applicability-domain control, safety-aware Bayesian optimization, and batch experiment selection.

## 1. Project Overview

This project targets **EVA polymer formulation design** and builds an end-to-end workflow from historical experimental data to model-based formulation recommendation.

The project jointly considers three core properties:

- **LOI** — regression, higher is better
- **UL-94** — classification of flame-retardant level
- **Transmittance** — regression, higher is better

The objective is not simply to obtain the highest model prediction. Instead, the workflow searches for formulations that balance **predicted performance, model reliability, applicability domain, and experimental executability**.

### R&D workflow

```text
Historical experimental data
        ↓
Data cleaning & validation
        ↓
LOI / UL-94 / Transmittance models
        ↓
Model diagnostics
        ↓
Multi-objective prediction
        ↓
Pareto analysis
        ↓
Bayesian optimization
        ↓
Gaussian-process uncertainty
        ↓
Applicability-domain / formulation-distance filtering
        ↓
Safety-aware candidate screening
        ↓
Batch experiment selection
(performance + uncertainty + diversity)
        ↓
Recommended next experiments
        ↓
Real experimental measurements
        ↓
Data validation & dataset update
        ↓
Model retraining / next optimization cycle
```

The final feedback/retraining stage is implemented as a workflow interface. **No synthetic experimental feedback is used.** It is activated only after real measurements become available.

---

## 2. Dataset

The current dataset contains **100 EVA formulation experiments**.

### Input variables

| Variable | Description |
|---|---|
| EVA_content | EVA content |
| Polymer_A | Polymer component A |
| Polymer_B | Polymer component B |
| FR_A | Flame retardant A |
| FR_B | Flame retardant B |
| FR_C | Flame retardant C |
| FR_D | Flame retardant D |
| Additive_1 | Additive 1 |
| Additive_2 | Additive 2 |

### Targets

| Target | Task | Role |
|---|---|---|
| LOI | Regression | Maximize |
| UL-94 | Classification | Improve flame-retardant level |
| Transmittance | Regression | Maximize |
| Haze | Experimental reference | Stored for feedback consistency |

All formulations are checked against the formulation-balance constraint:

```text
Total formulation ≈ 100 wt%
```

Experimental-resolution constraints are also respected when generating candidate formulations. Continuous optimizer outputs are converted to the practical precision used by the historical dataset before recommendation.

---

## 3. Predictive Models

### 3.1 LOI Regression

Models compared:

- Linear Regression
- Random Forest

Model selection is based primarily on **5-fold cross-validation RMSE**, while a hold-out set is retained for an intuitive independent check.

| Evaluation | MAE | RMSE | R² |
|---|---:|---:|---:|
| Hold-out | 0.674 | 0.843 | 0.897 |
| 5-fold CV | 0.538 ± 0.078 | 0.681 ± 0.095 | 0.926 ± 0.015 |

**Selected model: Linear Regression**

After the model family is selected, the final production model is retrained on all 100 available experimental samples before candidate screening.

### 3.2 UL-94 Classification

Models compared:

- Logistic Regression
- Random Forest

**Selected model: Logistic Regression**

| Metric | Result |
|---|---:|
| Accuracy | 0.900 |
| Macro Precision | 0.926 |
| Macro Recall | 0.926 |
| Macro F1 | 0.917 |

Current class coverage:

| UL-94 class | Samples |
|---|---:|
| NR | 37 |
| V-2 | 43 |
| V-1 | 20 |
| V-0 | 0 |

> **Important limitation:** the training data contain no V-0 samples. Therefore, the current classifier has learned only NR, V-2 and V-1 and must **not** be presented as a validated V-0 predictor. Real V-0 experimental samples are required before that class can be learned reliably.

### 3.3 Transmittance Regression

Models compared:

- Linear Regression
- Random Forest

**Selected model: Linear Regression**

| Evaluation | MAE | RMSE | R² |
|---|---:|---:|---:|
| Hold-out | 0.898 | 1.005 | 0.974 |
| 5-fold CV | 0.920 ± 0.109 | 1.124 ± 0.180 | 0.958 ± 0.018 |

---

## 4. Multi-Objective Formulation Optimization

The three predictive models are integrated into one formulation-search workflow.

The optimization objective is to balance:

```text
LOI
+
UL-94 fire-performance score
+
Transmittance
```

Because these properties can conflict, a single weighted optimum is not sufficient to describe the design space. Pareto analysis is therefore used to identify formulations for which no other formulation is simultaneously better in all objectives.

![Pareto Front](docs/images/pareto_loi_transmittance.png)

---

## 5. Bayesian Optimization

A Gaussian Process surrogate is used to explore candidate formulations and estimate both:

- predicted utility, \(\mu(x)\)
- predictive uncertainty, \(\sigma(x)\)

The acquisition logic balances exploitation and exploration rather than selecting only the highest predicted response.

A key observation from the project is that unconstrained optimization can push candidate formulations into regions that are poorly represented by the experimental dataset. This creates apparently attractive predictions that may be dominated by extrapolation risk.

---

## 6. Safety-Aware Bayesian Optimization

To reduce extrapolation risk, the project adds a practical safety-aware screening layer based on:

- formulation-balance constraints
- Gaussian-process uncertainty
- experimental property ranges
- applicability-domain distance
- distance from historical formulations
- multi-objective / Pareto filtering

This is intentionally described as **safety-aware Bayesian optimization**, not theoretical SafeOpt. The method is a practical engineering constraint layer for materials formulation screening and does not claim formal safety guarantees.

Latest closed-pipeline output:

```text
Experimental samples:                    100
Safe BO candidates loaded:               20
Within experimental performance ranges:  10
Safe BO Pareto candidates:                7
Final model-based recommendations:        5
```

The five final recommendations are **model-based candidates**, not experimentally validated formulations.

---

## 7. Batch Experiment Selection

A dedicated batch-selection layer is added after safety-aware Bayesian optimization.

Instead of simply taking the five candidates with the highest predicted score, the next-experiment strategy balances:

- **60% predicted performance / safety** — exploitation
- **25% model uncertainty** — exploration
- **15% formulation-space diversity** — avoid redundant experiments

The implementation uses a greedy batch-selection strategy in standardized formulation space.

### Current recommended next experiments

| Rank | EVA | Polymer A | Polymer B | Pred. LOI | Pred. UL-94 | Pred. Trans. | Utility STD | Selection reason |
|---:|---:|---:|---:|---:|---|---:|---:|---|
| 1 | 46.0 | 10 | 15 | 30.3 | V-1 | 70.9 | 0.070 | High predicted performance |
| 2 | 56.6 | 0 | 15 | 29.6 | V-1 | 72.6 | 0.056 | High predicted performance |
| 3 | 53.8 | 15 | 5 | 29.4 | V-1 | 72.1 | 0.069 | Batch diversity |
| 4 | 58.4 | 5 | 10 | 29.1 | V-1 | 73.0 | 0.071 | Batch diversity |
| 5 | 52.4 | 5 | 10 | 30.5 | V-1 | 69.4 | 0.081 | Exploration / high model uncertainty |

The complete formulations and model diagnostics are saved to:

```text
results/next_experiment_selection/next_experiments.csv
```

This layer is best interpreted as **practical batch experiment selection built on top of Bayesian optimization**, rather than as a separate theoretical Active Learning algorithm.

---

## 8. Experimental Feedback Interface

The project includes an explicit interface for future real experimental feedback.

### Create a blank measurement template

```bash
python src/update_experimental_data.py --create-template
```

This creates:

```text
data/new_experimental_results_template.csv
```

Model predictions are stored only as reference columns. The following columns must remain blank until real experiments are performed:

```text
LOI
UL_94
Transmittance
Haze
```

### Validate real feedback

After completing experiments:

1. copy the template
2. rename it to `new_experimental_results.csv`
3. fill in **real measured values only**
4. run a dry validation

```bash
python src/update_experimental_data.py
```

The validation checks:

- missing values
- numeric fields
- allowed UL-94 labels
- property ranges
- formulation balance

No dataset is modified during dry-run validation.

### Commit validated real measurements

Only after validation passes:

```bash
python src/update_experimental_data.py --commit
```

The script backs up the existing raw dataset before appending new experimental rows.

> Model predictions are never copied back as experimental labels. This prevents self-reinforcing data leakage.

---

## 9. Closed-Loop Retraining Workflow

Once **real new experimental measurements** have been committed, the full workflow can be rerun with:

```bash
python src/run_closed_loop.py
```

The pipeline automatically performs:

```text
Data cleaning
→ LOI retraining
→ UL-94 retraining
→ Transmittance retraining
→ Multi-objective integration
→ Pareto analysis
→ Bayesian optimization
→ Safety-aware Bayesian optimization
→ Final recommendation
→ Next-experiment batch selection
```

The pipeline is implemented and executable, but the current repository does **not** claim that a real feedback cycle has already been completed.

---

## 10. Model Diagnostics

The repository includes:

- Actual vs Predicted plots
- residual analysis
- regression coefficient analysis
- UL-94 confusion matrix
- classification report
- model-comparison tables
- Pareto visualization
- optimization summary outputs

![Actual vs Predicted](docs/images/actual_vs_predicted.png)

![Final Optimization Result](docs/images/final_optimization_result.png)

---

## 11. Project Structure

```text
EVA project/
│
├── data/
│   ├── EVA data 100.xlsx
│   ├── polymer_dataset_clean.csv
│   └── new_experimental_results_template.csv
│
├── docs/
│   └── images/
│       ├── actual_vs_predicted.png
│       ├── final_optimization_result.png
│       ├── pareto_loi_transmittance.png
│       └── project_summary.png
│
├── models/
│   ├── loi_model.pkl
│   ├── ul94_model.pkl
│   └── transmittance_model.pkl
│
├── src/
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── loi_model.py
│   ├── loi_diagnostics.py
│   ├── ul94_model.py
│   ├── ul94_diagnostics.py
│   ├── transmittance_model.py
│   ├── transmittance_diagnostics.py
│   ├── multi_objective_model.py
│   ├── pareto_analysis.py
│   ├── bayesian_optimization.py
│   ├── safe_bayesian_optimization.py
│   ├── final_recommendation.py
│   ├── next_experiment_recommendation.py
│   ├── update_experimental_data.py
│   ├── run_closed_loop.py
│   └── final_report.py
│
├── results/
│   ├── eda/
│   ├── loi_model/
│   ├── ul94_model/
│   ├── transmittance_model/
│   ├── multi_objective/
│   ├── pareto/
│   ├── bayesian_optimization/
│   ├── next_experiment_selection/
│   └── final_project_report/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 12. Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Core dependencies include:

- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Seaborn
- joblib
- openpyxl

The current repository uses practical formulation precision consistent with the historical experimental dataset rather than reporting optimizer outputs with unrealistic decimal precision.

---

## 13. Key Engineering Decisions

### 1. Small-data model selection

For the 100-sample materials dataset, model choice is based on cross-validation rather than assuming that a more complex model must perform better.

### 2. Full-data retraining after model selection

Hold-out/CV results are used for evaluation and model-family selection. Once the modeling approach is fixed, all available experimental data are used to train the final model for candidate screening.

### 3. Reliability before optimum

The project distinguishes between:

```text
highest predicted score
```

and

```text
high predicted performance within a defensible model domain
```

### 4. Practical experimental resolution

Candidate formulations are rounded/discretized to the same practical resolution as the experimental dataset before recommendation.

### 5. No synthetic feedback

A candidate prediction is not treated as a new experimental label. Only real measurements may enter the next model-training cycle.

---

## 14. Limitations

This repository intentionally keeps the following limitations explicit:

1. **Dataset size is limited to 100 historical experiments.** Model uncertainty and applicability-domain constraints remain important.
2. **No V-0 samples are present.** The UL-94 classifier cannot currently be considered a validated V-0 predictor.
3. **The five recommended formulations have not yet been experimentally validated.** They are next-experiment candidates generated by the model workflow.
4. **The closed-loop retraining interface is implemented, but no fake feedback cycle is reported.** Real measurements are required before retraining.
5. The current optimization is still dependent on the representativeness and quality of the historical experimental design space.

---

## 15. Why This Project Matters for AI-Driven Materials R&D

The project is designed around a materials-R&D question rather than an algorithm benchmark:

```text
How can limited historical formulation experiments be converted into
more efficient and more reliable decisions about what to test next?
```

The repository demonstrates a workflow covering:

- materials experimental-data cleaning
- small-data machine learning
- interpretable model evaluation
- multi-property trade-off analysis
- Bayesian optimization
- uncertainty-aware screening
- applicability-domain control
- experimentally executable formulation generation
- batch experiment planning
- real-data feedback interface

The main value is therefore not any single algorithm, but the integration of **materials-domain constraints + machine learning + optimization + experimental decision-making** into one reproducible workflow.

---

## 16. Project Status

**Implemented**

- data cleaning and validation
- LOI / UL-94 / transmittance models
- cross-validation and model diagnostics
- multi-objective prediction
- Pareto analysis
- Bayesian optimization
- safety-aware candidate screening
- applicability-domain filtering
- practical formulation rounding
- batch next-experiment selection
- experimental-feedback template and validation
- closed-loop retraining pipeline interface

**Pending real laboratory work**

- preparation of recommended candidate formulations
- LOI / UL-94 / transmittance / haze measurements
- feedback of real measurements into the dataset
- retraining and comparison of the next model iteration

> **Core idea:** use machine learning and optimization to turn limited formulation data into a constrained, interpretable, and experimentally actionable next-experiment strategy.
