import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# STEP 12 - PARETO FRONT ANALYSIS
# ============================================================

print("=" * 70)
print("STEP 12 - PARETO FRONT ANALYSIS")
print("=" * 70)


# ============================================================
# Path configuration
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "polymer_dataset_clean.csv"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "pareto"
)

os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# Model paths
# ============================================================

LOI_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "loi_model.pkl"
)

UL94_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "ul94_model.pkl"
)

TRANSMITTANCE_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "transmittance_model.pkl"
)


# ============================================================
# Feature configuration
# ============================================================

FEATURES = [
    "EVA_content",
    "Polymer_A",
    "Polymer_B",
    "FR_A",
    "FR_B",
    "FR_C",
    "FR_D",
    "Additive_1",
    "Additive_2"
]


# ============================================================
# Load dataset
# ============================================================

print("Loading cleaned dataset...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Cleaned dataset not found:\n{DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print(f"Samples: {len(df)}")


# ============================================================
# Load models
# ============================================================

print("Loading LOI model...")

if not os.path.exists(LOI_MODEL_PATH):
    raise FileNotFoundError(
        f"LOI model not found:\n{LOI_MODEL_PATH}"
    )

loi_model = joblib.load(LOI_MODEL_PATH)


print("Loading UL-94 model...")

if not os.path.exists(UL94_MODEL_PATH):
    raise FileNotFoundError(
        f"UL-94 model not found:\n{UL94_MODEL_PATH}"
    )

ul94_model = joblib.load(UL94_MODEL_PATH)


print("Loading Transmittance model...")

if not os.path.exists(TRANSMITTANCE_MODEL_PATH):
    raise FileNotFoundError(
        f"Transmittance model not found:\n{TRANSMITTANCE_MODEL_PATH}"
    )

transmittance_model = joblib.load(
    TRANSMITTANCE_MODEL_PATH
)


# ============================================================
# Prepare features
# ============================================================

X = df[FEATURES].copy()


# ============================================================
# Predict LOI
# ============================================================

print("Predicting multi-objective performance...")

predicted_loi = loi_model.predict(X)


# ============================================================
# Predict UL-94 probabilities
# ============================================================

ul94_probabilities = ul94_model.predict_proba(X)

ul94_classes = list(ul94_model.classes_)

probability_df = pd.DataFrame(
    ul94_probabilities,
    columns=[
        f"P({cls})"
        for cls in ul94_classes
    ]
)


# ============================================================
# UL-94 Fire Score
# ============================================================
#
# Scoring:
#
# NR  = 0
# V-2 = 0.3333
# V-1 = 0.6667
# V-0 = 1.0000
#
# The score is calculated using the expected value
# of UL-94 class probabilities.
# ============================================================

UL94_SCORE_MAP = {
    "NR": 0.0,
    "V-2": 1.0 / 3.0,
    "V-1": 2.0 / 3.0,
    "V-0": 1.0
}

fire_score = np.zeros(len(df))

for cls in ul94_classes:

    if cls in UL94_SCORE_MAP:

        fire_score += (
            probability_df[f"P({cls})"]
            * UL94_SCORE_MAP[cls]
        )


# ============================================================
# Predicted UL-94 class
# ============================================================

predicted_ul94_index = np.argmax(
    ul94_probabilities,
    axis=1
)

predicted_ul94 = np.array(
    ul94_classes
)[predicted_ul94_index]


# ============================================================
# Predict Transmittance
# ============================================================

predicted_transmittance = (
    transmittance_model.predict(X)
)


# ============================================================
# Build prediction dataframe
# ============================================================

prediction_df = df[
    ["sample_ID"]
].copy()

prediction_df["Predicted_LOI"] = predicted_loi

prediction_df["Predicted_Fire_Score"] = fire_score

prediction_df["Predicted_Transmittance"] = (
    predicted_transmittance
)

prediction_df["Predicted_UL94"] = (
    predicted_ul94
)


# ============================================================
# Pareto front calculation
# ============================================================
#
# Objectives:
#
# 1. Maximize LOI
# 2. Maximize Fire Score
# 3. Maximize Transmittance
#
# A solution is dominated if another solution is
# at least as good in every objective and strictly
# better in at least one objective.
# ============================================================

print("Calculating Pareto front...")


OBJECTIVES = prediction_df[
    [
        "Predicted_LOI",
        "Predicted_Fire_Score",
        "Predicted_Transmittance"
    ]
].values


def is_dominated(candidate, others):

    """
    Determine whether candidate is dominated.

    All objectives are maximized.
    """

    for other in others:

        no_worse = np.all(
            other >= candidate
        )

        strictly_better = np.any(
            other > candidate
        )

        if no_worse and strictly_better:
            return True

    return False


pareto_mask = np.ones(
    len(OBJECTIVES),
    dtype=bool
)


for i in range(len(OBJECTIVES)):

    if is_dominated(
        OBJECTIVES[i],
        np.delete(OBJECTIVES, i, axis=0)
    ):

        pareto_mask[i] = False


pareto_df = prediction_df[
    pareto_mask
].copy()


# ============================================================
# Sort Pareto solutions
# ============================================================

pareto_df = pareto_df.sort_values(
    by=[
        "Predicted_LOI",
        "Predicted_Fire_Score",
        "Predicted_Transmittance"
    ],
    ascending=[
        False,
        False,
        False
    ]
).reset_index(drop=True)


print()
print(f"Total samples: {len(prediction_df)}")
print(f"Pareto solutions: {len(pareto_df)}")


# ============================================================
# Display Pareto solutions
# ============================================================

print()
print("Pareto solutions:")

print(
    pareto_df.to_string(
        index=False
    )
)


# ============================================================
# Save all predictions
# ============================================================

all_predictions_path = os.path.join(
    RESULT_DIR,
    "pareto_all_predictions.csv"
)

prediction_df.to_csv(
    all_predictions_path,
    index=False
)


# ============================================================
# Save Pareto results
# ============================================================

pareto_path = os.path.join(
    RESULT_DIR,
    "pareto_solutions.csv"
)

pareto_df.to_csv(
    pareto_path,
    index=False
)


# ============================================================
# Save Pareto formulation data
# ============================================================

pareto_formulation_df = pd.merge(
    pareto_df,
    df,
    on="sample_ID",
    how="left"
)


pareto_formulation_path = os.path.join(
    RESULT_DIR,
    "pareto_formulations.csv"
)

pareto_formulation_df.to_csv(
    pareto_formulation_path,
    index=False
)


# ============================================================
# Pareto Plot 1
# LOI vs Transmittance
# ============================================================

print("Generating Pareto plots...")


plt.figure(figsize=(9, 7))

plt.scatter(
    prediction_df["Predicted_Transmittance"],
    prediction_df["Predicted_LOI"],
    alpha=0.55,
    label="All samples"
)

plt.scatter(
    pareto_df["Predicted_Transmittance"],
    pareto_df["Predicted_LOI"],
    s=70,
    label="Pareto front"
)

plt.xlabel(
    "Predicted Transmittance"
)

plt.ylabel(
    "Predicted LOI"
)

plt.title(
    "Pareto Front: LOI vs Transmittance"
)

plt.legend()

plt.grid(
    alpha=0.25
)

plt.tight_layout()

pareto_plot_1 = os.path.join(
    RESULT_DIR,
    "pareto_loi_transmittance.png"
)

plt.savefig(
    pareto_plot_1,
    dpi=300
)

plt.close()


# ============================================================
# Pareto Plot 2
# LOI vs Fire Score
# ============================================================

plt.figure(figsize=(9, 7))

plt.scatter(
    prediction_df["Predicted_Fire_Score"],
    prediction_df["Predicted_LOI"],
    alpha=0.55,
    label="All samples"
)

plt.scatter(
    pareto_df["Predicted_Fire_Score"],
    pareto_df["Predicted_LOI"],
    s=70,
    label="Pareto front"
)

plt.xlabel(
    "Predicted Fire Score"
)

plt.ylabel(
    "Predicted LOI"
)

plt.title(
    "Pareto Front: LOI vs Fire Score"
)

plt.legend()

plt.grid(
    alpha=0.25
)

plt.tight_layout()

pareto_plot_2 = os.path.join(
    RESULT_DIR,
    "pareto_loi_fire_score.png"
)

plt.savefig(
    pareto_plot_2,
    dpi=300
)

plt.close()


# ============================================================
# Pareto Plot 3
# Fire Score vs Transmittance
# ============================================================

plt.figure(figsize=(9, 7))

plt.scatter(
    prediction_df["Predicted_Transmittance"],
    prediction_df["Predicted_Fire_Score"],
    alpha=0.55,
    label="All samples"
)

plt.scatter(
    pareto_df["Predicted_Transmittance"],
    pareto_df["Predicted_Fire_Score"],
    s=70,
    label="Pareto front"
)

plt.xlabel(
    "Predicted Transmittance"
)

plt.ylabel(
    "Predicted Fire Score"
)

plt.title(
    "Pareto Front: Fire Score vs Transmittance"
)

plt.legend()

plt.grid(
    alpha=0.25
)

plt.tight_layout()

pareto_plot_3 = os.path.join(
    RESULT_DIR,
    "pareto_fire_score_transmittance.png"
)

plt.savefig(
    pareto_plot_3,
    dpi=300
)

plt.close()


# ============================================================
# Completion
# ============================================================

print()
print("Pareto front analysis completed.")

print()
print("Results saved to:")

print(RESULT_DIR)

print()
print("Generated files:")

print(
    os.path.basename(all_predictions_path)
)

print(
    os.path.basename(pareto_path)
)

print(
    os.path.basename(pareto_formulation_path)
)

print(
    os.path.basename(pareto_plot_1)
)

print(
    os.path.basename(pareto_plot_2)
)

print(
    os.path.basename(pareto_plot_3)
)

print()
print("=" * 70)