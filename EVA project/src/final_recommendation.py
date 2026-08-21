import os
import joblib
import numpy as np
import pandas as pd


# ============================================================
# STEP 17 - FINAL RECOMMENDATION
# ============================================================

print("=" * 70)
print("STEP 17 - FINAL RECOMMENDATION")
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
    "bayesian_optimization"
)

os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# File paths
# ============================================================

SAFE_BO_PATH = os.path.join(
    RESULT_DIR,
    "safe_bo_candidates.csv"
)

SAFE_BO_ALL_PATH = os.path.join(
    RESULT_DIR,
    "safe_bo_all_candidates.csv"
)

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

FINAL_RECOMMENDATION_PATH = os.path.join(
    RESULT_DIR,
    "final_recommendations.csv"
)

SAFE_PARETO_PATH = os.path.join(
    RESULT_DIR,
    "safe_bo_pareto.csv"
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
# Utility functions
# ============================================================

def calculate_pareto_front(df):
    """
    Pareto optimization objectives:

    1. LOI               -> maximize
    2. Fire Score        -> maximize
    3. Transmittance     -> maximize

    A candidate is dominated if another candidate
    is at least as good in all objectives and
    strictly better in at least one objective.
    """

    values = df[
        [
            "Predicted_LOI",
            "Predicted_Fire_Score",
            "Predicted_Transmittance"
        ]
    ].values

    n = len(values)

    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):

        if not is_pareto[i]:
            continue

        for j in range(n):

            if i == j:
                continue

            if (
                np.all(values[j] >= values[i])
                and np.any(values[j] > values[i])
            ):
                is_pareto[i] = False
                break

    return is_pareto


def min_max_normalize(series):
    """Normalize values to 0-1."""

    min_value = series.min()
    max_value = series.max()

    if max_value == min_value:
        return pd.Series(
            np.ones(len(series)),
            index=series.index
        )

    return (
        (series - min_value)
        / (max_value - min_value)
    )


# ============================================================
# Load experimental dataset
# ============================================================

print()
print("Loading experimental dataset...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Experimental dataset not found:\n{DATA_PATH}"
    )

experimental_df = pd.read_csv(DATA_PATH)

print(
    f"Experimental samples: {len(experimental_df)}"
)


# ============================================================
# Load Safe BO candidates
# ============================================================

print()
print("Loading Safe Bayesian Optimization candidates...")

if not os.path.exists(SAFE_BO_PATH):

    if os.path.exists(SAFE_BO_ALL_PATH):
        print(
            "safe_bo_candidates.csv not found."
        )
        print(
            "Using safe_bo_all_candidates.csv instead."
        )

        safe_bo_path = SAFE_BO_ALL_PATH

    else:
        raise FileNotFoundError(
            "Safe BO result file not found:\n"
            f"{SAFE_BO_PATH}"
        )

else:
    safe_bo_path = SAFE_BO_PATH


safe_df = pd.read_csv(safe_bo_path)

print(
    f"Safe BO candidates loaded: {len(safe_df)}"
)

print(
    f"Using file:\n{safe_bo_path}"
)


# ============================================================
# Check required columns
# ============================================================

required_columns = FEATURES + [
    "Predicted_LOI",
    "Predicted_Fire_Score",
    "Predicted_UL94",
    "Predicted_Transmittance"
]

missing_columns = [
    col for col in required_columns
    if col not in safe_df.columns
]

if missing_columns:

    raise ValueError(
        "Missing required columns in Safe BO data:\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# Recalculate utility if necessary
# ============================================================

print()
print("Normalizing multi-objective performance...")

safe_df["LOI_norm"] = min_max_normalize(
    safe_df["Predicted_LOI"]
)

safe_df["Fire_norm"] = min_max_normalize(
    safe_df["Predicted_Fire_Score"]
)

safe_df["Transmittance_norm"] = min_max_normalize(
    safe_df["Predicted_Transmittance"]
)


# ============================================================
# Multi-objective score
# ============================================================

print("Calculating multi-objective score...")

# Balanced objective weighting
#
# LOI:
# higher flame-retardant performance is preferred
#
# Fire Score:
# higher UL-94 performance is preferred
#
# Transmittance:
# higher optical transparency is preferred

safe_df["Multi_Objective_Score"] = (
    0.40 * safe_df["LOI_norm"]
    + 0.40 * safe_df["Fire_norm"]
    + 0.20 * safe_df["Transmittance_norm"]
)


# ============================================================
# Safety factor
# ============================================================

print("Calculating safety factor...")

if "Nearest_Experimental_Distance" in safe_df.columns:

    distance = safe_df[
        "Nearest_Experimental_Distance"
    ].astype(float)

else:

    # If STEP 15 did not store distance,
    # calculate formulation-space distance.

    X_exp = experimental_df[FEATURES].copy()

    # Normalize formulation space
    feature_min = X_exp.min()
    feature_max = X_exp.max()

    denominator = feature_max - feature_min

    denominator = denominator.replace(
        0,
        1
    )

    X_exp_norm = (
        X_exp - feature_min
    ) / denominator

    X_candidate_norm = (
        safe_df[FEATURES]
        - feature_min
    ) / denominator

    distances = []

    for _, row in X_candidate_norm.iterrows():

        diff = X_exp_norm.values - row.values

        dist = np.sqrt(
            np.sum(diff ** 2, axis=1)
        )

        distances.append(
            np.min(dist)
        )

    distance = pd.Series(
        distances,
        index=safe_df.index
    )

    safe_df[
        "Nearest_Experimental_Distance"
    ] = distance


# Smaller distance = safer
#
# Safety factor:
# 1.0 -> very close to experimental data
# 0.0 -> far from experimental data

distance = distance.astype(float)

distance_scale = (
    distance.quantile(0.95)
)

if distance_scale <= 0:
    distance_scale = distance.max()

if distance_scale <= 0:
    distance_scale = 1.0

safe_df["Safety_Factor"] = np.clip(
    1.0 - distance / distance_scale,
    0.0,
    1.0
)


# ============================================================
# Safety-adjusted score
# ============================================================

print("Calculating safety-adjusted score...")

safe_df["Safety_Adjusted_Score"] = (
    safe_df["Multi_Objective_Score"]
    * (
        0.5
        + 0.5 * safe_df["Safety_Factor"]
    )
)


# ============================================================
# Additional safety filters
# ============================================================

print()
print("Applying final recommendation constraints...")


# Experimental performance ranges

experimental_loi_min = experimental_df["LOI"].min()
experimental_loi_max = experimental_df["LOI"].max()

experimental_trans_min = (
    experimental_df["Transmittance"].min()
)

experimental_trans_max = (
    experimental_df["Transmittance"].max()
)


safe_df["Within_LOI_Range"] = (
    safe_df["Predicted_LOI"]
    .between(
        experimental_loi_min,
        experimental_loi_max
    )
)

safe_df["Within_Transmittance_Range"] = (
    safe_df["Predicted_Transmittance"]
    .between(
        experimental_trans_min,
        experimental_trans_max
    )
)


# Prefer candidates inside experimental
# performance ranges.

valid_df = safe_df[
    safe_df["Within_LOI_Range"]
    & safe_df["Within_Transmittance_Range"]
].copy()


print(
    f"Candidates within experimental "
    f"performance ranges: {len(valid_df)}"
)


# If filtering becomes too strict,
# retain all Safe BO candidates.

if len(valid_df) < 5:

    print(
        "Fewer than 5 candidates satisfy "
        "the final performance-range filter."
    )

    print(
        "Using all Safe BO candidates "
        "for final ranking."
    )

    valid_df = safe_df.copy()


# ============================================================
# Pareto front
# ============================================================

print()
print("Calculating Safe BO Pareto front...")

pareto_mask = calculate_pareto_front(
    valid_df.reset_index(drop=True)
)

pareto_df = (
    valid_df
    .reset_index(drop=True)
    .loc[pareto_mask]
    .copy()
)

print(
    f"Safe BO Pareto candidates: {len(pareto_df)}"
)


# ============================================================
# Final ranking
# ============================================================

print()
print("Ranking final candidates...")

pareto_df = pareto_df.sort_values(
    by="Safety_Adjusted_Score",
    ascending=False
).reset_index(drop=True)


# ============================================================
# Select top 5
# ============================================================

final_df = pareto_df.head(5).copy()

final_df.insert(
    0,
    "Final_Rank",
    range(1, len(final_df) + 1)
)


# ============================================================
# Save Pareto results
# ============================================================

pareto_df.to_csv(
    SAFE_PARETO_PATH,
    index=False
)

print()
print(
    "Safe BO Pareto results saved to:"
)
print(SAFE_PARETO_PATH)


# ============================================================
# Save final recommendations
# ============================================================

final_df.to_csv(
    FINAL_RECOMMENDATION_PATH,
    index=False
)


# ============================================================
# Display final recommendations
# ============================================================

display_columns = [
    "Final_Rank",
    "EVA_content",
    "Polymer_A",
    "Polymer_B",
    "FR_A",
    "FR_B",
    "FR_C",
    "FR_D",
    "Additive_1",
    "Additive_2",
    "Predicted_LOI",
    "Predicted_Fire_Score",
    "Predicted_UL94",
    "Predicted_Transmittance",
    "Predicted_Utility",
    "Nearest_Experimental_Distance",
    "Safety_Factor",
    "Multi_Objective_Score",
    "Safety_Adjusted_Score"
]


# Only display columns that exist
display_columns = [
    col
    for col in display_columns
    if col in final_df.columns
]


print()
print("=" * 70)
print("Final recommended formulations:")
print("=" * 70)

if len(final_df) == 0:

    print(
        "No final recommendations were generated."
    )

else:

    print(
        final_df[
            display_columns
        ].to_string(
            index=False
        )
    )


# ============================================================
# Summary statistics
# ============================================================

print()
print("=" * 70)
print("FINAL RECOMMENDATION SUMMARY")
print("=" * 70)

print(
    f"Experimental samples: "
    f"{len(experimental_df)}"
)

print(
    f"Safe BO candidates loaded: "
    f"{len(safe_df)}"
)

print(
    f"Candidates within experimental "
    f"performance ranges: "
    f"{len(valid_df)}"
)

print(
    f"Safe BO Pareto candidates: "
    f"{len(pareto_df)}"
)

print(
    f"Final recommendations: "
    f"{len(final_df)}"
)


# ============================================================
# Save compact final recommendation table
# ============================================================

compact_columns = [
    "Final_Rank",
    "EVA_content",
    "Polymer_A",
    "Polymer_B",
    "Predicted_LOI",
    "Predicted_Fire_Score",
    "Predicted_UL94",
    "Predicted_Transmittance",
    "Safety_Adjusted_Score"
]

compact_columns = [
    col
    for col in compact_columns
    if col in final_df.columns
]

compact_path = os.path.join(
    RESULT_DIR,
    "final_recommendation_summary.csv"
)

final_df[
    compact_columns
].to_csv(
    compact_path,
    index=False
)


# ============================================================
# Final output
# ============================================================

print()
print(
    "Final recommendations saved to:"
)
print(FINAL_RECOMMENDATION_PATH)

print()
print(
    "Final recommendation summary saved to:"
)
print(compact_path)

print()
print("=" * 70)
print("STEP 17 completed.")
print("=" * 70)