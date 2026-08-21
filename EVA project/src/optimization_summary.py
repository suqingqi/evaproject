import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# STEP 16 - OPTIMIZATION SUMMARY
# ============================================================

print("=" * 70)
print("STEP 16 - OPTIMIZATION SUMMARY")
print("=" * 70)


# ============================================================
# Path configuration
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data"
)

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "bayesian_optimization"
)

os.makedirs(RESULT_DIR, exist_ok=True)


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

def find_existing_file(directory, filenames):

    for filename in filenames:

        path = os.path.join(
            directory,
            filename
        )

        if os.path.exists(path):
            return path

    return None


def normalize_series(series):

    series = pd.to_numeric(
        series,
        errors="coerce"
    )

    min_value = series.min()
    max_value = series.max()

    if max_value - min_value < 1e-12:
        return pd.Series(
            np.ones(len(series)),
            index=series.index
        )

    return (
        (series - min_value)
        /
        (max_value - min_value)
    )


# ============================================================
# Load experimental dataset
# ============================================================

print()
print("Loading experimental dataset...")

DATA_PATH = find_existing_file(
    DATA_DIR,
    [
        "polymer_dataset_clean.csv"
    ]
)

if DATA_PATH is None:

    # Fallback: project root
    fallback = os.path.join(
        PROJECT_ROOT,
        "polymer_dataset_clean.csv"
    )

    if os.path.exists(fallback):
        DATA_PATH = fallback

if DATA_PATH is None:

    raise FileNotFoundError(
        "Experimental dataset not found.\n"
        "Expected:\n"
        f"{os.path.join(DATA_DIR, 'polymer_dataset_clean.csv')}"
    )


experimental_df = pd.read_csv(DATA_PATH)

print(
    f"Experimental samples: "
    f"{len(experimental_df)}"
)


# ============================================================
# Experimental reference points
# ============================================================

print()
print("Calculating experimental reference points...")

experimental_loi_max = experimental_df["LOI"].max()
experimental_loi_min = experimental_df["LOI"].min()

experimental_trans_max = (
    experimental_df["Transmittance"].max()
)

experimental_trans_min = (
    experimental_df["Transmittance"].min()
)

print(
    f"Experimental LOI range: "
    f"{experimental_loi_min:.2f} - "
    f"{experimental_loi_max:.2f}"
)

print(
    f"Experimental Transmittance range: "
    f"{experimental_trans_min:.2f} - "
    f"{experimental_trans_max:.2f}"
)


# ============================================================
# Load standard BO Pareto results
# ============================================================

print()
print("Loading standard BO Pareto results...")

STANDARD_BO_PATH = find_existing_file(
    RESULT_DIR,
    [
        "bo_pareto_candidates.csv",
        "bo_final_recommendations.csv",
        "bo_top_candidates.csv"
    ]
)

if STANDARD_BO_PATH is None:

    raise FileNotFoundError(
        "Standard BO result file not found in:\n"
        f"{RESULT_DIR}"
    )

standard_bo = pd.read_csv(
    STANDARD_BO_PATH
)

print(
    f"Standard BO file: "
    f"{os.path.basename(STANDARD_BO_PATH)}"
)

print(
    f"Standard BO candidates: "
    f"{len(standard_bo)}"
)


# ============================================================
# Load Safe BO results
# ============================================================

print()
print("Searching for Safe BO result file...")

SAFE_BO_PATH = find_existing_file(
    RESULT_DIR,
    [
        "safe_bo_candidates.csv",
        "safe_bo_all_candidates.csv"
    ]
)

if SAFE_BO_PATH is None:

    raise FileNotFoundError(
        "Safe BO result file not found.\n"
        "Expected one of:\n"
        "safe_bo_candidates.csv\n"
        "safe_bo_all_candidates.csv\n"
        f"Directory:\n{RESULT_DIR}"
    )


safe_bo = pd.read_csv(
    SAFE_BO_PATH
)

print(
    f"Safe BO file: "
    f"{os.path.basename(SAFE_BO_PATH)}"
)

print(
    f"Safe BO candidates: "
    f"{len(safe_bo)}"
)


# ============================================================
# Validate required columns
# ============================================================

required_standard_columns = [
    "Predicted_LOI",
    "Predicted_Transmittance",
    "Predicted_UL94"
]

for column in required_standard_columns:

    if column not in standard_bo.columns:

        raise ValueError(
            f"Missing column in standard BO result: "
            f"{column}"
        )


required_safe_columns = [
    "Predicted_LOI",
    "Predicted_Transmittance",
    "Predicted_UL94"
]

for column in required_safe_columns:

    if column not in safe_bo.columns:

        raise ValueError(
            f"Missing column in Safe BO result: "
            f"{column}"
        )


# ============================================================
# Select best Standard BO candidate
# ============================================================

print()
print("Selecting best standard BO candidate...")


# Standard BO ranking priority:
# 1. Utility
# 2. Fire score
# 3. Transmittance
# 4. Lower LOI is preferred

if "Predicted_Utility" in standard_bo.columns:

    standard_bo["_utility"] = pd.to_numeric(
        standard_bo["Predicted_Utility"],
        errors="coerce"
    )

else:

    standard_bo["_utility"] = 0.0


if "Predicted_Fire_Score" in standard_bo.columns:

    standard_bo["_fire"] = pd.to_numeric(
        standard_bo["Predicted_Fire_Score"],
        errors="coerce"
    )

else:

    standard_bo["_fire"] = 0.0


standard_bo["_trans"] = pd.to_numeric(
    standard_bo["Predicted_Transmittance"],
    errors="coerce"
)

standard_bo["_loi"] = pd.to_numeric(
    standard_bo["Predicted_LOI"],
    errors="coerce"
)


standard_bo_sorted = standard_bo.sort_values(
    by=[
        "_utility",
        "_fire",
        "_trans",
        "_loi"
    ],
    ascending=[
        False,
        False,
        False,
        True
    ]
)

best_standard = (
    standard_bo_sorted.iloc[0]
)


# ============================================================
# Select best Safe BO candidate
# ============================================================

print("Selecting best Safe BO candidate...")


if "Safe_UCB" in safe_bo.columns:

    safe_bo["_safe_score"] = pd.to_numeric(
        safe_bo["Safe_UCB"],
        errors="coerce"
    )

elif "Safety_Adjusted_Score" in safe_bo.columns:

    safe_bo["_safe_score"] = pd.to_numeric(
        safe_bo["Safety_Adjusted_Score"],
        errors="coerce"
    )

elif "Predicted_Utility" in safe_bo.columns:

    safe_bo["_safe_score"] = pd.to_numeric(
        safe_bo["Predicted_Utility"],
        errors="coerce"
    )

else:

    safe_bo["_safe_score"] = 0.0


safe_bo["_fire"] = pd.to_numeric(
    safe_bo.get(
        "Predicted_Fire_Score",
        pd.Series(
            np.zeros(len(safe_bo))
        )
    ),
    errors="coerce"
)

safe_bo["_trans"] = pd.to_numeric(
    safe_bo["Predicted_Transmittance"],
    errors="coerce"
)

safe_bo["_loi"] = pd.to_numeric(
    safe_bo["Predicted_LOI"],
    errors="coerce"
)


# Prefer candidates inside experimental performance ranges
safe_bo["_inside_performance_range"] = (
    (safe_bo["_loi"] >= experimental_loi_min)
    &
    (safe_bo["_loi"] <= experimental_loi_max)
    &
    (safe_bo["_trans"] >= experimental_trans_min)
    &
    (safe_bo["_trans"] <= experimental_trans_max)
)


# Prefer applicability-domain candidates when distance exists
if "Nearest_Experimental_Distance" in safe_bo.columns:

    safe_bo["_distance"] = pd.to_numeric(
        safe_bo["Nearest_Experimental_Distance"],
        errors="coerce"
    )

else:

    safe_bo["_distance"] = np.nan


# Safe ranking
safe_bo_sorted = safe_bo.sort_values(
    by=[
        "_inside_performance_range",
        "_safe_score",
        "_fire",
        "_trans",
        "_loi"
    ],
    ascending=[
        False,
        False,
        False,
        False,
        True
    ]
)

best_safe = (
    safe_bo_sorted.iloc[0]
)


# ============================================================
# Build comparison table
# ============================================================

comparison_rows = []


# ------------------------------------------------------------
# Experimental reference
# ------------------------------------------------------------

comparison_rows.append({
    "Method": "Experimental Dataset",
    "LOI": experimental_loi_max,
    "Transmittance": experimental_trans_max,
    "UL94": "Experimental",
    "Utility": np.nan
})


# ------------------------------------------------------------
# Standard BO
# ------------------------------------------------------------

standard_utility = (
    best_standard["Predicted_Utility"]
    if "Predicted_Utility" in best_standard.index
    else np.nan
)

comparison_rows.append({
    "Method": "Standard Bayesian Optimization",
    "LOI": best_standard["Predicted_LOI"],
    "Transmittance": (
        best_standard["Predicted_Transmittance"]
    ),
    "UL94": best_standard["Predicted_UL94"],
    "Utility": standard_utility
})


# ------------------------------------------------------------
# Safe BO
# ------------------------------------------------------------

safe_utility = (
    best_safe["Predicted_Utility"]
    if "Predicted_Utility" in best_safe.index
    else np.nan
)

comparison_rows.append({
    "Method": "Safe Bayesian Optimization",
    "LOI": best_safe["Predicted_LOI"],
    "Transmittance": (
        best_safe["Predicted_Transmittance"]
    ),
    "UL94": best_safe["Predicted_UL94"],
    "Utility": safe_utility
})


comparison = pd.DataFrame(
    comparison_rows
)


# ============================================================
# Save comparison
# ============================================================

summary_path = os.path.join(
    RESULT_DIR,
    "optimization_summary.csv"
)

comparison.to_csv(
    summary_path,
    index=False
)


# ============================================================
# Print summary
# ============================================================

print()
print("Optimization comparison:")

print(
    comparison.to_string(
        index=False
    )
)

print()
print("Optimization summary saved to:")
print(summary_path)


# ============================================================
# Save detailed selected candidates
# ============================================================

selected_path = os.path.join(
    RESULT_DIR,
    "optimization_selected_candidates.csv"
)


selected_rows = []


standard_record = {
    "Method": "Standard Bayesian Optimization"
}

for column in standard_bo.columns:

    if not column.startswith("_"):

        standard_record[column] = (
            best_standard[column]
        )

selected_rows.append(
    standard_record
)


safe_record = {
    "Method": "Safe Bayesian Optimization"
}

for column in safe_bo.columns:

    if not column.startswith("_"):

        safe_record[column] = (
            best_safe[column]
        )

selected_rows.append(
    safe_record
)


selected_df = pd.DataFrame(
    selected_rows
)

selected_df.to_csv(
    selected_path,
    index=False
)


print()
print("Selected candidate details saved to:")
print(selected_path)


# ============================================================
# Generate comparison plot
# ============================================================

print()
print("Generating optimization comparison plot...")


plot_df = comparison.copy()


fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)


# ------------------------------------------------------------
# LOI
# ------------------------------------------------------------

axes[0].bar(
    plot_df["Method"],
    plot_df["LOI"]
)

axes[0].set_title(
    "Predicted LOI"
)

axes[0].set_ylabel(
    "LOI"
)

axes[0].tick_params(
    axis="x",
    rotation=25
)


# ------------------------------------------------------------
# Transmittance
# ------------------------------------------------------------

axes[1].bar(
    plot_df["Method"],
    plot_df["Transmittance"]
)

axes[1].set_title(
    "Predicted Transmittance"
)

axes[1].set_ylabel(
    "Transmittance (%)"
)

axes[1].tick_params(
    axis="x",
    rotation=25
)


# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------

utility_plot = plot_df["Utility"].fillna(0)

axes[2].bar(
    plot_df["Method"],
    utility_plot
)

axes[2].set_title(
    "Optimization Utility"
)

axes[2].set_ylabel(
    "Utility"
)

axes[2].tick_params(
    axis="x",
    rotation=25
)


plt.tight_layout()


plot_path = os.path.join(
    RESULT_DIR,
    "optimization_comparison.png"
)

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print()
print("Optimization comparison plot saved to:")
print(plot_path)


# ============================================================
# Final output
# ============================================================

print()
print("STEP 16 completed.")
print("=" * 70)