import os
import glob
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


# ============================================================
# STEP 14 - BO RECOMMENDATION RELIABILITY CHECK
# ============================================================

print("=" * 70)
print("STEP 14 - BO RECOMMENDATION RELIABILITY CHECK")
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

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "bayesian_optimization"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
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

TARGET_LOI = "LOI"
TARGET_TRANSMITTANCE = "Transmittance"


# ============================================================
# Utility functions
# ============================================================

def find_bo_pareto_file():

    print()
    print("Searching for BO Pareto result file...")

    candidates = [
        os.path.join(
            RESULT_DIR,
            "bo_pareto_candidates.csv"
        ),
        os.path.join(
            RESULT_DIR,
            "bo_pareto.csv"
        ),
        os.path.join(
            RESULT_DIR,
            "bo_candidate_evaluation.csv"
        )
    ]

    for path in candidates:

        if os.path.exists(path):

            print("BO Pareto file found:")
            print(os.path.basename(path))

            return path

    raise FileNotFoundError(
        "No BO Pareto result file found in:\n"
        f"{RESULT_DIR}"
    )


# ============================================================
# Load experimental dataset
# ============================================================

def load_experimental_data():

    print()
    print("Loading experimental dataset...")

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            f"Experimental dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"Experimental samples: {len(df)}")

    return df


# ============================================================
# Load BO Pareto candidates
# ============================================================

def load_bo_candidates():

    path = find_bo_pareto_file()

    print()
    print("Loading BO Pareto candidates...")

    df = pd.read_csv(path)

    print(f"BO Pareto candidates: {len(df)}")

    required_columns = [
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
        "Predicted_Transmittance",
        "Predicted_UL94"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns in BO Pareto file:\n"
            + "\n".join(missing)
        )

    return df, path


# ============================================================
# Performance extrapolation
# ============================================================

def calculate_performance_extrapolation(
    experimental_df,
    bo_df
):

    print()
    print("Checking performance extrapolation...")

    loi_min = experimental_df["LOI"].min()
    loi_max = experimental_df["LOI"].max()

    trans_min = experimental_df["Transmittance"].min()
    trans_max = experimental_df["Transmittance"].max()

    print(
        f"Experimental LOI range: "
        f"{loi_min:.2f} - {loi_max:.2f}"
    )

    print(
        f"Experimental Transmittance range: "
        f"{trans_min:.2f} - {trans_max:.2f}"
    )

    loi_extrapolation = (
        (bo_df["Predicted_LOI"] < loi_min)
        |
        (bo_df["Predicted_LOI"] > loi_max)
    )

    trans_extrapolation = (
        (bo_df["Predicted_Transmittance"] < trans_min)
        |
        (bo_df["Predicted_Transmittance"] > trans_max)
    )

    bo_df["LOI_Extrapolation"] = loi_extrapolation

    bo_df["Transmittance_Extrapolation"] = (
        trans_extrapolation
    )

    bo_df["Performance_Extrapolation"] = (
        loi_extrapolation
        |
        trans_extrapolation
    )

    return bo_df


# ============================================================
# Formulation-space distance
# ============================================================

def calculate_formulation_distance(
    experimental_df,
    bo_df
):

    print()
    print("Calculating formulation-space distance...")

    X_exp = experimental_df[FEATURES].copy()
    X_bo = bo_df[FEATURES].copy()

    scaler = StandardScaler()

    X_exp_scaled = scaler.fit_transform(X_exp)

    X_bo_scaled = scaler.transform(X_bo)

    distances = pairwise_distances(
        X_bo_scaled,
        X_exp_scaled,
        metric="euclidean"
    )

    nearest_distance = distances.min(axis=1)

    nearest_index = distances.argmin(axis=1)

    nearest_ids = (
        experimental_df.iloc[nearest_index]["sample_ID"]
        .values
    )

    bo_df["Nearest_Experimental_Distance"] = (
        nearest_distance
    )

    bo_df["Nearest_Experimental_ID"] = (
        nearest_ids
    )

    # --------------------------------------------------------
    # Applicability domain threshold
    # --------------------------------------------------------

    experimental_distances = pairwise_distances(
        X_exp_scaled,
        X_exp_scaled,
        metric="euclidean"
    )

    np.fill_diagonal(
        experimental_distances,
        np.inf
    )

    nearest_exp_distance = (
        experimental_distances.min(axis=1)
    )

    threshold = np.percentile(
        nearest_exp_distance,
        95
    )

    print(
        f"95th percentile distance threshold: "
        f"{threshold:.3f}"
    )

    bo_df["Distance_Extrapolation"] = (
        bo_df["Nearest_Experimental_Distance"]
        > threshold
    )

    return bo_df, threshold


# ============================================================
# Reliability score
# ============================================================

def calculate_reliability(bo_df):

    bo_df["Reliable"] = ~(
        bo_df["Performance_Extrapolation"]
        |
        bo_df["Distance_Extrapolation"]
    )

    return bo_df


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    experimental_df = load_experimental_data()

    bo_df, bo_file = load_bo_candidates()

    # --------------------------------------------------------
    # Performance extrapolation
    # --------------------------------------------------------

    bo_df = calculate_performance_extrapolation(
        experimental_df,
        bo_df
    )

    # --------------------------------------------------------
    # Formulation-space distance
    # --------------------------------------------------------

    bo_df, distance_threshold = (
        calculate_formulation_distance(
            experimental_df,
            bo_df
        )
    )

    # --------------------------------------------------------
    # Reliability
    # --------------------------------------------------------

    bo_df = calculate_reliability(bo_df)

    # --------------------------------------------------------
    # Reliability summary
    # --------------------------------------------------------

    performance_count = int(
        bo_df["Performance_Extrapolation"].sum()
    )

    distance_count = int(
        bo_df["Distance_Extrapolation"].sum()
    )

    reliable_count = int(
        bo_df["Reliable"].sum()
    )

    print()
    print("Reliability summary:")

    print(
        f"Performance extrapolation: "
        f"{performance_count}"
    )

    print(
        f"Formulation-space extrapolation: "
        f"{distance_count}"
    )

    print(
        f"Reliable candidates: "
        f"{reliable_count}"
    )

    # --------------------------------------------------------
    # Candidate reliability table
    # --------------------------------------------------------

    output_columns = [
        "Pareto_Rank",
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
        "Nearest_Experimental_Distance",
        "Nearest_Experimental_ID",
        "LOI_Extrapolation",
        "Transmittance_Extrapolation",
        "Performance_Extrapolation",
        "Distance_Extrapolation",
        "Reliable"
    ]

    available_columns = [
        col for col in output_columns
        if col in bo_df.columns
    ]

    reliability_df = bo_df[
        available_columns
    ].copy()

    print()
    print("Candidate reliability:")

    print(
        reliability_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Reliable recommendations
    # --------------------------------------------------------

    reliable_df = bo_df[
        bo_df["Reliable"]
    ].copy()

    print()
    print("Reliable BO recommendations:")

    if len(reliable_df) == 0:

        print("No reliable candidates found.")

    else:

        print(
            reliable_df[
                available_columns
            ].to_string(index=False)
        )

    # --------------------------------------------------------
    # Save reliability result
    # --------------------------------------------------------

    reliability_path = os.path.join(
        RESULT_DIR,
        "bo_reliability_check.csv"
    )

    reliability_df.to_csv(
        reliability_path,
        index=False
    )

    # --------------------------------------------------------
    # Save reliable candidates
    # --------------------------------------------------------

    reliable_path = os.path.join(
        RESULT_DIR,
        "bo_reliable_candidates.csv"
    )

    reliable_df.to_csv(
        reliable_path,
        index=False
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = pd.DataFrame({

        "Metric": [
            "Experimental_Samples",
            "BO_Pareto_Candidates",
            "Performance_Extrapolation",
            "Distance_Extrapolation",
            "Reliable_Candidates",
            "Distance_Threshold"
        ],

        "Value": [
            len(experimental_df),
            len(bo_df),
            performance_count,
            distance_count,
            reliable_count,
            distance_threshold
        ]
    })

    summary_path = os.path.join(
        RESULT_DIR,
        "bo_reliability_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("BO reliability analysis completed.")

    print()
    print("Results saved to:")

    print(reliability_path)
    print(reliable_path)
    print(summary_path)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()