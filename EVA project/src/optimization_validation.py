import os
import numpy as np
import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "polymer_dataset_clean.csv"
)

BO_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "bayesian_optimization"
)

OUTPUT_DIR = BO_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FEATURE CONFIGURATION
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

TARGET_COLUMNS = [
    "Predicted_LOI",
    "Predicted_Transmittance"
]


# ============================================================
# LOAD DATA
# ============================================================

def load_experimental_data():

    print("Loading experimental dataset...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Experimental dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(
        f"Experimental samples: {len(df)}"
    )

    return df


def load_bo_results():

    # --------------------------------------------------------
    # Safe BO candidates
    # --------------------------------------------------------

    safe_path = os.path.join(
        BO_DIR,
        "safe_bo_candidates.csv"
    )

    if not os.path.exists(safe_path):
        raise FileNotFoundError(
            f"Safe BO candidates not found:\n{safe_path}"
        )

    print("\nLoading Safe BO candidates...")

    safe_bo = pd.read_csv(safe_path)

    print(
        f"Safe BO candidates: {len(safe_bo)}"
    )

    # --------------------------------------------------------
    # Safe BO Pareto
    # --------------------------------------------------------

    pareto_path = os.path.join(
        BO_DIR,
        "safe_bo_pareto.csv"
    )

    if not os.path.exists(pareto_path):
        raise FileNotFoundError(
            f"Safe BO Pareto file not found:\n{pareto_path}"
        )

    print("Loading Safe BO Pareto results...")

    pareto = pd.read_csv(pareto_path)

    print(
        f"Safe BO Pareto candidates: {len(pareto)}"
    )

    # --------------------------------------------------------
    # Final recommendations
    # --------------------------------------------------------

    final_path = os.path.join(
        BO_DIR,
        "final_recommendations.csv"
    )

    if not os.path.exists(final_path):
        raise FileNotFoundError(
            f"Final recommendations not found:\n{final_path}"
        )

    print("Loading final recommendations...")

    final_rec = pd.read_csv(final_path)

    print(
        f"Final recommendations: {len(final_rec)}"
    )

    return safe_bo, pareto, final_rec


# ============================================================
# FORMULATION DISTANCE
# ============================================================

def calculate_formulation_distance(
    candidate,
    experimental,
    feature_ranges
):

    distances = []

    for feature in FEATURES:

        value = candidate[feature]

        minimum = feature_ranges[feature]["min"]
        maximum = feature_ranges[feature]["max"]

        span = maximum - minimum

        if span == 0:
            span = 1.0

        normalized_difference = (
            experimental[feature] - value
        ) / span

        distances.append(
            normalized_difference ** 2
        )

    distance = np.sqrt(
        np.sum(distances, axis=1)
    )

    return distance.min()


# ============================================================
# VALIDATE PERFORMANCE
# ============================================================

def validate_performance(
    df,
    experimental
):

    loi_min = experimental["LOI"].min()
    loi_max = experimental["LOI"].max()

    trans_min = experimental["Transmittance"].min()
    trans_max = experimental["Transmittance"].max()

    df = df.copy()

    df["LOI_Within_Range"] = (
        (df["Predicted_LOI"] >= loi_min)
        &
        (df["Predicted_LOI"] <= loi_max)
    )

    df["Transmittance_Within_Range"] = (
        (df["Predicted_Transmittance"] >= trans_min)
        &
        (df["Predicted_Transmittance"] <= trans_max)
    )

    df["Performance_Within_Range"] = (
        df["LOI_Within_Range"]
        &
        df["Transmittance_Within_Range"]
    )

    df["LOI_Extrapolation"] = (
        ~df["LOI_Within_Range"]
    )

    df["Transmittance_Extrapolation"] = (
        ~df["Transmittance_Within_Range"]
    )

    df["Performance_Extrapolation"] = (
        ~df["Performance_Within_Range"]
    )

    return df


# ============================================================
# APPLICABILITY DOMAIN
# ============================================================

def calculate_applicability_domain(
    candidates,
    experimental
):

    print("\nCalculating formulation-space distances...")

    feature_ranges = {}

    for feature in FEATURES:

        feature_ranges[feature] = {
            "min": experimental[feature].min(),
            "max": experimental[feature].max()
        }

    distances = []

    nearest_ids = []

    for _, candidate in candidates.iterrows():

        candidate_distances = []

        for _, experimental_row in experimental.iterrows():

            squared_sum = 0.0

            for feature in FEATURES:

                minimum = feature_ranges[feature]["min"]
                maximum = feature_ranges[feature]["max"]

                span = maximum - minimum

                if span == 0:
                    span = 1.0

                difference = (
                    candidate[feature]
                    -
                    experimental_row[feature]
                ) / span

                squared_sum += difference ** 2

            candidate_distances.append(
                np.sqrt(squared_sum)
            )

        min_index = int(
            np.argmin(candidate_distances)
        )

        distances.append(
            candidate_distances[min_index]
        )

        nearest_ids.append(
            experimental.iloc[min_index]["sample_ID"]
        )

    return (
        np.array(distances),
        nearest_ids
    )


# ============================================================
# VALIDATE DATASET
# ============================================================

def validate_dataset(
    df,
    experimental,
    distance_threshold
):

    df = df.copy()

    distances, nearest_ids = (
        calculate_applicability_domain(
            df,
            experimental
        )
    )

    df[
        "Nearest_Experimental_Distance"
    ] = distances

    df[
        "Nearest_Experimental_ID"
    ] = nearest_ids

    df[
        "Distance_Extrapolation"
    ] = (
        df[
            "Nearest_Experimental_Distance"
        ]
        > distance_threshold
    )

    df["Inside_Applicability_Domain"] = (
        ~df["Distance_Extrapolation"]
    )

    df["Valid_Formulation"] = True

    df["Final_Safe"] = (
        df["Performance_Within_Range"]
        &
        df["Inside_Applicability_Domain"]
        &
        df["Valid_Formulation"]
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEP 19 - OPTIMIZATION VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load experimental data
    # --------------------------------------------------------

    experimental = load_experimental_data()

    # --------------------------------------------------------
    # Experimental reference ranges
    # --------------------------------------------------------

    loi_min = experimental["LOI"].min()
    loi_max = experimental["LOI"].max()

    trans_min = experimental["Transmittance"].min()
    trans_max = experimental["Transmittance"].max()

    print("\nExperimental performance ranges:")

    print(
        f"LOI: {loi_min:.2f} - {loi_max:.2f}"
    )

    print(
        f"Transmittance: "
        f"{trans_min:.2f} - {trans_max:.2f}"
    )

    # --------------------------------------------------------
    # Load latest Safe BO results
    # --------------------------------------------------------

    safe_bo, pareto, final_rec = (
        load_bo_results()
    )

    # --------------------------------------------------------
    # Determine applicability threshold
    # --------------------------------------------------------

    print(
        "\nCalculating applicability-domain threshold..."
    )

    experimental_distances = []

    feature_ranges = {}

    for feature in FEATURES:

        feature_ranges[feature] = {
            "min": experimental[feature].min(),
            "max": experimental[feature].max()
        }

    # Leave-one-out nearest-neighbor distances
    for i in range(len(experimental)):

        current = experimental.iloc[i]

        other = experimental.drop(
            experimental.index[i]
        )

        distance_list = []

        for _, row in other.iterrows():

            squared_sum = 0.0

            for feature in FEATURES:

                minimum = feature_ranges[feature]["min"]
                maximum = feature_ranges[feature]["max"]

                span = maximum - minimum

                if span == 0:
                    span = 1.0

                diff = (
                    current[feature]
                    -
                    row[feature]
                ) / span

                squared_sum += diff ** 2

            distance_list.append(
                np.sqrt(squared_sum)
            )

        experimental_distances.append(
            min(distance_list)
        )

    distance_threshold = float(
        np.percentile(
            experimental_distances,
            95
        )
    )

    print(
        f"Applicability distance threshold: "
        f"{distance_threshold:.3f}"
    )

    # --------------------------------------------------------
    # Validate Safe BO candidates
    # --------------------------------------------------------

    print(
        "\nChecking Safe BO candidate performance..."
    )

    validated_safe = validate_performance(
        safe_bo,
        experimental
    )

    validated_safe = validate_dataset(
        validated_safe,
        experimental,
        distance_threshold
    )

    # --------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------

    total_candidates = len(
        validated_safe
    )

    within_range = int(
        validated_safe[
            "Performance_Within_Range"
        ].sum()
    )

    inside_domain = int(
        validated_safe[
            "Inside_Applicability_Domain"
        ].sum()
    )

    final_safe = int(
        validated_safe[
            "Final_Safe"
        ].sum()
    )

    # --------------------------------------------------------
    # Validate Pareto candidates
    # --------------------------------------------------------

    print(
        "\nValidating Safe BO Pareto candidates..."
    )

    validated_pareto = validate_performance(
        pareto,
        experimental
    )

    validated_pareto = validate_dataset(
        validated_pareto,
        experimental,
        distance_threshold
    )

    pareto_valid_count = int(
        validated_pareto[
            "Final_Safe"
        ].sum()
    )

    # --------------------------------------------------------
    # Validate final recommendations
    # --------------------------------------------------------

    print(
        "\nValidating final recommendations..."
    )

    validated_final = validate_performance(
        final_rec,
        experimental
    )

    validated_final = validate_dataset(
        validated_final,
        experimental,
        distance_threshold
    )

    final_valid_count = int(
        validated_final[
            "Final_Safe"
        ].sum()
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("OPTIMIZATION VALIDATION SUMMARY")
    print("=" * 70)

    print(
        f"Total Safe BO candidates: "
        f"{total_candidates}"
    )

    print(
        f"Within experimental performance range: "
        f"{within_range}"
    )

    print(
        f"Inside applicability domain: "
        f"{inside_domain}"
    )

    print(
        f"Final safe candidates: "
        f"{final_safe}"
    )

    print(
        f"Safe BO Pareto candidates: "
        f"{len(pareto)}"
    )

    print(
        f"Valid Safe BO Pareto candidates: "
        f"{pareto_valid_count}"
    )

    print(
        f"Final recommendations: "
        f"{len(final_rec)}"
    )

    print(
        f"Valid final recommendations: "
        f"{final_valid_count}"
    )

    # --------------------------------------------------------
    # Save complete validation table
    # --------------------------------------------------------

    validation_path = os.path.join(
        OUTPUT_DIR,
        "optimization_validation.csv"
    )

    validated_safe.to_csv(
        validation_path,
        index=False
    )

    print(
        "\nOptimization validation saved to:"
    )

    print(validation_path)

    # --------------------------------------------------------
    # Save Pareto validation
    # --------------------------------------------------------

    pareto_validation_path = os.path.join(
        OUTPUT_DIR,
        "safe_bo_pareto_validation.csv"
    )

    validated_pareto.to_csv(
        pareto_validation_path,
        index=False
    )

    print(
        "Safe BO Pareto validation saved to:"
    )

    print(pareto_validation_path)

    # --------------------------------------------------------
    # Save final recommendation validation
    # --------------------------------------------------------

    final_validation_path = os.path.join(
        OUTPUT_DIR,
        "final_recommendation_validation.csv"
    )

    validated_final.to_csv(
        final_validation_path,
        index=False
    )

    print(
        "Final recommendation validation saved to:"
    )

    print(final_validation_path)

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = pd.DataFrame({

        "Metric": [
            "Experimental Samples",
            "Safe BO Candidates",
            "Candidates Within Experimental Performance Range",
            "Candidates Inside Applicability Domain",
            "Final Safe Candidates",
            "Safe BO Pareto Candidates",
            "Valid Safe BO Pareto Candidates",
            "Final Recommendations",
            "Valid Final Recommendations",
            "LOI Experimental Minimum",
            "LOI Experimental Maximum",
            "Transmittance Experimental Minimum",
            "Transmittance Experimental Maximum",
            "Applicability Distance Threshold"
        ],

        "Value": [
            len(experimental),
            total_candidates,
            within_range,
            inside_domain,
            final_safe,
            len(pareto),
            pareto_valid_count,
            len(final_rec),
            final_valid_count,
            loi_min,
            loi_max,
            trans_min,
            trans_max,
            distance_threshold
        ]
    })

    summary_path = os.path.join(
        OUTPUT_DIR,
        "optimization_validation_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    print(
        "Validation summary saved to:"
    )

    print(summary_path)

    # --------------------------------------------------------
    # Final recommendation validation display
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL RECOMMENDATION VALIDATION")
    print("=" * 70)

    display_columns = [
        "Final_Rank",
        "Predicted_LOI",
        "Predicted_UL94",
        "Predicted_Transmittance",
        "Nearest_Experimental_Distance",
        "Performance_Extrapolation",
        "Distance_Extrapolation",
        "Final_Safe"
    ]

    existing_columns = [
        column
        for column in display_columns
        if column in validated_final.columns
    ]

    print(
        validated_final[
            existing_columns
        ].to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("STEP 19 completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()