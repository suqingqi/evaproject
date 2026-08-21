import os
import joblib
import numpy as np
import pandas as pd


# ============================================================
# STEP 13-3
# BO CANDIDATE EVALUATION
# ============================================================

print("=" * 70)
print("STEP 13-3 - BO CANDIDATE EVALUATION")
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

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

BO_RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "bayesian_optimization"
)

os.makedirs(BO_RESULT_DIR, exist_ok=True)


# ============================================================
# File paths
# ============================================================

DATA_PATH = os.path.join(
    DATA_DIR,
    "polymer_dataset_clean.csv"
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


# ============================================================
# STEP 13-2 candidate file
# ============================================================

CANDIDATE_FILES = [
    "bo_top_candidates.csv",
    "bo_candidates.csv",
    "bayesian_candidates.csv",
    "bo_all_candidates.csv"
]


def find_candidate_file():

    print("Searching for latest BO candidate file...")

    for filename in CANDIDATE_FILES:

        path = os.path.join(
            BO_RESULT_DIR,
            filename
        )

        if os.path.exists(path):

            print(f"BO candidate file found:")
            print(filename)

            return path

    raise FileNotFoundError(
        "No BO candidate file found in:\n"
        f"{BO_RESULT_DIR}\n\n"
        "Expected one of:\n"
        + "\n".join(CANDIDATE_FILES)
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
# Load models
# ============================================================

def load_models():

    print()
    print("Loading LOI model...")

    if not os.path.exists(LOI_MODEL_PATH):
        raise FileNotFoundError(
            f"LOI model not found:\n{LOI_MODEL_PATH}"
        )

    loi_model = joblib.load(
        LOI_MODEL_PATH
    )

    print("Loading UL-94 model...")

    if not os.path.exists(UL94_MODEL_PATH):
        raise FileNotFoundError(
            f"UL-94 model not found:\n{UL94_MODEL_PATH}"
        )

    ul94_model = joblib.load(
        UL94_MODEL_PATH
    )

    print("Loading Transmittance model...")

    if not os.path.exists(TRANSMITTANCE_MODEL_PATH):
        raise FileNotFoundError(
            "Transmittance model not found:\n"
            f"{TRANSMITTANCE_MODEL_PATH}"
        )

    trans_model = joblib.load(
        TRANSMITTANCE_MODEL_PATH
    )

    return loi_model, ul94_model, trans_model


# ============================================================
# Load candidates
# ============================================================

def load_candidates():

    candidate_path = find_candidate_file()

    print()
    print("Loading Bayesian Optimization candidates...")

    candidates = pd.read_csv(
        candidate_path
    )

    print(
        f"Candidates loaded: {len(candidates)}"
    )

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in candidates.columns
    ]

    if missing_features:

        raise ValueError(
            "Candidate file is missing required features:\n"
            + "\n".join(missing_features)
        )

    # 如果文件中超过20个，只保留前20个
    if len(candidates) > 20:

        candidates = candidates.head(20).copy()

    candidates = candidates.reset_index(
        drop=True
    )

    return candidates, candidate_path


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

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        f"Experimental samples: {len(df)}"
    )

    return df


# ============================================================
# Fire score
# ============================================================

def calculate_fire_score(probabilities, classes):

    fire_score = np.zeros(
        len(probabilities)
    )

    for i, row in enumerate(probabilities):

        score = 0.0

        for probability, cls in zip(
            row,
            classes
        ):

            cls = str(cls)

            if cls == "V-0":
                score += probability * 1.0

            elif cls == "V-1":
                score += probability * 0.667

            elif cls == "V-2":
                score += probability * 0.333

            elif cls == "NR":
                score += probability * 0.0

        fire_score[i] = score

    return fire_score


# ============================================================
# Pareto calculation
# ============================================================

def calculate_pareto_front(df):

    objectives = df[
        [
            "Predicted_LOI",
            "Predicted_Fire_Score",
            "Predicted_Transmittance"
        ]
    ].values

    n = len(df)

    is_pareto = np.ones(
        n,
        dtype=bool
    )

    for i in range(n):

        if not is_pareto[i]:
            continue

        for j in range(n):

            if i == j:
                continue

            # 最小化 LOI
            loi_better_or_equal = (
                objectives[j, 0]
                <= objectives[i, 0]
            )

            # 最大化 Fire Score
            fire_better_or_equal = (
                objectives[j, 1]
                >= objectives[i, 1]
            )

            # 最大化 Transmittance
            trans_better_or_equal = (
                objectives[j, 2]
                >= objectives[i, 2]
            )

            at_least_one_strict = (
                objectives[j, 0]
                < objectives[i, 0]
                or
                objectives[j, 1]
                > objectives[i, 1]
                or
                objectives[j, 2]
                > objectives[i, 2]
            )

            if (
                loi_better_or_equal
                and fire_better_or_equal
                and trans_better_or_equal
                and at_least_one_strict
            ):

                is_pareto[i] = False
                break

    return is_pareto


# ============================================================
# Utility calculation
# ============================================================

def calculate_utility(
    loi,
    fire_score,
    transmittance,
    experimental_df
):

    loi_min = experimental_df[
        "LOI"
    ].min()

    loi_max = experimental_df[
        "LOI"
    ].max()

    trans_min = experimental_df[
        "Transmittance"
    ].min()

    trans_max = experimental_df[
        "Transmittance"
    ].max()

    # LOI 越低越好
    loi_score = (
        loi_max - loi
    ) / (
        loi_max - loi_min
    )

    # Transmittance 越高越好
    trans_score = (
        transmittance - trans_min
    ) / (
        trans_max - trans_min
    )

    loi_score = np.clip(
        loi_score,
        0,
        1
    )

    trans_score = np.clip(
        trans_score,
        0,
        1
    )

    fire_score = np.clip(
        fire_score,
        0,
        1
    )

    utility = (
        0.40 * loi_score
        + 0.30 * fire_score
        + 0.30 * trans_score
    )

    return utility


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Load candidates
    # --------------------------------------------------------

    candidates, candidate_path = load_candidates()

    print()
    print(
        f"Using candidate file:\n{candidate_path}"
    )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    loi_model, ul94_model, trans_model = (
        load_models()
    )

    # --------------------------------------------------------
    # Load experimental data
    # --------------------------------------------------------

    experimental_df = load_experimental_data()

    X = candidates[
        FEATURES
    ].copy()

    # --------------------------------------------------------
    # Predict LOI
    # --------------------------------------------------------

    print()
    print("Predicting LOI...")

    predicted_loi = loi_model.predict(
        X
    )

    # --------------------------------------------------------
    # Predict UL-94
    # --------------------------------------------------------

    print("Predicting UL-94 probabilities...")

    ul94_probabilities = (
        ul94_model.predict_proba(X)
    )

    ul94_classes = (
        ul94_model.classes_
    )

    predicted_ul94_index = (
        np.argmax(
            ul94_probabilities,
            axis=1
        )
    )

    predicted_ul94 = [
        ul94_classes[i]
        for i in predicted_ul94_index
    ]

    # --------------------------------------------------------
    # Fire score
    # --------------------------------------------------------

    predicted_fire_score = (
        calculate_fire_score(
            ul94_probabilities,
            ul94_classes
        )
    )

    # --------------------------------------------------------
    # Predict transmittance
    # --------------------------------------------------------

    print("Predicting Transmittance...")

    predicted_transmittance = (
        trans_model.predict(X)
    )

    # --------------------------------------------------------
    # Add predictions
    # --------------------------------------------------------

    results = candidates.copy()

    results[
        "Predicted_LOI"
    ] = predicted_loi

    results[
        "Predicted_Fire_Score"
    ] = predicted_fire_score

    results[
        "Predicted_UL94"
    ] = predicted_ul94

    results[
        "Predicted_Transmittance"
    ] = predicted_transmittance

    # --------------------------------------------------------
    # Utility
    # --------------------------------------------------------

    print()
    print(
        "Calculating multi-objective utility..."
    )

    results[
        "Predicted_Utility"
    ] = calculate_utility(
        results["Predicted_LOI"].values,
        results["Predicted_Fire_Score"].values,
        results["Predicted_Transmittance"].values,
        experimental_df
    )

    # --------------------------------------------------------
    # UCB
    # --------------------------------------------------------

    if "UCB" not in results.columns:

        if (
            "Utility_Mean" in results.columns
            and
            "Utility_STD" in results.columns
        ):

            results["UCB"] = (
                results["Utility_Mean"]
                +
                2.0 * results["Utility_STD"]
            )

        else:

            results["UCB"] = (
                results["Predicted_Utility"]
            )

    # --------------------------------------------------------
    # Pareto front
    # --------------------------------------------------------

    print()
    print(
        "Calculating Pareto front..."
    )

    pareto_mask = calculate_pareto_front(
        results
    )

    results[
        "Pareto"
    ] = pareto_mask

    pareto_results = (
        results[
            results["Pareto"]
        ]
        .copy()
        .reset_index(drop=True)
    )

    pareto_results[
        "Pareto_Rank"
    ] = np.arange(
        1,
        len(pareto_results) + 1
    )

    # --------------------------------------------------------
    # Sort Pareto results
    # --------------------------------------------------------

    pareto_results = pareto_results.sort_values(
        by=[
            "Predicted_Utility",
            "UCB"
        ],
        ascending=[
            False,
            False
        ]
    ).reset_index(
        drop=True
    )

    pareto_results[
        "Pareto_Rank"
    ] = np.arange(
        1,
        len(pareto_results) + 1
    )

    # --------------------------------------------------------
    # Final recommendations
    # --------------------------------------------------------

    final_recommendations = (
        pareto_results
        .head(5)
        .copy()
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print(
        f"Total BO candidates: {len(results)}"
    )

    print(
        f"Pareto candidates: {len(pareto_results)}"
    )

    print()
    print(
        "BO Pareto candidates:"
    )

    display_columns = [
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
        "Predicted_Utility",
        "UCB"
    ]

    print(
        pareto_results[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Final recommended candidates:"
    )

    print(
        final_recommendations[
            display_columns
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # Save files
    # ========================================================

    all_results_path = os.path.join(
        BO_RESULT_DIR,
        "bo_candidate_evaluation.csv"
    )

    pareto_path = os.path.join(
        BO_RESULT_DIR,
        "bo_pareto_candidates.csv"
    )

    final_path = os.path.join(
        BO_RESULT_DIR,
        "bo_final_recommendations.csv"
    )

    results.to_csv(
        all_results_path,
        index=False
    )

    pareto_results.to_csv(
        pareto_path,
        index=False
    )

    final_recommendations.to_csv(
        final_path,
        index=False
    )

    print()
    print(
        "BO candidate evaluation completed."
    )

    print()
    print(
        "Results saved to:"
    )

    print(all_results_path)
    print(pareto_path)
    print(final_path)

    print()


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()