import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

# ============================================================
# STEP 18 - NEXT EXPERIMENT SELECTION
# ============================================================
# Goal:
# Select a small experimental batch from the safety-aware BO
# candidate pool by balancing:
#   1) predicted performance / safety (exploitation),
#   2) GP uncertainty (exploration),
#   3) formulation diversity within the batch.
#
# This is a practical batch-selection layer built on top of the
# existing Bayesian optimization workflow. It does not claim a
# separate theoretical Active Learning algorithm.
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "polymer_dataset_clean.csv"
)

BO_RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "bayesian_optimization"
)

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "next_experiment_selection"
)

os.makedirs(RESULT_DIR, exist_ok=True)

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

# Keep the weighting simple and interpretable.
# Performance remains the main objective; uncertainty and diversity
# are used to avoid selecting five nearly identical high-score points.
PERFORMANCE_WEIGHT = 0.60
UNCERTAINTY_WEIGHT = 0.25
DIVERSITY_WEIGHT = 0.15

N_EXPERIMENTS = 5


def min_max(values):
    values = np.asarray(values, dtype=float)
    minimum = np.nanmin(values)
    maximum = np.nanmax(values)

    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise ValueError("Non-finite values found during normalization.")

    if maximum - minimum < 1e-12:
        return np.ones_like(values) * 0.5

    return (values - minimum) / (maximum - minimum)


def find_candidate_file():
    candidate_files = [
        "safe_bo_pareto.csv",
        "safe_bo_candidates.csv",
        "safe_bo_all_candidates.csv"
    ]

    for filename in candidate_files:
        path = os.path.join(BO_RESULT_DIR, filename)
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "No safety-aware BO candidate file found. Run "
        "safe_bayesian_optimization.py and final_recommendation.py first."
    )


def choose_performance_column(df):
    preferred = [
        "Safety_Adjusted_Score",
        "Safe_UCB",
        "Predicted_Utility",
        "Multi_Objective_Score"
    ]

    for column in preferred:
        if column in df.columns:
            return column

    raise ValueError(
        "No suitable performance score found in candidate file."
    )


def choose_uncertainty_column(df):
    preferred = [
        "Utility_STD"
    ]

    for column in preferred:
        if column in df.columns:
            return column

    # If GP uncertainty is unavailable, do not invent uncertainty.
    # Use a neutral constant score and make the limitation explicit.
    df["Utility_STD"] = 0.0
    return "Utility_STD"


def validate_input(candidate_df, experimental_df):
    missing_candidate = [
        feature for feature in FEATURES
        if feature not in candidate_df.columns
    ]

    missing_experimental = [
        feature for feature in FEATURES
        if feature not in experimental_df.columns
    ]

    if missing_candidate:
        raise ValueError(
            f"Candidate file missing features: {missing_candidate}"
        )

    if missing_experimental:
        raise ValueError(
            f"Experimental dataset missing features: {missing_experimental}"
        )


def prepare_candidate_scores(candidate_df):
    performance_column = choose_performance_column(candidate_df)
    uncertainty_column = choose_uncertainty_column(candidate_df)

    candidate_df = candidate_df.copy().reset_index(drop=True)

    candidate_df["Performance_Score"] = min_max(
        candidate_df[performance_column].values
    )

    candidate_df["Uncertainty_Score"] = min_max(
        candidate_df[uncertainty_column].values
    )

    candidate_df["Base_Acquisition_Score"] = (
        PERFORMANCE_WEIGHT * candidate_df["Performance_Score"]
        + UNCERTAINTY_WEIGHT * candidate_df["Uncertainty_Score"]
    )

    return candidate_df, performance_column, uncertainty_column


def greedy_batch_selection(candidate_df, experimental_df, n_select=5):
    """
    Greedy batch selection.

    First point:
        performance + uncertainty

    Later points:
        performance + uncertainty + distance to already-selected points

    Candidate-candidate distances are calculated in standardized
    formulation space using a scaler fitted on the historical
    experimental data.
    """

    n_select = min(n_select, len(candidate_df))

    scaler = StandardScaler()
    scaler.fit(experimental_df[FEATURES])

    candidate_scaled = scaler.transform(candidate_df[FEATURES])

    selected_indices = []
    remaining_indices = list(range(len(candidate_df)))
    selection_records = []

    for step in range(n_select):
        if step == 0:
            scores = candidate_df.loc[
                remaining_indices,
                "Base_Acquisition_Score"
            ].values

            best_local_position = int(np.argmax(scores))
            best_index = remaining_indices[best_local_position]
            diversity_score = 0.0
            final_score = float(scores[best_local_position])

        else:
            remaining_scaled = candidate_scaled[remaining_indices]
            selected_scaled = candidate_scaled[selected_indices]

            distances = pairwise_distances(
                remaining_scaled,
                selected_scaled,
                metric="euclidean"
            )

            # Distance to the nearest formulation already selected.
            min_distance_to_batch = distances.min(axis=1)
            diversity_scores = min_max(min_distance_to_batch)

            performance_scores = candidate_df.loc[
                remaining_indices,
                "Performance_Score"
            ].values

            uncertainty_scores = candidate_df.loc[
                remaining_indices,
                "Uncertainty_Score"
            ].values

            combined_scores = (
                PERFORMANCE_WEIGHT * performance_scores
                + UNCERTAINTY_WEIGHT * uncertainty_scores
                + DIVERSITY_WEIGHT * diversity_scores
            )

            best_local_position = int(np.argmax(combined_scores))
            best_index = remaining_indices[best_local_position]
            diversity_score = float(diversity_scores[best_local_position])
            final_score = float(combined_scores[best_local_position])

        selected_indices.append(best_index)
        remaining_indices.remove(best_index)

        selection_records.append({
            "Selection_Index": best_index,
            "Diversity_Score": diversity_score,
            "Next_Experiment_Score": final_score
        })

    selected = candidate_df.loc[selected_indices].copy().reset_index(drop=True)
    records = pd.DataFrame(selection_records)

    selected["Diversity_Score"] = records["Diversity_Score"].values
    selected["Next_Experiment_Score"] = records[
        "Next_Experiment_Score"
    ].values

    selected.insert(
        0,
        "Experiment_Rank",
        range(1, len(selected) + 1)
    )

    return selected


def add_selection_reason(selected):
    reasons = []

    for _, row in selected.iterrows():
        performance = row["Performance_Score"]
        uncertainty = row["Uncertainty_Score"]
        diversity = row["Diversity_Score"]

        if performance >= 0.75 and uncertainty >= 0.60:
            reason = "High performance + informative uncertainty"
        elif performance >= 0.75:
            reason = "High predicted performance"
        elif uncertainty >= 0.70:
            reason = "Exploration / high model uncertainty"
        elif diversity >= 0.70:
            reason = "Batch diversity"
        else:
            reason = "Balanced performance, uncertainty and diversity"

        reasons.append(reason)

    selected["Selection_Reason"] = reasons
    return selected



def format_practical_output(selected):
    """Match output precision to the historical experimental dataset."""
    selected = selected.copy()

    # Formulation resolution used in the original dataset.
    for column in [
        "EVA_content",
        "FR_A",
        "FR_B",
        "FR_C",
        "FR_D",
        "Additive_1",
        "Additive_2",
    ]:
        if column in selected.columns:
            selected[column] = selected[column].round(1)

    for column in ["Polymer_A", "Polymer_B"]:
        if column in selected.columns:
            selected[column] = selected[column].round().astype(int)

    # Experimental properties in the source data are recorded to 0.1.
    for column in [
        "Predicted_LOI",
        "Predicted_Transmittance",
    ]:
        if column in selected.columns:
            selected[column] = selected[column].round(1)

    # Keep internal model / acquisition diagnostics concise but informative.
    for column in [
        "Predicted_Fire_Score",
        "Utility_Mean",
        "Utility_STD",
        "Nearest_Experimental_Distance",
        "Safety_Factor",
        "Safety_Adjusted_Score",
        "Performance_Score",
        "Uncertainty_Score",
        "Diversity_Score",
        "Next_Experiment_Score",
    ]:
        if column in selected.columns:
            selected[column] = selected[column].round(3)

    return selected

def main():
    print("STEP 18 - NEXT EXPERIMENT SELECTION")
    print()

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Experimental dataset not found:\n{DATA_PATH}"
        )

    experimental_df = pd.read_csv(DATA_PATH)
    candidate_path = find_candidate_file()
    candidate_df = pd.read_csv(candidate_path)

    print(f"Historical experimental samples: {len(experimental_df)}")
    print(f"Candidate source: {candidate_path}")
    print(f"Candidate pool size: {len(candidate_df)}")

    validate_input(candidate_df, experimental_df)

    candidate_df, performance_column, uncertainty_column = (
        prepare_candidate_scores(candidate_df)
    )

    print(f"Performance score source: {performance_column}")
    print(f"Uncertainty source: {uncertainty_column}")
    print("Output precision: formulation/properties follow historical experimental resolution")
    print(
        "Selection weights: "
        f"performance={PERFORMANCE_WEIGHT:.2f}, "
        f"uncertainty={UNCERTAINTY_WEIGHT:.2f}, "
        f"diversity={DIVERSITY_WEIGHT:.2f}"
    )

    selected = greedy_batch_selection(
        candidate_df,
        experimental_df,
        n_select=N_EXPERIMENTS
    )

    selected = add_selection_reason(selected)
    selected = format_practical_output(selected)

    output_columns = [
        "Experiment_Rank",
        *FEATURES,
        "Predicted_LOI",
        "Predicted_Fire_Score",
        "Predicted_UL94",
        "Predicted_Transmittance",
        "Utility_Mean",
        "Utility_STD",
        "Nearest_Experimental_Distance",
        "Safety_Factor",
        "Safety_Adjusted_Score",
        "Performance_Score",
        "Uncertainty_Score",
        "Diversity_Score",
        "Next_Experiment_Score",
        "Selection_Reason"
    ]

    output_columns = [
        column for column in output_columns
        if column in selected.columns
    ]

    output_path = os.path.join(
        RESULT_DIR,
        "next_experiments.csv"
    )

    selected[output_columns].to_csv(
        output_path,
        index=False
    )

    print()
    print("Recommended next experiments:")

    display_columns = [
        "Experiment_Rank",
        "EVA_content",
        "Polymer_A",
        "Polymer_B",
        "Predicted_LOI",
        "Predicted_UL94",
        "Predicted_Transmittance",
        "Utility_STD",
        "Next_Experiment_Score",
        "Selection_Reason"
    ]

    display_columns = [
        column for column in display_columns
        if column in selected.columns
    ]

    print(
        selected[display_columns].to_string(
            index=False
        )
    )

    print()
    print("Next-experiment batch saved to:")
    print(output_path)
    print()
    print("STEP 18 completed.")


if __name__ == "__main__":
    main()
