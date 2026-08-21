import os
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel


# ============================================================
# STEP 15 - SAFE BAYESIAN OPTIMIZATION
# ============================================================

print("=" * 70)
print("STEP 15 - SAFE BAYESIAN OPTIMIZATION")
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

# Integer-like formulation variables
INTEGER_FEATURES = [
    "Polymer_A",
    "Polymer_B"
]


# ============================================================
# Load dataset
# ============================================================

def load_dataset():

    print()
    print("Loading experimental dataset...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Experimental dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"Samples: {len(df)}")
    print(f"Features: {len(FEATURES)}")

    return df


# ============================================================
# Load models
# ============================================================

def load_models():

    print()
    print("Loading LOI model...")

    loi_path = os.path.join(
        MODEL_DIR,
        "loi_model.pkl"
    )

    if not os.path.exists(loi_path):
        raise FileNotFoundError(
            f"LOI model not found:\n{loi_path}"
        )

    loi_model = joblib.load(loi_path)

    print("Loading UL-94 model...")

    ul94_path = os.path.join(
        MODEL_DIR,
        "ul94_model.pkl"
    )

    if not os.path.exists(ul94_path):
        raise FileNotFoundError(
            f"UL-94 model not found:\n{ul94_path}"
        )

    ul94_model = joblib.load(ul94_path)

    print("Loading Transmittance model...")

    trans_path = os.path.join(
        MODEL_DIR,
        "transmittance_model.pkl"
    )

    if not os.path.exists(trans_path):
        raise FileNotFoundError(
            f"Transmittance model not found:\n{trans_path}"
        )

    trans_model = joblib.load(trans_path)

    return loi_model, ul94_model, trans_model


# ============================================================
# Applicability domain
# ============================================================

def calculate_applicability_domain(df):

    print()
    print("Calculating applicability domain...")

    X = df[FEATURES].copy()

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    distance_matrix = pairwise_distances(
        X_scaled,
        X_scaled,
        metric="euclidean"
    )

    np.fill_diagonal(
        distance_matrix,
        np.inf
    )

    nearest_distance = distance_matrix.min(
        axis=1
    )

    threshold = np.percentile(
        nearest_distance,
        95
    )

    print(
        f"Applicability distance threshold: "
        f"{threshold:.3f}"
    )

    return scaler, threshold


# ============================================================
# Experimental performance ranges
# ============================================================

def calculate_performance_ranges(df):

    loi_min = df["LOI"].min()
    loi_max = df["LOI"].max()

    trans_min = df["Transmittance"].min()
    trans_max = df["Transmittance"].max()

    print(
        f"LOI experimental range: "
        f"{loi_min:.2f} - {loi_max:.2f}"
    )

    print(
        f"Transmittance experimental range: "
        f"{trans_min:.2f} - {trans_max:.2f}"
    )

    return (
        loi_min,
        loi_max,
        trans_min,
        trans_max
    )


# ============================================================
# Experimental objective
# ============================================================

def calculate_experimental_utility(
    df,
    loi_min,
    loi_max,
    trans_min,
    trans_max
):

    print()
    print("Calculating experimental objective values...")

    # --------------------------------------------------------
    # LOI objective
    # Higher LOI is better, but keep the experimental range
    # --------------------------------------------------------

    loi_score = (
        (df["LOI"] - loi_min)
        /
        (loi_max - loi_min)
    )

    # --------------------------------------------------------
    # Transmittance objective
    # --------------------------------------------------------

    trans_score = (
        (df["Transmittance"] - trans_min)
        /
        (trans_max - trans_min)
    )

    # --------------------------------------------------------
    # UL-94 score
    # --------------------------------------------------------

    ul94_score = df["UL94_score"].astype(float)

    ul94_max = ul94_score.max()

    if ul94_max > 0:
        fire_score = ul94_score / ul94_max
    else:
        fire_score = ul94_score

    # --------------------------------------------------------
    # Weighted objective
    # --------------------------------------------------------

    utility = (
        0.40 * loi_score
        +
        0.35 * fire_score
        +
        0.25 * trans_score
    )

    return utility


# ============================================================
# Gaussian Process
# ============================================================

def train_gaussian_process(
    df,
    experimental_utility
):

    print()
    print("Preparing Gaussian Process target...")

    X = df[FEATURES].copy()

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    y = np.asarray(
        experimental_utility,
        dtype=float
    )

    print("Training Gaussian Process...")

    kernel = (
        ConstantKernel(
            1.0,
            (1e-3, 1e3)
        )
        *
        Matern(
            length_scale=1.0,
            nu=2.5
        )
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-4,
        normalize_y=True,
        random_state=42,
        n_restarts_optimizer=3
    )

    gp.fit(
        X_scaled,
        y
    )

    print("Gaussian Process training completed.")

    return gp, scaler


# ============================================================
# Candidate generation
# ============================================================

def generate_candidates(
    df,
    n_candidates=30000,
    random_state=42
):

    print()
    print("Generating constrained candidates...")

    rng = np.random.default_rng(
        random_state
    )

    candidate_list = []

    min_values = df[FEATURES].min()
    max_values = df[FEATURES].max()

    # --------------------------------------------------------
    # Generate candidates
    # --------------------------------------------------------

    for _ in range(n_candidates):

        polymer_a = rng.choice(
            sorted(
                df["Polymer_A"]
                .unique()
            )
        )

        polymer_b = rng.choice(
            sorted(
                df["Polymer_B"]
                .unique()
            )
        )

        eva = rng.uniform(
            min_values["EVA_content"],
            max_values["EVA_content"]
        )

        fr_a = rng.uniform(
            min_values["FR_A"],
            max_values["FR_A"]
        )

        fr_b = rng.uniform(
            min_values["FR_B"],
            max_values["FR_B"]
        )

        fr_c = rng.uniform(
            min_values["FR_C"],
            max_values["FR_C"]
        )

        fr_d = rng.uniform(
            min_values["FR_D"],
            max_values["FR_D"]
        )

        additive_1 = rng.uniform(
            min_values["Additive_1"],
            max_values["Additive_1"]
        )

        additive_2 = rng.uniform(
            min_values["Additive_2"],
            max_values["Additive_2"]
        )

        # ----------------------------------------------------
        # Formulation total constraint
        #
        # EVA is calculated as the balance component so the
        # final formulation is exactly 100 wt%.
        # ----------------------------------------------------

        other_total = (
            polymer_a
            + polymer_b
            + fr_a
            + fr_b
            + fr_c
            + fr_d
            + additive_1
            + additive_2
        )

        eva_required = 100.0 - other_total

        # ----------------------------------------------------
        # Keep EVA inside experimental range
        # ----------------------------------------------------

        if (
            eva_required
            <
            min_values["EVA_content"]
        ):
            continue

        if (
            eva_required
            >
            max_values["EVA_content"]
        ):
            continue

        candidate_list.append([
            eva_required,
            polymer_a,
            polymer_b,
            fr_a,
            fr_b,
            fr_c,
            fr_d,
            additive_1,
            additive_2
        ])

    candidates = pd.DataFrame(
        candidate_list,
        columns=FEATURES
    )

    # Remove numerical duplicates
    candidates = candidates.drop_duplicates(
        subset=FEATURES
    ).reset_index(drop=True)

    print(
        f"Valid candidates generated: "
        f"{len(candidates)}"
    )

    print(
        f"Unique candidates: "
        f"{len(candidates)}"
    )

    return candidates


# ============================================================
# Applicability filtering
# ============================================================

def filter_applicability_domain(
    candidates,
    experimental_df,
    scaler,
    threshold
):

    X_exp = experimental_df[
        FEATURES
    ].copy()

    X_candidates = candidates[
        FEATURES
    ].copy()

    X_exp_scaled = scaler.transform(
        X_exp
    )

    X_candidate_scaled = scaler.transform(
        X_candidates
    )

    distances = pairwise_distances(
        X_candidate_scaled,
        X_exp_scaled,
        metric="euclidean"
    )

    nearest_distance = distances.min(
        axis=1
    )

    nearest_index = distances.argmin(
        axis=1
    )

    nearest_ids = (
        experimental_df
        .iloc[nearest_index]
        ["sample_ID"]
        .values
    )

    candidates[
        "Nearest_Experimental_Distance"
    ] = nearest_distance

    candidates[
        "Nearest_Experimental_ID"
    ] = nearest_ids

    candidates[
        "Inside_Applicability_Domain"
    ] = (
        nearest_distance <= threshold
    )

    safe_candidates = candidates[
        candidates[
            "Inside_Applicability_Domain"
        ]
    ].copy()

    return safe_candidates


# ============================================================
# Candidate prediction
# ============================================================

def predict_candidates(
    candidates,
    loi_model,
    ul94_model,
    trans_model
):

    print()
    print("Predicting candidate utility...")

    X = candidates[
        FEATURES
    ]

    print("Predicting LOI...")

    predicted_loi = loi_model.predict(
        X
    )

    print("Predicting UL-94 probabilities...")

    ul94_prob = ul94_model.predict_proba(
        X
    )

    ul94_classes = (
        ul94_model
        .classes_
    )

    # --------------------------------------------------------
    # Convert UL-94 probabilities into fire score
    # --------------------------------------------------------

    fire_score = np.zeros(
        len(candidates)
    )

    for i, class_name in enumerate(
        ul94_classes
    ):

        probability = ul94_prob[:, i]

        if class_name == "V-1":
            fire_score += (
                probability
                *
                2.0
                /
                3.0
            )

        elif class_name == "V-2":
            fire_score += (
                probability
                *
                1.0
                /
                3.0
            )

        elif class_name == "V-0":
            fire_score += probability

    predicted_ul94 = (
        ul94_classes[
            np.argmax(
                ul94_prob,
                axis=1
            )
        ]
    )

    print("Predicting Transmittance...")

    predicted_trans = (
        trans_model.predict(
            X
        )
    )

    candidates[
        "Predicted_LOI"
    ] = predicted_loi

    candidates[
        "Predicted_Fire_Score"
    ] = fire_score

    candidates[
        "Predicted_UL94"
    ] = predicted_ul94

    candidates[
        "Predicted_Transmittance"
    ] = predicted_trans

    return candidates


# ============================================================
# Multi-objective utility
# ============================================================

def calculate_candidate_utility(
    candidates,
    loi_min,
    loi_max,
    trans_min,
    trans_max
):

    # --------------------------------------------------------
    # LOI normalized score
    # --------------------------------------------------------

    loi_score = (
        candidates["Predicted_LOI"]
        - loi_min
    ) / (
        loi_max - loi_min
    )

    loi_score = np.clip(
        loi_score,
        0,
        1
    )

    # --------------------------------------------------------
    # Fire score
    # --------------------------------------------------------

    fire_score = np.clip(
        candidates[
            "Predicted_Fire_Score"
        ],
        0,
        1
    )

    # --------------------------------------------------------
    # Transmittance score
    # --------------------------------------------------------

    trans_score = (
        candidates[
            "Predicted_Transmittance"
        ]
        - trans_min
    ) / (
        trans_max - trans_min
    )

    trans_score = np.clip(
        trans_score,
        0,
        1
    )

    utility = (
        0.40 * loi_score
        +
        0.35 * fire_score
        +
        0.25 * trans_score
    )

    candidates[
        "Predicted_Utility"
    ] = utility

    return candidates


# ============================================================
# Gaussian Process prediction
# ============================================================

def predict_gp_utility(
    candidates,
    gp,
    gp_scaler
):

    X = candidates[
        FEATURES
    ]

    X_scaled = gp_scaler.transform(
        X
    )

    utility_mean, utility_std = (
        gp.predict(
            X_scaled,
            return_std=True
        )
    )

    candidates[
        "Utility_Mean"
    ] = utility_mean

    candidates[
        "Utility_STD"
    ] = utility_std

    candidates[
        "UCB"
    ] = (
        utility_mean
        +
        2.0
        *
        utility_std
    )

    return candidates


# ============================================================
# Safety factor
# ============================================================

def calculate_safety_factor(
    candidates,
    threshold
):

    # --------------------------------------------------------
    # Safety factor based on formulation distance
    #
    # distance = 0       -> safety factor = 1
    # distance = threshold -> safety factor = 0.5
    #
    # Since candidates are already inside the domain,
    # the factor remains between 0.5 and 1.0.
    # --------------------------------------------------------

    distance = candidates[
        "Nearest_Experimental_Distance"
    ].values

    normalized_distance = (
        distance / threshold
    )

    safety_factor = (
        1.0
        -
        0.5 * normalized_distance
    )

    safety_factor = np.clip(
        safety_factor,
        0.5,
        1.0
    )

    candidates[
        "Safety_Factor"
    ] = safety_factor

    candidates[
        "Safe_UCB"
    ] = (
        candidates["UCB"]
        *
        candidates["Safety_Factor"]
    )

    return candidates


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_dataset()

    loi_model, ul94_model, trans_model = (
        load_models()
    )

    # --------------------------------------------------------
    # Applicability domain
    # --------------------------------------------------------

    ad_scaler, ad_threshold = (
        calculate_applicability_domain(
            df
        )
    )

    # --------------------------------------------------------
    # Experimental ranges
    # --------------------------------------------------------

    (
        loi_min,
        loi_max,
        trans_min,
        trans_max
    ) = calculate_performance_ranges(
        df
    )

    # --------------------------------------------------------
    # Experimental objective
    # --------------------------------------------------------

    experimental_utility = (
        calculate_experimental_utility(
            df,
            loi_min,
            loi_max,
            trans_min,
            trans_max
        )
    )

    # --------------------------------------------------------
    # Gaussian Process
    # --------------------------------------------------------

    gp, gp_scaler = (
        train_gaussian_process(
            df,
            experimental_utility
        )
    )

    # --------------------------------------------------------
    # Candidate generation
    # --------------------------------------------------------

    candidates = generate_candidates(
        df,
        n_candidates=30000,
        random_state=42
    )

    # --------------------------------------------------------
    # Applicability filtering
    # --------------------------------------------------------

    candidates = filter_applicability_domain(
        candidates,
        df,
        ad_scaler,
        ad_threshold
    )

    print()
    print(
        f"Candidates inside applicability domain: "
        f"{len(candidates)}"
    )

    if len(candidates) == 0:

        raise RuntimeError(
            "No candidates remain after "
            "applicability-domain filtering."
        )

    # --------------------------------------------------------
    # Candidate predictions
    # --------------------------------------------------------

    candidates = predict_candidates(
        candidates,
        loi_model,
        ul94_model,
        trans_model
    )

    # --------------------------------------------------------
    # Multi-objective utility
    # --------------------------------------------------------

    candidates = calculate_candidate_utility(
        candidates,
        loi_min,
        loi_max,
        trans_min,
        trans_max
    )

    # --------------------------------------------------------
    # GP utility prediction
    # --------------------------------------------------------

    candidates = predict_gp_utility(
        candidates,
        gp,
        gp_scaler
    )

    # --------------------------------------------------------
    # Safety factor
    # --------------------------------------------------------

    candidates = calculate_safety_factor(
        candidates,
        ad_threshold
    )

    # --------------------------------------------------------
    # Sort by Safe UCB
    # --------------------------------------------------------

    candidates = candidates.sort_values(
        "Safe_UCB",
        ascending=False
    ).reset_index(
        drop=True
    )

    candidates[
        "Candidate_Rank"
    ] = (
        np.arange(
            len(candidates)
        )
        + 1
    )

    # --------------------------------------------------------
    # Top 20
    # --------------------------------------------------------

    top_candidates = candidates.head(
        20
    ).copy()

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print(
        "Top Safe Bayesian Optimization candidates:"
    )

    display_columns = [
        "Candidate_Rank",
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
        "Safe_UCB"
    ]

    print(
        top_candidates[
            display_columns
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save all candidates
    # --------------------------------------------------------

    all_candidates_path = os.path.join(
        RESULT_DIR,
        "safe_bo_all_candidates.csv"
    )

    candidates.to_csv(
        all_candidates_path,
        index=False
    )

    # --------------------------------------------------------
    # Save top candidates
    # --------------------------------------------------------

    top_candidates_path = os.path.join(
        RESULT_DIR,
        "safe_bo_candidates.csv"
    )

    top_candidates.to_csv(
        top_candidates_path,
        index=False
    )

    # --------------------------------------------------------
    # Save applicability-domain information
    # --------------------------------------------------------

    ad_summary = pd.DataFrame({

        "Metric": [
            "Experimental_Samples",
            "Initial_Candidates",
            "Safe_Candidates",
            "Applicability_Distance_Threshold",
            "LOI_Min",
            "LOI_Max",
            "Transmittance_Min",
            "Transmittance_Max"
        ],

        "Value": [
            len(df),
            30000,
            len(candidates),
            ad_threshold,
            loi_min,
            loi_max,
            trans_min,
            trans_max
        ]
    })

    ad_summary_path = os.path.join(
        RESULT_DIR,
        "safe_bo_summary.csv"
    )

    ad_summary.to_csv(
        ad_summary_path,
        index=False
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print(
        "Safe Bayesian Optimization completed."
    )

    print()
    print("Results saved to:")

    print(all_candidates_path)
    print(top_candidates_path)
    print(ad_summary_path)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()