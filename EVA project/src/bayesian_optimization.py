import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATH
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
# FEATURES
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
# MODEL PATHS
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
# STEP 13-1
# SEARCH SPACE ANALYSIS
# ============================================================

print("=" * 70)
print("STEP 13-1 - BAYESIAN OPTIMIZATION SEARCH SPACE")
print("=" * 70)

print("Loading cleaned dataset...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Cleaned dataset not found:\n{DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print(f"Samples: {len(df)}")
print(f"Features: {len(FEATURES)}")


print()
print("Analyzing formulation variable ranges...")


range_df = pd.DataFrame({
    "Feature": FEATURES,
    "Min": [
        df[col].min()
        for col in FEATURES
    ],
    "Max": [
        df[col].max()
        for col in FEATURES
    ],
    "Mean": [
        df[col].mean()
        for col in FEATURES
    ],
    "Std": [
        df[col].std()
        for col in FEATURES
    ]
})


print()
print("Variable ranges:")

print(
    range_df.to_string(
        index=False
    )
)


# ============================================================
# FORMULATION TOTAL
# ============================================================

print()
print("Checking formulation totals...")


formulation_totals = df[FEATURES].sum(axis=1)


print(
    f"Minimum: {formulation_totals.min():.4f}"
)

print(
    f"Maximum: {formulation_totals.max():.4f}"
)

print(
    f"Mean: {formulation_totals.mean():.4f}"
)

print(
    f"Std: {formulation_totals.std():.4f}"
)


within_range = (
    np.abs(formulation_totals - 100.0)
    <= 0.5
)


print(
    f"Formulations within 100 ± 0.5 wt%: "
    f"{within_range.sum()}"
)

print(
    f"Formulations outside 100 ± 0.5 wt%: "
    f"{(~within_range).sum()}"
)


# ============================================================
# VARIABLE TYPES
# ============================================================

print()
print("Analyzing variable behavior...")


INTEGER_LIKE = [
    "Polymer_A",
    "Polymer_B"
]

CONTINUOUS = [
    feature
    for feature in FEATURES
    if feature not in INTEGER_LIKE
]


print()
print("Integer-like variables:")

for feature in INTEGER_LIKE:
    print(f"- {feature}")


print()
print("Continuous variables:")

for feature in CONTINUOUS:
    print(f"- {feature}")


# ============================================================
# UNIQUE VALUES
# ============================================================

print()
print("Analyzing number of unique values...")


unique_df = pd.DataFrame({
    "Feature": FEATURES,
    "Unique_Values": [
        df[col].nunique()
        for col in FEATURES
    ],
    "Min": [
        df[col].min()
        for col in FEATURES
    ],
    "Max": [
        df[col].max()
        for col in FEATURES
    ]
})


print(
    unique_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE SEARCH SPACE
# ============================================================

search_space_path = os.path.join(
    RESULT_DIR,
    "search_space_summary.csv"
)

range_df.to_csv(
    search_space_path,
    index=False
)


unique_path = os.path.join(
    RESULT_DIR,
    "variable_unique_values.csv"
)

unique_df.to_csv(
    unique_path,
    index=False
)


print()
print("Search space analysis completed.")

print(
    f"Results saved to: {RESULT_DIR}"
)


# ============================================================
# STEP 13-2
# BAYESIAN OPTIMIZATION
# ============================================================

print()
print("=" * 70)
print("STEP 13-2 - BAYESIAN OPTIMIZATION")
print("=" * 70)


print("Loading cleaned dataset...")

print(f"Samples: {len(df)}")


# ============================================================
# LOAD MODELS
# ============================================================

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
        f"Transmittance model not found:\n"
        f"{TRANSMITTANCE_MODEL_PATH}"
    )

transmittance_model = joblib.load(
    TRANSMITTANCE_MODEL_PATH
)


# ============================================================
# EXPERIMENTAL OBJECTIVE
# ============================================================

print(
    "Calculating experimental objective values..."
)


X_exp = df[FEATURES].copy()


experimental_loi = (
    loi_model.predict(X_exp)
)


experimental_transmittance = (
    transmittance_model.predict(X_exp)
)


experimental_fire_prob = (
    ul94_model.predict_proba(X_exp)
)


ul94_classes = list(
    ul94_model.classes_
)


UL94_SCORE_MAP = {
    "NR": 0.0,
    "V-2": 1.0 / 3.0,
    "V-1": 2.0 / 3.0,
    "V-0": 1.0
}


experimental_fire_score = np.zeros(
    len(df)
)


for i, cls in enumerate(
    ul94_classes
):

    if cls in UL94_SCORE_MAP:

        experimental_fire_score += (
            experimental_fire_prob[:, i]
            * UL94_SCORE_MAP[cls]
        )


# ============================================================
# MULTI-OBJECTIVE UTILITY
# ============================================================

def normalize_array(values):

    minimum = np.min(values)
    maximum = np.max(values)

    if maximum - minimum < 1e-12:
        return np.ones_like(values) * 0.5

    return (
        (values - minimum)
        /
        (maximum - minimum)
    )


loi_norm = normalize_array(
    experimental_loi
)

fire_norm = normalize_array(
    experimental_fire_score
)

trans_norm = normalize_array(
    experimental_transmittance
)


# Equal-weight objective

experimental_utility = (
    loi_norm
    + fire_norm
    + trans_norm
) / 3.0


# ============================================================
# PREPARE GP INPUT
# ============================================================

print(
    "Preparing Gaussian Process input..."
)


scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X_exp
)


# ============================================================
# TRAIN GP
# ============================================================

print(
    "Training Gaussian Process..."
)


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
    alpha=0.01,
    normalize_y=True,
    random_state=42,
    n_restarts_optimizer=3
)


gp.fit(
    X_scaled,
    experimental_utility
)


print(
    "Gaussian Process training completed."
)


# ============================================================
# GENERATE CANDIDATES
# ============================================================

print(
    "Generating valid formulation candidates..."
)


RANDOM_SEED = 42

rng = np.random.default_rng(
    RANDOM_SEED
)


N_CANDIDATES = 20000


candidate_rows = []


# ------------------------------------------------------------
# Generate candidates by randomly selecting 8 variables
# and calculating EVA content as the balance to 100 wt%.
# ------------------------------------------------------------

for _ in range(N_CANDIDATES * 5):

    if len(candidate_rows) >= N_CANDIDATES:
        break

    candidate = {}


    # Polymer_A / Polymer_B
    # Integer-like variables
    candidate["Polymer_A"] = rng.choice(
        sorted(
            df["Polymer_A"]
            .unique()
        )
    )

    candidate["Polymer_B"] = rng.choice(
        sorted(
            df["Polymer_B"]
            .unique()
        )
    )


    # Continuous variables
    for feature in [
        "FR_A",
        "FR_B",
        "FR_C",
        "FR_D",
        "Additive_1",
        "Additive_2"
    ]:

        minimum = df[feature].min()
        maximum = df[feature].max()

        candidate[feature] = rng.uniform(
            minimum,
            maximum
        )


    # Calculate EVA as balance
    non_eva_sum = sum(
        candidate[feature]
        for feature in FEATURES
        if feature != "EVA_content"
    )


    eva_content = (
        100.0
        - non_eva_sum
    )


    eva_min = df["EVA_content"].min()
    eva_max = df["EVA_content"].max()


    if (
        eva_content < eva_min
        or
        eva_content > eva_max
    ):
        continue


    candidate["EVA_content"] = eva_content


    # Keep feature order consistent
    candidate_rows.append(
        [
            candidate[feature]
            for feature in FEATURES
        ]
    )


candidates = pd.DataFrame(
    candidate_rows,
    columns=FEATURES
)


print(
    f"Generated candidates: "
    f"{len(candidates)}"
)


# ============================================================
# FORMULATION CONSTRAINT CHECK
# ============================================================

candidate_totals = candidates[
    FEATURES
].sum(axis=1)


print(
    "Checking formulation constraint..."
)

print(
    f"Minimum total: "
    f"{candidate_totals.min():.6f}"
)

print(
    f"Maximum total: "
    f"{candidate_totals.max():.6f}"
)


valid_mask = (
    np.abs(
        candidate_totals - 100.0
    )
    <= 0.5
)


candidates = candidates[
    valid_mask
].copy()


# ============================================================
# PREDICT GP UTILITY
# ============================================================

X_candidates = scaler.transform(
    candidates[FEATURES]
)


utility_mean, utility_std = (
    gp.predict(
        X_candidates,
        return_std=True
    )
)


# ============================================================
# UCB
# ============================================================

BETA = 2.0


ucb = (
    utility_mean
    +
    BETA * utility_std
)


candidates["Utility_Mean"] = (
    utility_mean
)

candidates["Utility_STD"] = (
    utility_std
)

candidates["UCB"] = ucb


# ============================================================
# SORT CANDIDATES
# ============================================================

candidates = candidates.sort_values(
    by="UCB",
    ascending=False
).reset_index(
    drop=True
)


candidates.insert(
    0,
    "Candidate_Rank",
    np.arange(
        1,
        len(candidates) + 1
    )
)


# ============================================================
# TOP 20
# ============================================================

top_candidates = candidates.head(
    20
).copy()


print()
print(
    "Top Bayesian Optimization candidates:"
)

print(
    top_candidates.to_string(
        index=False
    )
)


# ============================================================
# SAVE ALL CANDIDATES
# ============================================================

all_candidates_path = os.path.join(
    RESULT_DIR,
    "bo_all_candidates.csv"
)

candidates.to_csv(
    all_candidates_path,
    index=False
)


# ============================================================
# SAVE TOP 20
# ============================================================

top_candidates_path = os.path.join(
    RESULT_DIR,
    "bo_top20_candidates.csv"
)

top_candidates.to_csv(
    top_candidates_path,
    index=False
)


# ============================================================
# SAVE GP MODEL
# ============================================================

gp_path = os.path.join(
    RESULT_DIR,
    "gaussian_process_model.pkl"
)

joblib.dump(
    {
        "model": gp,
        "scaler": scaler,
        "features": FEATURES
    },
    gp_path
)


# ============================================================
# ACQUISITION PLOT
# ============================================================

print(
    "Generating acquisition plot..."
)


plt.figure(
    figsize=(10, 7)
)

plt.scatter(
    candidates["Utility_Mean"],
    candidates["Utility_STD"],
    alpha=0.35,
    s=12
)

plt.scatter(
    top_candidates["Utility_Mean"],
    top_candidates["Utility_STD"],
    s=45,
    label="Top 20 candidates"
)

plt.xlabel(
    "Utility Mean"
)

plt.ylabel(
    "Utility Standard Deviation"
)

plt.title(
    "Bayesian Optimization Acquisition Space"
)

plt.legend()

plt.grid(
    alpha=0.25
)

plt.tight_layout()


plot_path = os.path.join(
    RESULT_DIR,
    "bo_acquisition_plot.png"
)

plt.savefig(
    plot_path,
    dpi=300
)

plt.close()


# ============================================================
# COMPLETION
# ============================================================

print()
print(
    "Bayesian Optimization candidate generation completed."
)

print(
    f"Results saved to: {RESULT_DIR}"
)

print()
print("=" * 70)