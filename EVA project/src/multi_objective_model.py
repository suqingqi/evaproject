import os
import joblib
import numpy as np
import pandas as pd


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
    "multi_objective"
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
# UL-94 class mapping
# ============================================================

UL94_CLASSES = [
    "NR",
    "V-2",
    "V-1",
    "V-0"
]


# ============================================================
# Load models
# ============================================================

def load_models():

    print("STEP 11 - MULTI-OBJECTIVE MODEL INTEGRATION")

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

    if not os.path.exists(
        TRANSMITTANCE_MODEL_PATH
    ):
        raise FileNotFoundError(
            "Transmittance model not found:\n"
            f"{TRANSMITTANCE_MODEL_PATH}"
        )

    transmittance_model = joblib.load(
        TRANSMITTANCE_MODEL_PATH
    )

    return (
        loi_model,
        ul94_model,
        transmittance_model
    )


# ============================================================
# Load experimental data
# ============================================================

def load_data():

    print("Loading cleaned dataset...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Cleaned dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        f"Samples: {len(df)}"
    )

    return df


# ============================================================
# Fire performance score
# ============================================================

def calculate_fire_score(
    ul94_model,
    X
):

    probabilities = ul94_model.predict_proba(
        X
    )

    classes = list(
        ul94_model.classes_
    )

    probability_dict = {}

    for i, class_name in enumerate(classes):

        probability_dict[
            f"P({class_name})"
        ] = probabilities[:, i]

    # --------------------------------------------------------
    # Fire score definition
    #
    # NR  = 0
    # V-2 = 0.3333
    # V-1 = 0.6667
    # V-0 = 1.0000
    # --------------------------------------------------------

    class_scores = {
        "NR": 0.0,
        "V-2": 1.0 / 3.0,
        "V-1": 2.0 / 3.0,
        "V-0": 1.0
    }

    fire_score = np.zeros(
        len(X)
    )

    for i, class_name in enumerate(
        classes
    ):

        score = class_scores.get(
            class_name,
            0.0
        )

        fire_score += (
            probabilities[:, i]
            * score
        )

    return (
        probability_dict,
        fire_score
    )


# ============================================================
# Unified prediction
# ============================================================

def predict_multi_objective(
    loi_model,
    ul94_model,
    transmittance_model,
    X
):

    loi_prediction = loi_model.predict(
        X
    )

    transmittance_prediction = (
        transmittance_model.predict(
            X
        )
    )

    ul94_prediction = (
        ul94_model.predict(
            X
        )
    )

    probability_dict, fire_score = (
        calculate_fire_score(
            ul94_model,
            X
        )
    )

    result = pd.DataFrame({

        "Predicted_LOI":
            loi_prediction,

        "Predicted_UL94":
            ul94_prediction,

        "Predicted_Fire_Score":
            fire_score,

        "Predicted_Transmittance":
            transmittance_prediction
    })

    for column, values in probability_dict.items():

        result[column] = values

    return result


# ============================================================
# Main
# ============================================================

def main():

    (
        loi_model,
        ul94_model,
        transmittance_model
    ) = load_models()

    df = load_data()

    # --------------------------------------------------------
    # Test formulation
    # --------------------------------------------------------

    test_formulation = pd.DataFrame([{

        "EVA_content": 68.1,

        "Polymer_A": 20.0,

        "Polymer_B": 0.0,

        "FR_A": 2.0,

        "FR_B": 2.4,

        "FR_C": 0.9,

        "FR_D": 2.6,

        "Additive_1": 3.1,

        "Additive_2": 0.9
    }])

    print()
    print("Testing unified prediction...")

    for feature in FEATURES:

        print(
            f"{feature}: "
            f"{test_formulation.iloc[0][feature]}"
        )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = predict_multi_objective(
        loi_model,
        ul94_model,
        transmittance_model,
        test_formulation[FEATURES]
    )

    print()
    print("Predicted performance:")

    print(
        f"LOI: "
        f"{prediction.iloc[0]['Predicted_LOI']:.3f}"
    )

    print(
        f"UL-94: "
        f"{prediction.iloc[0]['Predicted_UL94']}"
    )

    for class_name in UL94_CLASSES:

        column = f"P({class_name})"

        if column in prediction.columns:

            print(
                f"{column}: "
                f"{prediction.iloc[0][column]:.3f}"
            )

    print(
        f"Fire Score: "
        f"{prediction.iloc[0]['Predicted_Fire_Score']:.3f}"
    )

    print(
        f"Transmittance: "
        f"{prediction.iloc[0]['Predicted_Transmittance']:.3f}"
    )

    # --------------------------------------------------------
    # Save unified test prediction
    # --------------------------------------------------------

    output = test_formulation.copy()

    for column in prediction.columns:

        output[column] = prediction[
            column
        ].values

    output_path = os.path.join(
        RESULT_DIR,
        "unified_test_prediction.csv"
    )

    output.to_csv(
        output_path,
        index=False
    )

    print()
    print(
        "Unified prediction saved to:"
    )

    print(output_path)

    print()
    print(
        "Multi-objective model integration completed."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()