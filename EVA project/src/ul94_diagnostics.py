import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


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

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "ul94_model.pkl"
)

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "ul94_model"
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

TARGET = "UL_94"


# ============================================================
# Load dataset
# ============================================================

def load_data():

    print("STEP 8 - UL-94 MODEL DIAGNOSTICS")
    print("Loading cleaned dataset...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Cleaned dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"Samples: {len(df)}")

    return df


# ============================================================
# Load model
# ============================================================

def load_model():

    print("Loading trained UL-94 model...")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"UL-94 model not found:\n{MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    return model


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    # --------------------------------------------------------
    # Reproduce diagnostic test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------------
    # Diagnostic prediction
    # --------------------------------------------------------

    print("Training diagnostic model...")

    diagnostic_model = model

    diagnostic_model.fit(
        X_train,
        y_train
    )

    predictions = diagnostic_model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    macro_precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    print()
    print(
        f"Accuracy: {accuracy:.3f}"
    )

    print(
        f"Macro Precision: {macro_precision:.3f}"
    )

    print(
        f"Macro Recall: {macro_recall:.3f}"
    )

    print(
        f"Macro F1: {macro_f1:.3f}"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print()
    print("Classification report:")

    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    print(report)

    # --------------------------------------------------------
    # Save classification report
    # --------------------------------------------------------

    report_path = os.path.join(
        RESULT_DIR,
        "classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(report)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print("Generating confusion matrix...")

    labels = ["NR", "V-1", "V-2"]

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    display.plot(
        ax=ax,
        values_format="d"
    )

    ax.set_title(
        "UL-94 Confusion Matrix"
    )

    plt.tight_layout()

    confusion_path = os.path.join(
        RESULT_DIR,
        "confusion_matrix.png"
    )

    plt.savefig(
        confusion_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Fire performance score
    # --------------------------------------------------------

    # UL-94 fire performance score:
    #
    # NR  -> 0
    # V-2 -> 1/3
    # V-1 -> 2/3
    # V-0 -> 1
    #
    # The score is calculated from predicted probabilities
    # when probability information is available.

    if hasattr(
        diagnostic_model,
        "predict_proba"
    ):

        probabilities = diagnostic_model.predict_proba(
            X_test
        )

        class_names = list(
            diagnostic_model.classes_
        )

        fire_scores = []

        score_mapping = {
            "NR": 0.0,
            "V-2": 1.0 / 3.0,
            "V-1": 2.0 / 3.0,
            "V-0": 1.0
        }

        for row in probabilities:

            score = 0.0

            for class_name, probability in zip(
                class_names,
                row
            ):

                score += (
                    probability
                    * score_mapping.get(
                        class_name,
                        0.0
                    )
                )

            fire_scores.append(score)

        fire_performance_score = (
            sum(fire_scores)
            / len(fire_scores)
        )

    else:

        fire_performance_score = 0.0

    print()
    print("Fire performance score calculated.")

    # --------------------------------------------------------
    # Save diagnostic summary
    # --------------------------------------------------------

    summary = pd.DataFrame({

        "Metric": [
            "Accuracy",
            "Macro_Precision",
            "Macro_Recall",
            "Macro_F1",
            "Fire_Performance_Score"
        ],

        "Value": [
            accuracy,
            macro_precision,
            macro_recall,
            macro_f1,
            fire_performance_score
        ]
    })

    summary_path = os.path.join(
        RESULT_DIR,
        "diagnostic_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print()
    print("UL-94 diagnostics completed.")

    print(
        f"Results saved to: {RESULT_DIR}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()