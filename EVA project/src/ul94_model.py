import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
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

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "ul94_model"
)

os.makedirs(MODEL_DIR, exist_ok=True)
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

    print("STEP 7 - UL-94 CLASSIFICATION")
    print("Loading cleaned dataset...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Cleaned dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"Samples: {len(df)}")
    print(f"Features: {len(FEATURES)}")
    print(f"Target: {TARGET}")

    return df


# ============================================================
# Prepare data
# ============================================================

def prepare_data(df):

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    if X.isnull().any().any():
        raise ValueError(
            "Missing values found in feature data."
        )

    if y.isnull().any():
        raise ValueError(
            "Missing values found in UL-94 target."
        )

    return X, y


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_data()

    X, y = prepare_data(df)

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print()
    print("UL-94 class distribution:")

    print(
        y.value_counts()
    )

    # --------------------------------------------------------
    # Train-test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print()
    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Test samples: {len(X_test)}"
    )

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    print()
    print("Training LogisticRegression...")

    logistic_model = LogisticRegression(
        max_iter=2000,
        random_state=42
    )

    logistic_model.fit(
        X_train,
        y_train
    )

    logistic_pred = logistic_model.predict(
        X_test
    )

    logistic_accuracy = accuracy_score(
        y_test,
        logistic_pred
    )

    logistic_precision = precision_score(
        y_test,
        logistic_pred,
        average="macro",
        zero_division=0
    )

    logistic_recall = recall_score(
        y_test,
        logistic_pred,
        average="macro",
        zero_division=0
    )

    logistic_f1 = f1_score(
        y_test,
        logistic_pred,
        average="macro",
        zero_division=0
    )

    print(
        f"LogisticRegression: "
        f"Accuracy={logistic_accuracy:.3f}, "
        f"Macro Precision={logistic_precision:.3f}, "
        f"Macro Recall={logistic_recall:.3f}, "
        f"Macro F1={logistic_f1:.3f}"
    )

    print()
    print("Classification report:")

    print(
        classification_report(
            y_test,
            logistic_pred,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    print()
    print("Training RandomForest...")

    rf_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1
    )

    rf_model.fit(
        X_train,
        y_train
    )

    rf_pred = rf_model.predict(
        X_test
    )

    rf_accuracy = accuracy_score(
        y_test,
        rf_pred
    )

    rf_precision = precision_score(
        y_test,
        rf_pred,
        average="macro",
        zero_division=0
    )

    rf_recall = recall_score(
        y_test,
        rf_pred,
        average="macro",
        zero_division=0
    )

    rf_f1 = f1_score(
        y_test,
        rf_pred,
        average="macro",
        zero_division=0
    )

    print(
        f"RandomForest: "
        f"Accuracy={rf_accuracy:.3f}, "
        f"Macro Precision={rf_precision:.3f}, "
        f"Macro Recall={rf_recall:.3f}, "
        f"Macro F1={rf_f1:.3f}"
    )

    print()
    print("Classification report:")

    print(
        classification_report(
            y_test,
            rf_pred,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

    comparison = pd.DataFrame({

        "Model": [
            "LogisticRegression",
            "RandomForest"
        ],

        "Accuracy": [
            logistic_accuracy,
            rf_accuracy
        ],

        "Macro_Precision": [
            logistic_precision,
            rf_precision
        ],

        "Macro_Recall": [
            logistic_recall,
            rf_recall
        ],

        "Macro_F1": [
            logistic_f1,
            rf_f1
        ]
    })

    comparison_path = os.path.join(
        RESULT_DIR,
        "model_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False
    )

    print()
    print("Model comparison saved to:")

    print(comparison_path)

    # --------------------------------------------------------
    # Select best model
    # --------------------------------------------------------

    best_index = comparison[
        "Macro_F1"
    ].idxmax()

    best_model_name = comparison.loc[
        best_index,
        "Model"
    ]

    print()
    print(
        f"Best model based on Macro F1: "
        f"{best_model_name}"
    )

    if best_model_name == "LogisticRegression":

        final_model = LogisticRegression(
            max_iter=2000,
            random_state=42
        )

    else:

        final_model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1
        )

    # --------------------------------------------------------
    # Train final model on all experimental data
    # --------------------------------------------------------

    print()
    print("Training final UL-94 model...")

    final_model.fit(
        X,
        y
    )

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        "ul94_model.pkl"
    )

    joblib.dump(
        final_model,
        model_path
    )

    print()
    print("Final UL-94 model saved to:")

    print(model_path)

    print()
    print("UL-94 classification completed.")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()