import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
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
    "transmittance_model.pkl"
)

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "transmittance_model"
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

TARGET = "Transmittance"


# ============================================================
# Load dataset
# ============================================================

def load_data():

    print("STEP 10 - TRANSMITTANCE MODEL DIAGNOSTICS")
    print("Loading cleaned dataset...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Cleaned dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"Samples: {len(df)}")

    return df


# ============================================================
# Load trained model
# ============================================================

def load_model():

    print("Loading trained Transmittance model...")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Transmittance model not found:\n{MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    return model


# ============================================================
# Diagnostic evaluation
# ============================================================

def evaluate_model(df, model):

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print("Training diagnostic model...")

    # The saved model was trained on the full dataset.
    # For diagnostics, retrain a fresh model on the same
    # 80/20 split to reproduce the Step 9 hold-out result.

    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor

    model_name = type(model).__name__

    if model_name == "LinearRegression":

        diagnostic_model = LinearRegression()

    elif model_name == "RandomForestRegressor":

        diagnostic_model = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1
        )

    else:

        diagnostic_model = model

    diagnostic_model.fit(
        X_train,
        y_train
    )

    predictions = diagnostic_model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2: {r2:.3f}")

    return (
        diagnostic_model,
        X_test,
        y_test,
        predictions,
        mae,
        rmse,
        r2
    )


# ============================================================
# Actual vs predicted plot
# ============================================================

def plot_actual_vs_predicted(
    y_test,
    predictions
):

    print(
        "Generating actual vs predicted plot..."
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.scatter(
        y_test,
        predictions,
        alpha=0.8
    )

    min_value = min(
        y_test.min(),
        predictions.min()
    )

    max_value = max(
        y_test.max(),
        predictions.max()
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )

    plt.xlabel(
        "Actual Transmittance"
    )

    plt.ylabel(
        "Predicted Transmittance"
    )

    plt.title(
        "Transmittance: Actual vs Predicted"
    )

    plt.tight_layout()

    output_path = os.path.join(
        RESULT_DIR,
        "actual_vs_predicted.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()


# ============================================================
# Residual plot
# ============================================================

def plot_residuals(
    y_test,
    predictions
):

    print(
        "Generating residual plot..."
    )

    residuals = y_test - predictions

    plt.figure(
        figsize=(7, 6)
    )

    plt.scatter(
        predictions,
        residuals,
        alpha=0.8
    )

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.xlabel(
        "Predicted Transmittance"
    )

    plt.ylabel(
        "Residual"
    )

    plt.title(
        "Transmittance Residual Plot"
    )

    plt.tight_layout()

    output_path = os.path.join(
        RESULT_DIR,
        "residual_plot.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()


# ============================================================
# Regression coefficient extraction
# ============================================================

def extract_coefficients(model):

    print(
        "Extracting regression coefficients..."
    )

    if not hasattr(
        model,
        "coef_"
    ):

        print(
            "Model does not contain linear coefficients."
        )

        return None

    coefficients = pd.DataFrame({

        "Feature": FEATURES,

        "Coefficient": model.coef_
    })

    coefficients["Absolute_Coefficient"] = (
        coefficients["Coefficient"]
        .abs()
    )

    coefficients = coefficients.sort_values(
        "Absolute_Coefficient",
        ascending=False
    )

    coefficient_path = os.path.join(
        RESULT_DIR,
        "regression_coefficients.csv"
    )

    coefficients.to_csv(
        coefficient_path,
        index=False
    )

    return coefficients


# ============================================================
# Coefficient plot
# ============================================================

def plot_coefficients(
    coefficients
):

    if coefficients is None:
        return

    print(
        "Generating coefficient plot..."
    )

    plot_df = coefficients.sort_values(
        "Coefficient"
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.barh(
        plot_df["Feature"],
        plot_df["Coefficient"]
    )

    plt.xlabel(
        "Regression Coefficient"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Transmittance Regression Coefficients"
    )

    plt.tight_layout()

    output_path = os.path.join(
        RESULT_DIR,
        "coefficient_plot.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()


# ============================================================
# Main
# ============================================================

def main():

    df = load_data()

    model = load_model()

    (
        diagnostic_model,
        X_test,
        y_test,
        predictions,
        mae,
        rmse,
        r2
    ) = evaluate_model(
        df,
        model
    )

    # --------------------------------------------------------
    # Actual vs predicted
    # --------------------------------------------------------

    plot_actual_vs_predicted(
        y_test,
        predictions
    )

    # --------------------------------------------------------
    # Residual analysis
    # --------------------------------------------------------

    plot_residuals(
        y_test,
        predictions
    )

    # --------------------------------------------------------
    # Coefficients
    # --------------------------------------------------------

    coefficients = extract_coefficients(
        diagnostic_model
    )

    plot_coefficients(
        coefficients
    )

    # --------------------------------------------------------
    # Save diagnostic metrics
    # --------------------------------------------------------

    metrics = pd.DataFrame({

        "Metric": [
            "MAE",
            "RMSE",
            "R2"
        ],

        "Value": [
            mae,
            rmse,
            r2
        ]
    })

    metrics_path = os.path.join(
        RESULT_DIR,
        "diagnostic_metrics.csv"
    )

    metrics.to_csv(
        metrics_path,
        index=False
    )

    print()
    print(
        "Transmittance diagnostics completed."
    )

    print(
        "Results saved to:"
    )

    print(RESULT_DIR)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()