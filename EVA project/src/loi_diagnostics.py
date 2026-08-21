import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
    "loi_model.pkl"
)

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "loi_model"
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

TARGET = "LOI"


def load_data():

    print("STEP 6 - LOI MODEL DIAGNOSTICS")
    print("Loading cleaned dataset...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Cleaned dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"Samples: {len(df)}")

    return df


def load_model():

    print("Loading trained LOI model...")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"LOI model not found:\n{MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    return model


def calculate_metrics(y_true, y_pred):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = mean_squared_error(
        y_true,
        y_pred
    ) ** 0.5

    r2 = r2_score(
        y_true,
        y_pred
    )

    return mae, rmse, r2


def generate_actual_vs_predicted(
    y_test,
    predictions
):

    print("Generating actual vs predicted plot...")

    plt.figure(figsize=(7, 6))

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

    plt.xlabel("Actual LOI")
    plt.ylabel("Predicted LOI")
    plt.title("LOI Model: Actual vs Predicted")

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


def generate_residual_plot(
    y_test,
    predictions
):

    print("Generating residual plot...")

    residuals = y_test - predictions

    plt.figure(figsize=(7, 6))

    plt.scatter(
        predictions,
        residuals,
        alpha=0.8
    )

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.xlabel("Predicted LOI")
    plt.ylabel("Residual")
    plt.title("LOI Model Residual Plot")

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


def extract_coefficients(model):

    print("Extracting regression coefficients...")

    if not hasattr(model, "coef_"):

        print(
            "Selected model does not provide linear "
            "regression coefficients."
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

    output_path = os.path.join(
        RESULT_DIR,
        "coefficients.csv"
    )

    coefficients.to_csv(
        output_path,
        index=False
    )

    return coefficients


def generate_coefficient_plot(
    coefficients
):

    if coefficients is None:
        return

    print("Generating coefficient plot...")

    plot_df = coefficients.sort_values(
        "Coefficient"
    )

    plt.figure(figsize=(8, 6))

    plt.barh(
        plot_df["Feature"],
        plot_df["Coefficient"]
    )

    plt.xlabel("Regression Coefficient")
    plt.ylabel("Feature")
    plt.title("LOI Regression Coefficients")

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


def main():

    df = load_data()

    model = load_model()

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print("Training diagnostic model...")

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    mae, rmse, r2 = calculate_metrics(
        y_test,
        predictions
    )

    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2: {r2:.3f}")

    generate_actual_vs_predicted(
        y_test,
        predictions
    )

    generate_residual_plot(
        y_test,
        predictions
    )

    coefficients = extract_coefficients(
        model
    )

    generate_coefficient_plot(
        coefficients
    )

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
    print("LOI diagnostics completed.")
    print("Results saved to:")
    print(RESULT_DIR)


if __name__ == "__main__":
    main()