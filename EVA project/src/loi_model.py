import os
import sys
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
    "loi_model"
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

TARGET = "LOI"


# ============================================================
# Load dataset
# ============================================================

def load_data():

    print("STEP 4 - LOI REGRESSION")
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
        raise ValueError("Missing values found in feature data.")

    if y.isnull().any():
        raise ValueError("Missing values found in LOI target.")

    return X, y


# ============================================================
# Train and evaluate models
# ============================================================

def evaluate_model(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5
    r2 = r2_score(y_test, predictions)

    return model, mae, rmse, r2


# ============================================================
# Main
# ============================================================

def main():

    df = load_data()

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    print()
    print("Training LinearRegression...")

    linear_model = LinearRegression()

    linear_model, linear_mae, linear_rmse, linear_r2 = evaluate_model(
        linear_model,
        X_train,
        X_test,
        y_train,
        y_test
    )

    print(
        f"LinearRegression - "
        f"MAE: {linear_mae:.3f}, "
        f"RMSE: {linear_rmse:.3f}, "
        f"R2: {linear_r2:.3f}"
    )

    print()
    print("Training RandomForest...")

    rf_model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1
    )

    rf_model, rf_mae, rf_rmse, rf_r2 = evaluate_model(
        rf_model,
        X_train,
        X_test,
        y_train,
        y_test
    )

    print(
        f"RandomForest - "
        f"MAE: {rf_mae:.3f}, "
        f"RMSE: {rf_rmse:.3f}, "
        f"R2: {rf_r2:.3f}"
    )

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

    comparison = pd.DataFrame({
        "Model": [
            "LinearRegression",
            "RandomForest"
        ],
        "MAE": [
            linear_mae,
            rf_mae
        ],
        "RMSE": [
            linear_rmse,
            rf_rmse
        ],
        "R2": [
            linear_r2,
            rf_r2
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
    print("Saved model comparison to:")
    print(comparison_path)

    # --------------------------------------------------------
    # Select best model
    # --------------------------------------------------------

    best_index = comparison["RMSE"].idxmin()

    best_model_name = comparison.loc[
        best_index,
        "Model"
    ]

    if best_model_name == "LinearRegression":
        best_model = linear_model
    else:
        best_model = rf_model

    print()
    print(f"Best model: {best_model_name}")

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        "loi_model.pkl"
    )

    joblib.dump(
        best_model,
        model_path
    )

    print("Saved model to:")
    print(model_path)

    print()
    print("LOI regression completed.")


if __name__ == "__main__":
    main()