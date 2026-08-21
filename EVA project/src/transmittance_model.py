import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
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

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "transmittance_model"
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

TARGET = "Transmittance"


# ============================================================
# Load dataset
# ============================================================

def load_data():

    print("STEP 9 - TRANSMITTANCE REGRESSION")
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
            "Missing values found in Transmittance target."
        )

    return X, y


# ============================================================
# Evaluate model on hold-out test set
# ============================================================

def evaluate_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test
):

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

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

    return model, mae, rmse, r2


# ============================================================
# Cross-validation
# ============================================================

def cross_validation(model, X, y):

    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2"
    }

    cv_results = cross_validate(
        model,
        X,
        y,
        cv=kfold,
        scoring=scoring,
        n_jobs=-1
    )

    mae_scores = -cv_results["test_mae"]
    rmse_scores = -cv_results["test_rmse"]
    r2_scores = cv_results["test_r2"]

    return (
        mae_scores.mean(),
        mae_scores.std(),
        rmse_scores.mean(),
        rmse_scores.std(),
        r2_scores.mean(),
        r2_scores.std()
    )


# ============================================================
# Main
# ============================================================

def main():

    df = load_data()

    X, y = prepare_data(df)

    # --------------------------------------------------------
    # Train / test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Test samples: {len(X_test)}"
    )

    # --------------------------------------------------------
    # Linear Regression
    # --------------------------------------------------------

    print()
    print("Training LinearRegression...")

    linear_model = LinearRegression()

    (
        linear_model,
        linear_mae,
        linear_rmse,
        linear_r2
    ) = evaluate_model(
        linear_model,
        X_train,
        X_test,
        y_train,
        y_test
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    print()
    print("Training RandomForest...")

    rf_model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1
    )

    (
        rf_model,
        rf_mae,
        rf_rmse,
        rf_r2
    ) = evaluate_model(
        rf_model,
        X_train,
        X_test,
        y_train,
        y_test
    )

    # --------------------------------------------------------
    # Hold-out results
    # --------------------------------------------------------

    print()
    print("Hold-out test results:")

    print(
        f"LinearRegression: "
        f"MAE={linear_mae:.3f}, "
        f"RMSE={linear_rmse:.3f}, "
        f"R2={linear_r2:.3f}"
    )

    print(
        f"RandomForest: "
        f"MAE={rf_mae:.3f}, "
        f"RMSE={rf_rmse:.3f}, "
        f"R2={rf_r2:.3f}"
    )

    # --------------------------------------------------------
    # Cross-validation
    # --------------------------------------------------------

    print()
    print("5-fold cross-validation:")

    (
        linear_cv_mae,
        linear_cv_mae_std,
        linear_cv_rmse,
        linear_cv_rmse_std,
        linear_cv_r2,
        linear_cv_r2_std
    ) = cross_validation(
        LinearRegression(),
        X,
        y
    )

    (
        rf_cv_mae,
        rf_cv_mae_std,
        rf_cv_rmse,
        rf_cv_rmse_std,
        rf_cv_r2,
        rf_cv_r2_std
    ) = cross_validation(
        RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1
        ),
        X,
        y
    )

    print(
        f"LinearRegression: "
        f"MAE={linear_cv_mae:.3f} ± {linear_cv_mae_std:.3f}, "
        f"RMSE={linear_cv_rmse:.3f} ± {linear_cv_rmse_std:.3f}, "
        f"R2={linear_cv_r2:.3f} ± {linear_cv_r2_std:.3f}"
    )

    print(
        f"RandomForest: "
        f"MAE={rf_cv_mae:.3f} ± {rf_cv_mae_std:.3f}, "
        f"RMSE={rf_cv_rmse:.3f} ± {rf_cv_rmse_std:.3f}, "
        f"R2={rf_cv_r2:.3f} ± {rf_cv_r2_std:.3f}"
    )

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

    comparison = pd.DataFrame({

        "Model": [
            "LinearRegression",
            "RandomForest"
        ],

        "Holdout_MAE": [
            linear_mae,
            rf_mae
        ],

        "Holdout_RMSE": [
            linear_rmse,
            rf_rmse
        ],

        "Holdout_R2": [
            linear_r2,
            rf_r2
        ],

        "CV_MAE": [
            linear_cv_mae,
            rf_cv_mae
        ],

        "CV_MAE_STD": [
            linear_cv_mae_std,
            rf_cv_mae_std
        ],

        "CV_RMSE": [
            linear_cv_rmse,
            rf_cv_rmse
        ],

        "CV_RMSE_STD": [
            linear_cv_rmse_std,
            rf_cv_rmse_std
        ],

        "CV_R2": [
            linear_cv_r2,
            rf_cv_r2
        ],

        "CV_R2_STD": [
            linear_cv_r2_std,
            rf_cv_r2_std
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

    # --------------------------------------------------------
    # Select model using CV RMSE
    # --------------------------------------------------------

    if linear_cv_rmse <= rf_cv_rmse:

        best_model_name = "LinearRegression"

    else:

        best_model_name = "RandomForest"

    print()
    print(
        f"Selected model based on CV RMSE: "
        f"{best_model_name}"
    )

    # --------------------------------------------------------
    # Retrain selected model on all experimental data
    # --------------------------------------------------------

    if best_model_name == "LinearRegression":

        final_model = LinearRegression()

    else:

        final_model = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1
        )

    final_model.fit(
        X,
        y
    )

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        "transmittance_model.pkl"
    )

    joblib.dump(
        final_model,
        model_path
    )

    print()
    print(
        "Final Transmittance model saved to:"
    )

    print(model_path)

    print()
    print(
        "Transmittance regression completed."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()