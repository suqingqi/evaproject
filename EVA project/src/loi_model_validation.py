import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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

    print("STEP 5 - LOI MODEL VALIDATION")
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


def evaluate_holdout(model, X_train, X_test, y_train, y_test):

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

    results = cross_validate(
        model,
        X,
        y,
        cv=kfold,
        scoring=scoring,
        return_train_score=False
    )

    mae_scores = -results["test_mae"]
    rmse_scores = -results["test_rmse"]
    r2_scores = results["test_r2"]

    return {
        "mae_mean": mae_scores.mean(),
        "mae_std": mae_scores.std(),

        "rmse_mean": rmse_scores.mean(),
        "rmse_std": rmse_scores.std(),

        "r2_mean": r2_scores.mean(),
        "r2_std": r2_scores.std()
    }


def main():

    df = load_data()

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1
        )
    }

    holdout_results = {}
    cv_results = {}

    print()
    print("Hold-out test results:")

    for name, model in models.items():

        trained_model, mae, rmse, r2 = evaluate_holdout(
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        holdout_results[name] = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }

        print(
            f"{name}: "
            f"MAE={mae:.3f}, "
            f"RMSE={rmse:.3f}, "
            f"R2={r2:.3f}"
        )

    print()
    print("5-fold cross-validation:")

    for name, model in models.items():

        result = cross_validation(
            model,
            X,
            y
        )

        cv_results[name] = result

        print(
            f"{name}: "
            f"MAE={result['mae_mean']:.3f} ± "
            f"{result['mae_std']:.3f}, "
            f"RMSE={result['rmse_mean']:.3f} ± "
            f"{result['rmse_std']:.3f}, "
            f"R2={result['r2_mean']:.3f} ± "
            f"{result['r2_std']:.3f}"
        )

    best_model_name = min(
        cv_results,
        key=lambda name: cv_results[name]["rmse_mean"]
    )

    print()
    print(
        f"Selected model based on CV RMSE: "
        f"{best_model_name}"
    )

    final_model = models[best_model_name]

    final_model.fit(
        X,
        y
    )

    model_path = os.path.join(
        MODEL_DIR,
        "loi_model.pkl"
    )

    joblib.dump(
        final_model,
        model_path
    )

    print("Final LOI model saved to:")
    print(model_path)

    validation_rows = []

    for name in models:

        validation_rows.append({
            "Model": name,

            "Holdout_MAE":
                holdout_results[name]["MAE"],

            "Holdout_RMSE":
                holdout_results[name]["RMSE"],

            "Holdout_R2":
                holdout_results[name]["R2"],

            "CV_MAE_Mean":
                cv_results[name]["mae_mean"],

            "CV_MAE_STD":
                cv_results[name]["mae_std"],

            "CV_RMSE_Mean":
                cv_results[name]["rmse_mean"],

            "CV_RMSE_STD":
                cv_results[name]["rmse_std"],

            "CV_R2_Mean":
                cv_results[name]["r2_mean"],

            "CV_R2_STD":
                cv_results[name]["r2_std"]
        })

    validation_df = pd.DataFrame(
        validation_rows
    )

    validation_df["Selected"] = (
        validation_df["Model"]
        == best_model_name
    )

    output_path = os.path.join(
        RESULT_DIR,
        "model_validation.csv"
    )

    validation_df.to_csv(
        output_path,
        index=False
    )

    print()
    print("Validation results saved to:")
    print(output_path)

    print()
    print("LOI model validation completed.")


if __name__ == "__main__":
    main()