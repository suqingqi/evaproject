import argparse
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


# ============================================================
# STEP 19 - EXPERIMENTAL FEEDBACK UPDATE
# ============================================================
# This script creates a feedback template from the rounded batch
# recommended in STEP 18, validates REAL experimental measurements,
# and optionally appends them to the experimental dataset.
#
# Important:
# - Model predictions are kept only as reference columns.
# - Only measured LOI / UL_94 / Transmittance / Haze are accepted
#   as new training labels.
# - Default mode is DRY RUN. Use --commit only after validation.
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
RESULT_DIR = PROJECT_ROOT / "results" / "next_experiment_selection"
BACKUP_DIR = DATA_DIR / "backup"

RAW_DATA_PATH = DATA_DIR / "EVA data 100.xlsx"
CLEAN_DATA_PATH = DATA_DIR / "polymer_dataset_clean.csv"
NEXT_EXPERIMENT_PATH = RESULT_DIR / "next_experiments.csv"
TEMPLATE_PATH = DATA_DIR / "new_experimental_results_template.csv"
FEEDBACK_PATH = DATA_DIR / "new_experimental_results.csv"

FEATURES = [
    "EVA_content",
    "Polymer_A",
    "Polymer_B",
    "FR_A",
    "FR_B",
    "FR_C",
    "FR_D",
    "Additive_1",
    "Additive_2",
]

MEASURED_COLUMNS = [
    "LOI",
    "UL_94",
    "Transmittance",
    "Haze",
]

UL94_CLASSES = ["NR", "V-2", "V-1", "V-0"]


def practical_rounding(df):
    """Use the same practical resolution as the historical dataset."""
    df = df.copy()

    for column in [
        "EVA_content",
        "FR_A",
        "FR_B",
        "FR_C",
        "FR_D",
        "Additive_1",
        "Additive_2",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").round(1)

    for column in ["Polymer_A", "Polymer_B"]:
        if column in df.columns:
            numeric = pd.to_numeric(df[column], errors="coerce")
            df[column] = numeric.round().astype("Int64")

    for column in [
        "Model_Predicted_LOI",
        "Model_Predicted_Transmittance",
        "LOI",
        "Transmittance",
        "Haze",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").round(1)

    return df


def create_template():
    if not NEXT_EXPERIMENT_PATH.exists():
        raise FileNotFoundError(
            "STEP 18 output not found. Run next_experiment_recommendation .py first:\n"
            f"{NEXT_EXPERIMENT_PATH}"
        )

    selected = pd.read_csv(NEXT_EXPERIMENT_PATH)

    required = FEATURES + [
        "Predicted_LOI",
        "Predicted_UL94",
        "Predicted_Transmittance",
    ]

    missing = [column for column in required if column not in selected.columns]
    if missing:
        raise ValueError(f"STEP 18 output is missing required columns: {missing}")

    template = selected[[
        "Experiment_Rank",
        *FEATURES,
        "Predicted_LOI",
        "Predicted_UL94",
        "Predicted_Transmittance",
    ]].copy()

    template = template.rename(columns={
        "Predicted_LOI": "Model_Predicted_LOI",
        "Predicted_UL94": "Model_Predicted_UL94",
        "Predicted_Transmittance": "Model_Predicted_Transmittance",
    })

    # Real experimental measurements are intentionally left blank.
    template["LOI"] = pd.NA
    template["UL_94"] = pd.NA
    template["Transmittance"] = pd.NA
    template["Haze"] = pd.NA
    template["Notes"] = ""

    template = practical_rounding(template)
    template.to_csv(TEMPLATE_PATH, index=False)

    print(f"Feedback template created with {len(template)} experiment(s):")
    print(TEMPLATE_PATH)
    print()
    print("The formulation values use the same practical precision as STEP 18.")
    print("Do NOT copy model predictions into the measured-result columns.")


def validate_feedback(feedback):
    required = FEATURES + MEASURED_COLUMNS
    missing = [column for column in required if column not in feedback.columns]
    if missing:
        raise ValueError(f"Feedback file is missing required columns: {missing}")

    if feedback.empty:
        raise ValueError("Feedback file contains no rows.")

    incomplete = feedback[MEASURED_COLUMNS].isna().any(axis=1)
    if incomplete.any():
        rows = (feedback.index[incomplete] + 2).tolist()
        raise ValueError(
            "Measured experimental results are incomplete. "
            f"Check CSV row(s): {rows}"
        )

    numeric_columns = FEATURES + ["LOI", "Transmittance", "Haze"]
    for column in numeric_columns:
        feedback[column] = pd.to_numeric(feedback[column], errors="coerce")
        if feedback[column].isna().any():
            raise ValueError(f"Non-numeric value found in '{column}'.")

    invalid_ul94 = sorted(set(feedback["UL_94"].astype(str)) - set(UL94_CLASSES))
    if invalid_ul94:
        raise ValueError(
            f"Invalid UL-94 label(s): {invalid_ul94}. "
            f"Allowed: {UL94_CLASSES}"
        )

    range_rules = {
        "EVA_content": (0, 100),
        "Polymer_A": (0, 20),
        "Polymer_B": (0, 20),
        "FR_A": (0, 20),
        "FR_B": (0, 20),
        "FR_C": (0, 20),
        "FR_D": (0, 20),
        "Additive_1": (0, 10),
        "Additive_2": (0, 10),
        "LOI": (0, 100),
        "Transmittance": (0, 100),
        "Haze": (0, 100),
    }

    for column, (lower, upper) in range_rules.items():
        invalid = (feedback[column] < lower) | (feedback[column] > upper)
        if invalid.any():
            raise ValueError(
                f"'{column}' contains value(s) outside {lower}-{upper}."
            )

    formulation_total = feedback[FEATURES].sum(axis=1)
    invalid_total = ~formulation_total.between(99.5, 100.5)
    if invalid_total.any():
        details = [
            f"row {idx + 2}: {formulation_total.loc[idx]:.1f} wt%"
            for idx in feedback.index[invalid_total]
        ]
        raise ValueError(
            "Some formulations are outside 100 +/- 0.5 wt%:\n" + "\n".join(details)
        )

    return practical_rounding(feedback)


def next_sample_ids(existing_ids, n):
    numbers = []
    for value in existing_ids.astype(str):
        if value.startswith("F") and value[1:].isdigit():
            numbers.append(int(value[1:]))

    start = max(numbers, default=0) + 1
    return [f"F{number:03d}" for number in range(start, start + n)]


def prepare_new_rows(feedback, raw_df):
    new_rows = feedback[FEATURES + MEASURED_COLUMNS].copy()
    new_rows.insert(0, "sample_ID", next_sample_ids(raw_df["sample_ID"], len(new_rows)))
    return new_rows


def commit_feedback(new_rows, raw_df):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    raw_backup = BACKUP_DIR / f"EVA_data_before_feedback_{timestamp}.xlsx"
    shutil.copy2(RAW_DATA_PATH, raw_backup)

    updated_raw = pd.concat([raw_df, new_rows], ignore_index=True)
    updated_raw.to_excel(RAW_DATA_PATH, index=False)

    print()
    print("Feedback committed to raw experimental dataset.")
    print(f"New experimental rows: {len(new_rows)}")
    print(f"Updated total samples: {len(updated_raw)}")
    print(f"Backup created: {raw_backup}")
    print()
    print("Next: run data_cleaning.py before retraining the models.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--create-template",
        action="store_true",
        help="Create a blank measured-results template from STEP 18 recommendations.",
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Append validated REAL experimental results to the raw dataset.",
    )
    args = parser.parse_args()

    print("STEP 19 - EXPERIMENTAL FEEDBACK UPDATE")
    print()

    if args.create_template:
        create_template()
        print()

    if not FEEDBACK_PATH.exists():
        print("No completed experimental feedback file found yet.")
        print()
        print("Next action:")
        print("1. Copy new_experimental_results_template.csv")
        print("2. Rename the copy to new_experimental_results.csv")
        print("3. Fill only REAL measured LOI, UL_94, Transmittance and Haze values")
        print("4. Run this script without --commit for validation")
        print("5. Use --commit only after the dry run passes")
        print()
        print("No dataset was modified.")
        return

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw experimental dataset not found:\n{RAW_DATA_PATH}")

    feedback = pd.read_csv(FEEDBACK_PATH)
    feedback = validate_feedback(feedback)
    raw_df = pd.read_excel(RAW_DATA_PATH)
    new_rows = prepare_new_rows(feedback, raw_df)

    print(f"Validated real experimental feedback rows: {len(new_rows)}")
    print()
    print("Rows ready to append:")
    print(new_rows.to_string(index=False))

    if not args.commit:
        print()
        print("DRY RUN ONLY - no dataset was modified.")
        print("If these are real measured values and the table is correct, run:")
        print("python src/update_experimental_data.py --commit")
        return

    commit_feedback(new_rows, raw_df)


if __name__ == "__main__":
    main()
