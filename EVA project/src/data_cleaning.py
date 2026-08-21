import sys
from pathlib import Path

import pandas as pd


# Add src directory to Python module search path
SRC_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from config import RAW_DATA_PATH, CLEAN_DATA_PATH
# Required columns
REQUIRED_COLUMNS = [
    "sample_ID",
    "EVA_content",
    "Polymer_A",
    "Polymer_B",
    "FR_A",
    "FR_B",
    "FR_C",
    "FR_D",
    "Additive_1",
    "Additive_2",
    "LOI",
    "UL_94",
    "Transmittance",
    "Haze"
]


# Numerical columns
NUMERICAL_COLUMNS = [
    "EVA_content",
    "Polymer_A",
    "Polymer_B",
    "FR_A",
    "FR_B",
    "FR_C",
    "FR_D",
    "Additive_1",
    "Additive_2",
    "LOI",
    "Transmittance",
    "Haze"
]


# Expected UL-94 classes
UL94_CLASSES = ["NR", "V-2", "V-1", "V-0"]

# Numerical encoding used by the classification model
UL94_MAPPING = {
    "NR": 0,
    "V-2": 1,
    "V-1": 2,
    "V-0": 3
}


def load_raw_data():
    """Load the original Excel dataset."""

    print("STEP 2 - DATA CLEANING")
    print("Loading raw dataset...")

    if not Path(RAW_DATA_PATH).exists():
        raise FileNotFoundError(
            f"Raw dataset not found:\n{RAW_DATA_PATH}"
        )

    df = pd.read_excel(RAW_DATA_PATH)

    print(f"Loaded {len(df)} samples and {len(df.columns)} columns.")

    return df


def validate_columns(df):
    """Check whether all required columns are present."""

    print("\n1. Column validation")

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("All required columns are present.")


def check_missing_values(df):
    """Check missing values."""

    print("\n2. Missing value check")

    missing = df[REQUIRED_COLUMNS].isnull().sum()

    if missing.sum() > 0:
        print("Missing values found:")
        print(missing[missing > 0])

        raise ValueError(
            "Dataset contains missing values."
        )

    print("No missing values found.")


def check_duplicates(df):
    """Check duplicate rows."""

    print("\n3. Duplicate row check")

    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        print(f"Duplicated rows: {duplicate_count}")

        raise ValueError(
            "Dataset contains duplicated rows."
        )

    print("No duplicated rows found.")


def check_sample_ids(df):
    """Check sample ID uniqueness."""

    print("\n4. Sample ID check")

    duplicate_ids = df["sample_ID"].duplicated().sum()

    if duplicate_ids > 0:
        raise ValueError(
            f"Duplicated sample IDs found: {duplicate_ids}"
        )

    print("All sample IDs are unique.")


def validate_numerical_columns(df):
    """Validate numerical columns."""

    print("\n5. Numerical data validation")

    for column in NUMERICAL_COLUMNS:

        if not pd.api.types.is_numeric_dtype(df[column]):
            raise TypeError(
                f"Column '{column}' must be numerical."
            )

    print(
        f"Validated {len(NUMERICAL_COLUMNS)} numerical columns."
    )


def validate_ranges(df):
    """Validate physically reasonable value ranges."""

    print("\n6. Range validation")

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

        "Haze": (0, 100)
    }

    for column, (lower, upper) in range_rules.items():

        invalid = (
            (df[column] < lower)
            | (df[column] > upper)
        )

        if invalid.any():

            raise ValueError(
                f"Invalid values found in '{column}'."
            )

    print("All numerical range checks passed.")


def validate_ul94(df):
    """Validate UL-94 classification labels."""

    print("\n7. UL-94 validation")

    invalid_labels = set(df["UL_94"]) - set(UL94_CLASSES)

    if invalid_labels:

        raise ValueError(
            f"Unexpected UL-94 labels: {invalid_labels}"
        )

    print("UL-94 labels found:")

    print(df["UL_94"].value_counts())

    print("\nUL-94 encoding:")

    for label, score in UL94_MAPPING.items():

        print(f"{label} -> {score}")


def create_ul94_score(df):
    """Create numerical UL-94 score."""

    df = df.copy()

    df["UL94_score"] = (
        df["UL_94"]
        .map(UL94_MAPPING)
        .astype(int)
    )

    return df


def validate_formulation_total(df):
    """
    Check formulation composition.

    The dataset is expected to contain 100 wt% formulation.
    """

    print("\n8. Formulation total validation")

    formulation_columns = [
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

    formulation_total = df[formulation_columns].sum(axis=1)

    min_total = formulation_total.min()
    max_total = formulation_total.max()

    print(f"Minimum formulation total: {min_total:.4f}")
    print(f"Maximum formulation total: {max_total:.4f}")

    valid_count = (
        (formulation_total >= 99.5)
        & (formulation_total <= 100.5)
    ).sum()

    invalid_count = len(df) - valid_count

    print(
        f"Formulations within 100 ± 0.5 wt%: {valid_count}"
    )

    print(
        f"Formulations outside 100 ± 0.5 wt%: {invalid_count}"
    )

    if invalid_count > 0:

        raise ValueError(
            "Some formulations do not sum to approximately 100 wt%."
        )


def save_cleaned_data(df):
    """Save cleaned dataset."""

    print("\n9. Save cleaned data")

    output_path = Path(CLEAN_DATA_PATH)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Cleaned dataset saved to:\n{output_path}"
    )


def main():

    df = load_raw_data()

    validate_columns(df)

    check_missing_values(df)

    check_duplicates(df)

    check_sample_ids(df)

    validate_numerical_columns(df)

    validate_ranges(df)

    validate_ul94(df)

    validate_formulation_total(df)

    df = create_ul94_score(df)

    print("\n10. Final dataset check")

    print(f"Samples: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nFinal columns:")

    for index, column in enumerate(df.columns, start=1):

        print(f"{index:02d}. {column}")

    save_cleaned_data(df)

    print("\nSTEP 2 completed.")


if __name__ == "__main__":
    main()