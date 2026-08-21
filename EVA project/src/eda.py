import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Add src directory to Python module search path
SRC_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from config import CLEAN_DATA_PATH, RESULTS_DIR


# EDA output directory
EDA_DIR = RESULTS_DIR / "eda"


# Modeling features
FEATURE_COLUMNS = [
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


# Regression targets
REGRESSION_TARGETS = [
    "LOI",
    "Transmittance",
    "Haze"
]


def load_cleaned_data():
    """Load cleaned dataset."""

    print("STEP 3 - EXPLORATORY DATA ANALYSIS")
    print("Loading cleaned dataset...")

    if not CLEAN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found:\n{CLEAN_DATA_PATH}"
        )

    df = pd.read_csv(CLEAN_DATA_PATH)

    print(f"Samples: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    return df


def prepare_output_directory():
    """Create EDA result directory."""

    EDA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


def generate_descriptive_statistics(df):
    """Generate descriptive statistics."""

    print("\nGenerating descriptive statistics...")

    statistics = df.describe(
        include="all"
    ).transpose()

    output_path = EDA_DIR / "descriptive_statistics.csv"

    statistics.to_csv(output_path)

    print(
        f"Saved descriptive statistics to:\n{output_path}"
    )


def generate_feature_distributions(df):
    """Generate feature distribution plots."""

    print("Generating feature distributions...")

    for feature in FEATURE_COLUMNS:

        plt.figure(figsize=(7, 5))

        sns.histplot(
            df[feature],
            kde=True
        )

        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")

        plt.tight_layout()

        output_path = (
            EDA_DIR
            / f"distribution_{feature}.png"
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


def generate_target_distributions(df):
    """Generate regression target distributions."""

    print("Generating target distributions...")

    for target in REGRESSION_TARGETS:

        plt.figure(figsize=(7, 5))

        sns.histplot(
            df[target],
            kde=True
        )

        plt.title(f"Distribution of {target}")
        plt.xlabel(target)
        plt.ylabel("Frequency")

        plt.tight_layout()

        output_path = (
            EDA_DIR
            / f"distribution_{target}.png"
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


def generate_ul94_distribution(df):
    """Generate UL-94 class distribution."""

    print("Generating UL-94 distribution...")

    plt.figure(figsize=(7, 5))

    order = [
        "NR",
        "V-2",
        "V-1",
        "V-0"
    ]

    sns.countplot(
        data=df,
        x="UL_94",
        order=order
    )

    plt.title("UL-94 Classification Distribution")
    plt.xlabel("UL-94 Class")
    plt.ylabel("Sample Count")

    plt.tight_layout()

    output_path = (
        EDA_DIR / "ul94_distribution.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def generate_correlation_matrix(df):
    """Generate numerical feature correlation matrix."""

    print("Generating correlation matrix...")

    correlation_columns = (
        FEATURE_COLUMNS
        + REGRESSION_TARGETS
    )

    correlation_matrix = (
        df[correlation_columns]
        .corr()
    )

    plt.figure(
        figsize=(12, 10)
    )

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )

    plt.title(
        "Feature and Target Correlation Matrix"
    )

    plt.tight_layout()

    output_path = (
        EDA_DIR / "correlation_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # Also save numerical correlation matrix
    correlation_matrix.to_csv(
        EDA_DIR / "correlation_matrix.csv"
    )


def generate_target_relationships(df):
    """Generate relationships among target variables."""

    print("Generating LOI vs Transmittance plot...")

    plt.figure(figsize=(7, 5))

    sns.scatterplot(
        data=df,
        x="LOI",
        y="Transmittance",
        hue="UL_94"
    )

    plt.title(
        "LOI vs Transmittance"
    )

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "loi_vs_transmittance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Generating LOI vs Haze plot...")

    plt.figure(figsize=(7, 5))

    sns.scatterplot(
        data=df,
        x="LOI",
        y="Haze",
        hue="UL_94"
    )

    plt.title(
        "LOI vs Haze"
    )

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "loi_vs_haze.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Generating Transmittance vs Haze plot...")

    plt.figure(figsize=(7, 5))

    sns.scatterplot(
        data=df,
        x="Transmittance",
        y="Haze",
        hue="UL_94"
    )

    plt.title(
        "Transmittance vs Haze"
    )

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "transmittance_vs_haze.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def generate_feature_vs_target_plots(df):
    """Generate feature-target relationship plots."""

    print("Generating feature vs LOI plots...")

    for feature in FEATURE_COLUMNS:

        plt.figure(figsize=(7, 5))

        sns.scatterplot(
            data=df,
            x=feature,
            y="LOI"
        )

        plt.title(
            f"{feature} vs LOI"
        )

        plt.tight_layout()

        output_path = (
            EDA_DIR
            / f"{feature}_vs_loi.png"
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    print("Generating feature vs Transmittance plots...")

    for feature in FEATURE_COLUMNS:

        plt.figure(figsize=(7, 5))

        sns.scatterplot(
            data=df,
            x=feature,
            y="Transmittance"
        )

        plt.title(
            f"{feature} vs Transmittance"
        )

        plt.tight_layout()

        output_path = (
            EDA_DIR
            / f"{feature}_vs_transmittance.png"
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


def main():

    df = load_cleaned_data()

    prepare_output_directory()

    generate_descriptive_statistics(df)

    generate_feature_distributions(df)

    generate_target_distributions(df)

    generate_ul94_distribution(df)

    generate_correlation_matrix(df)

    generate_target_relationships(df)

    generate_feature_vs_target_plots(df)

    print("\nEDA completed.")

    print(
        f"Results saved to:\n{EDA_DIR}"
    )


if __name__ == "__main__":
    main()