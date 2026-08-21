import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATH CONFIGURATION
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "polymer_dataset_clean.csv"
)

BO_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "bayesian_optimization"
)

OUTPUT_DIR = os.path.join(
    BO_DIR
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEP 18 - FINAL OPTIMIZATION VISUALIZATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load experimental dataset
    # --------------------------------------------------------

    print("\nLoading experimental dataset...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Experimental dataset not found:\n{DATA_PATH}"
        )

    experimental = pd.read_csv(DATA_PATH)

    print(
        f"Experimental samples: {len(experimental)}"
    )

    # --------------------------------------------------------
    # Load Safe BO Pareto results
    # --------------------------------------------------------

    pareto_path = os.path.join(
        BO_DIR,
        "safe_bo_pareto.csv"
    )

    if not os.path.exists(pareto_path):
        raise FileNotFoundError(
            f"Safe BO Pareto results not found:\n{pareto_path}"
        )

    print("Loading Safe BO Pareto results...")

    pareto = pd.read_csv(pareto_path)

    print(
        f"Safe BO Pareto candidates: {len(pareto)}"
    )

    # --------------------------------------------------------
    # Load final recommendations
    # --------------------------------------------------------

    recommendation_path = os.path.join(
        BO_DIR,
        "final_recommendations.csv"
    )

    if not os.path.exists(recommendation_path):
        raise FileNotFoundError(
            f"Final recommendations not found:\n"
            f"{recommendation_path}"
        )

    print("Loading final recommendations...")

    recommendations = pd.read_csv(
        recommendation_path
    )

    print(
        f"Final recommendations: "
        f"{len(recommendations)}"
    )

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    required_columns = [
        "Predicted_LOI",
        "Predicted_Transmittance",
        "Predicted_Fire_Score"
    ]

    for column in required_columns:

        if column not in pareto.columns:
            raise ValueError(
                f"Missing required column in Pareto data: "
                f"{column}"
            )

        if column not in recommendations.columns:
            raise ValueError(
                f"Missing required column in recommendation data: "
                f"{column}"
            )

    # --------------------------------------------------------
    # Experimental reference ranges
    # --------------------------------------------------------

    exp_loi_min = experimental["LOI"].min()
    exp_loi_max = experimental["LOI"].max()

    exp_trans_min = experimental[
        "Transmittance"
    ].min()

    exp_trans_max = experimental[
        "Transmittance"
    ].max()

    print()
    print("Experimental reference ranges:")

    print(
        f"LOI: "
        f"{exp_loi_min:.2f} - "
        f"{exp_loi_max:.2f}"
    )

    print(
        f"Transmittance: "
        f"{exp_trans_min:.2f} - "
        f"{exp_trans_max:.2f}"
    )

    # --------------------------------------------------------
    # Generate final optimization plot
    # --------------------------------------------------------

    print("\nGenerating final optimization plot...")

    plt.figure(
        figsize=(10, 7)
    )

    # Experimental data
    plt.scatter(
        experimental["LOI"],
        experimental["Transmittance"],
        alpha=0.45,
        s=45,
        label="Experimental samples"
    )

    # Safe BO Pareto candidates
    plt.scatter(
        pareto["Predicted_LOI"],
        pareto["Predicted_Transmittance"],
        s=65,
        marker="^",
        label="Safe BO Pareto candidates"
    )

    # Final recommendations
    plt.scatter(
        recommendations["Predicted_LOI"],
        recommendations["Predicted_Transmittance"],
        s=140,
        marker="*",
        label="Final recommendations"
    )

    # Annotate final recommendations
    for _, row in recommendations.iterrows():

        rank = int(row["Final_Rank"])

        plt.annotate(
            f"Rank {rank}",
            (
                row["Predicted_LOI"],
                row["Predicted_Transmittance"]
            ),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9
        )

    # Experimental range box
    plt.axvline(
        exp_loi_min,
        linestyle="--",
        linewidth=1
    )

    plt.axvline(
        exp_loi_max,
        linestyle="--",
        linewidth=1
    )

    plt.axhline(
        exp_trans_min,
        linestyle="--",
        linewidth=1
    )

    plt.axhline(
        exp_trans_max,
        linestyle="--",
        linewidth=1
    )

    plt.xlabel(
        "LOI"
    )

    plt.ylabel(
        "Transmittance"
    )

    plt.title(
        "Final Safe Bayesian Optimization Result"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        "final_optimization_result.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Final optimization plot saved to:"
    )

    print(output_path)

    # --------------------------------------------------------
    # Generate final visualization data
    # --------------------------------------------------------

    visualization_data = recommendations[
        [
            "Final_Rank",
            "EVA_content",
            "Polymer_A",
            "Polymer_B",
            "Predicted_LOI",
            "Predicted_Fire_Score",
            "Predicted_UL94",
            "Predicted_Transmittance",
            "Predicted_Utility",
            "Nearest_Experimental_Distance",
            "Safety_Factor",
            "Multi_Objective_Score",
            "Safety_Adjusted_Score"
        ]
    ].copy()

    data_path = os.path.join(
        OUTPUT_DIR,
        "final_visualization_data.csv"
    )

    visualization_data.to_csv(
        data_path,
        index=False
    )

    print(
        "Final visualization data saved to:"
    )

    print(data_path)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL VISUALIZATION SUMMARY")
    print("=" * 70)

    print(
        f"Experimental samples: "
        f"{len(experimental)}"
    )

    print(
        f"Safe BO Pareto candidates: "
        f"{len(pareto)}"
    )

    print(
        f"Final recommendations: "
        f"{len(recommendations)}"
    )

    print("\nFinal recommendation points:")

    print(
        recommendations[
            [
                "Final_Rank",
                "Predicted_LOI",
                "Predicted_UL94",
                "Predicted_Transmittance",
                "Safety_Adjusted_Score"
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("STEP 18 completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()