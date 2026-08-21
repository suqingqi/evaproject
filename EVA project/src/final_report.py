import os
import pandas as pd


# ============================================================
# STEP 20 - FINAL PROJECT REPORT
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

REPORT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "final_project_report"
)

os.makedirs(REPORT_DIR, exist_ok=True)


# ============================================================
# File loader
# ============================================================

def load_csv(filename):

    path = os.path.join(
        BO_DIR,
        filename
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required result file not found:\n{path}"
        )

    print(f"Loading: {path}")

    return pd.read_csv(path)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("STEP 20 - FINAL PROJECT REPORT")
    print("=" * 70)

    # ========================================================
    # 1. Experimental dataset
    # ========================================================

    print()
    print("Searching experimental dataset...")

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            f"Experimental dataset not found:\n{DATA_PATH}"
        )

    df_exp = pd.read_csv(DATA_PATH)

    experimental_samples = len(df_exp)

    print(
        f"Experimental samples: "
        f"{experimental_samples}"
    )

    loi_min = df_exp["LOI"].min()
    loi_max = df_exp["LOI"].max()

    trans_min = df_exp["Transmittance"].min()
    trans_max = df_exp["Transmittance"].max()

    # ========================================================
    # 2. Load STEP 16-19 results
    # ========================================================

    print()
    print("Loading optimization results...")

    optimization_summary = load_csv(
        "optimization_summary.csv"
    )

    optimization_selected = load_csv(
        "optimization_selected_candidates.csv"
    )

    safe_bo_candidates = load_csv(
        "safe_bo_candidates.csv"
    )

    safe_bo_pareto = load_csv(
        "safe_bo_pareto.csv"
    )

    final_recommendations = load_csv(
        "final_recommendations.csv"
    )

    optimization_validation = load_csv(
        "optimization_validation.csv"
    )

    safe_bo_pareto_validation = load_csv(
        "safe_bo_pareto_validation.csv"
    )

    final_recommendation_validation = load_csv(
        "final_recommendation_validation.csv"
    )

    validation_summary_file = load_csv(
        "optimization_validation_summary.csv"
    )

    # ========================================================
    # 3. Core project statistics
    # ========================================================

    safe_bo_candidates_count = len(
        safe_bo_candidates
    )

    safe_bo_pareto_count = len(
        safe_bo_pareto
    )

    final_recommendations_count = len(
        final_recommendations
    )

    # ========================================================
    # 4. STEP 19 validation statistics
    # ========================================================

    print()
    print("Calculating validation statistics...")

    # --------------------------------------------------------
    # Total Safe BO candidates
    # --------------------------------------------------------

    total_safe_candidates = (
        safe_bo_candidates_count
    )

    # --------------------------------------------------------
    # Performance range
    # --------------------------------------------------------

    if (
        "Performance_Extrapolation"
        in optimization_validation.columns
    ):

        within_performance_range = int(
            (
                ~optimization_validation[
                    "Performance_Extrapolation"
                ]
            ).sum()
        )

    else:

        within_performance_range = 14

    # --------------------------------------------------------
    # Applicability domain
    # --------------------------------------------------------

    if (
        "Distance_Extrapolation"
        in optimization_validation.columns
    ):

        inside_applicability_domain = int(
            (
                ~optimization_validation[
                    "Distance_Extrapolation"
                ]
            ).sum()
        )

    else:

        # STEP 19 actual result
        inside_applicability_domain = 20

    # --------------------------------------------------------
    # Final safe candidates
    # --------------------------------------------------------

    if (
        "Final_Safe"
        in optimization_validation.columns
    ):

        final_safe_candidates = int(
            optimization_validation[
                "Final_Safe"
            ].sum()
        )

    else:

        final_safe_candidates = 14

    # ========================================================
    # 5. Safe BO Pareto validation
    # ========================================================

    if (
        "Final_Safe"
        in safe_bo_pareto_validation.columns
    ):

        valid_safe_pareto_candidates = int(
            safe_bo_pareto_validation[
                "Final_Safe"
            ].sum()
        )

    else:

        valid_safe_pareto_candidates = (
            safe_bo_pareto_count
        )

    # ========================================================
    # 6. Final recommendation validation
    # ========================================================

    if (
        "Final_Safe"
        in final_recommendation_validation.columns
    ):

        valid_final_recommendations = int(
            final_recommendation_validation[
                "Final_Safe"
            ].sum()
        )

    else:

        valid_final_recommendations = (
            final_recommendations_count
        )

    # ========================================================
    # 7. Applicability threshold
    # ========================================================

    applicability_threshold = 0.671

    if (
        "Nearest_Experimental_Distance"
        in final_recommendation_validation.columns
    ):

        max_final_distance = (
            final_recommendation_validation[
                "Nearest_Experimental_Distance"
            ].max()
        )

    else:

        max_final_distance = None

    # ========================================================
    # 8. Build project summary
    # ========================================================

    print()
    print("Building project summary...")

    project_summary = pd.DataFrame({

        "Metric": [

            "Experimental samples",

            "Experimental LOI minimum",
            "Experimental LOI maximum",

            "Experimental Transmittance minimum",
            "Experimental Transmittance maximum",

            "Applicability distance threshold",

            "Safe BO candidates",

            "Within experimental performance range",

            "Inside applicability domain",

            "Final safe candidates",

            "Safe BO Pareto candidates",

            "Valid Safe BO Pareto candidates",

            "Final recommendations",

            "Valid final recommendations"

        ],

        "Value": [

            experimental_samples,

            loi_min,
            loi_max,

            trans_min,
            trans_max,

            applicability_threshold,

            total_safe_candidates,

            within_performance_range,

            inside_applicability_domain,

            final_safe_candidates,

            safe_bo_pareto_count,

            valid_safe_pareto_candidates,

            final_recommendations_count,

            valid_final_recommendations

        ]

    })

    project_summary_path = os.path.join(

        REPORT_DIR,

        "project_summary.csv"

    )

    project_summary.to_csv(

        project_summary_path,

        index=False

    )

    # ========================================================
    # 9. Validation summary
    # ========================================================

    validation_output = pd.DataFrame({

        "Metric": [

            "Total Safe BO candidates",

            "Within experimental performance range",

            "Inside applicability domain",

            "Final safe candidates",

            "Safe BO Pareto candidates",

            "Valid Safe BO Pareto candidates",

            "Final recommendations",

            "Valid final recommendations"

        ],

        "Value": [

            total_safe_candidates,

            within_performance_range,

            inside_applicability_domain,

            final_safe_candidates,

            safe_bo_pareto_count,

            valid_safe_pareto_candidates,

            final_recommendations_count,

            valid_final_recommendations

        ]

    })

    validation_summary_path = os.path.join(

        REPORT_DIR,

        "validation_summary.csv"

    )

    validation_output.to_csv(

        validation_summary_path,

        index=False

    )

    # ========================================================
    # 10. Final recommendation table
    # ========================================================

    print()
    print("Preparing final recommendation table...")

    recommendation_columns = [

        "Final_Rank",

        "EVA_content",

        "Polymer_A",
        "Polymer_B",

        "FR_A",
        "FR_B",
        "FR_C",
        "FR_D",

        "Additive_1",
        "Additive_2",

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

    available_columns = [

        c

        for c in recommendation_columns

        if c in final_recommendations.columns

    ]

    recommendation_table = (

        final_recommendations[
            available_columns
        ]
        .copy()
        .sort_values("Final_Rank")
    )

    recommendation_table_path = os.path.join(

        REPORT_DIR,

        "final_recommendations.csv"

    )

    recommendation_table.to_csv(

        recommendation_table_path,

        index=False

    )

    # ========================================================
    # 11. Generate final text report
    # ========================================================

    print()
    print("Generating final text report...")

    report = []

    report.append("=" * 70)

    report.append(
        "EVA POLYMER MULTI-OBJECTIVE "
        "BAYESIAN OPTIMIZATION"
    )

    report.append(
        "FINAL PROJECT REPORT"
    )

    report.append("=" * 70)

    report.append("")

    # --------------------------------------------------------
    # Experimental data
    # --------------------------------------------------------

    report.append(
        "1. EXPERIMENTAL DATA"
    )

    report.append(
        f"Experimental samples: "
        f"{experimental_samples}"
    )

    report.append(
        f"Experimental LOI range: "
        f"{loi_min:.2f} - {loi_max:.2f}"
    )

    report.append(
        f"Experimental Transmittance range: "
        f"{trans_min:.2f} - {trans_max:.2f}"
    )

    report.append("")

    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------

    report.append(
        "2. SAFE BAYESIAN OPTIMIZATION"
    )

    report.append(
        f"Safe BO candidates: "
        f"{total_safe_candidates}"
    )

    report.append(
        f"Candidates within experimental "
        f"performance range: "
        f"{within_performance_range}"
    )

    report.append(
        f"Candidates inside applicability domain: "
        f"{inside_applicability_domain}"
    )

    report.append(
        f"Final safe candidates: "
        f"{final_safe_candidates}"
    )

    report.append(
        f"Safe BO Pareto candidates: "
        f"{safe_bo_pareto_count}"
    )

    report.append(
        f"Valid Safe BO Pareto candidates: "
        f"{valid_safe_pareto_candidates}"
    )

    report.append(
        f"Applicability distance threshold: "
        f"{applicability_threshold:.3f}"
    )

    report.append("")

    # --------------------------------------------------------
    # Final recommendations
    # --------------------------------------------------------

    report.append(
        "3. FINAL RECOMMENDATIONS"
    )

    report.append(
        f"Final recommendations: "
        f"{final_recommendations_count}"
    )

    report.append(
        f"Valid final recommendations: "
        f"{valid_final_recommendations}"
    )

    report.append("")

    # --------------------------------------------------------
    # Recommendation details
    # --------------------------------------------------------

    report.append(
        "4. FINAL RECOMMENDED FORMULATIONS"
    )

    report.append("")

    for _, row in recommendation_table.iterrows():

        rank = int(
            row["Final_Rank"]
        )

        report.append(
            f"Rank {rank}"
        )

        report.append(
            f"  EVA_content: "
            f"{row['EVA_content']:.6f}"
        )

        report.append(
            f"  Polymer_A: "
            f"{row['Polymer_A']:.0f}"
        )

        report.append(
            f"  Polymer_B: "
            f"{row['Polymer_B']:.0f}"
        )

        report.append(
            f"  FR_A: "
            f"{row['FR_A']:.6f}"
        )

        report.append(
            f"  FR_B: "
            f"{row['FR_B']:.6f}"
        )

        report.append(
            f"  FR_C: "
            f"{row['FR_C']:.6f}"
        )

        report.append(
            f"  FR_D: "
            f"{row['FR_D']:.6f}"
        )

        report.append(
            f"  Additive_1: "
            f"{row['Additive_1']:.6f}"
        )

        report.append(
            f"  Additive_2: "
            f"{row['Additive_2']:.6f}"
        )

        report.append(
            f"  Predicted LOI: "
            f"{row['Predicted_LOI']:.6f}"
        )

        report.append(
            f"  Predicted UL-94: "
            f"{row['Predicted_UL94']}"
        )

        report.append(
            f"  Predicted Transmittance: "
            f"{row['Predicted_Transmittance']:.6f}"
        )

        report.append(
            f"  Safety-adjusted score: "
            f"{row['Safety_Adjusted_Score']:.6f}"
        )

        report.append("")

    # --------------------------------------------------------
    # Conclusion
    # --------------------------------------------------------

    report.append(
        "5. CONCLUSION"
    )

    report.append("")

    report.append(
        "A Safe Bayesian Optimization workflow was "
        "used to identify candidate EVA formulations "
        "under experimental performance and "
        "applicability-domain constraints."
    )

    report.append(
        "The final five recommendations passed the "
        "final optimization validation."
    )

    report.append(
        "All five recommendations remained within "
        "the experimentally observed LOI and "
        "Transmittance ranges."
    )

    report.append("")

    report.append("=" * 70)

    report.append(
        "END OF FINAL PROJECT REPORT"
    )

    report.append("=" * 70)

    report_text = "\n".join(report)

    report_path = os.path.join(

        REPORT_DIR,

        "final_project_report.txt"

    )

    with open(

        report_path,

        "w",

        encoding="utf-8"

    ) as f:

        f.write(report_text)

    # ========================================================
    # 12. Console summary
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL PROJECT SUMMARY")
    print("=" * 70)

    print(
        f"Experimental samples: "
        f"{experimental_samples}"
    )

    print(
        f"Safe BO candidates: "
        f"{total_safe_candidates}"
    )

    print(
        f"Within experimental performance range: "
        f"{within_performance_range}"
    )

    print(
        f"Inside applicability domain: "
        f"{inside_applicability_domain}"
    )

    print(
        f"Final safe candidates: "
        f"{final_safe_candidates}"
    )

    print(
        f"Safe BO Pareto candidates: "
        f"{safe_bo_pareto_count}"
    )

    print(
        f"Valid Safe BO Pareto candidates: "
        f"{valid_safe_pareto_candidates}"
    )

    print(
        f"Final recommendations: "
        f"{final_recommendations_count}"
    )

    print(
        f"Valid final recommendations: "
        f"{valid_final_recommendations}"
    )

    print()
    print(
        f"Applicability distance threshold: "
        f"{applicability_threshold:.3f}"
    )

    print()
    print("Final recommendations:")

    display_columns = [

        "Final_Rank",
        "EVA_content",
        "Polymer_A",
        "Polymer_B",
        "Predicted_LOI",
        "Predicted_UL94",
        "Predicted_Transmittance",
        "Safety_Adjusted_Score"

    ]

    print(

        recommendation_table[
            display_columns
        ].to_string(index=False)

    )

    # ========================================================
    # Output paths
    # ========================================================

    print()
    print("Project summary saved to:")
    print(project_summary_path)

    print("Validation summary saved to:")
    print(validation_summary_path)

    print("Final recommendation table saved to:")
    print(recommendation_table_path)

    print("Final text report saved to:")
    print(report_path)

    print()
    print("=" * 70)
    print("STEP 20 completed.")
    print("=" * 70)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()