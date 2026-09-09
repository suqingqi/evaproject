import subprocess
import sys
from pathlib import Path
from time import perf_counter

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

PIPELINE = [
    ("Data cleaning", "data_cleaning.py"),
    ("LOI model", "loi_model.py"),
    ("UL-94 model", "ul94_model.py"),
    ("Transmittance model", "transmittance_model.py"),
    ("Multi-objective model integration", "multi_objective_model.py"),
    ("Pareto analysis", "pareto_analysis.py"),
    ("Bayesian optimization", "bayesian_optimization.py"),
    ("Safety-aware Bayesian optimization", "safe_bayesian_optimization.py"),
    ("Final recommendation", "final_recommendation.py"),
    ("Next experiment selection", "next_experiment_recommendation .py"),
]


def run_step(index, total, label, script_name):
    script_path = CURRENT_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Pipeline script not found: {script_path}")

    print()
    print(f"[{index}/{total}] {label}")
    print(f"Running: {script_name}")

    start = perf_counter()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        check=False,
    )
    elapsed = perf_counter() - start

    if result.returncode != 0:
        raise RuntimeError(
            f"Pipeline stopped because {script_name} failed "
            f"with exit code {result.returncode}."
        )

    print(f"Completed: {label} ({elapsed:.1f} s)")


def main():
    print("CLOSED-LOOP MODEL RETRAINING PIPELINE")
    print(f"Project root: {PROJECT_ROOT}")
    print()
    print("Purpose:")
    print("re-clean experimental data -> retrain models -> rerun optimization -> select next experiments")
    print()
    print("Important: run this only after REAL experimental results have been committed")
    print("with update_experimental_data.py --commit.")

    total = len(PIPELINE)
    pipeline_start = perf_counter()

    for index, (label, script_name) in enumerate(PIPELINE, start=1):
        run_step(index, total, label, script_name)

    elapsed = perf_counter() - pipeline_start

    print()
    print("CLOSED-LOOP PIPELINE COMPLETED")
    print(f"Total runtime: {elapsed:.1f} s")
    print()
    print("Updated outputs include:")
    print("- cleaned experimental dataset")
    print("- LOI / UL-94 / Transmittance models")
    print("- Pareto and Bayesian-optimization results")
    print("- safety-aware candidate screening")
    print("- next-experiment batch")
    print()
    print("Next, create a fresh blank feedback sheet for the new batch with:")
    print("python src/update_experimental_data.py --create-template")


if __name__ == "__main__":
    main()
