from pathlib import Path


# src directory
SRC_DIR = Path(__file__).resolve().parent


# Project root directory
PROJECT_ROOT = SRC_DIR.parent


# Project directories
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"


# Dataset paths
RAW_DATA_PATH = DATA_DIR / "EVA data 100.xlsx"
CLEAN_DATA_PATH = DATA_DIR / "polymer_dataset_clean.csv"