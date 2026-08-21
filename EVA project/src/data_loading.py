import sys
from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

from config import RAW_DATA_PATH


# ============================================================
# STEP 1 - RAW DATA LOADING
# ============================================================

print("STEP 1 - RAW DATA LOADING")

print("Loading raw dataset...")

df = pd.read_excel(RAW_DATA_PATH)


# ============================================================
# DATASET SHAPE
# ============================================================

print("\nDATASET SHAPE")

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")


# ============================================================
# COLUMN NAMES
# ============================================================

print("\nCOLUMN NAMES")

for i, column in enumerate(df.columns, start=1):
    print(f"{i:02d}. {column}")


# ============================================================
# FIRST 5 ROWS
# ============================================================

print("\nFIRST 5 ROWS")

print(df.head())


# ============================================================
# DATA TYPES
# ============================================================

print("\nDATA TYPES")

print(df.dtypes)


# ============================================================
# MISSING VALUES
# ============================================================

print("\nMISSING VALUES")

print(df.isnull().sum())


# ============================================================
# DUPLICATE ROWS
# ============================================================

print("\nDUPLICATE ROWS")

print(f"Duplicate rows: {df.duplicated().sum()}")


# ============================================================
# UL-94 DISTRIBUTION
# ============================================================

print("\nUL-94 DISTRIBUTION")

print(df["UL_94"].value_counts())


# ============================================================
# NUMERICAL SUMMARY
# ============================================================

print("\nNUMERICAL SUMMARY")

print(df.describe())


# ============================================================
# COMPLETION
# ============================================================

print("\nSTEP 1 COMPLETED")