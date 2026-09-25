"""Load the serological dataset and write quality control reports."""

from pathlib import Path
import numpy as np
import pandas as pd


def load_data(filepath: Path) -> pd.DataFrame:
    """Load the serological data from a CSV file and standardize the text fields."""
    df = pd.read_csv(filepath, na_values=['na', 'NA', ''])
    for col in ['sex', 'exposure', 'visit_reason']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()
    return df


def generate_data_inventory(df: pd.DataFrame, output_path: Path) -> str:
    """Write the cohort summary statistics and quality control metrics to a file."""
    lines = []
    lines.append("=" * 70)
    lines.append("PITCH COHORT DATA INVENTORY AND QUALITY CONTROL REPORT")
    lines.append("=" * 70)
    lines.append(f"Total Observations: {len(df)}")
    lines.append(f"Total Variables: {len(df.columns)}")
    lines.append(f"Unique Participants (pubid): {df['pubid'].nunique()}\n")

    lines.append("-" * 50)
    lines.append("1. DEMOGRAPHICS AND STUDY VISITS")
    lines.append("-" * 50)
    for col in ['sex', 'exposure', 'visit_reason']:
        if col in df.columns:
            lines.append(f"\n--- {col.upper()} ---")
            counts = df[col].value_counts(dropna=False)
            pcts = df[col].value_counts(dropna=False, normalize=True) * 100
            for k, v in counts.items():
                lines.append(f"  {str(k):<15} : {v:>5} ({pcts[k]:>5.2f}%)")

    lines.append("\n" + "-" * 50)
    lines.append("2. MISSING VALUES PER VARIABLE")
    lines.append("-" * 50)
    null_counts = df.isnull().sum()
    for col, count in null_counts.items():
        if count > 0:
            lines.append(f"  {col:<25}: {count:>5} missing ({count/len(df)*100:>5.1f}%)")

    lines.append("\n" + "-" * 50)
    lines.append("3. CONTINUOUS ANTIBODY MEASUREMENTS (RAW TITRES)")
    lines.append("-" * 50)
    num_cols = df.select_dtypes(include=[np.number]).columns
    desc = df[num_cols].describe().T[['count', 'mean', 'std', 'min', '50%', 'max']]
    desc.columns = ['N', 'Mean', 'SD', 'Min', 'Median', 'Max']
    lines.append(desc.to_string())

    report = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    return report
