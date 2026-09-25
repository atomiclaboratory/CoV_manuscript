# Systems Serology: SARS-CoV-2 Vaccine Response Analysis

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides the analysis pipeline for the manuscript:

> **"A Beta-to-Alpha seasonal coronavirus antibody signature identifies high and low responders to SARS-CoV-2 vaccination"**  
> *Stephanie Longet\*, Caolann Brady\*, Ivan Tomic\*, Tom Tipton, Salma Bessalah, Eleanor Barnes, Susanna Dunachie, Christopher J. A. Duncan, Paul Klenerman, Alex Richter, James E. D. Thaventhiran, Lance Turtle, Thushan I. de Silva, Adriana Tomic, Miles W. Carroll*

---

## Overview

This repository contains the analysis code for the PITCH study. The pipeline evaluates the association between seasonal human coronavirus antibodies and the SARS-CoV-2 vaccine response.

### Analysis Steps

1. **Preprocessing and Limit of Detection:** Floor raw electrochemiluminescence values at 1.0. Apply the log10 transformation to all antibody measurements.
2. **Cross-Reactivity Correction:** Regress MERS-CoV and SARS-CoV-1 background signals from endemic coronavirus titres with ordinary least squares.
3. **Engineered Features:**
   * **Beta to Alpha Ratio:** Calculate $\frac{\text{OC43}_{\text{corr}} + \text{HKU1}_{\text{corr}}}{\text{229E}_{\text{corr}} + \text{NL63}_{\text{corr}} + 10^{-5}}$.
   * **Endemic Breadth Score:** Count the corrected endemic coronavirus titres above the cohort median (range 0 to 4).
4. **Extreme Phenotyping and Feature Selection:** Classify the top 25% versus the bottom 25% post-dose 2 Spike IgG responders. Use L1-regularized logistic regression with recursive feature elimination.
5. **Permutation Testing:** Execute 1,000 label permutations to calculate the empirical significance ($P < 0.001$).
6. **Longitudinal Waning Analysis:** Model antibody decay between 28 and 180 days post-dose 2 with ordinary least squares regression.

---

## Directory Structure

```
public_code/
├── data/
│   └── pitch_serology_data.csv       # Serological dataset
├── src/
│   ├── __init__.py
│   ├── config.py                     # Configuration settings and paths
│   ├── data_loader.py                # Data loading and inventory reporting
│   ├── preprocessing.py              # Limit of detection and cross-reactivity correction
│   ├── feature_engineering.py        # Engineered features and cohort tables
│   ├── models.py                     # Statistical models and machine learning
│   └── plotting.py                   # Plotting functions for Figures 1 to 4
├── run_pipeline.py                   # Master analysis script
├── environment.yml                   # Conda environment definition
├── requirements.txt                  # Python package requirements
├── LICENSE                           # MIT License
└── README.md
```

The pipeline creates the `output/` directory dynamically at runtime.

---

## Installation and Execution

### Using Conda

Execute the following commands in your shell:

```bash
git clone https://github.com/atomiclaboratory/CoV_analasys.git
cd CoV_analasys

conda env create -f environment.yml
conda activate cov-analysis
python run_pipeline.py
```

### Using Pip

Execute the following commands in your shell:

```bash
git clone https://github.com/atomiclaboratory/CoV_analasys.git
cd CoV_analasys

python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
python run_pipeline.py
```

---

## Output Files

The pipeline generates the following files in `output/`:

* **`output/plots/figure1_n_vs_s_waning.pdf` (and `.png`):** Nucleocapsid versus Spike IgG trajectories across study visits.
* **`output/plots/figure2_extreme_phenotyping_roc.pdf` (and `.png`):** Receiver operating characteristic curve for extreme responders (AUC = 0.72).
* **`output/plots/figure3_permutation_test.pdf` (and `.png`):** Permutation test null distribution across 1,000 iterations ($P < 0.001$).
* **`output/plots/figure4_oc43_recall.pdf` (and `.png`):** Endemic Betacoronavirus (OC43) Spike IgG binding across vaccination timepoints.
* **`output/statistical_summary.txt`:** Statistical metrics and machine learning results.
* **`output/data/data_inventory.txt`:** Cohort summary statistics and quality control counts.
* **`output/checkpoints/`:** Intermediate data files and saved model objects.

---

## Contact

* **Laboratory:** aTomic Lab (https://atomic-lab.org)
* **Email:** Ivan Tomic (ivan@atomic-lab.org)

---

## License

This project uses the MIT License. See the [LICENSE](LICENSE) file for details.
