# AMLS_25_26_25101025  
UCL – Applied Machine Learning Systems (AMLS) Coursework

This repository contains the implementation and experimental evaluation for the AMLS assignment, comparing a classical machine learning model (Model A) and a deep learning model (Model B) on the BreastMNIST dataset.

---

## Project Structure

AMLS_25_26_25101025/
├── Code/
│ ├── model_A/ # Kernel SVM pipeline (raw pixels & HOG features)
│ └── model_B/ # ResNet-18 pipeline (deep learning model)
├── Datasets/ # Kept empty for submission (dataset loaded automatically)
├── results/
│ ├── summary.csv # Aggregated experimental results
│ ├── plots/ # Figures for report
│ └── curves/ # Training curves (CSV)
├── main.py # Main entry point for all experiments
├── plot_results.py # Generate report-ready figures from summary.csv
└── README.md


---

## Environment

Recommended environment:

- Python 3.10
- numpy, scipy, pandas
- scikit-learn
- scikit-image
- torch, torchvision
- medmnist
- matplotlib

A single environment is sufficient for both Model A and Model B.

---

## Running the Experiments

### Run all experiments (Model A and Model B)
```bash
python main.py
This is the default and recommended command for reproducing all results reported in the coursework.

Run individual models (optional)
python main.py --run A   # Model A only
python main.py --run B   # Model B only


A fixed random seed is used to ensure reproducibility.


## Outputs:

After execution, the following files are generated:

results/summary.csv
Aggregated test-set metrics for all experiments.

results/plots/
Figures used in the final report (capacity, augmentation, budget analysis).

results/curves/
Training curves (epoch-wise loss and validation F1).

Generating Report Figures (Model B)

To generate report-ready figures for Model B from the aggregated results:

python plot_results.py --summary results/summary.csv --outdir results/plots

This script produces augmentation, budget, and capacity comparison figures directly from summary.csv.


##Notes:

The Datasets/ directory is intentionally kept empty for submission.

All experiments can be reproduced on CPU within reasonable time; GPU acceleration is optional.