# AMLS_25_26_25101025  
UCL – Applied Machine Learning Systems (AMLS) Coursework

This repository contains the implementation and experimental evaluation for the AMLS assignment, comparing a classical machine learning model (Model A) and a deep learning model (Model B) on the BreastMNIST dataset.

---

## Project Structure

The project is organised as follows:

- `Code/model_A/`: Classical machine learning pipeline, including Kernel SVM
  experiments with raw pixel features and HOG feature extraction.
- `Code/model_B/`: Deep learning pipeline based on a ResNet-18 architecture.
- `Datasets/`: Kept empty for submission. The dataset will be provided and
  automatically loaded during assessment.
- `results/`: Stores experimental outputs, including aggregated results,
  training curves, and figures used in the report.
- `main.py`: Unified entry point for running all experiments.
- `plot_results.py`: Utility script for generating report-ready figures from
  the experimental results.
- `README.md`: Project documentation and usage instructions.


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