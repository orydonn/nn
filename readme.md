# Cluster Prediction Model

This project trains a neural network to predict in which of 17 product clusters a corporate client will appear after 12 months. It expects the following data files in the repository root:

- `train_data.pqt` – training data for months `month_1`…`month_3`.
- `test_data.pqt` – data for months `month_4`…`month_6` (predictions are made only for `month_6`).
- `cluster_weights.xlsx` – business weights for computing the weighted ROC‑AUC metric.
- `feature_description.xlsx` – description of 90 features used in the dataset.
- `sample_submission.csv` – example submission format.

## Setup
Install the required dependencies with
```bash
pip install -r requirements.txt
```
If you plan to run the LightGBM-based script (`model_new.py`) on macOS,
make sure the `libomp` runtime is installed as well:
```bash
brew install libomp
```
Without it LightGBM may fail to load with an error about `libomp.dylib`.

## Usage
Run the training script with optional hyperparameters:
```bash
python model.py --l2 0.0001 --lr 0.001 --batch-size 256 --epochs 150
```
This will produce a submission CSV with probabilities for each cluster.
