# Credit Card Fraud Detection with a PyOD AutoEncoder

This project builds and trains an unsupervised deep learning model that detects
fraudulent credit card transactions. It uses the **AutoEncoder** algorithm from
the [PyOD](https://github.com/yzhao062/pyod) library and an anonymized credit
card transactions dataset from Kaggle.

## Overview

An AutoEncoder is a neural network that learns to reconstruct its own input.
When it is trained mainly on normal transactions, it reconstructs them with a
small error. Fraudulent transactions behave differently, so the network
reconstructs them poorly and produces a large **reconstruction error**. By
setting a threshold on this error, the model flags likely fraud without ever
being told which transactions are fraudulent during training.

## Dataset

- **Source:** https://www.kaggle.com/datasets/whenamancodes/fraud-detection
- **File name:** `creditcard.csv`
- **Size:** 284,807 transactions, of which 492 are fraud (about 0.17%).
- **Features:** `V1`–`V28` are PCA-transformed (anonymized) features, plus the
  original `Time` and `Amount` columns. `Class` is the label
  (0 = normal, 1 = fraud).

Download the dataset from the link above and place `creditcard.csv` in the same
folder as the Python script. The file is not included in this repository
because of its size and licensing.

## Project structure

```
.
├── fraud_detection_autoencoder.py   # Main program
├── requirements.txt                 # Manifest file with all dependencies
├── README.md                        # This file
└── reconstruction_error.png         # Plot produced after a run
```

## Setup

1. Install Python 3.9 or newer from https://www.python.org/downloads/.
2. (Recommended) Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. Install the required packages using the manifest file:

   ```bash
   pip install -r requirements.txt
   ```

## How to run

Place `creditcard.csv` in the project folder and run:

```bash
python fraud_detection_autoencoder.py
```

You can also point to a dataset in another location:

```bash
python fraud_detection_autoencoder.py --data /path/to/creditcard.csv
```

## What the program does

1. **Loads** the dataset from the CSV file.
2. **Explores** the data by printing its shape, class distribution, and any
   missing values.
3. **Preprocesses** the data by separating the label and scaling the `Time`
   and `Amount` columns.
4. **Splits** the data into a training set and a test set.
5. **Builds and trains** the PyOD AutoEncoder model.
6. **Evaluates** the model using ROC AUC, average precision, a confusion
   matrix, and a classification report.
7. **Plots** the reconstruction error for normal and fraudulent transactions
   and saves it as `reconstruction_error.png`.

## Notes on the metrics

Because only about 0.17% of the transactions are fraud, plain accuracy is
misleading (a model that predicts "normal" every time would still score above
99%). This project therefore reports **ROC AUC** and **average precision**,
which are far more meaningful for rare-event detection.

## References

- PyOD documentation: https://pyod.readthedocs.io/en/latest/index.html
- PyOD AutoEncoder: https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.auto_encoder
- Dataset: https://www.kaggle.com/datasets/whenamancodes/fraud-detection
