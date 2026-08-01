"""
fraud_detection_autoencoder.py
================================

Hands-On Assignment 4: Use an Unsupervised Deep Learning Algorithm to Detect
Fraud with PyOD.

This program builds and trains an unsupervised fraud detection model using the
AutoEncoder algorithm provided by the PyOD (Python Outlier Detection) library.
The model is trained on an anonymized credit card transactions dataset from
Kaggle. An AutoEncoder is a neural network that learns to reconstruct its own
input. When it is trained mostly on normal (non-fraudulent) transactions, it
reconstructs those transactions with a small error. Fraudulent transactions,
which behave differently, produce a larger reconstruction error and can
therefore be flagged as anomalies.

Dataset:
    Credit Card Fraud Detection (creditcard.csv)
    https://www.kaggle.com/datasets/whenamancodes/fraud-detection

    The dataset contains 284,807 transactions, of which 492 are fraudulent
    (about 0.17 percent). The features V1 through V28 are the result of a PCA
    transformation applied to protect confidentiality. The columns "Time" and
    "Amount" are the only original features, and "Class" is the target
    (0 = normal, 1 = fraud).

"""

# ---------------------------------------------------------------------------
# Standard library imports
# ---------------------------------------------------------------------------
import os
import argparse

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

# The AutoEncoder detector is imported from PyOD. PyOD provides a single,
# consistent interface (fit / predict / decision_function) for more than 40
# outlier detection algorithms.
from pyod.models.auto_encoder import AutoEncoder


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
CSV_PATH = "creditcard.csv"      # Expected name of the Kaggle dataset file.
SEED_VALUE = 42                  # Fixed seed so that results are reproducible.
HOLDOUT_FRACTION = 0.30          # Proportion of data held out for testing.
FIGURE_PATH = "reconstruction_error.png"


# ---------------------------------------------------------------------------
# Function: read_transactions
# ---------------------------------------------------------------------------
def read_transactions(csv_path):
    """Load the credit card transactions dataset from a CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the creditcard.csv file downloaded from Kaggle.

    Returns
    -------
    pandas.DataFrame
        The full dataset loaded into memory.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            "Could not find '{}'. Please download the dataset from "
            "https://www.kaggle.com/datasets/whenamancodes/fraud-detection "
            "and place 'creditcard.csv' in the same folder as this "
            "script.".format(csv_path)
        )

    print("Loading dataset from '{}' ...".format(csv_path))
    transactions = pd.read_csv(csv_path)
    print("Dataset loaded. Shape: {} rows, {} columns".format(
        transactions.shape[0], transactions.shape[1]))
    return transactions


# ---------------------------------------------------------------------------
# Function: summarize_transactions
# ---------------------------------------------------------------------------
def summarize_transactions(transactions):
    """Print a short summary of the dataset to help understand its structure.

    Parameters
    ----------
    transactions : pandas.DataFrame
        The dataset to describe.
    """
    print("\n----- Dataset overview -----")
    print("First five rows:")
    print(transactions.head())

    # Count how many transactions are normal and how many are fraudulent.
    label_counts = transactions["Class"].value_counts()
    legit_count = int(label_counts.get(0, 0))
    fraud_count = int(label_counts.get(1, 0))
    total_count = legit_count + fraud_count

    print("\nClass distribution:")
    print("  Normal transactions : {} ({:.3f}%)".format(
        legit_count, 100.0 * legit_count / total_count))
    print("  Fraud transactions  : {} ({:.3f}%)".format(
        fraud_count, 100.0 * fraud_count / total_count))

    # Report any missing values so data quality problems are visible early.
    missing_total = int(transactions.isnull().sum().sum())
    print("\nTotal missing values in dataset: {}".format(missing_total))


# ---------------------------------------------------------------------------
# Function: prepare_features
# ---------------------------------------------------------------------------
def prepare_features(transactions):
    """Prepare the features and labels for modelling.

    The steps performed here are:
      1. Separate the input features (X) from the target label (y).
      2. Scale the "Time" and "Amount" columns so that they sit on a similar
         range to the PCA features V1 through V28. Neural networks train more
         reliably when all inputs have a comparable scale.

    Parameters
    ----------
    transactions : pandas.DataFrame
        The full dataset including the "Class" column.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        The scaled feature matrix X and the label vector y.
    """
    print("\n----- Preprocessing data -----")

    # The "Class" column is the ground-truth label. It is removed from the
    # feature set because the AutoEncoder learns in an unsupervised manner and
    # must not see the answer during training. The labels are kept only to
    # evaluate the model afterwards.
    feature_frame = transactions.drop(columns=["Class"])
    target_vector = transactions["Class"].values

    # Standardise the two original (non-PCA) columns. The V-columns are already
    # centred because they come from a PCA transformation.
    standardizer = StandardScaler()
    for field in ["Time", "Amount"]:
        if field in feature_frame.columns:
            feature_frame[field] = standardizer.fit_transform(
                feature_frame[[field]])

    print("Preprocessing complete. Feature matrix shape: {}".format(
        feature_frame.shape))
    return feature_frame.values, target_vector


# ---------------------------------------------------------------------------
# Function: train_detector
# ---------------------------------------------------------------------------
def train_detector(train_features, anomaly_ratio):
    """Create and train the PyOD AutoEncoder detector.

    Parameters
    ----------
    train_features : numpy.ndarray
        The training feature matrix.
    anomaly_ratio : float
        The expected proportion of anomalies (frauds) in the data. PyOD uses
        this value to choose the decision threshold on the reconstruction
        error.

    Returns
    -------
    pyod.models.auto_encoder.AutoEncoder
        The trained AutoEncoder model.
    """
    print("\n----- Building the AutoEncoder model -----")

    # The hidden_neuron_list defines the size of each hidden layer. The network
    # compresses the input down to a small "bottleneck" (16 neurons) and then
    # expands it back out. This forces the network to learn only the most
    # important patterns of a normal transaction.
    detector = AutoEncoder(
        hidden_neuron_list=[64, 32, 16, 32, 64],
        epoch_num=20,
        batch_size=256,
        contamination=anomaly_ratio,
        random_state=SEED_VALUE,
    )

    print("Training the model. This may take a few minutes ...")
    detector.fit(train_features)
    print("Training complete.")
    return detector


# ---------------------------------------------------------------------------
# Function: assess_detector
# ---------------------------------------------------------------------------
def assess_detector(detector, test_features, test_labels):
    """Evaluate the trained model and print performance metrics.

    Because the dataset is highly imbalanced, accuracy alone is misleading.
    The evaluation therefore reports the ROC AUC and the average precision
    (area under the precision-recall curve), which are better suited to rare
    event detection.

    Parameters
    ----------
    detector : pyod.models.auto_encoder.AutoEncoder
        The trained model.
    test_features : numpy.ndarray
        The test feature matrix.
    test_labels : numpy.ndarray
        The true labels for the test set.

    Returns
    -------
    numpy.ndarray
        The reconstruction error (anomaly score) for each test transaction.
    """
    print("\n----- Evaluating the model -----")

    # decision_function returns the raw anomaly score (reconstruction error).
    # A higher score means the transaction looks more abnormal.
    anomaly_scores = detector.decision_function(test_features)

    # predict returns a binary label: 1 for anomaly (fraud), 0 for normal.
    predicted_labels = detector.predict(test_features)

    auc_score = roc_auc_score(test_labels, anomaly_scores)
    precision_score = average_precision_score(test_labels, anomaly_scores)

    print("ROC AUC score            : {:.4f}".format(auc_score))
    print("Average precision score  : {:.4f}".format(precision_score))

    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(test_labels, predicted_labels))

    print("\nClassification report:")
    print(classification_report(
        test_labels, predicted_labels,
        target_names=["Normal", "Fraud"], digits=4))

    return anomaly_scores


# ---------------------------------------------------------------------------
# Function: draw_error_histogram
# ---------------------------------------------------------------------------
def draw_error_histogram(anomaly_scores, test_labels, figure_path):
    """Plot the reconstruction error for normal and fraudulent transactions.

    The plot makes it easy to see that fraudulent transactions generally have
    a higher reconstruction error than normal ones.

    Parameters
    ----------
    anomaly_scores : numpy.ndarray
        The reconstruction error for each test transaction.
    test_labels : numpy.ndarray
        The true labels for the test set.
    figure_path : str
        Path where the figure will be saved.
    """
    print("\n----- Creating reconstruction error plot -----")

    plt.figure(figsize=(10, 6))
    legit_errors = anomaly_scores[test_labels == 0]
    fraud_errors = anomaly_scores[test_labels == 1]

    plt.hist(legit_errors, bins=50, alpha=0.6, label="Normal", density=True)
    plt.hist(fraud_errors, bins=50, alpha=0.6, label="Fraud", density=True)
    plt.xlabel("Reconstruction error (anomaly score)")
    plt.ylabel("Density")
    plt.title("Reconstruction Error by Transaction Type")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    print("Plot saved to '{}'.".format(figure_path))


# ---------------------------------------------------------------------------
# Function: main
# ---------------------------------------------------------------------------
def main():
    """Run the full fraud detection workflow from start to finish."""
    arg_parser = argparse.ArgumentParser(
        description="Detect credit card fraud using a PyOD AutoEncoder.")
    arg_parser.add_argument(
        "--data", default=CSV_PATH,
        help="Path to the creditcard.csv dataset file.")
    parsed_args = arg_parser.parse_args()

    # Step 1: Load the data.
    transactions = read_transactions(parsed_args.data)

    # Step 2: Explore the data.
    summarize_transactions(transactions)

    # Step 3: Preprocess the data.
    feature_matrix, label_vector = prepare_features(transactions)

    # Step 4: Split the data into a training set and a test set. The split is
    # stratified so that the small number of frauds is shared proportionally
    # between both sets.
    train_features, test_features, train_labels, test_labels = train_test_split(
        feature_matrix, label_vector,
        test_size=HOLDOUT_FRACTION,
        random_state=SEED_VALUE,
        stratify=label_vector,
    )
    print("\nTraining set size: {} | Test set size: {}".format(
        len(train_features), len(test_features)))

    # The contamination is the proportion of frauds in the full dataset. It is
    # used only to set the model's decision threshold.
    anomaly_ratio = float(np.mean(label_vector))
    print("Estimated contamination (fraud rate): {:.5f}".format(anomaly_ratio))

    # Step 5: Build and train the model.
    detector = train_detector(train_features, anomaly_ratio)

    # Step 6: Evaluate the model on unseen data.
    anomaly_scores = assess_detector(detector, test_features, test_labels)

    # Step 7: Visualise the reconstruction error.
    draw_error_histogram(anomaly_scores, test_labels, FIGURE_PATH)

    print("\nWorkflow finished successfully.")


# This guard makes sure main() runs only when the file is executed directly,
# and not when it is imported as a module.
if __name__ == "__main__":
    main()