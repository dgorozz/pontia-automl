import pandas as pd
from sklearn.pipeline import Pipeline

import os

from src.config import RAW_DATASET_PATH, PROCESSED_DATASET_PATH


def load_raw_dataset():
    df = pd.read_csv(os.path.join(RAW_DATASET_PATH, "raw.csv"))
    return df


def load_processed_dataset():

    try:
        # load training data
        X_train = pd.read_csv(os.path.join(PROCESSED_DATASET_PATH, "X_train.csv"))
        y_train = pd.read_csv(os.path.join(PROCESSED_DATASET_PATH, "y_train.csv"))

        # load testing data
        X_test = pd.read_csv(os.path.join(PROCESSED_DATASET_PATH, "X_test.csv"))
        y_test = pd.read_csv(os.path.join(PROCESSED_DATASET_PATH, "y_test.csv"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Either X_train.csv, X_test.csv, y_train.csv or y_test.csv does not exists. Use preprocess.py script to create these files.")

    return X_train, y_train, X_test, y_test


    