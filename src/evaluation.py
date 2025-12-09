from src.config import MODELS_PATH

from glob import glob
import os

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def get_saved_model_paths():
    return glob(os.path.join(MODELS_PATH, "*.pkl"))
    

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted")
    }