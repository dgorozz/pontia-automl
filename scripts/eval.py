import joblib
import os
from glob import glob
import argparse

from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from pontia_automl.evaluation import plot_confusion_matrix, plot_roc_curve
from pontia_automl.config import PROCESSED_DATASET_PATH, SEED, MODELS_PATH



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Flag for show plots")
    args = parser.parse_args()

    # load cleaned dataset and split
    df = pd.read_csv(os.path.join(PROCESSED_DATASET_PATH, "cleaned.csv"))
    X = df.drop(columns="is_canceled")
    y = df["is_canceled"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    # get model paths
    model_paths = glob(os.path.join(MODELS_PATH, "**", "*.pkl"))

    # eval each model
    results = []
    for path in tqdm(model_paths, desc="Evaluating models"):
        # load model
        model: Pipeline = joblib.load(path)
        model_name = os.path.splitext(os.path.basename(path))[0]

        # eval
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # plots
        if args.plot:
            plot_roc_curve(model_name, y_test, y_pred_proba[:, 1])
            plot_confusion_matrix(model_name, y_test, y_pred)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="binary"),
            "recall": recall_score(y_test, y_pred, average="binary"),
            "f1": f1_score(y_test, y_pred, average="binary"),
            "roc-auc": roc_auc_score(y_test, y_pred_proba[:, 1])
        }
        results.append({"model": model_name} | metrics)

    df_results = pd.DataFrame(results)
    print("\n=== Results ===")
    print(df_results)

    best_model_name = df_results.loc[df_results['roc-auc'].idxmax(), 'model']
    print()
    print(f"Best model (based on 'roc-auc' metric): {best_model_name}")