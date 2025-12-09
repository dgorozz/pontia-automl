import joblib
import os

from tqdm import tqdm
import pandas as pd

from src.load_data import load_processed_dataset
from src.evaluation import get_saved_model_paths, evaluate_model


if __name__ == "__main__":

    # load dataset
    _, _, X_test, y_test = load_processed_dataset()

    # get model paths
    model_paths = get_saved_model_paths()

    # eval each model
    results = []
    for path in tqdm(model_paths, desc="Evaluating models"):
        model = joblib.load(path)
        result = evaluate_model(model, X_test, y_test)
        results.append({"model": os.path.basename(path)} | result)

    df_results = pd.DataFrame(results)
    print("\n=== Results ===")
    print(df_results)