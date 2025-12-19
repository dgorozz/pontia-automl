import argparse
import joblib
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from pontia_automl.preprocessing import build_pipeline
from pontia_automl.model_registry import get_model, VALID_MODEL_NAMES
from pontia_automl.config import PROCESSED_DATASET_PATH, MODELS_PATH, SEED


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=VALID_MODEL_NAMES, help="Model to be trained")
    args = parser.parse_args()

    # load cleaned dataset and split
    df = pd.read_csv(os.path.join(PROCESSED_DATASET_PATH, "cleaned.csv"))
    X = df.drop(columns="is_canceled")
    y = df["is_canceled"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    # get model from registry
    model = get_model(args.model)
    
    # build pipeline
    pipeline = build_pipeline(model=model)

    # train model/pipeline
    pipeline.fit(X_train, y_train)

    # save model
    model_path = os.path.join(MODELS_PATH, args.model, f"{args.model}.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)

    print(f"Model '{args.model}' succesfully trained. Saved in {Path(model_path).absolute().as_posix()}.")


if __name__ == "__main__":
    main()