import argparse
import joblib
import os

from src.model_registry import get_model, VALID_MODEL_NAMES, build_pipeline
from src.load_data import load_processed_dataset



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=VALID_MODEL_NAMES, help="Model to be trained")
    args = parser.parse_args()

    # load processed dataset
    X_train, y_train, _, _ = load_processed_dataset()

    # get model from registry
    model = get_model(args.model)
    
    # build pipeline
    numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude="number").columns.tolist()
    pipeline = build_pipeline(model=model, numeric_cols=numeric_cols, categorical_cols=categorical_cols)

    # train model/pipeline
    pipeline.fit(X_train, y_train)

    # save model
    model_path = os.path.join(f"{args.model}.pkl")
    joblib.dump(pipeline, model_path)

    print(f"Model {args.model} succesfully trained. Saved in {model_path}.")