from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler



MODEL_REGISTRY = {
    "LogisticRegression": LogisticRegression(solver="liblinear", random_state=28),
    "DecissionTree": DecisionTreeClassifier(random_state=28),
    "RandomForest": RandomForestClassifier(random_state=28),
    "XGBoost": XGBClassifier(random_state=28),
    # TODO: add keras sklearn wrapper (https://keras.io/api/utils/sklearn_wrappers/)
}
VALID_MODEL_NAMES = set(MODEL_REGISTRY.keys())


def get_model(model_name):
    model = MODEL_REGISTRY.get(model_name)
    if not model:
        raise ValueError(f"Model {model_name} does not exists in model registry. Please, provide one modelo from: {VALID_MODEL_NAMES}.")
    return model


def build_pipeline(model, numeric_cols, categorical_cols):

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    return pipe