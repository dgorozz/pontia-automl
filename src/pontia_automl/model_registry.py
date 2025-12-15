from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from pontia_automl.config import SEED


MODEL_REGISTRY = {
    "logreg": LogisticRegression(solver="liblinear", random_state=SEED),
    "tree": DecisionTreeClassifier(random_state=SEED),
    "rf": RandomForestClassifier(random_state=SEED),
    "xgb": XGBClassifier(random_state=SEED),
    # TODO: add keras sklearn wrapper (https://keras.io/api/utils/sklearn_wrappers/)
}
VALID_MODEL_NAMES = set(MODEL_REGISTRY.keys())


def get_model(model_name):
    model = MODEL_REGISTRY.get(model_name)
    if model is None:
        raise ValueError(f"Model {model_name} does not exists in model registry. Please, provide one modelo from: {VALID_MODEL_NAMES}.")
    return model