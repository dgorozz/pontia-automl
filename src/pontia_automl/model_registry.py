from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from scikeras.wrappers import KerasClassifier
from keras import layers, models
from xgboost import XGBClassifier

from pontia_automl.config import SEED


def create_keras_model(meta, hidden_layers=[64, 32]):
    n_features = meta["n_features_in_"]
    model = models.Sequential([
        layers.Input(shape=(n_features,)),
        *[layers.Dense(units, activation="relu") for units in hidden_layers],
        layers.Dense(1, activation="sigmoid")
    ])
    return model


MODEL_REGISTRY = {
    "logreg": LogisticRegression(solver="liblinear", random_state=SEED),
    "tree": DecisionTreeClassifier(random_state=SEED),
    "rf": RandomForestClassifier(random_state=SEED),
    "xgb": XGBClassifier(random_state=SEED),
    "keras": KerasClassifier(
        model=create_keras_model, 
        random_state=SEED, 
        epochs=5, 
        optimizer="adam", 
        loss="binary_crossentropy", 
        metrics=["accuracy"]
    )
}
VALID_MODEL_NAMES = set(MODEL_REGISTRY.keys())


def get_model(model_name):
    model = MODEL_REGISTRY.get(model_name)
    if model is None:
        raise ValueError(f"Model {model_name} does not exists in model registry. Please, provide one modelo from: {VALID_MODEL_NAMES}.")
    return model