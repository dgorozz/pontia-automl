from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CATEGORICAL_COLS = ["hotel", "meal", "market_segment", "distribution_channel", "deposit_type", "customer_type", "stay_period", "waiting_list_period", "arrival_quarter", "required_car_parking_spaces"]
NUMERIC_COLS = ["booking_changes", "total_of_special_requests", "adults", "children", "babies", "lead_time", "adr", "guest_cancel_ratio"]
KEEP_COLS = ["is_repeated_cols", "is_arrival_at_weekend", "room_changed", "is_arrival_at_weekend"]


def build_pipeline(model):

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS)
        ]
    )

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    return pipeline