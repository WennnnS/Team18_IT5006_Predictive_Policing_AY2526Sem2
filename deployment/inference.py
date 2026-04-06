from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

from utils import MODEL_PATH, check_file_exists


TARGET_CANDIDATES = [
    "target_hotspot_next_block",
    "target",
    "label",
    "y",
]

NON_FEATURE_COLUMNS = {
    "date",
    "event_date",
    "target_hotspot_next_block",
    "target",
    "label",
    "y",
    "predicted_label",
    "predicted_probability",
}


@st.cache_resource
def load_model_artifact():
    check_file_exists(MODEL_PATH, "Model file")
    artifact = joblib.load(MODEL_PATH)
    return artifact


def unpack_model_artifact(artifact):
    """
    Try to support common saved formats:
    1) pure sklearn model object
    2) dict with keys like:
       - model
       - feature_names
       - threshold
       - best_params
    """
DEPLOYMENT_DEFAULT_THRESHOLD = 0.28


def unpack_model_artifact(artifact):
    """
    Support common saved formats:
    1) pure sklearn model object
    2) dict with keys like:
       - model
       - feature_names
       - threshold
       - best_params
    """

    if isinstance(artifact, dict):
        model = artifact.get("model", artifact)
        feature_names = artifact.get("feature_names")

        # Prefer a saved threshold if the artifact already contains one.
        # If not, fall back to the Phase 2 deployment threshold for HGB.
        threshold = artifact.get("threshold", DEPLOYMENT_DEFAULT_THRESHOLD)
        meta = artifact
    else:
        model = artifact
        feature_names = getattr(model, "feature_names_in_", None)

        # The plain model object does not carry threshold information,
        # so we explicitly use the refined deployment threshold from Phase 2.
        threshold = DEPLOYMENT_DEFAULT_THRESHOLD
        meta = {}

    return model, feature_names, threshold, meta

    return model, feature_names, threshold, meta


def infer_feature_columns(df: pd.DataFrame):
    cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    return cols


def align_features(input_df: pd.DataFrame, feature_names):
    X = input_df.copy()

    if feature_names is None:
        return X

    for col in feature_names:
        if col not in X.columns:
            X[col] = 0

    extra_cols = [c for c in X.columns if c not in feature_names]
    if extra_cols:
        X = X.drop(columns=extra_cols)

    X = X[feature_names]
    return X


def predict_rows(input_df: pd.DataFrame):
    artifact = load_model_artifact()
    model, feature_names, threshold, meta = unpack_model_artifact(artifact)

    X = align_features(input_df, feature_names)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        raw_pred = model.predict(X)
        probs = raw_pred

    labels = (probs >= threshold).astype(int)

    result = pd.DataFrame({
        "predicted_probability": probs,
        "predicted_label": labels,
    })

    return result, threshold, meta