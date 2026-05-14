import logging
from typing import Tuple, Optional, Dict, List, Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.cluster import AgglomerativeClustering
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.similarity_utils import compute_cosine_similarity_matrix

logger = logging.getLogger(__name__)

CATEGORICAL_FEATURES = ["country_identifier", "issue", "issue_category", "bloc"]
NUMERIC_FEATURES = ["year", "country_entropy", "issue_salience"]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def _issue_category(value: Any) -> str:
    issue = str(value or "").strip()
    if not issue:
        return "unknown"
    if ":" in issue:
        return issue.split(":", 1)[0].strip().lower() or "unknown"
    words = issue.split()
    return words[0].strip().lower() if words else "unknown"


def _compute_country_entropy_map(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {}
    series = df.groupby("country_identifier")["vote"].apply(
        lambda x: float(entropy(x.value_counts(), base=2)) if len(x) else 0.0
    )
    return series.to_dict()


def _compute_issue_salience_map(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {}
    total = float(max(len(df), 1))
    counts = df["issue"].value_counts(dropna=False)
    return {str(issue): float(count / total) for issue, count in counts.items()}


def _compute_bloc_map(df: pd.DataFrame) -> dict[str, str]:
    """Estimate bloc membership from training-window vote similarity."""
    if df.empty or "rcid" not in df.columns:
        return {}

    matrix = (
        df.dropna(subset=["country_identifier", "rcid", "vote"])
        .pivot_table(
            index="country_identifier", columns="rcid", values="vote", aggfunc="first"
        )
        .fillna(0.0)
    )
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return {}

    similarity = compute_cosine_similarity_matrix(
        matrix, min_norm=1e-8, drop_zero_rows=True
    )
    if similarity.empty or similarity.shape[0] < 2:
        return {}

    n_countries = similarity.shape[0]
    n_clusters = max(2, min(8, int(np.sqrt(n_countries))))
    try:
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage="average",
        )
        labels = clustering.fit_predict(1 - similarity)
    except Exception as exc:
        logger.warning("Bloc feature fallback: clustering failed (%s)", exc)
        return {}

    return {
        country: f"bloc_{int(label)}"
        for country, label in zip(similarity.index.tolist(), labels, strict=False)
    }


def _build_feature_context(train_df: pd.DataFrame, test_start_year: int) -> dict:
    country_entropy_map = _compute_country_entropy_map(train_df)
    issue_salience_map = _compute_issue_salience_map(train_df)
    bloc_map = _compute_bloc_map(train_df)

    mean_entropy = (
        float(np.mean(list(country_entropy_map.values())))
        if country_entropy_map
        else 0.0
    )
    mean_salience = (
        float(np.mean(list(issue_salience_map.values()))) if issue_salience_map else 0.0
    )

    return {
        "country_entropy_map": country_entropy_map,
        "issue_salience_map": issue_salience_map,
        "bloc_map": bloc_map,
        "mean_country_entropy": mean_entropy,
        "mean_issue_salience": mean_salience,
        "default_prediction_year": int(test_start_year),
        "feature_columns": MODEL_FEATURES,
    }


def _engineer_features(frame: pd.DataFrame, context: Optional[dict]) -> pd.DataFrame:
    df = frame.copy()
    if "year" not in df.columns:
        fallback_year = int(context.get("default_prediction_year", 0)) if context else 0
        df["year"] = fallback_year

    df["issue"] = df["issue"].astype(str)
    df["country_identifier"] = df["country_identifier"].astype(str)
    df["issue_category"] = df["issue"].map(_issue_category)

    if context:
        entropy_map = context.get("country_entropy_map", {})
        salience_map = context.get("issue_salience_map", {})
        bloc_map = context.get("bloc_map", {})
        mean_entropy = float(context.get("mean_country_entropy", 0.0))
        mean_salience = float(context.get("mean_issue_salience", 0.0))
    else:
        entropy_map = {}
        salience_map = {}
        bloc_map = {}
        mean_entropy = 0.0
        mean_salience = 0.0

    df["country_entropy"] = (
        df["country_identifier"].map(entropy_map).fillna(mean_entropy).astype(float)
    )
    df["issue_salience"] = (
        df["issue"].map(salience_map).fillna(mean_salience).astype(float)
    )
    df["bloc"] = (
        df["country_identifier"].map(bloc_map).fillna("bloc_unknown").astype(str)
    )
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(
        int(context.get("default_prediction_year", 0)) if context else 0
    )

    return df


def prepare_features_for_model(
    model: Pipeline,
    frame: pd.DataFrame,
    prediction_year: Optional[int] = None,
) -> pd.DataFrame:
    """Build a feature frame compatible with both legacy and issue-aware models."""
    feature_columns = getattr(model, "feature_columns_", None)
    if not feature_columns:
        # Legacy model compatibility.
        return frame[["country_identifier", "issue"]].copy()

    context = getattr(model, "feature_context_", None)
    prepared = frame.copy()
    if prediction_year is not None:
        prepared["year"] = int(prediction_year)

    prepared = _engineer_features(prepared, context)
    for col in feature_columns:
        if col not in prepared.columns:
            prepared[col] = 0

    return prepared[feature_columns]


def train_vote_predictor(
    un_data_df: pd.DataFrame, train_yr_end: int, test_yr_start: int
) -> Tuple[Optional[Pipeline], float, Dict, Optional[np.ndarray], int, int]:
    """Train a vote predictor model."""
    if un_data_df is None or un_data_df.empty:
        return None, 0, {}, None, 0, 0

    base_features = ["country_identifier", "issue"]
    target = "vote"
    required_cols = base_features + [target, "date", "year"]

    if not all(col in un_data_df.columns for col in required_cols):
        missing = [c for c in required_cols if c not in un_data_df.columns]
        logger.warning(f"ML Skip: Missing cols: {missing}")
        return None, 0, {}, None, 0, 0

    if not pd.api.types.is_datetime64_any_dtype(un_data_df["date"]):
        un_data_df["date"] = pd.to_datetime(un_data_df["date"], errors="coerce")

    # Include rcid if available so bloc feature can be estimated.
    model_cols = required_cols + (["rcid"] if "rcid" in un_data_df.columns else [])
    df_ml = un_data_df[model_cols].dropna(subset=base_features + [target]).copy()

    train_df = df_ml[df_ml["year"] <= train_yr_end]
    test_df = df_ml[df_ml["year"] >= test_yr_start]

    if train_df.empty:
        return None, 0, {}, None, 0, len(test_df)

    feature_context = _build_feature_context(train_df, test_yr_start)

    train_features_df = _engineer_features(
        train_df[["country_identifier", "issue", "year"]], feature_context
    )
    X_train = train_features_df[MODEL_FEATURES]
    y_train = train_df[target]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=160,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    try:
        model_pipeline.fit(X_train, y_train)
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return None, 0, {}, None, len(train_df), len(test_df)

    accuracy = 0
    report = {}

    if not test_df.empty:
        test_features_df = _engineer_features(
            test_df[["country_identifier", "issue", "year"]],
            feature_context,
        )
        X_test = test_features_df[MODEL_FEATURES]
        y_test = test_df[target]
        try:
            y_pred = model_pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(
                y_test, y_pred, zero_division=0, output_dict=True
            )
        except Exception as e:
            logger.warning(f"Error evaluating model: {e}")

    # Attach context so inference can construct issue-aware features.
    model_pipeline.feature_context_ = feature_context
    model_pipeline.feature_columns_ = MODEL_FEATURES

    all_countries = (
        un_data_df["country_identifier"].unique()
        if "country_identifier" in un_data_df
        else None
    )
    return model_pipeline, accuracy, report, all_countries, len(train_df), len(test_df)


def predict_votes(
    model: Pipeline,
    countries: List[str],
    issue: str,
    prediction_year: Optional[int] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Predict votes for a given issue."""
    if model is None or countries is None or len(countries) == 0:
        return None, None

    try:
        # Create prediction DataFrame
        pred_df = pd.DataFrame(
            {
                "country_identifier": countries,
                "issue": [issue] * len(countries),
            }
        )
        if prediction_year is not None:
            pred_df["year"] = int(prediction_year)

        # Make predictions
        model_input = prepare_features_for_model(
            model,
            pred_df,
            prediction_year=prediction_year,
        )
        predictions = model.predict(model_input)

        # Create detailed predictions DataFrame
        detailed_predictions = pd.DataFrame(
            {
                "Country": countries,
                "Predicted Vote": predictions,
            }
        )

        # Create summary of predictions
        vote_counts = pd.DataFrame(
            detailed_predictions["Predicted Vote"].value_counts()
        ).reset_index()
        vote_counts.columns = ["Vote", "Count"]

        return vote_counts, detailed_predictions

    except Exception as e:
        logger.error(f"Error in vote prediction: {e}")
        return None, None


def save_model(model: Pipeline, filepath: str) -> None:
    """Save a trained model to disk.

    Args:
        model: The trained model to save
        filepath: Path where to save the model

    Raises:
        ValueError: If model is None or invalid
        Exception: For other errors during saving
    """
    if model is None:
        logger.error("Cannot save None model")
        raise ValueError("Model cannot be None")

    try:
        joblib.dump(model, filepath)
        logger.info(f"Model successfully saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise


def load_model(filepath: str) -> Pipeline:
    """Load a trained model from disk.

    Args:
        filepath: Path to the saved model

    Returns:
        The loaded model
    """
    try:
        model = joblib.load(filepath)
        logger.info(f"Model successfully loaded from {filepath}")
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise
