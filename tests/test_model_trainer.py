"""Unit tests for machine learning model training, evaluation, and inference."""

import numpy as np
import pandas as pd
import pytest

from src.data_processor import (
    ALL_FEATURE_COLUMNS,
    VALID_CLASSES,
    add_engineered_features,
    load_dataset,
    prepare_features_and_target,
    split_and_preprocess,
)
from src.model_trainer import predict_single_repository, train_and_evaluate_models


@pytest.fixture(scope="module")
def prepared_sample_data():
    df = load_dataset("data/sample_repositories.csv")
    X, y = prepare_features_and_target(df)
    return split_and_preprocess(X, y, test_size=0.2, random_state=42)


class TestModelTrainer:
    """Test model training pipeline, metric computation, and inference."""

    def test_train_and_evaluate_models(self, prepared_sample_data):
        split = prepared_sample_data
        result = train_and_evaluate_models(
            split["X_train"],
            split["X_test"],
            split["y_train"],
            split["y_test"],
            split["feature_names"],
        )

        assert "Baseline (Most Frequent)" in result["models"]
        assert "Logistic Regression" in result["models"]
        assert "Random Forest" in result["models"]

        comparison = result["comparison_table"]
        assert len(comparison) == 3
        assert set(comparison.columns) >= {"Model", "Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)"}

        # Check Random Forest performance is superior to Dummy baseline
        rf_acc = result["results"]["Random Forest"]["accuracy"]
        base_acc = result["results"]["Baseline (Most Frequent)"]["accuracy"]
        assert rf_acc > base_acc

        # Check confusion matrix shape
        cm = result["results"]["Random Forest"]["confusion_matrix"]
        assert cm.shape == (3, 3)

        # Check feature importance
        fi_df = result["feature_importance"]
        assert len(fi_df) == len(split["feature_names"])
        assert "Feature" in fi_df.columns
        assert "Importance" in fi_df.columns
        # Verify descending sort
        assert fi_df["Importance"].iloc[0] >= fi_df["Importance"].iloc[-1]

    def test_predict_single_repository(self, prepared_sample_data):
        split = prepared_sample_data
        model = train_and_evaluate_models(
            split["X_train"],
            split["X_test"],
            split["y_train"],
            split["y_test"],
            split["feature_names"],
        )["models"]["Random Forest"]

        # Healthy profile
        healthy_sample = pd.DataFrame([{
            "stars_count": 25000,
            "forks_count": 4500,
            "open_issues_count": 30,
            "closed_issues_count": 450,
            "pull_requests_count": 12,
            "days_since_last_commit": 2,
            "commits_last_year": 380,
            "contributors_count": 75,
            "repo_age_days": 1800,
            "has_license": 1,
            "has_readme": 1,
        }])
        healthy_featured = add_engineered_features(healthy_sample)

        prediction = predict_single_repository(model, split["scaler"], healthy_featured)
        assert prediction["predicted_class"] in VALID_CLASSES
        assert prediction["predicted_class"] == "Healthy"
        assert 0.0 <= prediction["confidence"] <= 1.0
        assert sum(prediction["probabilities"].values()) == pytest.approx(1.0, rel=1e-2)

    def test_predict_single_repository_missing_column_raises(self, prepared_sample_data):
        split = prepared_sample_data
        model = train_and_evaluate_models(
            split["X_train"],
            split["X_test"],
            split["y_train"],
            split["y_test"],
            split["feature_names"],
        )["models"]["Random Forest"]

        bad_df = pd.DataFrame([{"stars_count": 100}])
        with pytest.raises(ValueError, match="Missing required feature columns"):
            predict_single_repository(model, split["scaler"], bad_df)
