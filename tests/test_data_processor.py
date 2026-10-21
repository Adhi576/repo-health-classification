"""Unit tests for dataset loading, validation, feature engineering, and leakage-free splitting."""

import io
import numpy as np
import pandas as pd
import pytest

from src.data_processor import (
    ALL_FEATURE_COLUMNS,
    BASE_NUMERIC_FEATURES,
    VALID_CLASSES,
    add_engineered_features,
    assess_data_quality,
    convert_live_data_to_features,
    load_dataset,
    prepare_features_and_target,
    split_and_preprocess,
    validate_dataset,
)


@pytest.fixture
def sample_valid_df():
    data = {
        "repo_name": ["repo-a", "repo-b", "repo-c", "repo-d", "repo-e", "repo-f"],
        "stars_count": [1000, 50, 500, 20, 10, 2000],
        "forks_count": [200, 5, 80, 2, 1, 400],
        "open_issues_count": [15, 25, 30, 4, 10, 2],
        "closed_issues_count": [150, 5, 60, 1, 0, 300],
        "pull_requests_count": [5, 2, 8, 0, 1, 3],
        "days_since_last_commit": [4, 180, 25, 450, 600, 2],
        "commits_last_year": [120, 5, 45, 0, 1, 350],
        "contributors_count": [25, 2, 8, 1, 1, 40],
        "repo_age_days": [1500, 800, 600, 900, 1200, 2000],
        "has_license": [1, 1, 1, 0, 0, 1],
        "has_readme": [1, 1, 1, 1, 0, 1],
        "health_status": ["Healthy", "At Risk", "Healthy", "Likely Abandoned", "Likely Abandoned", "Healthy"],
    }
    return pd.DataFrame(data)


class TestDataProcessor:
    """Test data loading, validation, quality auditing, and leakage-free scaling."""

    def test_load_dataset_valid_buffer(self, sample_valid_df):
        csv_buffer = io.StringIO(sample_valid_df.to_csv(index=False))
        loaded = load_dataset(csv_buffer)
        assert len(loaded) == 6
        assert "health_status" in loaded.columns

    def test_load_dataset_empty_raises(self):
        empty_buffer = io.StringIO("")
        with pytest.raises(ValueError, match="Failed to read CSV dataset"):
            load_dataset(empty_buffer)

    def test_validate_dataset_success(self, sample_valid_df):
        is_valid, issues = validate_dataset(sample_valid_df)
        assert is_valid is True
        assert len(issues) == 0

    def test_validate_dataset_missing_target(self, sample_valid_df):
        df_no_target = sample_valid_df.drop(columns=["health_status"])
        is_valid, issues = validate_dataset(df_no_target)
        assert is_valid is False
        assert any("Missing required target column" in issue for issue in issues)

    def test_validate_dataset_invalid_label(self, sample_valid_df):
        df_bad = sample_valid_df.copy()
        df_bad.loc[0, "health_status"] = "SuperActive"
        is_valid, issues = validate_dataset(df_bad)
        assert is_valid is False
        assert any("Unexpected target labels found" in issue for issue in issues)

    def test_assess_data_quality(self, sample_valid_df):
        quality = assess_data_quality(sample_valid_df)
        assert quality["total_rows"] == 6
        assert quality["total_cols"] == 13
        assert quality["duplicate_rows"] == 0
        assert quality["target_distribution"]["Healthy"] == 3
        assert quality["target_distribution"]["Likely Abandoned"] == 2

    def test_engineered_features(self, sample_valid_df):
        enriched = add_engineered_features(sample_valid_df)
        assert "issue_close_rate" in enriched.columns
        assert "commit_frequency_per_month" in enriched.columns

        # Verify issue_close_rate calculation
        # row 0: closed=150, open=15 -> close rate = 150 / 165 ~ 0.9091
        close_rate_0 = enriched.loc[0, "issue_close_rate"]
        assert 0.90 <= close_rate_0 <= 0.92

        # row 0: commits=120 -> commits per month = 10.0
        assert enriched.loc[0, "commit_frequency_per_month"] == 10.0

    def test_split_and_preprocess_leakage_prevention(self, sample_valid_df):
        # Duplicate to have enough samples for stratified split
        expanded_df = pd.concat([sample_valid_df] * 6, ignore_index=True)
        X, y = prepare_features_and_target(expanded_df)

        split_dict = split_and_preprocess(X, y, test_size=0.25, random_state=42)

        scaler = split_dict["scaler"]
        X_train_raw = split_dict["X_train_raw"]
        X_test_raw = split_dict["X_test_raw"]

        # CRITICAL: Verify scaler mean matches training set mean, NOT overall or test set mean
        expected_train_mean = X_train_raw.mean().values
        np.testing.assert_allclose(scaler.mean_, expected_train_mean, rtol=1e-5)

        # Scaled shapes
        assert split_dict["X_train"].shape[0] == len(X_train_raw)
        assert split_dict["X_test"].shape[0] == len(X_test_raw)
        assert split_dict["X_train"].shape[1] == len(ALL_FEATURE_COLUMNS)

    def test_convert_live_data_to_features(self):
        live_repo_data = {
            "stars_count": 500,
            "forks_count": 80,
            "open_issues_count": 20,
            "open_prs_count": 5,
            "days_since_last_commit": 14,
            "repo_age_days": 730,
            "has_license": 1,
            "has_readme": 1,
            "commits_sample_count": 30,
        }

        feature_df = convert_live_data_to_features(live_repo_data)
        assert feature_df is not None
        assert list(feature_df.columns) == ALL_FEATURE_COLUMNS
        assert len(feature_df) == 1
        assert feature_df["stars_count"].iloc[0] == 500
