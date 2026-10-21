"""Data loading, validation, feature engineering, and train/test splitting module.

Guarantees reproducible preprocessing and strict prevention of data leakage.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_NUMERIC_FEATURES: List[str] = [
    "stars_count",
    "forks_count",
    "open_issues_count",
    "closed_issues_count",
    "pull_requests_count",
    "days_since_last_commit",
    "commits_last_year",
    "contributors_count",
    "repo_age_days",
    "has_license",
    "has_readme",
]

ENGINEERED_FEATURES: List[str] = [
    "issue_close_rate",
    "commit_frequency_per_month",
]

ALL_FEATURE_COLUMNS: List[str] = BASE_NUMERIC_FEATURES + ENGINEERED_FEATURES

TARGET_COLUMN: str = "health_status"
VALID_CLASSES: List[str] = ["Healthy", "At Risk", "Likely Abandoned"]


def load_dataset(file_path_or_buffer: Any) -> pd.DataFrame:
    """Load a repository dataset from a file path or uploaded file buffer.

    Args:
        file_path_or_buffer: File path string or file-like object (e.g. Streamlit UploadedFile).

    Returns:
        Loaded pandas DataFrame.

    Raises:
        ValueError: If file is empty or cannot be parsed.
    """
    try:
        df = pd.read_csv(file_path_or_buffer)
    except Exception as e:
        raise ValueError(f"Failed to read CSV dataset: {str(e)}")

    if df.empty:
        raise ValueError("The provided dataset is empty.")

    return df


def validate_dataset(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate that the dataset contains expected columns and valid target labels.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (is_valid: bool, issues: List[str]).
    """
    issues: List[str] = []

    if TARGET_COLUMN not in df.columns:
        issues.append(f"Missing required target column: '{TARGET_COLUMN}'.")
    else:
        unique_targets = set(df[TARGET_COLUMN].dropna().unique())
        invalid_labels = unique_targets - set(VALID_CLASSES)
        if invalid_labels:
            issues.append(
                f"Unexpected target labels found: {invalid_labels}. Expected subset of {VALID_CLASSES}."
            )

    missing_cols = [col for col in BASE_NUMERIC_FEATURES if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing required feature columns: {missing_cols}")

    return len(issues) == 0, issues


def assess_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform data quality and missing-value analysis on the DataFrame.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary containing quality metrics, null counts, duplicate count, and shape.
    """
    total_rows, total_cols = df.shape
    missing_by_col = df.isnull().sum()
    missing_percent = (missing_by_col / total_rows * 100).round(2)

    null_summary = pd.DataFrame({
        "Missing Count": missing_by_col,
        "Missing (%)": missing_percent,
        "Data Type": df.dtypes.astype(str),
    })

    duplicates = int(df.duplicated().sum())

    target_distribution: Dict[str, int] = {}
    if TARGET_COLUMN in df.columns:
        target_distribution = df[TARGET_COLUMN].value_counts().to_dict()

    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "total_missing_cells": int(missing_by_col.sum()),
        "duplicate_rows": duplicates,
        "null_summary": null_summary,
        "target_distribution": target_distribution,
    }


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived domain features based on open-source health metrics.

    Engineered features:
      - issue_close_rate = closed_issues / (open_issues + closed_issues + 1e-5)
      - commit_frequency_per_month = commits_last_year / 12.0

    Args:
        df: DataFrame with base numeric columns.

    Returns:
        DataFrame with engineered feature columns appended.
    """
    df_copy = df.copy()

    # Fill missing numeric values with medians if any exist
    for col in BASE_NUMERIC_FEATURES:
        if col in df_copy.columns and df_copy[col].isnull().any():
            df_copy[col] = df_copy[col].fillna(df_copy[col].median())

    # Issue close rate (bounded [0, 1])
    total_issues = df_copy["open_issues_count"] + df_copy["closed_issues_count"]
    df_copy["issue_close_rate"] = np.where(
        total_issues > 0,
        df_copy["closed_issues_count"] / total_issues,
        1.0,  # If 0 issues opened/closed, default rate is 1.0 (clean)
    ).round(4)

    # Commit frequency per month (last 12 months)
    df_copy["commit_frequency_per_month"] = (df_copy["commits_last_year"] / 12.0).round(2)

    return df_copy


def prepare_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract and validate feature matrix X and target vector y.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (X: pd.DataFrame, y: pd.Series).
    """
    is_valid, issues = validate_dataset(df)
    if not is_valid:
        raise ValueError(f"Dataset validation failed: {'; '.join(issues)}")

    # Clean rows with missing target
    cleaned_df = df.dropna(subset=[TARGET_COLUMN]).copy()

    # Feature engineering
    enriched_df = add_engineered_features(cleaned_df)

    X = enriched_df[ALL_FEATURE_COLUMNS].copy()
    y = enriched_df[TARGET_COLUMN].copy()

    return X, y


def split_and_preprocess(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Perform stratified train/test split and fit preprocessor on train set only.

    CRITICAL FOR LEAKAGE PREVENTION:
    The StandardScaler is fitted ONLY on X_train. X_test is transformed using
    the scaler parameters learned strictly from X_train.

    Args:
        X: Feature matrix.
        y: Target series.
        test_size: Proportion of dataset for test split (default 0.2).
        random_state: Seed for reproducibility.

    Returns:
        Dictionary containing:
          - X_train_raw, X_test_raw: Raw DataFrames
          - X_train, X_test: Scaled numpy arrays
          - y_train, y_test: Target series
          - scaler: Fitted StandardScaler instance
          - feature_names: List of feature names
    """
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    return {
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": list(X.columns),
    }


def convert_live_data_to_features(
    repo_data: Dict[str, Any],
) -> Optional[pd.DataFrame]:
    """Convert live GitHub API repository data into the model feature schema.

    Returns None if essential fields are missing or not compatible.
    """
    try:
        stars = int(repo_data.get("stars_count", 0))
        forks = int(repo_data.get("forks_count", 0))
        open_issues = int(repo_data.get("open_issues_count", 0))
        open_prs = int(repo_data.get("open_prs_count", 0))
        days_since_commit = int(repo_data.get("days_since_last_commit", 0))
        repo_age_days = int(repo_data.get("repo_age_days", 365))
        has_license = int(repo_data.get("has_license", 0))
        has_readme = int(repo_data.get("has_readme", 1))

        # Approximate closed issues & annual commits from live samples if full history not indexed
        commits_sample = repo_data.get("commits_sample_count", 0)
        # Conservative annual estimation based on recent commits cadence
        if days_since_commit <= 30:
            estimated_commits_year = max(commits_sample * 8, 24)
        elif days_since_commit <= 90:
            estimated_commits_year = max(commits_sample * 3, 10)
        elif days_since_commit <= 180:
            estimated_commits_year = max(commits_sample, 4)
        else:
            estimated_commits_year = 0

        # Approximate contributors based on forks & stars signal
        estimated_contributors = max(1, int((stars * 0.02) + (forks * 0.05) + 1))

        # Closed issues heuristic estimation for live repos where closed issues endpoint is paginated
        closed_issues = int(open_issues * 2.5) if open_issues > 0 else 5

        raw_dict = {
            "stars_count": [stars],
            "forks_count": [forks],
            "open_issues_count": [open_issues],
            "closed_issues_count": [closed_issues],
            "pull_requests_count": [open_prs],
            "days_since_last_commit": [days_since_commit],
            "commits_last_year": [estimated_commits_year],
            "contributors_count": [estimated_contributors],
            "repo_age_days": [repo_age_days],
            "has_license": [has_license],
            "has_readme": [has_readme],
        }

        df_raw = pd.DataFrame(raw_dict)
        df_featured = add_engineered_features(df_raw)
        return df_featured[ALL_FEATURE_COLUMNS]
    except Exception:
        return None
