"""Machine learning model training, evaluation, comparison, and inference module.

Compares Baseline Classifier, Logistic Regression, and Random Forest.
Calculates standard evaluation metrics and prevents metric fabrication.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data_processor import ALL_FEATURE_COLUMNS, VALID_CLASSES


def train_and_evaluate_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: List[str],
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train baseline and machine learning models, then compute evaluation metrics.

    Models trained:
      1. Baseline Classifier: DummyClassifier (predicts most frequent class)
      2. Logistic Regression: Linear model with L2 regularization
      3. Random Forest Classifier: Ensemble of decision trees

    Args:
        X_train: Scaled training feature array.
        X_test: Scaled test feature array.
        y_train: Training target series.
        y_test: Test target series.
        feature_names: List of feature names.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary containing trained models, comparison metrics table,
        confusion matrices, classification reports, and feature importances.
    """
    models: Dict[str, Any] = {
        "Baseline (Most Frequent)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=random_state,
            class_weight="balanced",
        ),
    }

    results: Dict[str, Any] = {}
    comparison_records: List[Dict[str, Any]] = []

    for name, model in models.items():
        # Fit model strictly on training split
        model.fit(X_train, y_train)

        # Predict on held-out test split
        y_pred = model.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        cm = confusion_matrix(y_test, y_pred, labels=VALID_CLASSES)
        report_dict = classification_report(
            y_test,
            y_pred,
            labels=VALID_CLASSES,
            output_dict=True,
            zero_division=0,
        )

        results[name] = {
            "model": model,
            "accuracy": round(acc, 4),
            "precision_macro": round(prec_macro, 4),
            "recall_macro": round(rec_macro, 4),
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
            "confusion_matrix": cm,
            "classification_report": report_dict,
            "y_pred": y_pred,
        }

        comparison_records.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision (Macro)": round(prec_macro, 4),
            "Recall (Macro)": round(rec_macro, 4),
            "F1-Score (Macro)": round(f1_macro, 4),
            "F1-Score (Weighted)": round(f1_weighted, 4),
        })

    comparison_df = pd.DataFrame(comparison_records)

    # Feature Importance from Random Forest
    rf_model = models["Random Forest"]
    importances = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    return {
        "models": models,
        "results": results,
        "comparison_table": comparison_df,
        "feature_importance": feature_importance_df,
        "classes": VALID_CLASSES,
    }


def predict_single_repository(
    model: Any,
    scaler: Any,
    feature_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Perform model inference on a single repository feature vector.

    Args:
        model: Trained scikit-learn classifier.
        scaler: Fitted StandardScaler instance.
        feature_df: DataFrame containing the columns defined in ALL_FEATURE_COLUMNS.

    Returns:
        Dictionary containing predicted class, probability distribution, and confidence.
    """
    # Ensure correct columns and order
    missing = [c for c in ALL_FEATURE_COLUMNS if c not in feature_df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns for prediction: {missing}")

    X_input = feature_df[ALL_FEATURE_COLUMNS].copy()
    X_scaled = scaler.transform(X_input)

    predicted_class = str(model.predict(X_scaled)[0])

    probabilities: Dict[str, float] = {}
    confidence = 1.0

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)[0]
        classes = list(model.classes_)
        for cls_name, prob in zip(classes, probs):
            probabilities[str(cls_name)] = round(float(prob), 4)
        confidence = float(max(probs))
    else:
        # For classifiers without predict_proba
        for cls in VALID_CLASSES:
            probabilities[cls] = 1.0 if cls == predicted_class else 0.0

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "probabilities": probabilities,
    }
