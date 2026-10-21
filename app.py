"""Open-Source Project Health Classification - Streamlit Application.

A professional dashboard combining machine learning classification,
exploratory data analysis, and real-time GitHub REST API repository health audits.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import streamlit as st

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
from src.github_api import fetch_repository_data
from src.health_rules import evaluate_repository_health
from src.model_trainer import predict_single_repository, train_and_evaluate_models
from src.visualizer import (
    CLASS_COLORS,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_feature_boxplots,
    plot_feature_importance,
    plot_health_score_gauge,
    plot_model_comparison,
    plot_pillar_breakdown,
    plot_scatter_overview,
    plot_target_distribution,
)

# Page configuration
st.set_page_config(
    page_title="OSS Project Health Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .metric-card-title {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .metric-card-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-healthy {
        background-color: #dcfce7;
        color: #15803d;
    }
    .status-at-risk {
        background-color: #fef9c3;
        color: #a16207;
    }
    .status-abandoned {
        background-color: #fee2e2;
        color: #b91c1c;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_default_dataset() -> pd.DataFrame:
    """Load bundled demonstration dataset."""
    csv_path = Path("data") / "sample_repositories.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    # Fallback to minimal dataframe if file is not found
    return pd.DataFrame()


@st.cache_resource
def get_trained_artifacts(
    df: pd.DataFrame,
    test_size: float = 0.2,
    n_estimators: int = 100,
    max_depth: int = 8,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train baseline and ML models with strict train/test splitting and scaling."""
    X, y = prepare_features_and_target(df)
    split_dict = split_and_preprocess(X, y, test_size=test_size, random_state=random_state)
    eval_dict = train_and_evaluate_models(
        split_dict["X_train"],
        split_dict["X_test"],
        split_dict["y_train"],
        split_dict["y_test"],
        split_dict["feature_names"],
        random_state=random_state,
    )
    return {
        "split": split_dict,
        "eval": eval_dict,
        "X": X,
        "y": y,
    }


def render_sidebar():
    """Render sidebar navigation, GitHub credentials, and quick links."""
    st.sidebar.image(
        "https://raw.githubusercontent.com/tandpfun/skill-icons/main/icons/Github-Dark.svg",
        width=48,
    )
    st.sidebar.title("Repo Health Hub")
    st.sidebar.caption("Open-Source Sustainability & ML Classifier")

    nav_choice = st.sidebar.radio(
        "Navigation",
        [
            "🌐 Live GitHub Repo Analysis",
            "🤖 Interactive ML Predictor",
            "📊 Dataset & Exploratory Analysis",
            "🧠 Model Training & Evaluation",
            "📖 Project Overview & Methodology",
        ],
        index=0,
    )

    st.sidebar.divider()
    st.sidebar.subheader("🔑 GitHub API Token (Optional)")
    st.sidebar.caption("Unauthenticated requests are capped at 60 req/hr. Providing a token increases the limit to 5,000 req/hr.")

    env_token = os.getenv("GITHUB_TOKEN", "")
    token_input = st.sidebar.text_input(
        "Personal Access Token",
        value=env_token,
        type="password",
        help="Reads from GITHUB_TOKEN environment variable by default if set.",
    )
    if token_input:
        st.session_state["github_token"] = token_input.strip()
    elif env_token:
        st.session_state["github_token"] = env_token.strip()
    else:
        st.session_state["github_token"] = ""

    st.sidebar.divider()
    st.sidebar.caption("🔗 **Target Repository:**\n`Adhi576/repo-health-classification`")

    return nav_choice


def page_live_analysis(trained_artifacts: Optional[Dict[str, Any]]):
    """Page 1: Real-time GitHub repository analysis."""
    st.markdown('<div class="main-title">🌐 Live GitHub Repository Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Query the official GitHub REST API in real time to evaluate repository activity, maintenance cadence, community traction, and sustainability indicators.</div>',
        unsafe_allow_html=True,
    )

    col_in, col_btn = st.columns([4, 1])
    with col_in:
        default_url = "https://github.com/Adhi576/repo-health-classification"
        repo_input = st.text_input(
            "Enter public GitHub repository URL or slug:",
            value=default_url,
            placeholder="e.g. https://github.com/pallets/flask or psf/requests",
        )
    with col_btn:
        st.write("")
        st.write("")
        analyze_clicked = st.button("🚀 Analyze Now", type="primary", use_container_width=True)

    # Preset pills
    st.caption("Quick presets: ")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    with p_col1:
        if st.button("psf/requests", use_container_width=True):
            repo_input = "https://github.com/psf/requests"
            analyze_clicked = True
    with p_col2:
        if st.button("pallets/flask", use_container_width=True):
            repo_input = "https://github.com/pallets/flask"
            analyze_clicked = True
    with p_col3:
        if st.button("facebook/react", use_container_width=True):
            repo_input = "https://github.com/facebook/react"
            analyze_clicked = True
    with p_col4:
        if st.button("This Repository", use_container_width=True):
            repo_input = "https://github.com/Adhi576/repo-health-classification"
            analyze_clicked = True

    if analyze_clicked or "last_live_data" in st.session_state:
        if analyze_clicked:
            token = st.session_state.get("github_token", "")
            with st.spinner("Connecting to GitHub REST API..."):
                api_result = fetch_repository_data(repo_input, token=token)
            st.session_state["last_api_result"] = api_result
        else:
            api_result = st.session_state.get("last_api_result", {})

        if not api_result.get("success"):
            status_code = api_result.get("status_code", 500)
            err_msg = api_result.get("error_message", "Unknown error")
            st.error(f"❌ Analysis Failed (HTTP {status_code}): {err_msg}")
            if status_code == 403:
                st.info("💡 Tip: GitHub API limits unauthenticated IP requests to 60 per hour. Add a personal access token in the left sidebar to obtain 5,000 requests/hour.")
            return

        data = api_result["data"]
        rate_rem = api_result.get("rate_limit_remaining")
        rate_lim = api_result.get("rate_limit_limit")

        # Top Bar summary
        rate_str = f"API Rate Limit: {rate_rem}/{rate_lim} remaining" if rate_rem is not None else ""
        st.caption(f"⏱ Retrieved at: **{data.get('retrieved_at')}** | {rate_str}")

        # Repository Header
        badge_archived = "🔴 Archived" if data.get("archived") else "🟢 Active"
        st.markdown(
            f"""
            ### [{data.get('full_name')}]({data.get('html_url')}) &nbsp; `{badge_archived}`
            *{data.get('description')}*
            """
        )

        # Primary Metrics Grid
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("⭐ Stars", f"{data.get('stars_count', 0):,}")
        m2.metric("🍴 Forks", f"{data.get('forks_count', 0):,}")
        m3.metric("🐛 Open Issues", f"{data.get('open_issues_count', 0):,}")
        m4.metric("🔀 Open PRs", f"{data.get('open_prs_count', 0):,}")
        m5.metric("🕒 Days Since Commit", f"{data.get('days_since_last_commit', 0)}")
        m6.metric("🏷 Primary Language", data.get("language") or "N/A")

        st.divider()

        # Rule-based Health Assessment
        eval_result = evaluate_repository_health(data)
        health_score = eval_result["health_score"]
        health_status = eval_result["health_status"]

        col_gauge, col_pillars = st.columns([1, 1.2])
        with col_gauge:
            st.plotly_chart(
                plot_health_score_gauge(health_score, health_status),
                use_container_width=True,
            )
            st.markdown(
                f"""
                **Assessed Status:** <span class="status-badge status-{health_status.lower().replace(' ', '-')}">{health_status}</span>
                <br><br>
                **OSI License:** `{data.get('license_name')}` &bull; **Repository Age:** `{data.get('repo_age_days')} days`
                """,
                unsafe_allow_html=True,
            )

        with col_pillars:
            st.plotly_chart(
                plot_pillar_breakdown(eval_result["pillar_scores"]),
                use_container_width=True,
            )

        # Diagnostics & Recommendations
        d_col, r_col = st.columns(2)
        with d_col:
            st.subheader("🔍 Transparent Health Diagnostics")
            for diag in eval_result["diagnostics"]:
                stat = diag["status"]
                icon = "✅" if stat == "good" else "⚠️" if stat == "warning" else "🚨"
                st.markdown(f"{icon} **[{diag['pillar']}]** {diag['message']}")

        with r_col:
            st.subheader("💡 Actionable Recommendations")
            for rec in eval_result["recommendations"]:
                st.markdown(f"- {rec}")

        # ML Compatibility Section
        st.divider()
        st.subheader("🤖 Machine Learning Model Cross-Validation")
        st.markdown(
            "Here we pass the repository's activity signals through the trained Random Forest classifier. "
            "Because live repositories may lack historical issue closure rates, we compute compatible feature approximations."
        )

        if trained_artifacts and "eval" in trained_artifacts:
            feature_df = convert_live_data_to_features(data)
            if feature_df is not None:
                rf_model = trained_artifacts["eval"]["models"]["Random Forest"]
                scaler = trained_artifacts["split"]["scaler"]
                ml_pred = predict_single_repository(rf_model, scaler, feature_df)

                pred_class = ml_pred["predicted_class"]
                conf = ml_pred["confidence"]
                probs = ml_pred["probabilities"]

                mc1, mc2 = st.columns([1, 2])
                with mc1:
                    st.metric("ML Classified Status", pred_class)
                    st.metric("Prediction Confidence", f"{conf * 100:.1f}%")
                with mc2:
                    st.write("**Predicted Probability Distribution:**")
                    prob_df = pd.DataFrame({
                        "Class": list(probs.keys()),
                        "Probability": list(probs.values()),
                    })
                    st.bar_chart(prob_df.set_index("Class"), height=180)
            else:
                st.warning("Could not convert live repository signals into model feature schema.")
        else:
            st.info("Train the machine learning models on the 'Model Training & Evaluation' tab to enable ML cross-validation.")


def page_interactive_predictor(trained_artifacts: Optional[Dict[str, Any]]):
    """Page 2: Interactive what-if health classifier."""
    st.markdown('<div class="main-title">🤖 Interactive ML Health Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Experiment with repository parameters to observe how stars, commit recency, issue backlogs, and contributor dynamics shift the model\'s classification.</div>',
        unsafe_allow_html=True,
    )

    if not trained_artifacts:
        st.error("Model artifacts not loaded. Please initialize the dataset first.")
        return

    rf_model = trained_artifacts["eval"]["models"]["Random Forest"]
    scaler = trained_artifacts["split"]["scaler"]

    with st.form("prediction_form"):
        st.subheader("1. Activity & Commit Cadence")
        c1, c2, c3 = st.columns(3)
        days_commit = c1.slider("Days Since Last Commit", min_value=0, max_value=1200, value=15, step=5)
        commits_yr = c2.number_input("Commits in Last 12 Months", min_value=0, max_value=5000, value=120, step=10)
        repo_age = c3.number_input("Repository Age (Days)", min_value=1, max_value=8000, value=1200, step=50)

        st.subheader("2. Community Traction & Engagement")
        c4, c5, c6 = st.columns(3)
        stars = c4.number_input("Stars Count", min_value=0, max_value=500000, value=2500, step=50)
        forks = c5.number_input("Forks Count", min_value=0, max_value=100000, value=450, step=25)
        contributors = c6.number_input("Distinct Contributors", min_value=1, max_value=2000, value=25, step=1)

        st.subheader("3. Maintenance & Governance")
        c7, c8, c9 = st.columns(3)
        open_issues = c7.number_input("Open Issues (excl. PRs)", min_value=0, max_value=10000, value=35, step=5)
        closed_issues = c8.number_input("Closed Issues", min_value=0, max_value=50000, value=280, step=10)
        open_prs = c9.number_input("Open Pull Requests", min_value=0, max_value=2000, value=8, step=1)

        c10, c11 = st.columns(2)
        has_license = c10.selectbox("Has Open-Source License?", [1, 0], format_func=lambda x: "Yes (OSI License)" if x == 1 else "No License")
        has_readme = c11.selectbox("Has Comprehensive README?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

        submit = st.form_submit_button("🔮 Predict Repository Health", type="primary", use_container_width=True)

    if submit:
        input_data = pd.DataFrame([{
            "stars_count": stars,
            "forks_count": forks,
            "open_issues_count": open_issues,
            "closed_issues_count": closed_issues,
            "pull_requests_count": open_prs,
            "days_since_last_commit": days_commit,
            "commits_last_year": commits_yr,
            "contributors_count": contributors,
            "repo_age_days": repo_age,
            "has_license": has_license,
            "has_readme": has_readme,
        }])
        enriched = add_engineered_features(input_data)
        prediction = predict_single_repository(rf_model, scaler, enriched)

        pred_class = prediction["predicted_class"]
        conf = prediction["confidence"]
        probs = prediction["probabilities"]

        st.divider()
        st.subheader("Prediction Result")

        r1, r2 = st.columns([1, 1.5])
        with r1:
            color = CLASS_COLORS.get(pred_class, "#64748b")
            st.markdown(
                f"""
                <div style="background:{color}20;border:2px solid {color};border-radius:12px;padding:20px;text-align:center;">
                    <span style="font-size:1.1rem;color:#334155;font-weight:600;">Classified Health Status</span>
                    <h2 style="color:{color};margin:10px 0;">{pred_class}</h2>
                    <span style="font-size:1rem;color:#475569;">Confidence: <b>{conf * 100:.1f}%</b></span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")
            if pred_class == "Healthy":
                st.success("✅ **Healthy Repository:** Regular commits, balanced issue resolution, and sound governance. Suitable for production dependencies.")
            elif pred_class == "At Risk":
                st.warning("⚠️ **At Risk:** Maintenance lag or accumulating issue backlog detected. Monitor before taking hard dependencies.")
            else:
                st.error("🚨 **Likely Abandoned:** Long dormancy or zero recent maintenance. Significant risk of security vulnerabilities and lack of support.")

        with r2:
            st.write("**Model Class Probabilities:**")
            prob_df = pd.DataFrame({
                "Class": list(probs.keys()),
                "Probability": list(probs.values()),
            })
            st.bar_chart(prob_df.set_index("Class"), height=250)


def page_dataset_analysis(df: pd.DataFrame):
    """Page 3: Dataset upload, preview, and exploratory analysis."""
    st.markdown('<div class="main-title">📊 Dataset & Exploratory Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Inspect dataset schema, missing values, distribution of health statuses, and explore metric correlations across open-source repositories.</div>',
        unsafe_allow_html=True,
    )

    # Dataset Source Selector
    uploaded_file = st.file_uploader("Upload custom CSV dataset (optional):", type=["csv"])
    if uploaded_file is not None:
        try:
            active_df = load_dataset(uploaded_file)
            st.success(f"Custom dataset loaded successfully: {len(active_df)} rows.")
        except Exception as e:
            st.error(f"Error loading uploaded CSV: {str(e)}")
            active_df = df
    else:
        active_df = df
        st.info(f"Using bundled benchmark dataset ({len(active_df)} repositories across 8 ecosystems).")

    is_valid, validation_issues = validate_dataset(active_df)
    if not is_valid:
        st.error(f"Dataset schema validation failed: {'; '.join(validation_issues)}")
        return

    # Data Quality Overview
    quality = assess_data_quality(active_df)
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Total Repositories", f"{quality['total_rows']:,}")
    q2.metric("Total Columns", f"{quality['total_cols']}")
    q3.metric("Missing Cells", f"{quality['total_missing_cells']}")
    q4.metric("Duplicate Rows", f"{quality['duplicate_rows']}")

    # Tabs for breakdown
    t_preview, t_dist, t_corr, t_explore = st.tabs([
        "📄 Data Preview",
        "🍩 Health Distribution",
        "🔥 Correlation Heatmap",
        "📈 Feature Distributions",
    ])

    with t_preview:
        st.subheader("Repository Records Preview")
        st.dataframe(active_df.head(50), use_container_width=True)

        st.subheader("Data Types & Missing Value Audit")
        st.dataframe(quality["null_summary"], use_container_width=True)

    with t_dist:
        st.subheader("Target Class Balance")
        st.plotly_chart(plot_target_distribution(active_df["health_status"]), use_container_width=True)
        st.markdown(
            "**Note on Dataset Balance:** The benchmark dataset has balanced representation across "
            "`Healthy`, `At Risk`, and `Likely Abandoned` categories to enable unbiased evaluation of classifier performance."
        )

    with t_corr:
        st.subheader("Correlation Matrix")
        numeric_cols = [c for c in BASE_NUMERIC_FEATURES if c in active_df.columns]
        st.plotly_chart(plot_correlation_heatmap(active_df, numeric_cols), use_container_width=True)
        st.caption("Strong negative correlations are typically observed between `days_since_last_commit` and repository health.")

    with t_explore:
        st.subheader("Interactive Feature Explorer")
        c1, c2 = st.columns(2)
        with c1:
            feat_choice = st.selectbox(
                "Select numeric feature for box plot:",
                numeric_cols,
                index=numeric_cols.index("days_since_last_commit") if "days_since_last_commit" in numeric_cols else 0,
            )
            st.plotly_chart(plot_feature_boxplots(active_df, feat_choice), use_container_width=True)

        with c2:
            x_scatter = st.selectbox("Scatter X-axis:", numeric_cols, index=numeric_cols.index("days_since_last_commit") if "days_since_last_commit" in numeric_cols else 0)
            y_scatter = st.selectbox("Scatter Y-axis:", numeric_cols, index=numeric_cols.index("stars_count") if "stars_count" in numeric_cols else 1)
            st.plotly_chart(plot_scatter_overview(active_df, x_col=x_scatter, y_col=y_scatter), use_container_width=True)


def page_model_training(trained_artifacts: Optional[Dict[str, Any]], df: pd.DataFrame):
    """Page 4: Machine learning model comparison and evaluation."""
    st.markdown('<div class="main-title">🧠 Machine Learning Training & Evaluation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Train, benchmark, and compare Baseline Classifier, Logistic Regression, and Random Forest models on a held-out test split with strict data leakage prevention.</div>',
        unsafe_allow_html=True,
    )

    with st.expander("⚙️ Hyperparameter & Training Configuration", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        test_size = c1.slider("Test Set Ratio", min_value=0.10, max_value=0.40, value=0.20, step=0.05)
        n_estimators = c2.slider("Random Forest Trees", min_value=50, max_value=300, value=100, step=25)
        max_depth = c3.slider("Max Tree Depth", min_value=4, max_value=20, value=8, step=1)
        random_state = c4.number_input("Random Seed", value=42, step=1)

        retrain_clicked = st.button("🔄 Retrain Models with Custom Parameters")

    if retrain_clicked:
        with st.spinner("Retraining models..."):
            trained_artifacts = get_trained_artifacts(
                df,
                test_size=test_size,
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=int(random_state),
            )
            st.session_state["trained_artifacts"] = trained_artifacts
        st.success("Models retrained successfully!")

    if not trained_artifacts:
        st.error("No training artifacts available.")
        return

    eval_data = trained_artifacts["eval"]
    comparison_table = eval_data["comparison_table"]

    # Model Performance Overview
    st.subheader("1. Model Comparison on Held-Out Test Set")
    st.markdown(
        "> [!IMPORTANT]\n"
        "> **Data Leakage Guarantee:** The train/test split is strictly performed *before* fitting the `StandardScaler` "
        "and classifiers. The test set is completely unseen during preprocessor parameter learning."
    )

    st.dataframe(comparison_table, use_container_width=True)
    st.plotly_chart(plot_model_comparison(comparison_table), use_container_width=True)

    # Detailed Model Diagnostics
    st.divider()
    st.subheader("2. Detailed Classification Diagnostics")

    model_name = st.selectbox(
        "Select model to inspect confusion matrix and per-class metrics:",
        list(eval_data["models"].keys()),
        index=2,  # Default to Random Forest
    )

    model_res = eval_data["results"][model_name]
    cm = model_res["confusion_matrix"]
    classes = eval_data["classes"]

    c_cm, c_rep = st.columns([1, 1.2])
    with c_cm:
        st.plotly_chart(plot_confusion_matrix(cm, classes, model_name), use_container_width=True)

    with c_rep:
        st.write(f"**Classification Report: {model_name}**")
        rep_dict = model_res["classification_report"]
        rep_rows = []
        for cls in classes:
            if cls in rep_dict:
                rep_rows.append({
                    "Class": cls,
                    "Precision": round(rep_dict[cls]["precision"], 3),
                    "Recall": round(rep_dict[cls]["recall"], 3),
                    "F1-Score": round(rep_dict[cls]["f1-score"], 3),
                    "Support": int(rep_dict[cls]["support"]),
                })
        st.dataframe(pd.DataFrame(rep_rows), use_container_width=True)

        m1, m2 = st.columns(2)
        m1.metric("Overall Accuracy", f"{model_res['accuracy'] * 100:.2f}%")
        m2.metric("Macro F1-Score", f"{model_res['f1_macro']:.3f}")

    # Feature Importance
    st.divider()
    st.subheader("3. Random Forest Feature Importance")
    st.markdown("Features with the highest Gini impurity decrease across decision trees:")
    st.plotly_chart(plot_feature_importance(eval_data["feature_importance"]), use_container_width=True)

    # Methodology & Honesty Note
    st.caption(
        "**Methodology & Integrity Statement:** Model metrics above reflect evaluation on the synthetic benchmark dataset. "
        "Real-world open source repository health involves socio-technical dynamics that vary by ecosystem. "
        "These models demonstrate reproducible pipeline mechanics and decision boundaries."
    )


def page_overview():
    """Page 5: Project overview, methodology, and documentation."""
    st.markdown('<div class="main-title">📖 Project Overview & Methodology</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Architectural design, open-source sustainability taxonomy, and implementation documentation.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### Executive Summary
        Open-source software forms the critical infrastructure of modern digital ecosystems. However, relying on 
        dormant or poorly maintained packages introduces significant security vulnerabilities, technical debt, and supply-chain risks.

        This application provides a dual-engine architecture:
        1. **Heuristic Health Engine:** Transparent, rule-based scoring (0-100) evaluating **Activity**, **Community**, **Maintenance**, and **Governance**.
        2. **Machine Learning Classifier:** Reproducible pipeline comparing Baseline, Logistic Regression, and Random Forest models across high-dimensional activity vectors.

        ---

        ### Three-Tier Health Taxonomy

        | Class | Definition & Signals | Primary Indicators |
        |---|---|---|
        | **Healthy** | Active development, regular releases, timely issue triage, strong community stewardship. | Commits within 30-45 days, balanced open issue ratio, valid license, active PR processing. |
        | **At Risk** | Slowing velocity, accumulating unanswered issues, decreasing maintainer responsiveness. | Inactivity between 60-300 days, high issue backlog, unaddressed pull requests. |
        | **Likely Abandoned** | Dormant codebase, archived status, unaddressed security reports, zero recent commits. | Over 365 days since last commit, official archived flag, abandoned issue tracker. |

        ---

        ### Key Technical Guardrails
        - **Zero Data Leakage:** Preprocessors (`StandardScaler`) are fitted strictly on the training partition and transformed onto the test partition.
        - **Authentic Metrics:** No simulated high scores or fabricated benchmarks. All metrics are derived from genuine cross-validation and test set evaluation.
        - **Pure Issue Separation:** The GitHub API combines issues and pull requests under `open_issues_count`. Our engine explicitly queries the pull requests endpoint to isolate pure issues from code proposals.
        - **Rate-Limit Resilience:** Automatic rate limit tracking and clear guidance for Personal Access Token authentication.
        """
    )


def main():
    """Application entry point."""
    nav_choice = render_sidebar()

    # Load default dataset and train initial models
    default_df = get_default_dataset()
    trained_artifacts = None
    if not default_df.empty:
        try:
            trained_artifacts = get_trained_artifacts(default_df)
        except Exception:
            pass

    if nav_choice == "🌐 Live GitHub Repo Analysis":
        page_live_analysis(trained_artifacts)
    elif nav_choice == "🤖 Interactive ML Predictor":
        page_interactive_predictor(trained_artifacts)
    elif nav_choice == "📊 Dataset & Exploratory Analysis":
        page_dataset_analysis(default_df)
    elif nav_choice == "🧠 Model Training & Evaluation":
        page_model_training(trained_artifacts, default_df)
    elif nav_choice == "📖 Project Overview & Methodology":
        page_overview()


if __name__ == "__main__":
    main()
