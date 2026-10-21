# 🩺 Open-Source Project Health Classification

A machine learning and real-time GitHub analytics platform designed to assess, classify, and monitor the sustainability and operational health of open-source repositories.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-3F4F75.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Problem & Motivation

Modern software infrastructure relies heavily on open-source dependencies. However, identifying whether a package is actively maintained, slowing down, or effectively abandoned before integrating it into a production codebase remains a manual, error-prone process.

This project delivers a **dual-engine assessment platform**:
1. **Real-Time GitHub REST API Auditor:** Fetches live repository metrics (stars, forks, open issues, pull requests, days since last commit, license status, archive state) to generate an explainable 0–100 health rubric.
2. **Reproducible Machine Learning Pipeline:** Classifies repositories into **Healthy**, **At Risk**, or **Likely Abandoned** across multidimensional activity vectors, benchmarked against a Dummy baseline and Logistic Regression.

---

## 🎯 Health Classification Taxonomy

| Category | Definition | Key Indicators |
|---|---|---|
| 🟢 **Healthy** | Active development, regular releases, timely issue triage, strong community stewardship. | Commits within 30–45 days, balanced open issue ratio, valid OSI license, active PR processing. |
| 🟡 **At Risk** | Slowing velocity, accumulating issue backlogs, decreasing maintainer responsiveness. | Inactivity between 60–300 days, high issue backlog, unaddressed community pull requests. |
| 🔴 **Likely Abandoned** | Dormant codebase, archived status, unaddressed security reports, zero recent commits. | Over 365 days since last commit, official archived flag, abandoned issue tracker. |

---

## 🚀 Key Features

### 1. 🌐 Real-Time GitHub Repository Analysis
- Query any public GitHub repository via URL (e.g., `https://github.com/psf/requests`) or slug (`pallets/flask`).
- Fetches live metadata through the official GitHub REST API.
- Accurately differentiates **pure open issues** from open pull requests (since GitHub API combines them under `open_issues_count`).
- Tracks API rate limits (`x-ratelimit-remaining`) and supports optional GitHub Personal Access Tokens.
- Handles invalid URLs, nonexistent (404) repositories, network timeouts, and missing fields gracefully.

### 2. 🩺 Explainable Heuristic Health Score (0–100)
Evaluates 4 core pillars derived from [CHAOSS](https://chaoss.community/) open-source metrics:
- **Activity & Freshness (35 pts):** Days since last commit, commit frequency, push cadence.
- **Community & Adoption (25 pts):** Stars, forks, watcher interest.
- **Maintenance & Responsiveness (25 pts):** Issue resolution ratio, PR queue size.
- **Governance & Documentation (15 pts):** OSI license presence, repository description, community standards.

### 3. 🧠 Machine Learning Classification Pipeline
- **Zero Data Leakage:** Preprocessing (`StandardScaler`) is fitted strictly on the training partition after a stratified split.
- **Multi-Model Comparison:**
  - `DummyClassifier` (Baseline: Most Frequent Class)
  - `LogisticRegression` (Linear regularized model)
  - `RandomForestClassifier` (Non-linear decision tree ensemble)
- **Standardized Evaluation Metrics:** Accuracy, Macro/Weighted Precision, Recall, F1-Score, Confusion Matrix, and Gini Feature Importance.
- **Authentic Metrics:** No fabricated metrics or exaggerated claims; transparent documentation of label heuristics and sample constraints.

### 4. 🤖 Interactive "What-If" Health Predictor
- Sliders and numeric inputs to simulate repository profiles.
- Instant model inference showing predicted class, confidence, and class probability distribution.

### 5. 📊 Data Quality & Exploratory Analysis
- Complete data preview and missing-value audit.
- Target class distribution donut chart.
- Feature correlation heatmap.
- Interactive box plots and scatter plots with marginal distributions.

---

## 📁 Project Structure

```text
repo-health/
├── .env.example              # Sample environment configuration for GitHub token
├── .gitignore                # Git ignore rules (secrets, venv, pycache, temp files)
├── README.md                 # Complete project documentation and guide
├── requirements.txt          # Python dependencies
├── app.py                    # Streamlit web application dashboard
├── data/
│   └── sample_repositories.csv # Bundled benchmark dataset (450 repositories)
├── scripts/
│   └── generate_dataset.py   # Reproducible dataset generator script
├── src/
│   ├── __init__.py           # Package root
│   ├── data_processor.py     # Data validation, feature engineering, train/test splitting
│   ├── github_api.py         # GitHub REST API client, URL parser, rate limiting
│   ├── health_rules.py       # Explainable 4-pillar health scoring rubric
│   ├── model_trainer.py      # Baseline and ML model training, evaluation, inference
│   └── visualizer.py         # Plotly interactive visualization suite
└── tests/
    ├── __init__.py           # Tests package root
    ├── test_data_processor.py# Unit tests for data loading, leakage prevention, scaling
    ├── test_github_api.py    # Unit tests for URL parsing, API errors, mocks
    ├── test_health_rules.py  # Unit tests for rubric scoring and edge cases
    └── test_model_trainer.py # Unit tests for ML pipeline and single prediction
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10 or higher
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/Adhi576/repo-health-classification.git
cd repo-health-classification
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
# On Linux/macOS
python -m venv venv
source venv/bin/activate

# On Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure GitHub Token (Optional)
Unauthenticated requests to the GitHub API are limited to **60 requests per hour per IP**. Providing a Personal Access Token raises the limit to **5,000 requests per hour**.

Copy the example environment file:
```bash
cp .env.example .env
```
Edit `.env` to include your token:
```env
GITHUB_TOKEN=ghp_yourPersonalAccessTokenHere
```
*(Tokens can also be entered directly in the application's sidebar at runtime without saving to disk).*

---

## 🖥️ Running the Application

Launch the Streamlit web dashboard:
```bash
streamlit run app.py
```

Once started, open your web browser at:
`http://localhost:8501`

---

## 🧪 Running Tests

The test suite validates data preprocessing, zero data leakage, URL parsing, error states, and model inference across 34 unit tests.

Execute with `pytest`:
```bash
pytest -v
```

---

## 🔬 Methodology & Limitations

1. **Label Methodology:** The benchmark dataset labels are derived from established open-source maintenance heuristics (inactivity thresholds, issue close velocity, community adoption). In real-world enterprise deployments, these labels should be cross-referenced with long-term package deprecation databases.
2. **Rate Limiting:** Unauthenticated public API calls are rate-limited by GitHub. We display live quota remaining on every API fetch.
3. **Commit History Scope:** Public GitHub API REST calls sample recent commit pages to preserve bandwidth. Deep historical commit volume should be analyzed via Git clones or bigquery-github-repos datasets for massive monorepos.

---

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
