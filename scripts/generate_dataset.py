"""Generator script for realistic open-source project benchmark dataset.

Generates 450 repository records spanning Python, JavaScript, Go, Rust, Java,
TypeScript, and C++ with transparently documented health metrics.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd

np.random.seed(42)

LANGUAGES = ["Python", "JavaScript", "TypeScript", "Go", "Rust", "Java", "C++", "Ruby"]

HEALTHY_NAMES = [
    "django/django", "psf/requests", "pallets/flask", "fastapi/fastapi", "scikit-learn/scikit-learn",
    "facebook/react", "vercel/next.js", "vuejs/core", "angular/angular", "tailwindlabs/tailwindcss",
    "golang/go", "kubernetes/kubernetes", "moby/moby", "gin-gonic/gin", "prometheus/prometheus",
    "rust-lang/rust", "tokio-rs/tokio", "BurntSushi/ripgrep", "sharkdp/bat", "alacritty/alacritty",
    "spring-projects/spring-boot", "apache/spark", "elastic/elasticsearch", "quarkusio/quarkus",
    "protocolbuffers/protobuf", "opencv/opencv", "pytorch/pytorch", "pandas-dev/pandas", "numpy/numpy"
]

def generate_dataset(n_samples: int = 450) -> pd.DataFrame:
    rows = []
    n_per_class = n_samples // 3

    # 1. Healthy Repositories (Active, responsive, maintained)
    for i in range(n_per_class):
        base_name = HEALTHY_NAMES[i % len(HEALTHY_NAMES)] if i < len(HEALTHY_NAMES) else f"community-org/healthy-tool-{i+1}"
        stars = int(np.random.exponential(scale=12000) + 150)
        forks = int(stars * np.random.uniform(0.12, 0.35))
        days_commit = int(np.random.exponential(scale=12) + 1)
        days_commit = min(days_commit, 50)
        commits_yr = int(np.random.exponential(scale=250) + 40)
        open_issues = int(np.random.exponential(scale=80) + 5)
        closed_issues = int(open_issues * np.random.uniform(2.5, 9.0) + 20)
        open_prs = int(np.random.exponential(scale=18) + 2)
        contributors = int(np.random.exponential(scale=45) + 6)
        repo_age = int(np.random.uniform(300, 4500))
        has_license = 1 if np.random.rand() > 0.03 else 0
        has_readme = 1
        lang = np.random.choice(LANGUAGES)

        rows.append({
            "repo_name": base_name if i < len(HEALTHY_NAMES) else f"active-oss/project-{i+1}",
            "language": lang,
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
            "health_status": "Healthy",
        })

    # 2. At Risk Repositories (Slowing activity, accumulating issues, backlog)
    for i in range(n_per_class):
        stars = int(np.random.exponential(scale=1800) + 30)
        forks = int(stars * np.random.uniform(0.08, 0.22))
        days_commit = int(np.random.uniform(65, 340))
        commits_yr = int(np.random.uniform(3, 35))
        open_issues = int(np.random.exponential(scale=120) + 30)
        closed_issues = int(open_issues * np.random.uniform(0.2, 1.1) + 5)
        open_prs = int(np.random.exponential(scale=25) + 8)
        contributors = int(np.random.uniform(1, 12))
        repo_age = int(np.random.uniform(400, 3800))
        has_license = 1 if np.random.rand() > 0.15 else 0
        has_readme = 1 if np.random.rand() > 0.05 else 0
        lang = np.random.choice(LANGUAGES)

        rows.append({
            "repo_name": f"stagnant-team/utility-{i+1}",
            "language": lang,
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
            "health_status": "At Risk",
        })

    # 3. Likely Abandoned Repositories (Dormant > 1yr, zero recent commits, unhandled issues)
    for i in range(n_samples - (2 * n_per_class)):
        stars = int(np.random.exponential(scale=400) + 5)
        forks = int(stars * np.random.uniform(0.02, 0.15))
        days_commit = int(np.random.uniform(380, 2200))
        commits_yr = int(np.random.choice([0, 0, 0, 1, 2]))
        open_issues = int(np.random.exponential(scale=45) + 5)
        closed_issues = int(np.random.uniform(0, 15))
        open_prs = int(np.random.exponential(scale=10) + 1)
        contributors = int(np.random.choice([1, 1, 2, 3]))
        repo_age = int(np.random.uniform(600, 4200))
        has_license = 1 if np.random.rand() > 0.35 else 0
        has_readme = 1 if np.random.rand() > 0.20 else 0
        lang = np.random.choice(LANGUAGES)

        rows.append({
            "repo_name": f"legacy-archive/abandoned-pkg-{i+1}",
            "language": lang,
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
            "health_status": "Likely Abandoned",
        })

    df = pd.DataFrame(rows)
    # Shuffle
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df

if __name__ == "__main__":
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = generate_dataset(450)
    out_path = out_dir / "sample_repositories.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} sample repository records to {out_path}")
    print("Class breakdown:\n", df["health_status"].value_counts())
