"""Transparent rule-based heuristic health scorer and diagnostic engine.

Provides explainable, objective health evaluation based on CHAOSS
metrics and open-source sustainability indicators.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def evaluate_repository_health(repo_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate an explainable Open-Source Project Health score and status.

    Evaluates four core pillars:
      1. Activity & Freshness (Max 35 points)
      2. Community & Adoption (Max 25 points)
      3. Maintenance & Responsiveness (Max 25 points)
      4. Governance & Documentation (Max 15 points)

    Total max score is 100 points.

    Args:
        repo_data: Dictionary of repository metrics (from fetch_repository_data
                   or compatible schema).

    Returns:
        Dictionary containing:
          - health_score: int (0 to 100)
          - health_status: str ("Healthy", "At Risk", "Likely Abandoned")
          - pillar_scores: dict of breakdown by pillar
          - diagnostics: list of dicts with label, status ("good", "warning", "critical") and detail
          - recommendations: list of strings
    """
    diagnostics: List[Dict[str, str]] = []
    recommendations: List[str] = []

    archived = bool(repo_data.get("archived", False))
    days_since_commit = repo_data.get("days_since_last_commit", 0)
    stars = repo_data.get("stars_count", 0)
    forks = repo_data.get("forks_count", 0)
    open_issues = repo_data.get("open_issues_count", 0)
    open_prs = repo_data.get("open_prs_count", 0)
    has_license = bool(repo_data.get("has_license", 0))
    license_name = repo_data.get("license_name", "No License")
    repo_age_days = repo_data.get("repo_age_days", 1)

    # 1. Activity & Freshness (35 pts max)
    activity_score = 0
    if archived:
        activity_score = 0
        diagnostics.append({
            "pillar": "Activity",
            "status": "critical",
            "message": "Repository is officially archived. No further commits, issues, or releases can occur.",
        })
    elif days_since_commit <= 14:
        activity_score = 35
        diagnostics.append({
            "pillar": "Activity",
            "status": "good",
            "message": f"High activity: Latest commit was {days_since_commit} days ago (within 2 weeks).",
        })
    elif days_since_commit <= 45:
        activity_score = 30
        diagnostics.append({
            "pillar": "Activity",
            "status": "good",
            "message": f"Active maintenance: Latest commit was {days_since_commit} days ago.",
        })
    elif days_since_commit <= 90:
        activity_score = 22
        diagnostics.append({
            "pillar": "Activity",
            "status": "warning",
            "message": f"Moderate activity: Latest commit was {days_since_commit} days ago (within 3 months).",
        })
    elif days_since_commit <= 180:
        activity_score = 14
        diagnostics.append({
            "pillar": "Activity",
            "status": "warning",
            "message": f"Slowing cadence: Latest commit was {days_since_commit} days ago (3-6 months ago).",
        })
        recommendations.append("Check if the maintainers have moved discussions or development to another branch or repo.")
    elif days_since_commit <= 365:
        activity_score = 6
        diagnostics.append({
            "pillar": "Activity",
            "status": "warning",
            "message": f"Stagnant: No commits in {days_since_commit} days (over 6 months).",
        })
        recommendations.append("Project shows signs of dormancy. Review PR queue and issue responsiveness before relying on it.")
    else:
        activity_score = 0
        diagnostics.append({
            "pillar": "Activity",
            "status": "critical",
            "message": f"Severe dormancy: No commits for {days_since_commit} days (over 1 year).",
        })
        recommendations.append("Project appears unmaintained. Look for active forks or alternative libraries.")

    # 2. Community & Adoption (25 pts max)
    community_score = 0
    # Evaluate stars
    if stars >= 1000:
        community_score += 15
        diagnostics.append({
            "pillar": "Community",
            "status": "good",
            "message": f"Strong adoption: {stars:,} stars indicates wide community trust and usage.",
        })
    elif stars >= 250:
        community_score += 12
        diagnostics.append({
            "pillar": "Community",
            "status": "good",
            "message": f"Established interest: {stars:,} stars.",
        })
    elif stars >= 50:
        community_score += 8
        diagnostics.append({
            "pillar": "Community",
            "status": "good",
            "message": f"Moderate traction: {stars:,} stars.",
        })
    elif stars >= 10:
        community_score += 5
        diagnostics.append({
            "pillar": "Community",
            "status": "warning",
            "message": f"Early stage / Niche: {stars:,} stars.",
        })
    else:
        community_score += 2
        diagnostics.append({
            "pillar": "Community",
            "status": "warning",
            "message": f"Low visibility: {stars:,} stars.",
        })

    # Evaluate forks
    if forks >= 200:
        community_score += 10
    elif forks >= 50:
        community_score += 8
    elif forks >= 10:
        community_score += 5
    elif forks >= 2:
        community_score += 3
    else:
        community_score += 1

    # 3. Maintenance & Responsiveness (25 pts max)
    maintenance_score = 0
    # Evaluate ratio of open issues vs community size
    if archived:
        maintenance_score = 0
    elif open_issues == 0 and days_since_commit <= 90:
        maintenance_score = 25
        diagnostics.append({
            "pillar": "Maintenance",
            "status": "good",
            "message": "Zero open issues with active development indicates clean issue resolution.",
        })
    else:
        # Check issue backlog proportionality
        if stars > 0:
            issue_to_star_ratio = open_issues / max(stars, 1)
        else:
            issue_to_star_ratio = open_issues / 10.0

        if issue_to_star_ratio <= 0.05:
            maintenance_score = 25
            diagnostics.append({
                "pillar": "Maintenance",
                "status": "good",
                "message": f"Healthy issue management: {open_issues} open issues relative to project size.",
            })
        elif issue_to_star_ratio <= 0.15:
            maintenance_score = 20
            diagnostics.append({
                "pillar": "Maintenance",
                "status": "good",
                "message": f"Balanced issue volume: {open_issues} open issues.",
            })
        elif issue_to_star_ratio <= 0.35:
            maintenance_score = 14
            diagnostics.append({
                "pillar": "Maintenance",
                "status": "warning",
                "message": f"Moderate issue accumulation: {open_issues} open issues.",
            })
            recommendations.append("Triage open issues to reduce maintenance burden.")
        else:
            maintenance_score = 8
            diagnostics.append({
                "pillar": "Maintenance",
                "status": "warning",
                "message": f"Significant issue backlog: {open_issues} open issues relative to stars.",
            })
            recommendations.append("High issue backlog may signal maintainer burnout or unaddressed bug reports.")

        # Check PR queue
        if open_prs > 50 and days_since_commit > 60:
            maintenance_score = max(2, maintenance_score - 6)
            diagnostics.append({
                "pillar": "Maintenance",
                "status": "warning",
                "message": f"Large pending pull request queue ({open_prs} open PRs) with slow recent commits.",
            })
            recommendations.append("Review community pull requests to keep external contributors engaged.")

    # 4. Governance & Documentation (15 pts max)
    governance_score = 0
    if has_license:
        governance_score += 10
        diagnostics.append({
            "pillar": "Governance",
            "status": "good",
            "message": f"Recognized open-source license: '{license_name}'. Safe for downstream use.",
        })
    else:
        diagnostics.append({
            "pillar": "Governance",
            "status": "critical",
            "message": "No open-source license detected. Code is all-rights-reserved by default.",
        })
        recommendations.append("Add an OSI-approved license (e.g. MIT, Apache-2.0) to clarify reuse rights.")

    desc = repo_data.get("description", "")
    if desc and len(desc.strip()) > 10 and desc != "No description provided":
        governance_score += 5
        diagnostics.append({
            "pillar": "Governance",
            "status": "good",
            "message": "Repository includes a descriptive purpose statement.",
        })
    else:
        recommendations.append("Provide a concise repository description to assist potential adopters.")

    total_score = min(100, max(0, activity_score + community_score + maintenance_score + governance_score))

    # Determine classification
    if archived:
        health_status = "Likely Abandoned"
        total_score = min(total_score, 25)
    elif days_since_commit > 365 and total_score < 50:
        health_status = "Likely Abandoned"
    elif total_score >= 70:
        health_status = "Healthy"
    elif total_score >= 40:
        health_status = "At Risk"
    else:
        health_status = "Likely Abandoned"

    if not recommendations:
        recommendations.append("Repository demonstrates strong operational health and maintenance cadence.")

    return {
        "health_score": total_score,
        "health_status": health_status,
        "pillar_scores": {
            "Activity & Freshness": {"score": activity_score, "max": 35},
            "Community & Adoption": {"score": community_score, "max": 25},
            "Maintenance & Responsiveness": {"score": maintenance_score, "max": 25},
            "Governance & Documentation": {"score": governance_score, "max": 15},
        },
        "diagnostics": diagnostics,
        "recommendations": recommendations,
    }
