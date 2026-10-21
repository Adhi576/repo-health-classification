"""Unit tests for transparent health rules and diagnostic engine."""

from src.health_rules import evaluate_repository_health


class TestHealthRules:
    """Test heuristic health rules, status categorization, and diagnostics."""

    def test_healthy_active_repository(self):
        repo_data = {
            "stars_count": 15000,
            "forks_count": 3200,
            "open_issues_count": 45,
            "open_prs_count": 10,
            "days_since_last_commit": 3,
            "repo_age_days": 1800,
            "has_license": 1,
            "license_name": "MIT License",
            "archived": False,
            "description": "A popular, production-ready framework.",
        }

        eval_result = evaluate_repository_health(repo_data)
        assert eval_result["health_status"] == "Healthy"
        assert eval_result["health_score"] >= 70
        assert eval_result["pillar_scores"]["Activity & Freshness"]["score"] == 35
        assert eval_result["pillar_scores"]["Governance & Documentation"]["score"] == 15
        assert len(eval_result["diagnostics"]) > 0

    def test_at_risk_slowing_repository(self):
        repo_data = {
            "stars_count": 800,
            "forks_count": 85,
            "open_issues_count": 120,
            "open_prs_count": 35,
            "days_since_last_commit": 120,
            "repo_age_days": 1200,
            "has_license": 1,
            "license_name": "Apache-2.0",
            "archived": False,
            "description": "A utility library.",
        }

        eval_result = evaluate_repository_health(repo_data)
        assert eval_result["health_status"] == "At Risk"
        assert 40 <= eval_result["health_score"] < 70
        assert any(d["status"] == "warning" for d in eval_result["diagnostics"])
        assert len(eval_result["recommendations"]) > 0

    def test_abandoned_dormant_repository(self):
        repo_data = {
            "stars_count": 25,
            "forks_count": 2,
            "open_issues_count": 40,
            "open_prs_count": 8,
            "days_since_last_commit": 750,
            "repo_age_days": 1500,
            "has_license": 0,
            "license_name": "No License",
            "archived": False,
            "description": "",
        }

        eval_result = evaluate_repository_health(repo_data)
        assert eval_result["health_status"] == "Likely Abandoned"
        assert eval_result["health_score"] < 40
        assert any(d["status"] == "critical" for d in eval_result["diagnostics"])

    def test_archived_repository_is_always_likely_abandoned(self):
        repo_data = {
            "stars_count": 50000,
            "forks_count": 12000,
            "open_issues_count": 0,
            "open_prs_count": 0,
            "days_since_last_commit": 5,
            "repo_age_days": 2000,
            "has_license": 1,
            "license_name": "MIT",
            "archived": True,  # Key flag
            "description": "Deprecated and archived repository.",
        }

        eval_result = evaluate_repository_health(repo_data)
        assert eval_result["health_status"] == "Likely Abandoned"
        assert eval_result["health_score"] <= 25
        assert any("archived" in d["message"].lower() for d in eval_result["diagnostics"])

    def test_score_boundaries(self):
        for days in [0, 5, 30, 80, 200, 500, 2000]:
            result = evaluate_repository_health({
                "stars_count": 100,
                "forks_count": 10,
                "open_issues_count": 10,
                "open_prs_count": 2,
                "days_since_last_commit": days,
                "has_license": 1,
                "archived": False,
            })
            assert 0 <= result["health_score"] <= 100
