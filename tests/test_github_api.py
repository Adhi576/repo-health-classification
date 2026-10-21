"""Unit tests for GitHub REST API integration and URL parsing."""

from unittest.mock import MagicMock, patch
import pytest
import requests

from src.github_api import fetch_repository_data, get_auth_headers, parse_github_url


class TestParseGithubUrl:
    """Test URL parsing for various formats and error conditions."""

    def test_standard_https_url(self):
        owner, repo = parse_github_url("https://github.com/psf/requests")
        assert owner == "psf"
        assert repo == "requests"

    def test_https_with_git_suffix(self):
        owner, repo = parse_github_url("https://github.com/Adhi576/repo-health-classification.git")
        assert owner == "Adhi576"
        assert repo == "repo-health-classification"

    def test_http_url(self):
        owner, repo = parse_github_url("http://github.com/torvalds/linux")
        assert owner == "torvalds"
        assert repo == "linux"

    def test_no_scheme_url(self):
        owner, repo = parse_github_url("github.com/facebook/react")
        assert owner == "facebook"
        assert repo == "react"

    def test_simple_slug(self):
        owner, repo = parse_github_url("pallets/flask")
        assert owner == "pallets"
        assert repo == "flask"

    def test_trailing_slash(self):
        owner, repo = parse_github_url("https://github.com/golang/go/")
        assert owner == "golang"
        assert repo == "go"

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_github_url("")

    def test_single_word_raises(self):
        with pytest.raises(ValueError, match="Invalid GitHub repository identifier"):
            parse_github_url("just-a-name")

    def test_invalid_url_raises(self):
        with pytest.raises(ValueError, match="Invalid GitHub repository identifier"):
            parse_github_url("https://gitlab.com/owner/repo")


class TestAuthHeaders:
    """Test header generation with and without tokens."""

    def test_headers_without_token(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        headers = get_auth_headers()
        assert "Authorization" not in headers
        assert "Accept" in headers
        assert "User-Agent" in headers

    def test_headers_with_explicit_token(self):
        headers = get_auth_headers("token_abc123")
        assert headers["Authorization"] == "Bearer token_abc123"

    def test_headers_with_env_token(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "token_env456")
        headers = get_auth_headers()
        assert headers["Authorization"] == "Bearer token_env456"


class TestFetchRepositoryData:
    """Test API response handling, error cases, and metric extraction."""

    def test_invalid_url_returns_error(self):
        result = fetch_repository_data("invalid-repo-slug")
        assert result["success"] is False
        assert result["status_code"] == 400
        assert "Invalid GitHub repository identifier" in result["error_message"]

    @patch("requests.get")
    def test_successful_fetch(self, mock_get):
        # Mock main repo response
        mock_repo_resp = MagicMock()
        mock_repo_resp.status_code = 200
        mock_repo_resp.headers = {"x-ratelimit-remaining": "59", "x-ratelimit-limit": "60"}
        mock_repo_resp.json.return_value = {
            "name": "flask",
            "full_name": "pallets/flask",
            "description": "The Python micro framework for building web applications.",
            "html_url": "https://github.com/pallets/flask",
            "stargazers_count": 68000,
            "forks_count": 16000,
            "subscribers_count": 2100,
            "open_issues_count": 25,
            "archived": False,
            "fork": False,
            "language": "Python",
            "license": {"key": "bsd-3-clause", "name": "BSD 3-Clause"},
            "created_at": "2010-04-06T11:11:59Z",
            "pushed_at": "2026-10-20T10:00:00Z",
            "updated_at": "2026-10-21T08:00:00Z",
        }

        # Mock PRs response
        mock_prs_resp = MagicMock()
        mock_prs_resp.status_code = 200
        mock_prs_resp.headers = {}
        mock_prs_resp.json.return_value = [{"id": 101}, {"id": 102}]

        # Mock Commits response
        mock_commits_resp = MagicMock()
        mock_commits_resp.status_code = 200
        mock_commits_resp.json.return_value = [
            {"commit": {"committer": {"date": "2026-10-20T10:00:00Z"}}}
        ]

        mock_get.side_effect = [mock_repo_resp, mock_prs_resp, mock_commits_resp]

        result = fetch_repository_data("pallets/flask")
        assert result["success"] is True
        assert result["status_code"] == 200
        data = result["data"]
        assert data["stars_count"] == 68000
        assert data["forks_count"] == 16000
        assert data["language"] == "Python"
        assert data["has_license"] == 1
        assert data["archived"] is False
        assert data["open_prs_count"] == 2
        assert data["open_issues_count"] == 23  # 25 - 2 PRs

    @patch("requests.get")
    def test_repository_not_found_404(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.headers = {"x-ratelimit-remaining": "55", "x-ratelimit-limit": "60"}
        mock_get.return_value = mock_resp

        result = fetch_repository_data("nonexistent-owner/does-not-exist")
        assert result["success"] is False
        assert result["status_code"] == 404
        assert "not found on GitHub" in result["error_message"]

    @patch("requests.get")
    def test_rate_limit_exceeded_403(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.headers = {
            "x-ratelimit-remaining": "0",
            "x-ratelimit-limit": "60",
            "x-ratelimit-reset": "1729512000",
        }
        mock_get.return_value = mock_resp

        result = fetch_repository_data("pallets/flask")
        assert result["success"] is False
        assert result["status_code"] == 403
        assert "rate limit exceeded" in result["error_message"].lower()

    @patch("requests.get")
    def test_network_timeout(self, mock_get):
        mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")
        result = fetch_repository_data("psf/requests")
        assert result["success"] is False
        assert result["status_code"] == 408
        assert "timed out" in result["error_message"].lower()
