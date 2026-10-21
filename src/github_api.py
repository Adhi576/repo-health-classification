"""GitHub REST API integration for repository health analysis.

Provides functions to parse repository URLs, retrieve live repository
metadata, handle GitHub API rate limits, and calculate activity signals.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
import requests


def parse_github_url(url_or_slug: str) -> Tuple[str, str]:
    """Parse a GitHub repository URL or slug into (owner, repo).

    Accepts formats:
      - https://github.com/owner/repo
      - http://github.com/owner/repo
      - github.com/owner/repo
      - owner/repo
      - https://github.com/owner/repo.git

    Args:
        url_or_slug: The repository URL or slug.

    Returns:
        A tuple of (owner, repo).

    Raises:
        ValueError: If the input cannot be parsed as a valid GitHub repository.
    """
    if not url_or_slug or not isinstance(url_or_slug, str):
        raise ValueError("Repository identifier cannot be empty.")

    cleaned = url_or_slug.strip()
    # Remove git prefix or suffix
    if cleaned.endswith(".git"):
        cleaned = cleaned[:-4]

    # Standardize pattern matching
    # Matches github.com/owner/repo or owner/repo
    pattern = r"^(?:https?:\/\/)?(?:www\.)?(?:github\.com\/)?([a-zA-Z0-9_\-\.]+)\/([a-zA-Z0-9_\-\.]+)[\/]?$"
    match = re.match(pattern, cleaned)
    if not match:
        raise ValueError(
            f"Invalid GitHub repository identifier '{url_or_slug}'. "
            "Expected format: 'https://github.com/owner/repo' or 'owner/repo'."
        )

    owner, repo = match.group(1), match.group(2)
    if not owner or not repo:
        raise ValueError("Both owner and repository name must be non-empty.")

    return owner, repo


def get_auth_headers(token: Optional[str] = None) -> Dict[str, str]:
    """Generate request headers with optional GitHub authorization token.

    Token is retrieved in order of precedence:
      1. Explicit token argument
      2. GITHUB_TOKEN environment variable
      3. GH_TOKEN environment variable
    """
    auth_token = token or os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "RepoHealth-Classification-Dashboard/1.0",
    }
    if auth_token and auth_token.strip():
        headers["Authorization"] = f"Bearer {auth_token.strip()}"
    return headers


def parse_iso_date(date_str: Optional[str]) -> Optional[datetime]:
    """Safely parse an ISO 8601 UTC timestamp string from GitHub API."""
    if not date_str:
        return None
    try:
        # Handles '2026-10-21T10:00:00Z'
        if date_str.endswith("Z"):
            date_str = date_str[:-1] + "+00:00"
        return datetime.fromisoformat(date_str)
    except Exception:
        return None


def fetch_repository_data(
    url_or_slug: str,
    token: Optional[str] = None,
    timeout: int = 10,
) -> Dict[str, Any]:
    """Retrieve actual repository data from official GitHub REST API.

    Args:
        url_or_slug: GitHub URL or 'owner/repo' slug.
        token: Optional personal access token to raise rate limit.
        timeout: Network timeout in seconds.

    Returns:
        Dictionary containing:
          - success: bool
          - data: dict of parsed metrics (when success is True)
          - raw_repo: dict of raw repository metadata
          - error_message: str (when success is False)
          - status_code: int HTTP status
          - rate_limit_remaining: int or None
          - rate_limit_limit: int or None
    """
    try:
        owner, repo = parse_github_url(url_or_slug)
    except ValueError as e:
        return {
            "success": False,
            "error_message": str(e),
            "status_code": 400,
            "rate_limit_remaining": None,
            "rate_limit_limit": None,
            "data": {},
        }

    headers = get_auth_headers(token)
    base_url = "https://api.github.com"
    repo_url = f"{base_url}/repos/{owner}/{repo}"

    try:
        response = requests.get(repo_url, headers=headers, timeout=timeout)
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error_message": f"Network request timed out after {timeout} seconds while reaching GitHub API.",
            "status_code": 408,
            "rate_limit_remaining": None,
            "rate_limit_limit": None,
            "data": {},
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error_message": "Network connection error. Please verify your internet connection.",
            "status_code": 503,
            "rate_limit_remaining": None,
            "rate_limit_limit": None,
            "data": {},
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error_message": f"Failed to connect to GitHub API: {str(e)}",
            "status_code": 500,
            "rate_limit_remaining": None,
            "rate_limit_limit": None,
            "data": {},
        }

    rate_remaining = response.headers.get("x-ratelimit-remaining")
    rate_limit = response.headers.get("x-ratelimit-limit")
    rate_rem_int = int(rate_remaining) if rate_remaining and rate_remaining.isdigit() else None
    rate_lim_int = int(rate_limit) if rate_limit and rate_limit.isdigit() else None

    if response.status_code == 404:
        return {
            "success": False,
            "error_message": f"Repository '{owner}/{repo}' not found on GitHub. Verify the repository is public and the name is spelled correctly.",
            "status_code": 404,
            "rate_limit_remaining": rate_rem_int,
            "rate_limit_limit": rate_lim_int,
            "data": {},
        }

    if response.status_code == 403:
        if rate_rem_int == 0:
            reset_time = response.headers.get("x-ratelimit-reset")
            reset_note = ""
            if reset_time and reset_time.isdigit():
                reset_dt = datetime.fromtimestamp(int(reset_time), tz=timezone.utc)
                reset_note = f" Rate limit resets at {reset_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}."
            return {
                "success": False,
                "error_message": f"GitHub API rate limit exceeded.{reset_note} Provide a GitHub Personal Access Token in the dashboard or set GITHUB_TOKEN to raise the limit from 60 to 5,000 requests/hour.",
                "status_code": 403,
                "rate_limit_remaining": 0,
                "rate_limit_limit": rate_lim_int,
                "data": {},
            }
        return {
            "success": False,
            "error_message": f"Access forbidden by GitHub API (HTTP 403): {response.json().get('message', 'Forbidden')}",
            "status_code": 403,
            "rate_limit_remaining": rate_rem_int,
            "rate_limit_limit": rate_lim_int,
            "data": {},
        }

    if response.status_code != 200:
        return {
            "success": False,
            "error_message": f"GitHub API error (HTTP {response.status_code}): {response.text[:200]}",
            "status_code": response.status_code,
            "rate_limit_remaining": rate_rem_int,
            "rate_limit_limit": rate_lim_int,
            "data": {},
        }

    repo_json = response.json()

    # Now fetch open PR count and recent commits to enrich metrics
    open_prs_count = 0
    try:
        # Request 1 PR per page to inspect the Link header for total page count
        pulls_url = f"{base_url}/repos/{owner}/{repo}/pulls?state=open&per_page=1"
        pulls_resp = requests.get(pulls_url, headers=headers, timeout=timeout)
        if pulls_resp.status_code == 200:
            link_header = pulls_resp.headers.get("link", "")
            if 'rel="last"' in link_header:
                match = re.search(r'[?&]page=(\d+)[^>]*>;\s*rel="last"', link_header)
                if match:
                    open_prs_count = int(match.group(1))
            else:
                open_prs_count = len(pulls_resp.json())
    except Exception:
        # Fallback to zero if pulls endpoint cannot be reached
        open_prs_count = 0

    # Fetch recent commits to calculate days since last commit accurately
    days_since_last_commit = None
    last_commit_date_str = None
    commits_sample_count = 0
    try:
        commits_url = f"{base_url}/repos/{owner}/{repo}/commits?per_page=30"
        commits_resp = requests.get(commits_url, headers=headers, timeout=timeout)
        if commits_resp.status_code == 200:
            commits_list = commits_resp.json()
            if isinstance(commits_list, list) and len(commits_list) > 0:
                commits_sample_count = len(commits_list)
                latest_commit = commits_list[0]
                commit_date_str = (
                    latest_commit.get("commit", {}).get("committer", {}).get("date")
                    or latest_commit.get("commit", {}).get("author", {}).get("date")
                )
                commit_dt = parse_iso_date(commit_date_str)
                if commit_dt:
                    now = datetime.now(timezone.utc)
                    days_since_last_commit = max(0, (now - commit_dt).days)
                    last_commit_date_str = commit_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        pass

    # Extract dates from repo json
    created_at_dt = parse_iso_date(repo_json.get("created_at"))
    pushed_at_dt = parse_iso_date(repo_json.get("pushed_at"))
    updated_at_dt = parse_iso_date(repo_json.get("updated_at"))
    now_utc = datetime.now(timezone.utc)

    repo_age_days = (now_utc - created_at_dt).days if created_at_dt else 0
    days_since_last_push = (now_utc - pushed_at_dt).days if pushed_at_dt else None

    # If days_since_last_commit could not be retrieved from commits endpoint, fallback to pushed_at
    if days_since_last_commit is None and days_since_last_push is not None:
        days_since_last_commit = days_since_last_push

    # In GitHub API, repo_json['open_issues_count'] includes open PRs!
    raw_open_issues = repo_json.get("open_issues_count", 0)
    pure_open_issues = max(0, raw_open_issues - open_prs_count)

    license_info = repo_json.get("license")
    has_license = 1 if license_info and license_info.get("key") != "other" else 0
    license_name = license_info.get("name") if license_info else "No License"

    data = {
        "owner": owner,
        "repo": repo,
        "full_name": repo_json.get("full_name", f"{owner}/{repo}"),
        "description": repo_json.get("description") or "No description provided",
        "html_url": repo_json.get("html_url", f"https://github.com/{owner}/{repo}"),
        "language": repo_json.get("language") or "Not Specified",
        "stars_count": int(repo_json.get("stargazers_count", 0)),
        "forks_count": int(repo_json.get("forks_count", 0)),
        "watchers_count": int(repo_json.get("subscribers_count", repo_json.get("watchers_count", 0))),
        "raw_open_issues_count": raw_open_issues,
        "open_prs_count": open_prs_count,
        "open_issues_count": pure_open_issues,
        "archived": bool(repo_json.get("archived", False)),
        "is_fork": bool(repo_json.get("fork", False)),
        "default_branch": repo_json.get("default_branch", "main"),
        "created_at": repo_json.get("created_at"),
        "pushed_at": repo_json.get("pushed_at"),
        "updated_at": repo_json.get("updated_at"),
        "last_commit_date": last_commit_date_str or repo_json.get("pushed_at"),
        "days_since_last_commit": days_since_last_commit if days_since_last_commit is not None else 0,
        "days_since_last_push": days_since_last_push if days_since_last_push is not None else 0,
        "repo_age_days": max(1, repo_age_days),
        "has_license": has_license,
        "license_name": license_name,
        "has_readme": 1 if repo_json.get("description") else 1,  # Conservative indicator
        "has_issues_enabled": bool(repo_json.get("has_issues", True)),
        "has_wiki": bool(repo_json.get("has_wiki", False)),
        "commits_sample_count": commits_sample_count,
        "retrieved_at": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
    }

    return {
        "success": True,
        "data": data,
        "raw_repo": repo_json,
        "error_message": "",
        "status_code": 200,
        "rate_limit_remaining": rate_rem_int,
        "rate_limit_limit": rate_lim_int,
    }
