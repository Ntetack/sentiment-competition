"""
update_leaderboard.py — Merges new results into leaderboard/results.json and recomputes ranks.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

LEADERBOARD_PATH = Path(__file__).parent.parent / "leaderboard" / "results.json"


def load_leaderboard() -> dict:
    if LEADERBOARD_PATH.exists():
        with open(LEADERBOARD_PATH) as f:
            return json.load(f)
    return {"submissions": [], "last_updated": ""}


def save_leaderboard(data: dict):
    LEADERBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LEADERBOARD_PATH, "w") as f:
        json.dump(data, f, indent=2)


def compute_ranks(submissions: list) -> list:
    """Best accuracy per team gets the rank. All submissions shown."""
    valid = [s for s in submissions if s.get("status") == "success"]
    invalid = [s for s in submissions if s.get("status") != "success"]

    # Best per team
    best_per_team = {}
    for s in valid:
        team = s["team_name"]
        if team not in best_per_team or s["accuracy"] > best_per_team[team]["accuracy"]:
            best_per_team[team] = s

    # Sort teams by best accuracy
    ranked_teams = sorted(best_per_team.values(), key=lambda x: x["accuracy"], reverse=True)
    team_rank = {s["team_name"]: i + 1 for i, s in enumerate(ranked_teams)}

    for s in valid:
        s["rank"] = team_rank.get(s["team_name"], 999)

    return valid + invalid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    args = parser.parse_args()

    with open(args.results) as f:
        new_results = json.load(f)

    new_results["submitted_at"] = datetime.now(timezone.utc).isoformat()

    leaderboard = load_leaderboard()

    # Remove previous entry for same issue (re-evaluation case)
    leaderboard["submissions"] = [
        s for s in leaderboard["submissions"]
        if s.get("issue_number") != new_results.get("issue_number")
    ]

    leaderboard["submissions"].append(new_results)
    leaderboard["submissions"] = compute_ranks(leaderboard["submissions"])
    leaderboard["last_updated"] = datetime.now(timezone.utc).isoformat()
    leaderboard["total_submissions"] = len(leaderboard["submissions"])

    # Write back rank into results file (for GitHub Actions comment)
    current_rank = next(
        (s["rank"] for s in leaderboard["submissions"]
         if s.get("issue_number") == new_results.get("issue_number")),
        999
    )
    new_results["rank"] = current_rank
    with open(args.results, "w") as f:
        json.dump(new_results, f, indent=2)

    save_leaderboard(leaderboard)

    print(f"[INFO] Leaderboard updated — Rank #{current_rank}")
    print(f"[INFO] Total submissions: {leaderboard['total_submissions']}")


if __name__ == "__main__":
    main()
