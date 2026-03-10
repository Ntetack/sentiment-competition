"""
update_leaderboard.py — Merges new evaluation results into the leaderboard JSON.
"""

import argparse
import json
import os
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
    """Sort submissions and assign ranks (best F1 = rank 1)."""
    # Only rank successful submissions
    valid = [s for s in submissions if s.get("status") == "success"]
    invalid = [s for s in submissions if s.get("status") != "success"]
    
    # Best submission per team
    best_per_team = {}
    for sub in valid:
        team = sub["team_name"]
        if team not in best_per_team or sub["f1_score"] > best_per_team[team]["f1_score"]:
            best_per_team[team] = sub
    
    # Sort by F1 score
    sorted_teams = sorted(best_per_team.values(), key=lambda x: x["f1_score"], reverse=True)
    
    for i, sub in enumerate(sorted_teams):
        sub["rank"] = i + 1
    
    # All submissions keep their rank reference
    team_ranks = {s["team_name"]: s["rank"] for s in sorted_teams}
    
    for sub in valid:
        sub["rank"] = team_ranks.get(sub["team_name"], 999)
    
    return valid + invalid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to results JSON")
    parser.add_argument("--issue-number", type=int, required=True)
    args = parser.parse_args()
    
    # Load new results
    with open(args.results) as f:
        new_results = json.load(f)
    
    # Load metadata if available
    metadata_path = "/tmp/submission_metadata.json"
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = {"team_name": "Unknown", "model_name": "Unknown"}
    
    # Build submission entry
    entry = {
        "issue_number": args.issue_number,
        "team_name": metadata.get("team_name", "Unknown"),
        "model_name": metadata.get("model_name", "Unknown"),
        "architecture": metadata.get("architecture", "Unknown"),
        "description": metadata.get("description", ""),
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "status": new_results.get("status", "error"),
        "f1_score": new_results.get("f1_score", 0.0),
        "accuracy": new_results.get("accuracy", 0.0),
        "per_class_f1": new_results.get("per_class_f1", [0.0] * 5),
        "inference_time_ms": new_results.get("inference_time_ms", 0.0),
        "total_samples": new_results.get("total_samples", 0),
        "error_message": new_results.get("error_message", ""),
        "rank": 0,
    }
    
    # Load & update leaderboard
    leaderboard = load_leaderboard()
    
    # Remove any previous entry from same issue (re-evaluation)
    leaderboard["submissions"] = [
        s for s in leaderboard["submissions"]
        if s.get("issue_number") != args.issue_number
    ]
    leaderboard["submissions"].append(entry)
    
    # Recompute ranks
    leaderboard["submissions"] = compute_ranks(leaderboard["submissions"])
    leaderboard["last_updated"] = datetime.now(timezone.utc).isoformat()
    leaderboard["total_submissions"] = len(leaderboard["submissions"])
    
    # Update results JSON with rank
    current_rank = next(
        (s["rank"] for s in leaderboard["submissions"]
         if s.get("issue_number") == args.issue_number),
        999
    )
    new_results["rank"] = current_rank
    with open(args.results, "w") as f:
        json.dump(new_results, f, indent=2)
    
    # Save leaderboard
    save_leaderboard(leaderboard)
    print(f"[INFO] Leaderboard updated — Rank #{current_rank} for issue #{args.issue_number}")
    print(f"[INFO] Total submissions: {leaderboard['total_submissions']}")


if __name__ == "__main__":
    main()
