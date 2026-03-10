"""
evaluate.py — Evaluates a participant's CSV submission against the secret labels.
"""

import argparse
import json
import sys

import pandas as pd
from sklearn.metrics import accuracy_score

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def evaluate(submission_path: str, ground_truth_path: str, metadata_path: str, output_path: str):

    # Load metadata
    with open(metadata_path) as f:
        metadata = json.load(f)

    # Load ground truth
    gt = pd.read_csv(ground_truth_path)
    gt.columns = gt.columns.str.strip().str.lower()
    gt["image_id"] = gt["image_id"].str.strip()
    gt["label"] = gt["label"].astype(int)

    # Load submission
    sub = pd.read_csv(submission_path)
    sub.columns = sub.columns.str.strip().str.lower()
    sub["image_id"] = sub["image_id"].str.strip()
    sub["label"] = sub["label"].astype(int)

    # Check all images are present
    missing = set(gt["image_id"]) - set(sub["image_id"])
    if missing:
        raise ValueError(
            f"{len(missing)} images missing from submission: {list(missing)[:5]}..."
        )

    extra = set(sub["image_id"]) - set(gt["image_id"])
    if extra:
        print(f"[WARN] {len(extra)} extra image_ids in submission (ignored)")

    # Merge on image_id
    merged = gt.merge(sub, on="image_id", suffixes=("_true", "_pred"))
    y_true = merged["label_true"].tolist()
    y_pred = merged["label_pred"].tolist()

    # Global accuracy
    accuracy = accuracy_score(y_true, y_pred)
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    total = len(y_true)

    print(f"[RESULT] Accuracy: {accuracy:.4f} ({correct}/{total})")

    # Per-class accuracy
    per_class = {}
    for idx, emotion in enumerate(EMOTIONS):
        indices = [i for i, t in enumerate(y_true) if t == idx]
        if not indices:
            per_class[emotion] = {"correct": 0, "total": 0, "acc": 0.0}
            continue
        class_correct = sum(1 for i in indices if y_pred[i] == idx)
        class_total = len(indices)
        per_class[emotion] = {
            "correct": class_correct,
            "total": class_total,
            "acc": round(class_correct / class_total, 4),
        }
        print(f"  {emotion:10s}: {class_correct}/{class_total} ({per_class[emotion]['acc']*100:.1f}%)")

    results = {
        "status": "success",
        "team_name": metadata["team_name"],
        "model_name": metadata["model_name"],
        "description": metadata.get("description", ""),
        "issue_number": metadata["issue_number"],
        "accuracy": round(accuracy, 6),
        "correct": correct,
        "total": total,
        "per_class": per_class,
        "rank": 0,  # assigned by update_leaderboard.py
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[INFO] Results saved to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        evaluate(args.submission, args.ground_truth, args.metadata, args.output)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

        error = {
            "status": "error",
            "error_message": str(e),
            "accuracy": 0.0,
            "correct": 0,
            "total": 0,
            "per_class": {},
            "rank": 0,
        }
        with open(args.output, "w") as f:
            json.dump(error, f, indent=2)
        sys.exit(1)


if __name__ == "__main__":
    main()
