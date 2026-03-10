"""
process_submission.py — Parses GitHub Issue body and extracts CSV + metadata.
"""

import argparse
import json
import os
import re
import sys


def parse_issue_body(body: str) -> dict:
    """Parse GitHub Issue form body into a dict."""
    data = {}
    pattern = re.compile(r"### (.+?)\n\n(.+?)(?=\n\n###|\Z)", re.DOTALL)
    for header, value in pattern.findall(body):
        header = header.strip()
        value = value.strip()
        field_map = {
            "👤 Team / Participant Name": "team_name",
            "🏷️ Model / Run Name": "model_name",
            "📄 Approach Description": "description",
            "🔢 Submission Number (today)": "submission_number",
            "📎 CSV Predictions": "csv_content",
        }
        if header in field_map:
            data[field_map[header]] = value
    return data


def clean_csv(csv_text: str) -> str:
    """Remove markdown fences and extra whitespace."""
    csv_text = re.sub(r"^```.*?\n", "", csv_text, flags=re.MULTILINE)
    csv_text = re.sub(r"```$", "", csv_text, flags=re.MULTILINE)
    return csv_text.strip()


def validate_csv(csv_text: str) -> list:
    """Validate CSV format and return list of (image_id, label) tuples."""
    lines = [l.strip() for l in csv_text.strip().splitlines() if l.strip()]

    if not lines:
        raise ValueError("CSV is empty.")

    # Check header
    header = lines[0].lower().replace(" ", "")
    if header != "image_id,label":
        raise ValueError(
            f"Invalid header: '{lines[0]}'. Expected: 'image_id,label'"
        )

    rows = []
    for i, line in enumerate(lines[1:], start=2):
        parts = line.split(",")
        if len(parts) != 2:
            raise ValueError(f"Line {i}: expected 2 columns, got {len(parts)}: '{line}'")

        image_id = parts[0].strip()
        label_str = parts[1].strip()

        if not image_id:
            raise ValueError(f"Line {i}: empty image_id")

        try:
            label = int(label_str)
        except ValueError:
            raise ValueError(f"Line {i}: label '{label_str}' is not an integer")

        if label < 0 or label > 6:
            raise ValueError(f"Line {i}: label {label} is out of range [0-6]")

        rows.append((image_id, label))

    if len(rows) == 0:
        raise ValueError("CSV has no prediction rows.")

    print(f"[INFO] CSV validated: {len(rows)} predictions")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue-number", required=True)
    parser.add_argument("--issue-title", required=True)
    args = parser.parse_args()

    issue_body = os.environ.get("ISSUE_BODY", "")

    print(f"[INFO] Processing Issue #{args.issue_number}: {args.issue_title}")

    # Parse
    data = parse_issue_body(issue_body)

    if "csv_content" not in data or not data["csv_content"].strip():
        print("[ERROR] No CSV content found in issue.", file=sys.stderr)
        sys.exit(1)

    if "team_name" not in data or not data["team_name"].strip():
        print("[ERROR] No team name found.", file=sys.stderr)
        sys.exit(1)

    # Validate CSV
    csv_text = clean_csv(data["csv_content"])
    rows = validate_csv(csv_text)

    # Save cleaned CSV
    with open("/tmp/submission.csv", "w") as f:
        f.write("image_id,label\n")
        for image_id, label in rows:
            f.write(f"{image_id},{label}\n")

    print(f"[INFO] Saved {len(rows)} predictions to /tmp/submission.csv")

    # Save metadata
    metadata = {
        "team_name": data.get("team_name", "Unknown").strip(),
        "model_name": data.get("model_name", "Unknown").strip(),
        "description": data.get("description", "").strip(),
        "submission_number": data.get("submission_number", "1").strip(),
        "issue_number": int(args.issue_number),
        "issue_title": args.issue_title,
    }
    with open("/tmp/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[INFO] Team: {metadata['team_name']} | Model: {metadata['model_name']}")
    print("[INFO] Submission parsed successfully ✓")


if __name__ == "__main__":
    main()
