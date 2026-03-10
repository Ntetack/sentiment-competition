"""
process_submission.py — Parses GitHub Issue body, downloads CSV from URL, validates it.
"""

import argparse
import json
import os
import re
import sys
import urllib.request


def parse_issue_body(body: str) -> dict:
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
            "🔗 CSV File URL": "csv_url",
        }
        if header in field_map:
            data[field_map[header]] = value
    return data


def normalize_url(url: str) -> str:
    url = url.strip()
    # Google Drive: /file/d/ID/view → direct download
    gdrive = re.match(r"https://drive\.google\.com/file/d/([^/?\s]+)", url)
    if gdrive:
        file_id = gdrive.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    # Dropbox: dl=0 → dl=1
    if "dropbox.com" in url:
        return re.sub(r"dl=0", "dl=1", url)
    return url


def download_csv(url: str, dest: str) -> str:
    url = normalize_url(url)
    print(f"[INFO] Downloading CSV from: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        content = response.read().decode("utf-8")
    with open(dest, "w") as f:
        f.write(content)
    lines = content.strip().splitlines()
    print(f"[INFO] Downloaded {len(lines)-1} prediction rows")
    return content


def validate_csv(csv_text: str) -> list:
    lines = [l.strip() for l in csv_text.strip().splitlines() if l.strip()]
    if not lines:
        raise ValueError("CSV is empty.")
    header = lines[0].lower().replace(" ", "")
    if header != "image_id,label":
        raise ValueError(f"Invalid header: '{lines[0]}'. Expected: 'image_id,label'")
    rows = []
    for i, line in enumerate(lines[1:], start=2):
        parts = line.split(",")
        if len(parts) != 2:
            raise ValueError(f"Line {i}: expected 2 columns, got {len(parts)}")
        image_id = parts[0].strip()
        label_str = parts[1].strip()
        if not image_id:
            raise ValueError(f"Line {i}: empty image_id")
        try:
            label = int(label_str)
        except ValueError:
            raise ValueError(f"Line {i}: label '{label_str}' is not an integer")
        if label < 0 or label > 6:
            raise ValueError(f"Line {i}: label {label} out of range [0-6]")
        rows.append((image_id, label))
    if not rows:
        raise ValueError("CSV has no prediction rows.")
    print(f"[INFO] CSV validated: {len(rows)} predictions ✓")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue-number", required=True)
    parser.add_argument("--issue-title", required=True)
    args = parser.parse_args()

    issue_body = os.environ.get("ISSUE_BODY", "")
    print(f"[INFO] Processing Issue #{args.issue_number}: {args.issue_title}")

    data = parse_issue_body(issue_body)

    if not data.get("team_name", "").strip():
        print("[ERROR] Missing team name.", file=sys.stderr)
        sys.exit(1)

    if not data.get("csv_url", "").strip():
        print("[ERROR] Missing CSV URL.", file=sys.stderr)
        sys.exit(1)

    csv_text = download_csv(data["csv_url"], "/tmp/submission.csv")
    validate_csv(csv_text)

    metadata = {
        "team_name": data.get("team_name", "Unknown").strip(),
        "model_name": data.get("model_name", "Unknown").strip(),
        "description": data.get("description", "").strip(),
        "submission_number": data.get("submission_number", "1").strip(),
        "csv_url": data.get("csv_url", "").strip(),
        "issue_number": int(args.issue_number),
        "issue_title": args.issue_title,
    }
    with open("/tmp/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[INFO] Team: {metadata['team_name']} | Model: {metadata['model_name']}")
    print("[INFO] Submission ready for evaluation ✓")


if __name__ == "__main__":
    main()
