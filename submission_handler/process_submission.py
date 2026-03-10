"""
process_submission.py — Parses a GitHub Issue body and prepares files for evaluation.
"""

import argparse
import os
import re
import sys
import textwrap
import urllib.request
from pathlib import Path


def parse_issue_body(body: str) -> dict:
    """
    Parse the structured GitHub Issue form body.
    GitHub Forms render as markdown with ### headers.
    """
    data = {}
    
    # Match ### Header\n\nValue patterns
    pattern = re.compile(r"### (.+?)\n\n(.+?)(?=\n\n###|\Z)", re.DOTALL)
    matches = pattern.findall(body)
    
    field_map = {
        "👤 Team / Participant Name": "team_name",
        "📧 Contact Email": "email",
        "🏷️ Model Name": "model_name",
        "🏗️ Base Architecture": "architecture",
        "🔗 Model File URL (.pth)": "model_url",
        "📝 Model Class Code": "model_class",
        "📄 Model Description": "description",
    }
    
    for header, value in matches:
        header = header.strip()
        value = value.strip()
        if header in field_map:
            data[field_map[header]] = value
    
    return data


def clean_code(code: str) -> str:
    """Remove markdown code fences if present."""
    code = re.sub(r"^```python\n?", "", code, flags=re.MULTILINE)
    code = re.sub(r"^```\n?", "", code, flags=re.MULTILINE)
    return code.strip()


def download_model(url: str, dest: str):
    """Download model file from URL."""
    # Handle Google Drive links
    gdrive_match = re.match(
        r"https://drive\.google\.com/file/d/([^/]+)/", url
    )
    if gdrive_match:
        file_id = gdrive_match.group(1)
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    # Handle Dropbox links
    if "dropbox.com" in url and "dl=0" in url:
        url = url.replace("dl=0", "dl=1")
    
    print(f"[INFO] Downloading model from: {url}")
    
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as response:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB
        
        with open(dest, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"  {downloaded / 1e6:.1f} MB / {total / 1e6:.1f} MB ({pct:.0f}%)")
    
    size_mb = Path(dest).stat().st_size / 1e6
    print(f"[INFO] Downloaded {size_mb:.1f} MB to {dest}")
    
    if size_mb > 500:
        raise ValueError(f"Model file too large: {size_mb:.1f} MB > 500 MB limit")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue-body", required=True)
    parser.add_argument("--issue-number", required=True)
    parser.add_argument("--issue-title", required=True)
    args = parser.parse_args()
    
    print(f"[INFO] Processing submission — Issue #{args.issue_number}")
    print(f"[INFO] Title: {args.issue_title}")
    
    # Parse issue
    data = parse_issue_body(args.issue_body)
    
    required_fields = ["team_name", "model_name", "model_url", "model_class"]
    for field in required_fields:
        if field not in data or not data[field].strip():
            print(f"[ERROR] Missing required field: {field}", file=sys.stderr)
            sys.exit(1)
    
    print(f"[INFO] Team: {data['team_name']}")
    print(f"[INFO] Model: {data['model_name']}")
    
    # Save model class code
    model_code = clean_code(data["model_class"])
    
    # Add standard imports if not present
    if "import torch" not in model_code:
        model_code = "import torch\nimport torch.nn as nn\n\n" + model_code
    
    model_code_path = "/tmp/model_class.py"
    with open(model_code_path, "w") as f:
        f.write(model_code)
    print(f"[INFO] Model class saved to {model_code_path}")
    
    # Save metadata
    import json
    metadata = {
        "team_name": data.get("team_name", "Unknown"),
        "model_name": data.get("model_name", "Unknown"),
        "architecture": data.get("architecture", "Unknown"),
        "description": data.get("description", ""),
        "email": data.get("email", ""),
        "issue_number": int(args.issue_number),
        "issue_title": args.issue_title,
    }
    with open("/tmp/submission_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Download model
    download_model(data["model_url"], "/tmp/submission_model.pth")
    
    print("[INFO] Submission parsed successfully ✓")


if __name__ == "__main__":
    main()
