"""Verify partner handoff files locally without importing ML libraries or training."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "models/artifact_manifest.json").read_text(encoding="utf-8"))
    problems = []
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        if not path.is_file():
            problems.append(f"Missing: {entry['path']} ({entry['delivery']})")
            continue
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if path.stat().st_size != entry["bytes"] or digest != entry["sha256"]:
            problems.append(f"Artifact differs from evaluated version: {entry['path']}")
    if problems:
        raise SystemExit("\n".join(problems) + "\nSee README.md: obtain the exact final weight files; do not retrain.")
    print(f"Verified {len(manifest['files'])} final model artifacts. Ready for local inference.")


if __name__ == "__main__":
    main()
