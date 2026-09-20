"""Explicit MEDIC labels and an auditable, fail-visible image manifest."""

import hashlib
from pathlib import Path

import pandas as pd

LABELS = {"informative": 0, "not_informative": 1}
SEVERITIES = {"little_or_none": 0, "mild": 1, "severe": 2}


def source_from_path(path):
    parts = Path(path).parts
    return parts[1] if len(parts) > 1 and parts[0] == "data" else parts[0]


def build_manifests(data_dir):
    """Train wins overlap, then dev. Hash bytes, not unreliable image_id fields.

    Exact hashes do not detect visually similar images or re-encoded duplicates.
    Missing/corrupt images are counted rather than silently reducing denominators.
    """
    root = Path(data_dir).resolve()
    seen = {}
    frames, audit = {}, {}
    for split in ("train", "dev", "test"):
        path = root / f"MEDIC_{split}.tsv"
        frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        counts = dict(input_rows=len(frame), invalid_labels=0, missing_files=0,
                      duplicate_within_split=0, overlap_earlier_split=0)
        rows = []
        for row in frame.to_dict("records"):
            if row["informative"] not in LABELS or row["damage_severity"] not in SEVERITIES:
                counts["invalid_labels"] += 1
                continue
            full = (root / row["image_path"]).resolve()
            if not full.is_relative_to(root):
                raise ValueError(f"Image path escapes data directory: {row['image_path']}")
            if not full.is_file():
                counts["missing_files"] += 1
                continue
            digest = hashlib.sha256(full.read_bytes()).hexdigest()
            if digest in seen:
                key = "duplicate_within_split" if seen[digest] == split else "overlap_earlier_split"
                counts[key] += 1
                continue
            seen[digest] = split
            rows.append(dict(path=row["image_path"], sha256=digest,
                             label=LABELS[row["informative"]],
                             severity=SEVERITIES[row["damage_severity"]],
                             source=source_from_path(row["image_path"])))
        frames[split] = pd.DataFrame(rows)
        counts["retained_before_decode"] = len(rows)
        audit[split] = counts
    return frames, audit
