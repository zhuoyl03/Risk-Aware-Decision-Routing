"""Delivery checks that need no training, model downloads, or network access."""
import hashlib
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path


def main():
    env = dict(os.environ, PYTHONPATH=str(Path("src").resolve()))
    command = [sys.executable, "-m", "radr.predict", "--feature-cache", "out/features/test.npz", "--budget", "1.2"]
    invalid = subprocess.run(command, env=env, capture_output=True, text=True)
    assert invalid.returncode == 2 and "budget must be" in invalid.stderr
    fake = Path(".local/tmp/invalid-cache.npz")
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_bytes(b"not an extracted feature cache")
    wrong = subprocess.run([sys.executable, "-m", "radr.predict", "--feature-cache", str(fake)],
                           env=env, capture_output=True, text=True)
    assert wrong.returncode != 0 and "does not match" in wrong.stderr
    checked_links = 0
    for p in [Path("README.md"), Path("REVIEW.zh-CN.md"), *Path("docs").glob("*.md"), Path("reports/results.md")]:
        for url in re.findall(r"\]\(([^)]+)\)", p.read_text()):
            if not url.startswith(("http:", "https:", "#")):
                assert (p.parent/url.split("#")[0]).exists(), (str(p), url)
                checked_links += 1
    html = Path("reports/index.html")
    for url in re.findall(r'(?:src|href)="([^"]+)"', html.read_text()):
        assert (html.parent/url).exists(), url
    wheel = Path("out/dist/risk_aware_decision_routing-0.2.0-py3-none-any.whl")
    with zipfile.ZipFile(wheel) as z:
        for p in Path("src/radr").glob("*.py"):
            assert z.read("radr/"+p.name) == p.read_bytes(), p.name
    provenance = json.loads(Path("reports/provenance.json").read_text())
    for p, digest in provenance["source_sha256"].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest, p
    tests = Path("out/tests.log").read_text()
    assert "Ran 23 tests" in tests and tests.rstrip().endswith("OK")
    repeat = json.loads(Path("reports/reproducibility.json").read_text())
    assert repeat["same_seed_report_csvs_identical"]
    check = subprocess.run(["git", "diff", "--check"], capture_output=True, text=True)
    assert check.returncode == 0, check.stdout
    for path in ["data/MEDIC_train.tsv", "out/models/stage2_best.pth", ".local/original-src/train.py", "build/"]:
        assert subprocess.run(["git", "check-ignore", "-q", path]).returncode == 0, path
    validation = dict(
        date="2026-09-18", environment="WSL Ubuntu-20.04, RADR Conda, RTX 4070 Laptop GPU",
        tests=dict(command="PYTHONPATH=src python -m unittest discover -s tests -v", passed=23, failed=0),
        full_pipeline=dict(command="RADR_PYTHON=/home/leo/anaconda3/envs/RADR/bin/python bash scripts/reproduce.sh --full",
                           outcome="passed; extraction, three-seed experiments, real-image stress, figures and demo"),
        repeatability=dict(command="bash scripts/reproduce.sh --cached", outcome="all report CSVs byte-identical to the final full run"),
        runtime=dict(command="PYTHONPATH=src python scripts/validate_runtime.py", outcome="passed", details="runtime_validation.json"),
        packaging=dict(command="python -m pip wheel . --no-deps --no-build-isolation -w out/dist",
                       outcome="passed; wheel source matches working tree; isolated target install and console help tested"),
        invalid_budget_exit_code=invalid.returncode, mismatched_feature_cache="rejected before loading",
        markdown_local_links_checked=checked_links, html_local_resources="all present",
        scientific_figures="five PNGs visually inspected; SVG counterparts generated",
        git_diff_check="passed", original_source="preserved byte-for-byte; see runtime_validation.json",
        not_run=["backbone retraining from scratch", "fresh event-held-out evaluation", "observed human-review experiment", "formal robustness certification"],
        publication="local only; no commit, push, or upload")
    Path("reports/validation.json").write_text(json.dumps(validation, indent=2))
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
