"""Check real inference, saved decisions, report accounting, and preserved source."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from radr.predict import image_features, route_batch
from radr.risk import metrics


def main():
    torch.set_num_threads(4)
    bundle = joblib.load("out/routing/router_seed17.joblib")
    d = dict(np.load("out/features/test.npz", allow_pickle=False))
    paths = [str(Path("data")/p) for p in d["paths"][:20]]
    features, logits = image_features(paths, "out/models/stage2_best.pth",
                                     bundle["preprocessing"]["checkpoint_sha256"])
    np.testing.assert_allclose(features, d["features"][:20], atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(logits, d["logits"][:20], atol=1e-4, rtol=1e-4)
    direct = route_batch(bundle, features, logits, .2)
    cached = route_batch(bundle, d["features"][:20], d["logits"][:20], .2)
    for i in [0, 2, 3]:
        np.testing.assert_array_equal(direct[i], cached[i])
    assert direct[2].sum() == 4
    actions, scores, review, pred, reason = route_batch(bundle, d["features"], d["logits"], .2)
    saved = dict(np.load("out/routing/test_decisions.npz", allow_pickle=False))
    for key, current in [("actions", actions), ("scores", scores), ("review", review), ("predictions", pred)]:
        np.testing.assert_allclose(current, saved[f"17_guarded_{key}"], atol=1e-7)
    actual = metrics(pred, d["labels"], d["severity"], review)
    frame = pd.read_csv("reports/routing_curves.csv")
    row = frame[(frame.seed == 17)&(frame.policy == "guarded")&(frame.budget == .2)].iloc[0]
    for key in ["mean_cost", "cvar90", "selective_accuracy", "severe_errors_remaining"]:
        assert np.isclose(actual[key], row[key])
    history = json.loads(Path("out/routing/training_history.json").read_text())
    for seed, item in history.items():
        parts = [set(item[k]) for k in ["fit_indices", "calibration_indices", "guard_indices"]]
        assert not parts[0]&parts[1] and not parts[0]&parts[2] and not parts[1]&parts[2]
        assert len(set.union(*parts)) == 6131
    preserved = {}
    for name in ["train.py", "eval.py", "models.py", "process_data.py"]:
        new = hashlib.sha256((Path("archive/legacy")/name).read_bytes()).hexdigest()
        old = hashlib.sha256((Path(".local/original-src")/name).read_bytes()).hexdigest()
        assert new == old
        preserved[name] = new
    result = dict(real_image_inference="20 images; matches cached features and routing",
                  max_feature_difference=float(np.abs(features-d["features"][:20]).max()),
                  max_logit_difference=float(np.abs(logits-d["logits"][:20]).max()),
                  batch_review_slots=int(direct[2].sum()),
                  full_test_deployment="matches saved experiment actions, scores, masks and predictions",
                  recomputed_metrics="matches CSV report", dev_partition_check="disjoint and exhaustive for all seeds",
                  original_source_sha256=preserved, policy_reason=reason)
    Path("reports/runtime_validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
