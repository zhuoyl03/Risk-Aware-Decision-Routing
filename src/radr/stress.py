"""Frozen-router stress test on actual blurred and JPEG-compressed MEDIC images."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax

from .models import LossMixture, predict_loss
from .predict import image_features
from .risk import allocate, choose_actions, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="out/features/test.npz")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--router", default="out/routing/router_seed17.joblib")
    parser.add_argument("--checkpoint", default="out/models/stage2_best.pth")
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--output", default="reports")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("samples must be positive")
    torch.set_num_threads(4)
    d = dict(np.load(args.features, allow_pickle=False))
    ids = np.sort(np.random.default_rng(2026).choice(len(d["features"]), min(args.samples, len(d["features"])), replace=False))
    d = {k: v[ids] for k,v in d.items()}
    bundle = joblib.load(args.router)
    paths = [str(Path(args.data_dir)/p) for p in d["paths"]]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows, timings = [], {}
    for corruption in ["clean", "blur", "jpeg"]:
        start = time.monotonic()
        print(f"Stress: {corruption}, {len(ids)} real images", flush=True)
        if corruption == "clean":
            f, logits = d["features"], d["logits"]
        else:
            f, logits = image_features(paths, args.checkpoint, bundle["preprocessing"]["checkpoint_sha256"], corruption)
        timings[corruption] = time.monotonic()-start
        z = bundle["expert_scaler"].transform(f)
        probs = np.stack([softmax(logits, axis=1)]+[h.predict_proba(z) for h in bundle["heads"]], axis=1)
        raw = np.concatenate([bundle["pca"].transform(z), probs.reshape(len(z), -1)], axis=1)
        model = LossMixture(bundle["input_dim"], 3)
        model.load_state_dict(bundle["model_state"])
        model.eval()
        q = predict_loss(model, bundle["router_scaler"].transform(raw).astype(np.float32))
        base_entropy = -(probs[:, 0]*np.log(np.maximum(probs[:, 0], 1e-12))).sum(1)
        policies = dict(entropy=(np.zeros(len(ids), dtype=int), base_entropy),
                        moe_mean=choose_actions(q, 0), moe_tail=choose_actions(q, 1))
        for policy, (actions, scores) in policies.items():
            pred = probs.argmax(-1)[np.arange(len(ids)), actions]
            for budget in [0., .1, .2]:
                m = metrics(pred, d["labels"], d["severity"], allocate(scores, budget))
                rows.append(dict(corruption=corruption, policy=policy, budget=budget, **m))
    pd.DataFrame(rows).to_csv(out/"image_corruptions.csv", index=False)
    (out/"corruption_protocol.json").write_text(json.dumps(dict(
        seed=2026, router_seed=bundle["seed"], sample_count=len(ids), test_indices=ids.tolist(),
        feature_sha256=hashlib.sha256(Path(args.features).read_bytes()).hexdigest(),
        blur="PIL GaussianBlur radius=2 after Resize256/CenterCrop224", jpeg="PIL JPEG quality=15 after crop",
        seconds=timings, policy_frozen=True, no_recalibration=True,
        note="Paired corruption sensitivity on a test subset, not adversarial robustness or unseen-event validation"), indent=2))
    print(pd.DataFrame(rows).query("budget == 0.2").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
