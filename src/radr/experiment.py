"""Repeatable experiments on cached real-image features; no test-time labels in routing."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .models import fit_loss_model, loss_targets, predict_loss
from .risk import (SEVERITY_COST, allocate, choose_actions, conditional_risk,
                   empirical_cvar, metrics, realized_loss)

BUDGETS = [0., .05, .1, .2, .3, .4]
TAIL_WEIGHTS = [0., .25, .5, .75, 1.]
EXPERTS = ["original_vit", "linear_erm", "severity_weighted_linear"]


def load_features(directory):
    data = {s: dict(np.load(Path(directory)/f"{s}.npz", allow_pickle=False)) for s in ("train", "dev", "test")}
    for s, d in data.items():
        n = len(d["labels"])
        if not n or any(len(v) != n for v in d.values()):
            raise ValueError(f"Empty or unaligned cache: {s}")
        if not np.isfinite(d["features"]).all() or not np.isfinite(d["logits"]).all():
            raise ValueError(f"Non-finite cache: {s}")
        if len(set(d["hashes"])) != n:
            raise ValueError(f"Duplicate image hashes in {s}")
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        if set(data[a]["hashes"]) & set(data[b]["hashes"]):
            raise ValueError(f"Image overlap between {a} and {b}")
    return data


def train_experts(data):
    train = data["train"]
    scaler = StandardScaler().fit(train["features"])
    z = {s: scaler.transform(d["features"]) for s, d in data.items()}
    heads = []
    for weighted in (False, True):
        model = LogisticRegression(C=.01, max_iter=500, random_state=2026)
        weights = SEVERITY_COST[train["severity"]] if weighted else None
        model.fit(z["train"], train["labels"], sample_weight=weights)
        heads.append(model)
    probabilities = {s: np.stack([softmax(d["logits"], axis=1)] +
                                 [head.predict_proba(z[s]) for head in heads], axis=1)
                     for s, d in data.items()}
    # PCA is fitted on training images, never dev/test.
    pca = PCA(n_components=16, random_state=2026, svd_solver="randomized").fit(z["train"])
    features = {s: np.concatenate([pca.transform(z[s]), probabilities[s].reshape(len(z[s]), -1)], axis=1)
                for s in data}
    return probabilities, features, dict(expert_scaler=scaler, heads=heads, pca=pca)


def evaluate_choice(probabilities, scores, actions, data, budget):
    predictions = probabilities.argmax(-1)[np.arange(len(actions)), actions]
    review = allocate(scores, budget)
    result = metrics(predictions, data["labels"], data["severity"], review)
    return result, predictions, review


def paired_bootstrap(delta, seed=2026, repetitions=1000, unit="individual test image"):
    rng = np.random.default_rng(seed)
    means = [delta[rng.integers(0, len(delta), size=len(delta))].mean() for _ in range(repetitions)]
    low, high = np.quantile(means, [.025, .975])
    return dict(mean=float(delta.mean()), ci_low=float(low), ci_high=float(high),
                resamples=repetitions, unit=unit, conditional_on_fitted_models=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="out/features")
    parser.add_argument("--output", default="reports")
    parser.add_argument("--artifacts", default="out/routing")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 73])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        parser.error("seeds must be unique nonnegative integers")
    out, artifacts = Path(args.output), Path(args.artifacts)
    out.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    start = time.monotonic()
    data = load_features(args.features)
    fingerprints = {s: hashlib.sha256((Path(args.features)/f"{s}.npz").read_bytes()).hexdigest() for s in data}
    print("Training shared frozen-feature expert heads...", flush=True)
    probabilities, raw_features, shared = train_experts(data)
    expert_rows = []
    for split in ("dev", "test"):
        for j, name in enumerate(EXPERTS):
            pred = probabilities[split][:, j].argmax(-1)
            expert_rows.append(dict(split=split, expert=name, **metrics(pred, data[split]["labels"],
                              data[split]["severity"], np.zeros(len(pred), dtype=bool))))
    pd.DataFrame(expert_rows).to_csv(out/"expert_baselines.csv", index=False)
    rows, slices, stress, selections, histories, bootstrap_rows, guards = [], [], [], [], {}, [], []
    masks = {}
    for seed in args.seeds:
        print(f"Fitting routing seed {seed}...", flush=True)
        dev = data["dev"]
        fit, held = train_test_split(np.arange(len(dev["labels"])), test_size=.5, random_state=seed,
                                   stratify=dev["labels"]*3+dev["severity"])
        cal, guard = train_test_split(held, test_size=.5, random_state=seed,
                                     stratify=(dev["labels"]*3+dev["severity"])[held])
        scaler = StandardScaler().fit(raw_features["dev"][fit])
        x = {s: scaler.transform(f).astype(np.float32) for s, f in raw_features.items()}
        target = loss_targets(probabilities["dev"].argmax(-1), dev["labels"], dev["severity"])
        single, hist_single = fit_loss_model(x["dev"][fit], target[fit], x["dev"][cal], target[cal], seed, components=1)
        mixture, hist_mix = fit_loss_model(x["dev"][fit], target[fit], x["dev"][cal], target[cal], seed, components=3)
        histories[str(seed)] = dict(single=hist_single, mixture=hist_mix,
                                    fit_indices=fit.tolist(), calibration_indices=cal.tolist(),
                                    guard_indices=guard.tolist())
        cp = predict_loss(mixture, x["dev"][cal])
        cal_data = {k: v[cal] for k, v in dev.items()}
        candidates = []
        for weight in TAIL_WEIGHTS:
            actions, scores = choose_actions(cp, weight)
            m, _, _ = evaluate_choice(probabilities["dev"][cal], scores, actions, cal_data, .2)
            objective = m["mean_cost"]+.25*m["cvar90"]
            candidates.append((objective, weight))
            selections.append(dict(seed=seed, tail_weight=weight, objective=objective, **m))
        _, chosen_weight = min(candidates)
        print(f"seed {seed}: calibration selected tail weight {chosen_weight}", flush=True)
        gp = predict_loss(mixture, x["dev"][guard])
        ga, gs = choose_actions(gp, chosen_weight)
        guard_data = {k: v[guard] for k, v in dev.items()}
        _, gpred, greview = evaluate_choice(probabilities["dev"][guard], gs, ga, guard_data, .2)
        gb = probabilities["dev"][guard, 0]
        ge = -(gb*np.log(np.maximum(gb, 1e-12))).sum(1)
        baseline_loss = realized_loss(gb.argmax(-1), guard_data["labels"], guard_data["severity"])
        candidate_loss = realized_loss(gpred, guard_data["labels"], guard_data["severity"])
        gd = np.where(greview, 0., candidate_loss)-np.where(allocate(ge, .2), 0., baseline_loss)
        guard_evidence = paired_bootstrap(gd, seed=2026+seed, unit="individual guard image")
        accepted = guard_evidence["ci_high"] < 0
        guards.append(dict(seed=seed, candidate_tail_weight=chosen_weight, accepted=bool(accepted),
                           guard_n=len(guard), **guard_evidence))
        print(f"seed {seed}: guard accepted={accepted}, interval={guard_evidence['ci_low']:.4f}, {guard_evidence['ci_high']:.4f}", flush=True)
        test_p = probabilities["test"]
        q = predict_loss(mixture, x["test"])
        qs = predict_loss(single, x["test"])
        mean, _ = conditional_risk(q)
        base = test_p[:, 0]
        n = len(base)
        zero = np.zeros(n, dtype=int)
        entropy = -(base*np.log(np.maximum(base, 1e-12))).sum(1)
        weighted_p = test_p[:, 2]
        weighted_entropy = -(weighted_p*np.log(np.maximum(weighted_p, 1e-12))).sum(1)
        policies = {
            "random": (zero, np.random.default_rng(seed).random(n)),
            "entropy": (zero, entropy),
            "weighted_entropy": (np.full(n, 2, dtype=int), weighted_entropy),
            "single_model_risk": (zero, mean[:, 0]),
            "l2d_single_head": choose_actions(qs, 0),
            "moe_mean": choose_actions(q, 0),
            "moe_tail": choose_actions(q, 1),
            "calibrated_tail": choose_actions(q, chosen_weight),
            "guarded": choose_actions(q, chosen_weight) if accepted else (zero, entropy),
        }
        seed_losses = {}
        for policy, (actions, scores) in policies.items():
            for budget in BUDGETS:
                used_actions, used_scores = (zero, entropy) if policy == "guarded" and budget != .2 else (actions, scores)
                m, pred, review = evaluate_choice(test_p, used_scores, used_actions, data["test"], budget)
                rows.append(dict(seed=seed, policy=policy, budget=budget, tail_weight=chosen_weight if policy=="calibrated_tail" else None, **m))
                if budget != .2:
                    continue
                losses = realized_loss(pred, data["test"]["labels"], data["test"]["severity"])
                residual = np.where(review, 0., losses)
                seed_losses[policy] = residual
                masks[f"{seed}_{policy}_actions"] = actions
                masks[f"{seed}_{policy}_scores"] = scores
                masks[f"{seed}_{policy}_predictions"] = pred
                masks[f"{seed}_{policy}_review"] = review
                for source in sorted(set(data["test"]["sources"])):
                    ix = data["test"]["sources"] == source
                    sm = metrics(pred[ix], data["test"]["labels"][ix], data["test"]["severity"][ix], review[ix])
                    slices.append(dict(seed=seed, policy=policy, source=source, **sm))
                for reviewer_error in [0., .05, .1, .2]:
                    # Expected cost, not CVaR of a sampled or real reviewer.
                    expected = residual+review*reviewer_error*SEVERITY_COST[data["test"]["severity"]]
                    stress.append(dict(seed=seed, policy=policy, scenario="imperfect_reviewer", level=reviewer_error,
                                       mean_cost=float(expected.mean())))
                severe = data["test"]["severity"] == 2
                for severe_share in [.25, .5, .75]:
                    reweighted = severe_share*residual[severe].mean()+(1-severe_share)*residual[~severe].mean()
                    stress.append(dict(seed=seed, policy=policy, scenario="severe_case_mix", level=severe_share,
                                       mean_cost=float(reweighted)))
        if seed == args.seeds[0]:
            for policy in policies:
                if policy != "entropy":
                    bootstrap_rows.append(dict(policy=policy, reference="entropy", seed=seed, budget=.2,
                                          **paired_bootstrap(seed_losses[policy]-seed_losses["entropy"])))
        bundle = dict(**shared, router_scaler=scaler, model_state=mixture.state_dict(),
                      input_dim=x["dev"].shape[1], tail_weight=chosen_weight, experts=EXPERTS,
                      seed=seed, preprocessing=json.loads((Path(args.features)/"metadata.json").read_text()),
                      default_budget=.2, guard_accepted=bool(accepted), guard_evidence=guard_evidence,
                      feature_fingerprints=fingerprints)
        joblib.dump(bundle, artifacts/f"router_seed{seed}.joblib")
    curves = pd.DataFrame(rows)
    curves.to_csv(out/"routing_curves.csv", index=False)
    pd.DataFrame(slices).to_csv(out/"source_slices.csv", index=False)
    pd.DataFrame(stress).to_csv(out/"stress_tests.csv", index=False)
    pd.DataFrame(selections).to_csv(out/"calibration_search.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(out/"paired_bootstrap.csv", index=False)
    pd.DataFrame(guards).to_csv(out/"guard_decisions.csv", index=False)
    (artifacts/"training_history.json").write_text(json.dumps(histories))
    np.savez_compressed(artifacts/"test_decisions.npz", **masks,
                        labels=data["test"]["labels"], severity=data["test"]["severity"],
                        sources=data["test"]["sources"], hashes=data["test"]["hashes"])
    summary = curves[curves.budget == .2].groupby("policy")[["mean_cost", "cvar90", "selective_accuracy", "severe_error_capture"]].agg(["mean", "std"])
    summary.to_csv(out/"summary.csv")
    provenance = dict(seeds=args.seeds, budgets=BUDGETS, tail_weights=TAIL_WEIGHTS,
                      selection_objective="calibration mean_cost + 0.25 * CVaR90 at budget 0.2",
                      severity_costs=SEVERITY_COST.tolist(), python=platform.python_version(),
                      torch=torch.__version__, sklearn=sklearn.__version__, seconds=time.monotonic()-start,
                      feature_sha256=fingerprints,
                      feature_metadata=json.loads((Path(args.features)/"metadata.json").read_text()),
                      source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted(Path("src/radr").glob("*.py"))},
                      protocol_sha256=hashlib.sha256(Path("docs/experiment-plan.md").read_bytes()).hexdigest())
    (out/"provenance.json").write_text(json.dumps(provenance, indent=2))
    print(summary.to_string(), flush=True)
    print(f"Completed in {time.monotonic()-start:.1f}s; reports in {out}", flush=True)


if __name__ == "__main__":
    main()
