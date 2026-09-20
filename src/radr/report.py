"""Render presentation figures and a local report from saved experiment tables."""

import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = {"random": "Random review", "entropy": "Entropy baseline", "single_model_risk": "Learned risk / fixed model",
          "weighted_entropy": "Weighted head / entropy",
          "l2d_single_head": "Single-head L2D", "moe_mean": "Mixture / mean risk", "moe_tail": "Mixture / tail risk",
          "calibrated_tail": "Calibrated candidate", "guarded": "Guarded deployment"}
COLORS = {"random": "#9da7b1", "entropy": "#187f82", "moe_mean": "#315d9c", "moe_tail": "#c37838",
          "weighted_entropy": "#729044",
          "guarded": "#23354d", "single_model_risk": "#96618a", "l2d_single_head": "#ae665d"}


def save(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 14, "axes.titleweight": "bold", "axes.labelcolor": "#344356",
                         "text.color": "#21334a", "axes.edgecolor": "#ccd4dc", "axes.spines.top": False,
                         "axes.spines.right": False, "grid.color": "#dce3e9", "grid.alpha": .55,
                         "figure.facecolor": "#fbfcfd", "axes.facecolor": "#fbfcfd", "legend.frameon": False,
                         "svg.fonttype": "none"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", default="reports")
    args = parser.parse_args()
    root = Path(args.reports)
    figures = root/"figures"
    figures.mkdir(parents=True, exist_ok=True)
    curves = pd.read_csv(root/"routing_curves.csv")
    guards = pd.read_csv(root/"guard_decisions.csv")
    provenance = json.loads((root/"provenance.json").read_text())
    n_test = int(provenance["feature_metadata"]["test"]["rows"])
    seeds = sorted(curves.seed.unique())
    n_seeds = len(seeds)
    accepted_count = int(guards.accepted.sum())
    guard_n = ", ".join(str(v) for v in sorted(guards.guard_n.unique()))
    style()
    policies = ["random", "entropy", "weighted_entropy", "moe_mean", "moe_tail"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for ax, metric, label in zip(axes, ["mean_cost", "severe_errors_remaining"],
                                 ["Unresolved weighted error per image", "Severe errors left unreviewed"]):
        for policy in policies:
            g = curves[curves.policy == policy].groupby("budget")[metric].agg(["mean", "std"])
            x, y, sd = g.index.to_numpy()*100, g["mean"].to_numpy(), g["std"].fillna(0).to_numpy()
            ax.plot(x, y, marker="o", markersize=4, linewidth=2, color=COLORS[policy], label=LABELS[policy])
            ax.fill_between(x, y-sd, y+sd, color=COLORS[policy], alpha=.12)
        ax.set(xlabel="Review capacity (%)", ylabel=label, ylim=(0, None))
        ax.grid(axis="y")
    axes[0].set_title("A strong baseline is hard to beat", loc="left")
    axes[1].set_title("Cost and severe-error counts differ", loc="left")
    axes[0].legend(fontsize=9)
    fig.supxlabel(f"{n_test:,} deduplicated test images · bands: ±1 SD over {n_seeds} router seeds · perfect-review simulation", fontsize=9)
    save(fig, figures/"risk_coverage")

    operating = curves[curves.budget == .2]
    display = ["entropy", "weighted_entropy", "single_model_risk", "l2d_single_head", "moe_mean", "moe_tail", "guarded"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, metric, title in zip(axes, ["mean_cost", "severe_errors_remaining"],
                                 ["Weighted error / image", "Severe errors left unreviewed"]):
        grouped = operating.groupby("policy")[metric].agg(["mean", "std"]).loc[display]
        y = np.arange(len(display))
        ax.barh(y, grouped["mean"], xerr=grouped["std"].fillna(0), height=.62,
                color=[COLORS[p] for p in display], error_kw={"capsize": 3, "ecolor": "#344356"})
        ax.set_yticks(y, [LABELS[p] for p in display] if ax is axes[0] else [""]*len(y))
        ax.invert_yaxis()
        ax.grid(axis="x")
        ax.set_title(title+" ↓", loc="left")
    fig.suptitle("At 20% review, keep the tradeoff visible", x=.05, ha="left", fontsize=17, weight="bold")
    fig.supxlabel(f"Same review budget for every policy · error bars: ±1 SD · {accepted_count}/{n_seeds} candidates passed the guard", fontsize=9)
    save(fig, figures/"operating_point")

    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    for i, row in guards.iterrows():
        ax.errorbar(row["mean"], i, xerr=[[row["mean"]-row.ci_low], [row.ci_high-row["mean"]]],
                    fmt="o", color="#315d9c", capsize=6, linewidth=2, markersize=7)
    ax.axvline(0, color="#c37838", linestyle="--", linewidth=1.5)
    ax.set_yticks(range(len(guards)), [f"Seed {s}" for s in guards.seed])
    ax.set(xlabel="Candidate cost minus entropy cost (negative favors candidate)",
           title=f"The guard supported {accepted_count} of {n_seeds} candidate switches")
    ax.grid(axis="x")
    fig.supxlabel(f"{guard_n} guard images per seed · paired 95% bootstrap intervals · model selection used a different dev subset", fontsize=9)
    save(fig, figures/"guard_evidence")

    stress = pd.read_csv(root/"stress_tests.csv")
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    for policy in ["entropy", "moe_mean", "moe_tail"]:
        g = stress[(stress.policy == policy)&(stress.scenario == "imperfect_reviewer")].groupby("level").mean(numeric_only=True)
        ax.plot(g.index*100, g.mean_cost, "o-", color=COLORS[policy], label=LABELS[policy], linewidth=2)
    ax.set(xlabel="Assumed reviewer error probability (%)", ylabel="Expected system weighted error / image",
           title="Deferral helps only if the reviewer can help")
    ax.legend()
    ax.grid(axis="y")
    fig.supxlabel("20% review · reviewer errors are hypothetical, independent, and equally likely for every deferred case", fontsize=9)
    save(fig, figures/"reviewer_sensitivity")

    sections = [
        ("risk_coverage", "What does the review budget buy?", "These curves compare equal batch capacities. Shading is variation across router seeds, not a confidence interval. Deferred cases have zero residual loss here: a perfect-review upper bound."),
        ("operating_point", "One operating point, two priorities", "Weighted cost and the number of severe mistakes answer different questions. Routing to a different model changes which errors exist, so severe-error capture percentages alone are not directly comparable across models."),
        ("guard_evidence", "Evidence before a switch", f"The candidate must beat entropy on a separate guard subset before deployment. {accepted_count} of {n_seeds} candidates passed. This is an empirical check, not a guarantee under distribution shift."),
        ("reviewer_sensitivity", "Review is a resource, not an oracle", "No human reviewer was observed in these experiments. This sensitivity analysis assigns a hypothetical independent reviewer error rate and includes the cost of introducing errors on previously correct cases.")]
    if (root/"image_corruptions.csv").is_file():
        corruption = pd.read_csv(root/"image_corruptions.csv")
        corruption_meta = json.loads((root/"corruption_protocol.json").read_text())
        if corruption_meta["feature_sha256"] != provenance["feature_sha256"]["test"]:
            raise ValueError("Corruption results use different features; rerun radr.stress before making this report")
        fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
        conditions = ["clean", "blur", "jpeg"]
        for j, policy in enumerate(["entropy", "moe_mean", "moe_tail"]):
            ys = corruption[(corruption.policy == policy)&(corruption.budget == .2)].set_index("corruption").loc[conditions, "mean_cost"]
            ax.bar(np.arange(3)+(j-1)*.24, ys, width=.22, color=COLORS[policy], label=LABELS[policy])
        ax.set_xticks(range(3), ["Original images", "Gaussian blur (r=2)", "JPEG (quality=15)"])
        ax.set(ylabel="Unresolved weighted error / image", title="A frozen router under image degradation")
        ax.legend()
        ax.grid(axis="y")
        fig.supxlabel(f"Same {corruption_meta['sample_count']:,} real test images in each condition · 20% review · seed {corruption_meta['router_seed']} · no recalibration", fontsize=9)
        save(fig, figures/"image_corruptions")
        sections.append(("image_corruptions", "Actual image corruption, not just score noise", "Blur and JPEG degradation were applied to the same real images and passed through ViT again. This paired subset study measures corruption sensitivity; it does not establish adversarial or unseen-event robustness."))
    means = operating.groupby("policy").mean(numeric_only=True)
    entropy_cost = means.loc["entropy", "mean_cost"]
    candidate_cost = means.loc["moe_mean", "mean_cost"]
    ratio = (candidate_cost/entropy_cost-1)*100
    chosen_weights = ", ".join(f"seed {int(r.seed)}: {r.candidate_tail_weight:g}" for r in guards.itertuples())
    table = "\n".join(f"| {LABELS[p]} | {means.loc[p, 'mean_cost']:.4f} | {means.loc[p, 'cvar90']:.4f} | {means.loc[p, 'selective_accuracy']:.2%} | {means.loc[p, 'severe_errors_remaining']:.1f} |" for p in display)
    markdown = f"""# What the experiments actually showed

The original ViT remains a useful baseline. On {n_test:,} deduplicated test images,
reviewing the 20% with highest entropy leaves weighted error cost **{entropy_cost:.4f}**
per image. The mean-risk mixture leaves **{candidate_cost:.4f}**, {abs(ratio):.1f}%
{'higher' if ratio > 0 else 'lower'} cost. The empirical guard supported {accepted_count}/{n_seeds} switches.

| Policy | Mean cost ↓ | CVaR90 ↓ | Selective accuracy ↑ | Severe errors left ↓ |
|---|---:|---:|---:|---:|
{table}

Values are means over seeds {', '.join(map(str, seeds))}. See summary.csv for standard deviations.
Review count is {int(n_test*.2):,}/{n_test:,} ({int(n_test*.2)/n_test:.3%}), identical across policies. Selective accuracy
describes only unreviewed cases. Costs 1/3/8 are illustrative severity weights.

## The useful negative result

Selected tail weights: {chosen_weights}. Pure conditional CVaR
prioritizes severe-error protection differently, but it did not improve aggregate
weighted cost here. More mixture components also do not guarantee a better router.
The first, unguarded run is preserved in pilot/; the guard was an exploratory
extension after seeing that result. Neither run is a blind final evaluation.

When fewer than 10% of all cases retain nonzero loss, empirical CVaR90 equals
10 times mean loss and contributes no independent ranking information. Check the
table before treating these metrics as separate wins. Conditional predicted CVaR
can still change routing scores. That distinction matters when choosing metrics.

## Reading the evidence

- guard_decisions.csv: paired intervals on an independent router guard subset.
- paired_bootstrap.csv: paired test-image cost differences for seed {provenance['seeds'][0]}, conditional
  on fitted models; not variability from backbone retraining or event sampling.
- source_slices.csv: the same global queue evaluated by source. These are source
  slices, not leave-one-event-out or out-of-distribution generalization results.
- stress_tests.csv: assumed reviewer error and reweighted severity mixtures. The
  reweighting keeps decisions fixed, so group-specific review rates can change.
- image_corruptions.csv: a paired 3,000-image corruption study, with seed 17 frozen.
- provenance.json: source hashes, environment, checkpoint fingerprint and exclusions.

## Limits that change interpretation

The backbone was selected using the original dev split. Router fit/selection/guard
are disjoint, but dev is not fully independent of backbone selection. Historical
test results were viewed before this work. Exact hashes remove byte duplicates,
not near-duplicates. Reviewer outcomes and real-world harm costs are unobserved.
All three model heads share the same backbone and execute before routing, so this
experiment does not demonstrate compute savings. Further evidence needs real
review outcomes, event-held-out data and a fresh final evaluation.
"""
    (root/"results.md").write_text(markdown, encoding="utf-8")
    cards = "".join(f'<section><h2>{html.escape(title)}</h2><p>{html.escape(text)}</p><img src="figures/{name}.png" alt="{html.escape(title)}"></section>' for name,title,text in sections)
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Risk-Aware Decision Routing — Experiment report</title>
<style>body{{margin:0;background:#eef2f5;color:#24344a;font:17px/1.65 system-ui,sans-serif}}main{{max-width:1060px;margin:40px auto;padding:0 24px}}header{{background:#20374c;color:#f5f8fb;padding:44px;border-radius:16px}}header p{{max-width:780px;color:#dce7ef}}.eyebrow{{font-size:13px;letter-spacing:.15em;text-transform:uppercase;color:#9fced0}}h1{{font-size:42px;line-height:1.15;margin:15px 0}}h2{{font-size:25px;line-height:1.25}}section{{background:white;padding:30px 36px;border-radius:14px;margin:24px 0}}img{{width:100%;height:auto}}a{{color:#187f82}}.stats{{display:flex;gap:30px;flex-wrap:wrap;margin-top:24px}}.stat strong{{display:block;font-size:30px}}.stat span{{font-size:14px;color:#cee0e8}}footer{{font-size:14px;padding:20px 4px 40px}}code{{background:#edf2f5;padding:2px 5px;border-radius:3px}}</style>
<main><header><div class="eyebrow">MEDIC · empirical AI safety · local research report</div><h1>Knowing when to keep<br>the simpler decision rule.</h1><p>We tested learned expert routing and tail-risk scores against an entropy baseline. A separate evidence check decides whether the candidate earns a switch.</p><div class="stats"><div class="stat"><strong>{n_test:,}</strong><span>deduplicated test images</span></div><div class="stat"><strong>{n_seeds} seeds</strong><span>router fits, one fixed backbone</span></div><div class="stat"><strong>{accepted_count} / {n_seeds}</strong><span>candidates passed the guard</span></div></div></header>
<section><h2>What changed at 20% review?</h2><p>Entropy leaves cost <strong>{entropy_cost:.4f}</strong> per image; the mean-risk mixture leaves <strong>{candidate_cost:.4f}</strong>. The project's output is a reproducible comparison and an explicit fallback contract. No state-of-the-art performance claim is made.</p><p><a href="results.md">Detailed findings</a> · <a href="routing_curves.csv">Every policy and seed</a> · <a href="provenance.json">Provenance</a> · <a href="../docs/method.md">Method and assumptions</a></p></section>{cards}
<section><h2>What is still unknown?</h2><p>The costs and human reviewer quality are assumptions. Test data was examined during development; the backbone also used dev data for checkpoint selection. These results support an exploratory engineering study, not a certified safety claim.</p></section><footer>Generated from local CSV evidence by <code>python -m radr.report</code>. No image examples, dataset files, or weights are embedded. All measurements remain local.</footer></main></html>"""
    (root/"index.html").write_text(page, encoding="utf-8")
    print(f"Wrote {len(sections)} figures, results.md, and index.html")


if __name__ == "__main__":
    main()
