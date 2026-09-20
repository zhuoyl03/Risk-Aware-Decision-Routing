# What the experiments actually showed

The original ViT remains a useful baseline. On 15,634 deduplicated test images,
reviewing the 20% with highest entropy leaves weighted error cost **0.0614**
per image. The mean-risk mixture leaves **0.0652**, 6.2%
higher cost. The empirical guard supported 0/3 switches.

| Policy | Mean cost ↓ | CVaR90 ↓ | Selective accuracy ↑ | Severe errors left ↓ |
|---|---:|---:|---:|---:|
| Entropy baseline | 0.0614 | 0.6140 | 95.29% | 37.0 |
| Weighted head / entropy | 0.0697 | 0.6972 | 93.55% | 27.0 |
| Learned risk / fixed model | 0.0656 | 0.6560 | 94.34% | 32.7 |
| Single-head L2D | 0.0825 | 0.8253 | 93.16% | 43.7 |
| Mixture / mean risk | 0.0652 | 0.6520 | 94.18% | 29.7 |
| Mixture / tail risk | 0.0828 | 0.8283 | 91.42% | 24.3 |
| Guarded deployment | 0.0614 | 0.6140 | 95.29% | 37.0 |

Values are means over seeds 17, 42, 73. See summary.csv for standard deviations.
Review count is 3,126/15,634 (19.995%), identical across policies. Selective accuracy
describes only unreviewed cases. Costs 1/3/8 are illustrative severity weights.

## The useful negative result

Selected tail weights: seed 17: 0, seed 42: 0, seed 73: 0. Pure conditional CVaR
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
- paired_bootstrap.csv: paired test-image cost differences for seed 17, conditional
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
