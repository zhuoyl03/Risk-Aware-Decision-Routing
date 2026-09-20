# Experiment protocol — frozen before routing evaluation

Date: 2026-09-18. This extends the existing MEDIC informativeness classifier.

Question: at a fixed review capacity, does predicting the distribution of
severity-weighted mistakes help route costly errors better than confidence alone?

## Data and separation

- Keep the existing stage-2 ViT fixed. It was trained on the official training
  split and selected on official development loss. Historical test accuracy was
  already viewed; this is a retrospective benchmark, not an untouched final test.
- Extract deterministic image features, logits, labels, severity and source.
  Labels/severity are training targets and evaluation metadata, never router inputs.
- Audit missing files, exact byte duplicates and overlap across official splits.
  Exclude cross-split duplicates from downstream splits, giving train precedence,
  then dev. Keep one copy within each split. Report every exclusion.
- Fit lightweight expert heads on a deterministic subset of at most 12,000
  training images. All experts share the same frozen image encoder.
- Split dev into router fitting and calibration halves, stratified by label and
  severity. Fit routers only on fitting data; choose hyperparameters on calibration.
  The encoder's earlier dev-based checkpoint selection remains a limitation.
- Evaluate the frozen choices on test. Seeds 17, 42, 73 repeat router fitting and
  dev partitioning, not backbone training. Keep all runs, including negative ones.

## Costs and decisions

- Task labels: informative=0, not_informative=1, matching the original sorted map.
- Error costs: little_or_none=1, mild=3, severe=8. Correct predictions cost zero.
  These are illustrative policy weights, not measured humanitarian harm.
- Compare entropy, random review, learned expected error cost, and a conditional
  loss-distribution router. Compare single base model and learned expert choice.
- Human review is a simulation: report unresolved model error cost with deferred
  cases set to zero (perfect-review upper bound), plus an imperfect-review
  sensitivity analysis. Do not call selective accuracy end-to-end system accuracy.
- Review budgets: 0, 5, 10, 20, 30, 40 percent. Batch top-k allocation uses only
  predictions; exactly floor(budget * batch size) cases may be reviewed.
- CVaR90 is the average loss in the worst 10 percent of cases, with fractional
  boundary mass. Conditional CVaR is a routing score, not a statistical guarantee.

## Comparison and reporting

Primary operating point: 20 percent review. Report mean unresolved weighted
error, CVaR90, severe-error recall, selective accuracy and actual review fraction.
Use calibration mean/CVaR objective to choose the tail weight from a small fixed
grid; retain plain expected-cost and tail-only ablations. Report paired bootstrap
intervals over test images for a representative seed, clearly conditional on the
fitted models. Report seed spread separately.

Stress checks: review capacity reduction; source-specific slices; severity-mixture
reweighting; imperfect reviewer costs. These are sensitivity analyses, not proof
of unseen-event robustness or adversarial robustness. The router must never see
true test severity, labels, or expert correctness when choosing actions.

Do not optimize the README headline after seeing test results. More experts and
CVaR can fail. A well-measured negative result is part of the deliverable.
