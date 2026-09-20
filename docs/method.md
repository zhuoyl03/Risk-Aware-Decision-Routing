# How a batch becomes a review queue

The task is binary image informativeness. The original label map is preserved:
informative=0, not_informative=1. Damage severity is a separate annotation used
to define training/evaluation costs. True severity is never available to routing.

## From images to model choices

One fixed ViT produces a 768-dimensional image representation. Three heads make
predictions: the original fine-tuned head, an L2-regularized logistic head, and an
otherwise identical logistic head fitted with severity sample weights. New heads
use 12,000 randomly selected, deduplicated train images (subset seed 2026).

Image features are standardized using training statistics. A 16-component PCA is
also fitted on train. The router sees those components and six class probabilities
from the three heads. A final scaler is fitted only on the router-fit subset.
Neither labels, severity, source identifiers nor expert correctness are inputs.

Image inference uses float32 without autocast; cuDNN and matrix-multiplication
TF32 are disabled in both extraction and the image CLI. A real-image check found
batch-dependent differences with the earlier default TF32 setting. Rebuilding the
cache with the explicit setting reduced the maximum checked feature discrepancy
to about 1.5e-6 and produced identical predictions and review choices. Final results
were regenerated; the earlier cache is not used for the presented tables.

Each risk component is a 22→32→12 MLP with tanh. A softmax gate mixes three
components. Each component emits a four-bin loss distribution for each of the
three model actions. The output is:

`q[j,k|x] = sum_m gate[m|x] * component[m,j,k|x]`.

The loss support is `[0,1,3,8]`: zero if the model is correct, otherwise the
severity weight. Negative log likelihood is averaged over all three observed
model outcomes on router-fit images. A one-component model is an ablation.
AdamW, learning rate .005, weight decay .01, at most 250 full-batch epochs and
25-epoch patience are fixed. Early stopping uses selection NLL.

This is direct learning of conditional outcome costs. It is related to two-stage
learning to defer, but is not an implementation of SARD, PiCCE or a consistent
surrogate theorem from those papers. Complete classifier labels permit evaluating
all three model outcomes offline. Actual human decisions are unavailable.

## Mean versus tail priority

For each action, expected cost is `sum_k q[j,k|x] * cost[k]`.
Conditional CVaR90 averages the worst 10% probability mass of the predicted cost
distribution. Fractional mass at the boundary is included; merely averaging
values greater than a percentile is incorrect for discrete distributions.

Priority is `(1-lambda)*expected_cost + lambda*conditional_CVaR90`.
Choose the model action with minimum priority. Review the highest-priority
`floor(batch_size * budget)` images; other images use their selected model.
Ties are deterministic in input order. This is batch allocation, not an online
queueing system with latency or arrival-rate guarantees.

Lambda is selected from 0/.25/.5/.75/1 on the selection subset at 20% review,
minimizing empirical mean residual cost + .25*empirical CVaR90. A lower lambda
wins ties. All three runs selected zero.

## The adoption guard

Dev is split into 3,065 router-fit, 1,533 selection and 1,533 guard images per
seed, stratified by label×severity. On guard images, compare the selected
candidate with original-model entropy routing under the same 20% capacity.
Use paired image-level bootstrap resampling (1,000 draws) of their cost difference.
Accept only if the 97.5th percentile is strictly below zero; otherwise use entropy.
The guard's supported budget is part of the contract; changing capacity falls back.

The guard set is independent of *router* fitting and selection. It is not fully
independent of the historical encoder, which was selected on original dev loss.
Percentile-bootstrap intervals are empirical evidence with assumptions, not
distribution-free or shift-robust safety certificates.

## Metrics with honest denominators

- **Mean unresolved cost:** sum of unreviewed model-error costs divided by all
  batch images. Perfect review is assumed only for this residual metric.
- **Empirical CVaR90:** worst 10% of per-image residual costs, over the whole batch.
- **Selective accuracy:** accuracy among unreviewed images only; undefined if none.
- **Severe-error capture:** fraction of a policy's own severe model errors reviewed;
  undefined when it makes no severe errors. Switching heads changes the denominator.
- **Severe errors remaining:** an absolute count, useful for comparing policies
  that choose different heads. It is not equivalent to severity-weighted total cost.

If fewer than 10% of all images have positive residual loss, CVaR90 is exactly
mean_loss/0.1. That happens at the main 20% budget. The two metrics should not be
counted as independent wins. Conditional CVaR of a predicted distribution is a
different object and can still change actions and ranking.

## Stress tests and scope

Source slices keep the global routing decisions fixed. Severity-mixture stress
reweights severe/nonsevere losses to specified mixture proportions without
rerouting; the resulting review share can differ. Reviewer-error stress assigns
an independent probability of an incorrect human decision on each deferred image,
with the same severity error cost. It reports expected mean cost, not CVaR of a
real or simulated reviewer trajectory.

The image study chooses 3,000 test images once with seed 2026, applies blur radius
2 or JPEG quality 15 after the fixed crop, and reruns ViT. The seed-17 router is
frozen. These are paired corruption checks, not adaptive attacks or event holdouts.

All heads use a shared backbone and run before routing. Cost of inference and
human-review delay are not optimized or measured as deployment benefits.
