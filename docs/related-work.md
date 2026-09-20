# Related work and what this project takes from it

Checked against primary paper pages on 2026-09-18. This is a focused reading list,
not a systematic novelty review. An arXiv posting is not necessarily peer-reviewed.

| Work | Why it matters here | What we actually do |
|---|---|---|
| [Mozannar & Sontag, ICML 2020: Consistent Estimators for Learning to Defer to an Expert](https://proceedings.mlr.press/v119/mozannar20b.html) | Deferral belongs in the decision problem, with expert behavior taken seriously. | Learn model-action costs; simulate reviewer quality explicitly because observed human decisions are absent. No consistency guarantee is imported. |
| [Montreuil et al., ICML 2025: Adversarial Robustness in Two-Stage Learning-to-Defer](https://arxiv.org/abs/2502.01027) | Routing itself can be manipulated, even when the underlying predictors are fixed. | Freeze policies for corruption checks. We do not implement SARD, optimize adversarial attacks, or claim adversarial robustness. |
| [Liu et al., February 2026: When More Experts Hurt](https://arxiv.org/abs/2602.17144) | Multi-expert L2D can underfit because identifying suitable experts is difficult. | Include one-component and multi-component cost predictors, plus fixed-head baselines. The negative result is consistent with the need for caution; it does not verify the paper's mechanism or reproduce PiCCE. |
| [Mulumudi et al., February 2026: On the Generalization and Robustness in CVaR](https://arxiv.org/abs/2602.18053) | Tail estimation and quantile sensitivity can make CVaR decisions unstable, especially with scarce tail evidence. | Use an explicit discrete loss distribution, exact fractional tails, repeated fits, and an empirical adoption guard. We do not implement truncated median-of-means CVaR or inherit its guarantees. |
| [Montreuil et al., revised May 2026: Learning-to-Defer with Expert-Conditional Advice](https://arxiv.org/abs/2603.14324) | Choosing an expert can be coupled to choosing additional information for that expert. | Keep the action space small. Advice acquisition is deferred because MEDIC does not provide its outcomes or costs. |
| [Rockafellar & Uryasev: Optimization of Conditional Value-at-Risk](https://sites.math.washington.edu/~rtr/papers.html) | The foundational tail-risk formulation. | For the finite four-point support, compute tail mass directly, including fractional probability at the boundary. |
| [Alam et al., 2021: MEDIC](https://arxiv.org/abs/2108.12828) | Supplies disaster-image annotations from multiple sources and tasks. | Evaluate informativeness with damage severity as a cost annotation; audit exact duplicates and report source slices. |

## A defensible contribution statement

This repository provides an auditable, local experiment connecting severity-aware
cost distributions, model selection, review capacity, and an empirical fallback
decision. It shows a concrete tradeoff: tail-focused routing can reduce the number
of unreviewed severe mistakes while increasing total weighted error. The adoption
guard declines unsupported switches in this study.

This integration and its evidence are useful engineering work. It is not yet an
established new algorithm, theorem, benchmark standard, or superior routing method.
The connection to robust systems is the explicit deployment contract and fallback;
the connection to empirical AI safety is the evaluation of expensive mistakes,
reviewer assumptions, and behavior under image degradation.

## Questions worth pursuing next

1. With actual specialist/reviewer outcomes, can the router learn *comparative*
   advantage rather than assume that review fixes a mistake?
2. Does a guard calibrated on one event continue to make useful adoption decisions
   on another, under a strict capacity budget?
3. Which tail metric remains informative when successful deferral makes most
   residual costs zero? CVaR90 becomes proportional to the mean in our main table.
4. Can a cheaper observable representation support routing before all expert heads
   execute? The current experiment measures no compute saving.

These need fresh evaluation data and a more complete literature comparison before
being presented as novel research contributions.
