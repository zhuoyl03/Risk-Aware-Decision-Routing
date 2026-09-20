"""The review-capacity contract used by the command-line demo."""

import numpy as np

from .risk import allocate, choose_actions


def guarded_decision(base_probabilities, loss_probabilities, tail_weight, accepted, budget):
    base = np.asarray(base_probabilities, dtype=float)
    if base.ndim != 2 or base.shape[1] != 2 or not np.isfinite(base).all():
        raise ValueError("Expected finite binary class probabilities")
    if (base < 0).any() or not np.allclose(base.sum(1), 1, atol=1e-5):
        raise ValueError("Class probabilities must be nonnegative and sum to one")
    if accepted and budget == .2:
        actions, scores = choose_actions(loss_probabilities, tail_weight)
        if len(actions) != len(base):
            raise ValueError("Unaligned action probabilities")
        reason = "candidate_passed_guard_at_this_budget"
    else:
        actions = np.zeros(len(base), dtype=int)
        scores = -(base*np.log(np.maximum(base, 1e-12))).sum(1)
        reason = "candidate_not_supported_by_guard" if not accepted else "budget_not_audited"
    return actions, scores, allocate(scores, budget), reason
