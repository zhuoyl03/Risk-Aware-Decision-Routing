"""Loss distributions, tail risk, and exact-capacity batch allocation.

No function that chooses actions accepts true labels or severity.
"""

import numpy as np

LOSS_VALUES = np.array([0., 1., 3., 8.])
SEVERITY_COST = LOSS_VALUES[1:]


def _probabilities(probabilities):
    p = np.asarray(probabilities, dtype=float)
    if p.ndim < 1 or p.shape[-1] != len(LOSS_VALUES):
        raise ValueError("Expected four loss probabilities per action")
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(-1), 1, atol=1e-5):
        raise ValueError("Probabilities must be finite, nonnegative, and sum to one")
    return p


def conditional_risk(probabilities, alpha=.9):
    """Return expected loss and CVaR_alpha of each discrete cost distribution."""
    p = _probabilities(probabilities)
    if not 0 <= alpha < 1:
        raise ValueError("alpha must be in [0, 1)")
    remaining = np.full(p.shape[:-1], 1-alpha, dtype=float)
    total = np.zeros_like(remaining)
    for k in range(len(LOSS_VALUES)-1, -1, -1):
        mass = np.minimum(p[..., k], remaining)
        total += mass * LOSS_VALUES[k]
        remaining = np.maximum(remaining-mass, 0)
    return p @ LOSS_VALUES, total / (1-alpha)


def empirical_cvar(losses, alpha=.9):
    """Average exactly the top (1-alpha) mass, including fractional boundary."""
    x = np.asarray(losses, dtype=float)
    if x.ndim != 1 or len(x) == 0 or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("losses must be a nonempty finite nonnegative vector")
    if not 0 <= alpha < 1:
        raise ValueError("alpha must be in [0, 1)")
    x = np.sort(x)[::-1]
    mass = (1-alpha)*len(x)
    full = int(np.floor(mass))
    boundary = (mass-full)*x[full] if full < len(x) else 0.
    return float((x[:full].sum()+boundary)/mass)


def allocate(scores, budget):
    """Review the top floor(n*budget) cases; stable input-order tie breaking."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("scores must be a finite vector")
    if not np.isfinite(budget) or not 0 <= budget <= 1:
        raise ValueError("budget must be in [0, 1]")
    review = np.zeros(len(scores), dtype=bool)
    review[np.argsort(-scores, kind="stable")[:int(len(scores)*budget)]] = True
    return review


def choose_actions(probabilities, tail_weight=0., alpha=.9):
    if not np.isfinite(tail_weight) or not 0 <= tail_weight <= 1:
        raise ValueError("tail_weight must be in [0, 1]")
    mean, tail = conditional_risk(probabilities, alpha)
    if mean.ndim != 2 or mean.shape[1] == 0:
        raise ValueError("Expected [images, actions, 4] probabilities")
    risk = (1-tail_weight)*mean + tail_weight*tail
    actions = risk.argmin(axis=1)
    return actions, risk[np.arange(len(actions)), actions]


def realized_loss(predictions, labels, severity):
    pred, y, s = map(np.asarray, (predictions, labels, severity))
    if pred.shape != y.shape or s.shape != y.shape or y.ndim != 1:
        raise ValueError("Predictions, labels, and severity must be aligned vectors")
    if not np.isin(pred, [0, 1]).all() or not np.isin(y, [0, 1]).all() or not np.isin(s, [0, 1, 2]).all():
        raise ValueError("Unknown class or severity label")
    return (pred != y)*SEVERITY_COST[s.astype(int)]


def metrics(predictions, labels, severity, review):
    loss = realized_loss(predictions, labels, severity)
    review = np.asarray(review)
    if review.dtype != bool or review.shape != loss.shape or len(loss) == 0:
        raise ValueError("review must be a nonempty aligned boolean mask")
    residual = np.where(review, 0., loss)
    errors = np.asarray(predictions) != labels
    severe_errors = errors & (np.asarray(severity) == 2)
    severe_total = int(severe_errors.sum())
    kept = ~review
    return dict(n=len(loss), review_fraction=float(review.mean()),
                selective_accuracy=float((~errors[kept]).mean()) if kept.any() else None,
                mean_cost=float(residual.mean()), cvar90=empirical_cvar(residual),
                severe_error_capture=float((severe_errors & review).sum()/severe_total) if severe_total else None,
                severe_errors_before_review=severe_total,
                severe_errors_remaining=int((severe_errors & kept).sum()),
                weighted_error_capture=float((loss-residual).sum()/loss.sum()) if loss.sum() else None)
