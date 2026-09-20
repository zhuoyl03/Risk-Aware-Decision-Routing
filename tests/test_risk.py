import unittest

import numpy as np

from radr.risk import (allocate, choose_actions, conditional_risk, empirical_cvar,
                       metrics, realized_loss)


class RiskTests(unittest.TestCase):
    def test_fractional_tail(self):
        # Worst 1.5 observations: 8 plus half of 3.
        self.assertAlmostEqual(empirical_cvar([0, 1, 3, 8], .625), 9.5/1.5)

    def test_tail_endpoints(self):
        self.assertEqual(empirical_cvar([0, 1, 3, 8], 0), 3)
        self.assertEqual(empirical_cvar([0, 1, 3, 8], .99), 8)

    def test_conditional_tail_exact(self):
        mean, tail = conditional_risk([.8, .1, .05, .05], .9)
        self.assertAlmostEqual(float(mean), .65)
        self.assertAlmostEqual(float(tail), 5.5)

    def test_tail_matches_empirical_distribution(self):
        for alpha in [0, .5, .625, .9, .99]:
            _, tail = conditional_risk([.25]*4, alpha)
            self.assertAlmostEqual(float(tail), empirical_cvar([0, 1, 3, 8], alpha))

    def test_invalid_probabilities(self):
        for p in [[1, 0], [1, -1, 0, 1], [np.nan, 0, 0, 1], [0, 0, 0, 0]]:
            with self.assertRaises(ValueError):
                conditional_risk(p)

    def test_invalid_loss_and_alpha(self):
        for x in [[], [np.nan], [-1], [[1]]]:
            with self.assertRaises(ValueError):
                empirical_cvar(x)
        for a in [-.1, 1, np.nan]:
            with self.assertRaises(ValueError):
                empirical_cvar([1], a)

    def test_exact_capacity_and_ties(self):
        np.testing.assert_array_equal(allocate([1, 1, 1, 1, 1], .4), [1, 1, 0, 0, 0])
        self.assertEqual(allocate(np.arange(11), .2).sum(), 2)
        self.assertEqual(allocate([1, 2], 0).sum(), 0)
        self.assertEqual(allocate([1, 2], 1).sum(), 2)

    def test_invalid_allocation(self):
        for budget in [-1, 1.1, np.nan]:
            with self.assertRaises(ValueError):
                allocate([1, 2], budget)
        with self.assertRaises(ValueError):
            allocate([np.inf], .1)

    def test_tail_can_change_expert(self):
        q = [[[.9, 0, 0, .1], [0, 1, 0, 0]]]
        # Expert 0: mean .8, CVaR8. Expert1: mean1, CVaR1.
        self.assertEqual(choose_actions(q, 0)[0][0], 0)
        self.assertEqual(choose_actions(q, 1)[0][0], 1)

    def test_label_free_choice_contract(self):
        import inspect
        self.assertEqual(list(inspect.signature(choose_actions).parameters),
                         ["probabilities", "tail_weight", "alpha"])

    def test_loss_mapping(self):
        np.testing.assert_equal(realized_loss([0, 0, 0], [1, 1, 1], [0, 1, 2]), [1, 3, 8])
        np.testing.assert_equal(realized_loss([1, 1], [1, 1], [2, 2]), [0, 0])

    def test_invalid_labels_fail(self):
        with self.assertRaises(ValueError):
            realized_loss([0], [1], [3])

    def test_selective_and_system_risk_differ(self):
        result = metrics(np.array([0, 0]), np.array([1, 0]), np.array([2, 0]), np.array([True, False]))
        self.assertEqual(result["mean_cost"], 0)
        self.assertEqual(result["selective_accuracy"], 1)
        self.assertEqual(result["severe_error_capture"], 1)

    def test_empty_coverage_is_not_perfect_accuracy(self):
        m = metrics([0], [1], [2], np.array([True]))
        self.assertIsNone(m["selective_accuracy"])

    def test_no_severe_errors_has_no_capture_denominator(self):
        m = metrics([0], [0], [2], np.array([False]))
        self.assertIsNone(m["severe_error_capture"])


if __name__ == "__main__":
    unittest.main()
