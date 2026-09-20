import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from radr.data import build_manifests, source_from_path
from radr.deployment import guarded_decision
from radr.models import LossMixture, fit_loss_model, loss_targets, predict_loss
from radr.risk import allocate, choose_actions


class PipelineTests(unittest.TestCase):
    def test_guard_falls_back_when_evidence_fails(self):
        base = np.array([[.5, .5], [.9, .1], [.7, .3], [.8, .2], [.6, .4]])
        actions, _, review, reason = guarded_decision(base, None, 1, False, .2)
        np.testing.assert_equal(actions, 0)
        np.testing.assert_equal(review, [True, False, False, False, False])
        self.assertEqual(reason, "candidate_not_supported_by_guard")

    def test_guard_does_not_generalize_across_capacity(self):
        _, _, _, reason = guarded_decision([[.5, .5]], None, 1, True, .4)
        self.assertEqual(reason, "budget_not_audited")

    def test_guard_uses_supported_candidate(self):
        q = np.array([[[0, 0, 0, 1], [1, 0, 0, 0]]]*5)
        actions, _, review, reason = guarded_decision([[.5, .5]]*5, q, 0, True, .2)
        np.testing.assert_equal(actions, 1)
        self.assertEqual(review.sum(), 1)
        self.assertEqual(reason, "candidate_passed_guard_at_this_budget")

    def test_mixture_is_valid_distribution_and_trains(self):
        torch.set_num_threads(2)
        rng = np.random.default_rng(1)
        x = rng.normal(size=(100, 5)).astype("float32")
        targets = np.column_stack([np.where(x[:, 0] > 0, 3, 0), np.ones(100)]).astype(int)
        model, history = fit_loss_model(x[:80], targets[:80], x[80:], targets[80:], 1, epochs=60)
        q = predict_loss(model, x)
        self.assertEqual(q.shape, (100, 2, 4))
        np.testing.assert_allclose(q.sum(-1), 1, atol=1e-6)
        self.assertLess(history[-1]["train_nll"], history[0]["train_nll"])
        actions, scores = choose_actions(q)
        self.assertEqual(actions.shape, (100,))
        self.assertEqual(allocate(scores, .2).sum(), 20)

    def test_loss_targets(self):
        result = loss_targets(np.array([[0, 1], [1, 0]]), np.array([0, 0]), np.array([2, 1]))
        np.testing.assert_equal(result, [[0, 3], [2, 0]])

    def test_content_overlap_and_missing_files_are_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"a.jpg").write_bytes(b"image-a")
            (root/"copy.jpg").write_bytes(b"image-a")
            (root/"b.jpg").write_bytes(b"image-b")
            for split, paths in {"train": ["a.jpg", "copy.jpg"], "dev": ["copy.jpg", "b.jpg"], "test": ["missing.jpg"]}.items():
                pd.DataFrame(dict(image_path=paths, informative=["informative"]*len(paths),
                                  damage_severity=["severe"]*len(paths))).to_csv(root/f"MEDIC_{split}.tsv", sep="\t", index=False)
            frames, audit = build_manifests(root)
            self.assertEqual(len(frames["train"]), 1)
            self.assertEqual(audit["train"]["duplicate_within_split"], 1)
            self.assertEqual(audit["dev"]["overlap_earlier_split"], 1)
            self.assertEqual(audit["test"]["missing_files"], 1)

    def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame(dict(image_path=["../escape.jpg"], informative=["informative"],
                              damage_severity=["severe"])).to_csv(root/"MEDIC_train.tsv", sep="\t", index=False)
            with self.assertRaises(ValueError):
                build_manifests(root)

    def test_source_is_path_based(self):
        self.assertEqual(source_from_path("data/crisismmd/event/image.jpg"), "crisismmd")


if __name__ == "__main__":
    unittest.main()
