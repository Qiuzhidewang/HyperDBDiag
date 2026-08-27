import unittest
from types import SimpleNamespace

import numpy as np

from structured_evidence_judge import StructuredEvidenceJudge


class _PoolH:
    candidates = ((0,), (1,), (0, 1))

    def predict_candidate_pool_indices(self, values):
        return np.tile(np.asarray([[0, 1]], dtype=np.int64), (len(values), 1))


class StructuredEvidenceJudgeTests(unittest.TestCase):
    def setUp(self):
        self.x = np.asarray(
            [[2.0, 0.0], [0.0, 2.0], [2.0, 2.0], [1.8, 0.1], [0.1, 1.8], [1.8, 1.8]],
            dtype=float,
        )
        self.labels = (("a",), ("b",), ("a", "b"), ("a",), ("b",), ("a", "b"))
        self.groups = (0, 0, 0, 1, 1, 1)

    def test_fit_uses_group_balanced_training_profiles(self):
        judge = StructuredEvidenceJudge().fit(
            self.x, self.labels, self.groups, ("a", "b"), ("m1", "m2")
        )
        metadata = judge.metadata()
        self.assertFalse(metadata["trained_second_diagnosis_model"])
        self.assertFalse(metadata["root_pair_failure_memory"])
        self.assertIn("group-balanced", metadata["profile_source"])

    def test_direct_semantic_evidence_exposes_one_challenger(self):
        judge = StructuredEvidenceJudge().fit(
            self.x, self.labels, self.groups, ("a", "b"), ("m1", "m2")
        )
        observations = ((SimpleNamespace(atoms=("atom_b",)),),)
        cards = {
            "a": {"semantic_observables": []},
            "b": {"semantic_observables": [{"atom": "atom_b", "role": "direct"}]},
        }
        decision = judge.propose(
            _PoolH(),
            np.asarray([[2.0, 0.0]]),
            ("candidate-a", "candidate-b", "candidate-ab"),
            observations,
            cards,
        )[0]
        self.assertEqual(decision.local_candidate_id, "candidate-a")
        self.assertEqual(decision.challenger_candidate_id, "candidate-b")
        self.assertEqual(decision.status, "direct_evidence_challenger")
        self.assertTrue(decision.reviewable)
        self.assertIn("atom_b", " ".join(decision.challenger_card.supporting_atoms))

    def test_without_independent_evidence_pool_is_retained(self):
        judge = StructuredEvidenceJudge().fit(
            self.x, self.labels, self.groups, ("a", "b"), ("m1", "m2")
        )
        decision = judge.propose(
            _PoolH(),
            np.asarray([[2.0, 0.0]]),
            ("candidate-a", "candidate-b", "candidate-ab"),
        )[0]
        self.assertEqual(decision.local_candidate_id, "candidate-a")
        self.assertIsNone(decision.challenger_candidate_id)
        self.assertEqual(decision.status, "broad_pool_retained")
        self.assertFalse(decision.reviewable)

    def test_profile_alignment_can_expose_challenger_without_sql_atoms(self):
        judge = StructuredEvidenceJudge().fit(
            self.x, self.labels, self.groups, ("a", "b"), ("m1", "m2")
        )
        cards = {
            "a": {"semantic_observables": []},
            "b": {"semantic_observables": []},
        }
        decision = judge.propose(
            _PoolH(),
            np.asarray([[0.0, 2.0]]),
            ("candidate-a", "candidate-b", "candidate-ab"),
            mechanism_cards=cards,
        )[0]
        self.assertEqual(decision.local_candidate_id, "candidate-a")
        self.assertEqual(decision.challenger_candidate_id, "candidate-b")
        self.assertEqual(decision.status, "profile_evidence_challenger")
        self.assertTrue(decision.reviewable)


if __name__ == "__main__":
    unittest.main()
