import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from hyperdbdiag_ablation import (
    AblationDataset,
    AblationSplit,
    _StructuredHandoff,
    _allocate_llm_review_budget,
    _convergence_summary,
    _direct_handoff,
    _new_h_model,
    _llm_policy_from_oof,
    _load_dbmags,
    _load_root_mechanism_cards,
    _relation_policies_from_oof,
    _retain_semantic_challengers,
    _root_profile_items,
    _run_remediation_advice,
    _select_remediation_jobs,
    _run_dataset,
    _training_oof_pair_audit,
)
from hyperdbdiag_pipeline import EvidenceItem, LLMReviewEvidence
from sql_semantic_evidence import SemanticObservation


class HyperDBDiagAblationTests(unittest.TestCase):
    def test_dbmags_uses_only_registered_replicate_holdout(self):
        cards, _ = _load_root_mechanism_cards()
        root = Path("data/dbmags_interaction_v10_metric_only")
        dataset = _load_dbmags(root, cards)
        self.assertEqual(len(dataset.splits), 6)
        self.assertTrue(
            all(len(split.train_ids) == 550 for split in dataset.splits)
        )
        self.assertTrue(all(len(split.eval_ids) == 110 for split in dataset.splits))
        self.assertEqual(
            dataset.metadata["outer_split"],
            "leave_one_replicate_index_out",
        )
        self.assertEqual(dataset.metadata["semantic_evidence"]["case_count"], 396)
        self.assertEqual(
            dataset.metadata["semantic_evidence"]["case_inventory_count"], 660
        )

    def test_direct_handoff_uses_hypergraph_pool_without_local_evidence(self):
        model = types.SimpleNamespace(
            candidates=((0,), (1,), (0, 1)),
            predict_candidate_pool_indices=lambda values: np.asarray(
                [[0, 2], [1, 0]], dtype=np.int64
            ),
        )
        handoff = _direct_handoff(
            model,
            np.asarray([[1.0], [2.0]]),
            ("candidate-0", "candidate-1", "candidate-2"),
            ("a", "b"),
        )
        self.assertEqual(handoff.local_indices, (0, 1))
        self.assertEqual(handoff.alternative_indices, (2, 0))
        self.assertEqual(handoff.candidate_cards[0][1].root_labels, ("a", "b"))
        self.assertFalse(handoff.candidate_cards[0][1].supporting_atoms)
        self.assertFalse(handoff.conflicts)

    def test_convergence_summary_uses_set_level_coverage(self):
        summary = _convergence_summary(
            (
                {"expected_labels": ["a"], "predicted_labels": ["a", "b"]},
                {"expected_labels": ["a", "b"], "predicted_labels": ["a"]},
            )
        )
        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["mean_predicted_roots"], 1.5)
        self.assertEqual(summary["full_coverage"], 0.5)
        self.assertEqual(summary["root_count_accuracy"], 0.0)
        self.assertAlmostEqual(summary["component_redundancy"], 1.0 / 3.0)

    @staticmethod
    def _remediation_evidence(*roots):
        return LLMReviewEvidence(
            query_items=(
                EvidenceItem(
                    "query",
                    "query_metric_trajectory",
                    "Current metric trajectory is elevated.",
                ),
            ),
            training_items=tuple(
                EvidenceItem(
                    f"profile:{root}",
                    "training_root_profile",
                    f"Training profile for {root}.",
                    (root,),
                )
                for root in roots
            ),
            mechanism_items=tuple(
                EvidenceItem(
                    f"mechanism:{root}",
                    "mechanism_card",
                    f"Mechanism card for {root}.",
                    (root,),
                )
                for root in roots
            ),
        )

    def test_remediation_sample_is_balanced_without_truth(self):
        evidence_a = self._remediation_evidence("a")
        evidence_b = self._remediation_evidence("b")
        jobs = (
            ("case-a1", ("a",), evidence_a),
            ("case-a2", ("a",), evidence_a),
            ("case-b1", ("b",), evidence_b),
        )
        self.assertEqual(_select_remediation_jobs(jobs, 2), (0, 2))

    def test_remediation_advice_is_reported_without_diagnosis_mutation(self):
        evidence_a = self._remediation_evidence("a")
        evidence_b = self._remediation_evidence("b")
        jobs = (
            ("case-a", ("a",), evidence_a),
            ("case-b", ("b",), evidence_b),
        )

        def client(payload):
            if payload["task"] == "blinded_remediation_quality_review":
                return {
                    "evidence_grounding_score": 5,
                    "root_relevance_score": 5,
                    "actionability_score": 4,
                    "verification_quality_score": 4,
                    "issue_codes": ["NONE"],
                    "confidence": "HIGH",
                }
            return {
                "action_type": "INSPECT_QUERY_PLAN",
                "steps": ["Inspect the captured plan shape in the same observation window."],
                "preconditions": ["Use a read-only diagnostic view."],
                "verification": ["Check whether the observed metric family follows the mechanism."],
                "rollback": ["No runtime state is changed; discard an unsupported hypothesis."],
                "evidence_ids": [
                    payload["evidence"]["mechanism_cards"][0]["evidence_id"],
                    payload["evidence"]["current_observations"][0]["evidence_id"],
                ],
                "risk_level": "LOW",
            }

        with patch("hyperdbdiag_ablation.LLM_MAX_WORKERS", 1):
            rows, audit = _run_remediation_advice(client, jobs, budget=1)
        self.assertEqual(rows["case-a"]["status"], "accepted_recommendation")
        self.assertEqual(
            rows["case-b"]["status"], "not_sampled_fixed_evaluation_budget"
        )
        self.assertEqual(audit["accepted_recommendation_count"], 1)
        self.assertEqual(audit["diagnosis_mutation_count"], 0)
        self.assertEqual(
            audit["blinded_quality_review"]["quality_pass_count"], 1
        )

    def test_semantic_gate_only_retains_evidence_supported_challenger(self):
        dataset = AblationDataset(
            name="semantic-gate",
            labels=("a", "b"),
            feature_count=1,
            splits=(),
            activation_threshold=0.5,
            decoder_criterion="gini",
            decoder_n_estimators=11,
            metadata={},
            training_selection={},
            evidence_eligibility={},
            require_discriminating_semantic_for_llm=True,
        )
        handoff = _StructuredHandoff(
            hypergraph_indices=(0, 0),
            local_indices=(0, 0),
            alternative_indices=(1, 1),
            candidate_cards=(("cards",), ("cards",)),
            conflicts=(),
        )
        marker_a = SemanticObservation(("atom_a",), ("catalog_plan",))
        marker_b = SemanticObservation(("atom_b",), ("catalog_plan",))
        retained = _retain_semantic_challengers(
            dataset,
            types.SimpleNamespace(candidates=((0,), (1,))),
            handoff,
            ((marker_b,), (marker_a,)),
            (("a",), ("b",), ("a",), ("b",)),
            (0, 0, 1, 1),
            ((marker_a,), (marker_b,), (marker_a,), (marker_b,)),
        )
        self.assertEqual(retained.alternative_indices, (1, None))
        self.assertEqual(retained.candidate_cards, (("cards",), ()))

    def test_root_profiles_learn_anonymous_semantic_associations(self):
        profiles = _root_profile_items(
            np.asarray([[1.0], [0.0], [1.1], [0.1]]),
            (("a",), ("b",), ("a",), ("b",)),
            (0, 0, 1, 1),
            ("a", "b"),
            ("metric",),
            (
                (SemanticObservation(("mixed_access_paths",), ("catalog_plan",)),),
                (SemanticObservation(("sequential_scan_only_path",), ("catalog_plan",)),),
                (SemanticObservation(("mixed_access_paths",), ("catalog_plan",)),),
                (SemanticObservation(("sequential_scan_only_path",), ("catalog_plan",)),),
            ),
        )
        by_root = {profile.root_labels[0]: profile.summary for profile in profiles}
        self.assertIn("mixed_access_paths", by_root["a"])
        self.assertIn("sequential_scan_only_path", by_root["b"])
        self.assertIn("not a decision rule", by_root["a"])
        self.assertIn("every training group", by_root["a"])

    def test_review_budget_is_even_and_bounded(self):
        self.assertEqual(_allocate_llm_review_budget((10, 0, 1, 3), 8), (4, 0, 1, 3))
        self.assertEqual(_allocate_llm_review_budget((1, 1), 8), (1, 1))

    def test_llm_override_policy_requires_cross_group_positive_value(self):
        self.assertFalse(
            _llm_policy_from_oof(
                ({"corrected": 2, "harmed": 0, "changed": 2},),
                minimum_decisions=2,
                minimum_active_groups=2,
                sign_test_alpha=0.10,
            )["allow_override"]
        )
        self.assertTrue(
            _llm_policy_from_oof(
                (
                    {"corrected": 3, "harmed": 0, "changed": 3},
                    {"corrected": 2, "harmed": 0, "changed": 2},
                ),
                minimum_decisions=2,
                minimum_active_groups=2,
                sign_test_alpha=0.10,
            )["allow_override"]
        )

    def test_relation_policy_is_root_agnostic(self):
        policies = _relation_policies_from_oof(
            {
                "CONFLICT": (
                    {"corrected": 3, "harmed": 0, "changed": 3},
                    {"corrected": 2, "harmed": 0, "changed": 2},
                )
            }
        )
        self.assertTrue(policies["CONFLICT"]["allow_override"])
        self.assertFalse(policies["COMPLEMENTARY"]["allow_override"])

    def test_training_pair_audit_only_uses_supplied_training_truth(self):
        handoff = _StructuredHandoff(
            hypergraph_indices=(0, 0, 0),
            local_indices=(0, 0, 0),
            alternative_indices=(1, 1, None),
            candidate_cards=((), (), ()),
            conflicts=(),
        )
        audit = _training_oof_pair_audit(
            (("b",), ("a",), ("a", "b")),
            ("a", "b"),
            types.SimpleNamespace(candidates=((0,), (1,), (0, 1))),
            handoff,
        )
        self.assertEqual(audit["eligible_pair_count"], 2)
        self.assertEqual(audit["recoverable_error_count"], 1)

    def _dataset(self):
        train_x = np.asarray(
            [[2.0, 0.0], [0.0, 2.0], [2.0, 2.0], [1.8, 0.1], [0.1, 1.8], [1.8, 1.8]],
            dtype=float,
        )
        labels = (("a",), ("b",), ("a", "b"), ("a",), ("b",), ("a", "b"))
        split = AblationSplit(
            split_id="test",
            train_ids=tuple(f"train-{index}" for index in range(6)),
            eval_ids=("eval-a", "eval-b", "eval-ab"),
            train_x=train_x,
            eval_x=train_x[:3],
            train_labels=labels,
            eval_labels=labels[:3],
            oof_groups=(0, 0, 0, 1, 1, 1),
        )
        return AblationDataset(
            name="synthetic",
            labels=("a", "b"),
            feature_count=2,
            splits=(split,),
            activation_threshold=0.5,
            decoder_criterion="gini",
            decoder_n_estimators=11,
            metadata={},
            training_selection={},
            evidence_eligibility={},
            feature_names=("metric_one", "metric_two"),
            mechanism_cards={"a": {"summary": "a"}, "b": {"summary": "b"}},
        )

    def test_missing_llm_configuration_keeps_metric_rows_reproducible(self):
        with patch.dict(
            "os.environ",
            {
                "HYPERDBDIAG_LLM_API_KEY": "",
                "HYPERDBDIAG_LLM_MODEL": "",
                "HYPERDBDIAG_LLM_BASE_URL": "",
            },
            clear=False,
        ):
            report = _run_dataset(self._dataset(), seed=3)
        self.assertEqual(
            report["methods"]["hypergraph"]["overall"]["exact_set_accuracy"],
            report["methods"]["hypergraph_local_judge"]["overall"]["exact_set_accuracy"],
        )
        self.assertEqual(
            report["methods"]["hyperdbdiag"]["stage"]["estimand_status"],
            "not_run_missing_explicit_llm_configuration",
        )

    def test_epdg_is_disabled_without_registered_metric_or_semantic_paths(self):
        model = _new_h_model(self._dataset(), seed=3)
        self.assertFalse(model.enable_epdg)


if __name__ == "__main__":
    unittest.main()
