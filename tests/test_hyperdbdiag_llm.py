import unittest

import numpy as np

from hyperdbdiag_llm import (
    BlindedRemediationQualityReviewer,
    PostDiagnosisLLMAdvisor,
)
from hyperdbdiag_pipeline import EvidenceItem, HyperDBDiagPipeline
from hypergraph_core import LightweightHDiffusionClassifier


class PostDiagnosisLLMAdvisorTests(unittest.TestCase):
    def setUp(self):
        self.roots = ("lock_waits", "order_by")
        self.mechanisms = (
            EvidenceItem(
                "mechanism:lock",
                "mechanism_card",
                "Lock-wait pressure can indicate blocked transactional work.",
                ("lock_waits",),
            ),
            EvidenceItem(
                "mechanism:sort",
                "mechanism_card",
                "Ordering work can require temporary execution resources.",
                ("order_by",),
            ),
        )
        self.observations = (
            EvidenceItem(
                "trajectory:current",
                "query_metric_trajectory",
                "Current anonymous trajectory has elevated lock-wait and execution-pressure families.",
            ),
            EvidenceItem(
                "semantic:current",
                "semantic_observation",
                "Current anonymous query shape contains an ordering clause.",
                ("order_by",),
            ),
        )

    @staticmethod
    def _valid_response(payload):
        return {
            "action_type": "REVIEW_LOCK_CHAIN",
            "steps": ["Inspect the captured lock-wait chain and correlate its time window with the query shape."],
            "preconditions": ["Use a read-only diagnostic view from the same observation window."],
            "verification": ["Confirm whether lock-wait pressure falls when the blocking chain is absent."],
            "rollback": ["No runtime state is changed; discard the hypothesis if the timing does not align."],
            "evidence_ids": [
                payload["evidence"]["mechanism_cards"][0]["evidence_id"],
                payload["evidence"]["current_observations"][0]["evidence_id"],
            ],
            "risk_level": "LOW",
        }

    def test_accepts_evidence_cited_read_only_recommendation(self):
        seen = []

        def client(payload):
            seen.append(payload)
            return self._valid_response(payload)

        result = PostDiagnosisLLMAdvisor(client).advise(
            self.roots, self.mechanisms, self.observations
        )
        self.assertEqual(result.status, "accepted_recommendation")
        self.assertEqual(result.action_type, "REVIEW_LOCK_CHAIN")
        self.assertEqual(result.risk_level, "LOW")
        self.assertTrue(result.cited_evidence_ids)
        self.assertEqual(seen[0]["selected_root_set"], list(self.roots))
        self.assertTrue(seen[0]["rules"]["diagnosis_is_frozen"])
        self.assertFalse(seen[0]["rules"]["may_change_root_set"])

    def test_negated_change_instruction_is_not_treated_as_unsafe(self):
        def client(payload):
            response = self._valid_response(payload)
            response["preconditions"] = [
                "Do not create, drop, or alter indexes, predicates, configuration, sessions, or data during review."
            ]
            return response

        result = PostDiagnosisLLMAdvisor(client).advise(
            self.roots, self.mechanisms, self.observations
        )
        self.assertEqual(result.status, "accepted_recommendation")

    def test_missing_observation_citation_fails_closed(self):
        def client(payload):
            response = self._valid_response(payload)
            response["evidence_ids"] = [
                payload["evidence"]["mechanism_cards"][0]["evidence_id"]
            ]
            return response

        result = PostDiagnosisLLMAdvisor(client).advise(
            self.roots, self.mechanisms, self.observations
        )
        self.assertEqual(result.status, "fallback_invalid_recommendation")
        self.assertEqual(result.action_type, "NO_ACTIONABLE_RECOMMENDATION")

    def test_invalid_action_fails_closed(self):
        def client(payload):
            response = self._valid_response(payload)
            response["action_type"] = "CHANGE_DATABASE_CONFIGURATION"
            return response

        result = PostDiagnosisLLMAdvisor(client).advise(
            self.roots, self.mechanisms, self.observations
        )
        self.assertEqual(result.status, "fallback_invalid_recommendation")

    def test_observation_cannot_introduce_an_unselected_root(self):
        observations = (
            EvidenceItem(
                "semantic:other",
                "semantic_observation",
                "Anonymous query-shape fact for another proposed mechanism.",
                ("missing_index",),
            ),
        )
        with self.assertRaisesRegex(ValueError, "unselected root"):
            PostDiagnosisLLMAdvisor(lambda _: {}).advise(
                self.roots, self.mechanisms, observations
            )

    def test_destructive_recommendation_is_rejected(self):
        def client(payload):
            response = self._valid_response(payload)
            response["steps"] = ["KILL the blocking session before collecting more evidence."]
            return response

        result = PostDiagnosisLLMAdvisor(client).advise(
            self.roots, self.mechanisms, self.observations
        )
        self.assertEqual(result.status, "fallback_unsafe_recommendation")

    def test_explicit_abstention_is_accepted_only_when_empty(self):
        result = PostDiagnosisLLMAdvisor(
            lambda _: {
                "action_type": "NO_ACTIONABLE_RECOMMENDATION",
                "steps": [],
                "preconditions": [],
                "verification": [],
                "rollback": [],
                "evidence_ids": [],
                "risk_level": "LOW",
            }
        ).advise(self.roots, self.mechanisms, self.observations)
        self.assertEqual(result.status, "accepted_no_action")
        self.assertFalse(result.steps)

    def test_remediation_call_cannot_change_diagnosis_predictions(self):
        vectors = np.asarray(
            [[2.0, 0.0], [0.0, 2.0], [2.0, 2.0], [1.8, 0.1], [0.1, 1.8], [1.8, 1.8]]
        )
        labels = [["a"], ["b"], ["a", "b"], ["a"], ["b"], ["a", "b"]]
        groups = [0, 0, 0, 1, 1, 1]
        model = LightweightHDiffusionClassifier(decoder_n_estimators=11, seed=3).fit(
            vectors, labels, ("a", "b"), groups
        )
        pipeline = HyperDBDiagPipeline(("a", "b"), metric_model=model)
        before = pipeline.predict(vectors)
        PostDiagnosisLLMAdvisor(lambda payload: self._valid_response(payload)).advise(
            self.roots, self.mechanisms, self.observations
        )
        self.assertEqual(pipeline.predict(vectors), before)

    def test_blinded_quality_review_uses_no_truth_or_case_identity(self):
        recommendation = PostDiagnosisLLMAdvisor(
            lambda payload: self._valid_response(payload)
        ).advise(self.roots, self.mechanisms, self.observations)
        seen = []

        def client(payload):
            seen.append(payload)
            return {
                "evidence_grounding_score": 5,
                "root_relevance_score": 5,
                "actionability_score": 4,
                "verification_quality_score": 4,
                "issue_codes": ["NONE"],
                "confidence": "HIGH",
            }

        result = BlindedRemediationQualityReviewer(client).review(
            self.roots, self.mechanisms, self.observations, recommendation
        )
        self.assertTrue(result.quality_pass)
        self.assertEqual(result.status, "accepted_blinded_quality_review")
        packet_text = str(seen[0])
        self.assertNotIn("expected", packet_text)
        self.assertNotIn("case_id", packet_text)
        self.assertNotIn("dataset", packet_text)
        self.assertNotIn("method", packet_text)

    def test_blinded_quality_review_fails_critical_issue(self):
        recommendation = PostDiagnosisLLMAdvisor(
            lambda payload: self._valid_response(payload)
        ).advise(self.roots, self.mechanisms, self.observations)
        result = BlindedRemediationQualityReviewer(
            lambda _: {
                "evidence_grounding_score": 2,
                "root_relevance_score": 4,
                "actionability_score": 3,
                "verification_quality_score": 3,
                "issue_codes": ["UNSUPPORTED_FACT"],
                "confidence": "MEDIUM",
            }
        ).review(self.roots, self.mechanisms, self.observations, recommendation)
        self.assertFalse(result.quality_pass)


if __name__ == "__main__":
    unittest.main()
