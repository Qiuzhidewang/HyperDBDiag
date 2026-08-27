import unittest

import numpy as np

from hyperdbdiag_pipeline import (
    CandidateBoundLLMReviewer,
    CandidateRecord,
    EvidenceItem,
    HyperDBDiagPipeline,
    LLMReviewEvidence,
)
from hypergraph_core import LightweightHDiffusionClassifier


class CandidateBoundLLMReviewerTests(unittest.TestCase):
    def setUp(self):
        self.local = CandidateRecord("candidate-000", ("a",))
        self.alternative = CandidateRecord("candidate-001", ("a", "b"))
        self.evidence = LLMReviewEvidence(
            query_items=(EvidenceItem("query-trajectory", "query_metric_trajectory", "current trajectory"),),
            training_items=(EvidenceItem("training-profile:b", "training_root_profile", "profile b", ("b",)),),
            mechanism_items=(EvidenceItem("mechanism-card:b", "mechanism_card", "mechanism b", ("b",)),),
            candidate_items=(
                EvidenceItem("candidate-card:local", "candidate_evidence_card", "local card", ("a",)),
                EvidenceItem("candidate-card:alternative", "candidate_evidence_card", "alternative card", ("a", "b")),
            ),
        )

    @staticmethod
    def _response(payload, selected="B", relation="COMPLEMENTARY"):
        ids = [item["evidence_id"] for group in payload["evidence"].values() for item in group]
        return {
            "action": "SELECT_CANDIDATE",
            "candidate_id": selected,
            "evidence_ids": ids,
            "reason_code": "evidence_closure",
            "relation_type": relation,
            "recommendation": (
                "RETAIN_COMPLEMENTARY_ROOTS" if relation == "COMPLEMENTARY" else "REPLACE_CONFLICTING_ROOTS"
            ),
        }

    def test_selected_existing_candidate_is_accepted(self):
        result = CandidateBoundLLMReviewer(
            client=lambda payload: self._response(payload), allow_override=True
        ).review(self.local, self.alternative, self.evidence)
        self.assertEqual(result.status, "accepted_candidate_selection")
        self.assertEqual(result.final_candidate_id, self.alternative.candidate_id)
        self.assertEqual(result.response_count, 1)

    def test_invalid_selection_fails_closed(self):
        def client(payload):
            response = self._response(payload)
            response["evidence_ids"] = []
            return response

        result = CandidateBoundLLMReviewer(client=client, allow_override=True).review(
            self.local, self.alternative, self.evidence
        )
        self.assertEqual(result.status, "fallback_invalid_candidate_selection")
        self.assertEqual(result.final_candidate_id, self.local.candidate_id)
        self.assertEqual(result.response_count, 1)

    def test_transport_failure_is_not_counted_as_response(self):
        result = CandidateBoundLLMReviewer(
            client=lambda _payload: (_ for _ in ()).throw(RuntimeError("offline")),
            allow_override=True,
        ).review(self.local, self.alternative, self.evidence)
        self.assertEqual(result.status, "fallback_transport_or_timeout")
        self.assertEqual(result.response_count, 0)

    def test_direct_semantic_requirement_is_fail_closed(self):
        evidence = LLMReviewEvidence(
            query_items=self.evidence.query_items,
            training_items=self.evidence.training_items,
            mechanism_items=self.evidence.mechanism_items,
            candidate_items=self.evidence.candidate_items,
            semantic_items=(
                EvidenceItem(
                    "semantic-01",
                    "semantic_observation",
                    "Contextual/non-discriminating observation",
                ),
            ),
            requires_direct_evidence=True,
        )
        called = []
        result = CandidateBoundLLMReviewer(
            client=lambda payload: called.append(payload), allow_override=True
        ).review(self.local, self.alternative, evidence)
        self.assertEqual(result.status, "skipped_no_direct_semantic_evidence")
        self.assertFalse(called)

    def test_selection_cannot_remove_semantically_supported_root(self):
        evidence = LLMReviewEvidence(
            query_items=self.evidence.query_items,
            training_items=self.evidence.training_items,
            mechanism_items=self.evidence.mechanism_items,
            candidate_items=self.evidence.candidate_items,
            semantic_items=(
                EvidenceItem(
                    "semantic-01",
                    "semantic_observation",
                    "Strong cross-group training-associated candidate-discriminating observation",
                    ("b",),
                ),
            ),
            requires_direct_evidence=True,
        )

        def client(payload):
            ids = [
                item["evidence_id"]
                for group in payload["evidence"].values()
                for item in group
            ]
            return {
                "action": "SELECT_CANDIDATE",
                "candidate_id": "A",
                "evidence_ids": ids,
                "reason_code": "candidate_contradicted",
                "relation_type": "COVERAGE",
                "recommendation": "PRUNE_COVERED_ROOTS",
            }

        result = CandidateBoundLLMReviewer(client=client, allow_override=True).review(
            self.local, self.alternative, evidence
        )
        self.assertEqual(result.status, "fallback_invalid_candidate_selection")
        self.assertEqual(result.final_candidate_id, self.local.candidate_id)


class HyperDBDiagPipelineTests(unittest.TestCase):
    def test_pipeline_matches_hypergraph_without_llm_client(self):
        vectors = np.asarray(
            [[2.0, 0.0], [0.0, 2.0], [2.0, 2.0], [1.8, 0.1], [0.1, 1.8], [1.8, 1.8]]
        )
        roots = [["a"], ["b"], ["a", "b"], ["a"], ["b"], ["a", "b"]]
        groups = [0, 0, 0, 1, 1, 1]
        model = LightweightHDiffusionClassifier(decoder_n_estimators=11, seed=3).fit(
            vectors, roots, ("a", "b"), groups
        )
        pipeline = HyperDBDiagPipeline(("a", "b"), metric_model=model)
        expected = model.predict(vectors, ("a", "b"))
        self.assertEqual(pipeline.predict(vectors), expected)


if __name__ == "__main__":
    unittest.main()
